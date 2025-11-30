import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import torch.nn.init as init
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DeepSets(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(DeepSets, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x: (batch, set_size, embedding_dim)
        x = F.relu(self.fc1(x))
        x = x.mean(dim=1) # Mean Pooling (Permutation Invariant)
        x = F.relu(self.fc2(x))
        return x

class Actor(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, action_dim, slate_size=1, log_std_min=-20, log_std_max=2, item_embeddings=None, gru_state_dict=None):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.slate_size = slate_size
        self.action_dim = action_dim
        
        if item_embeddings is not None:
            self.click_embedding = nn.Embedding.from_pretrained(item_embeddings, freeze=False, padding_idx=0)
        else:
            self.click_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
            
        self.deepsets = DeepSets(embedding_dim, hidden_size)
        
        # Input to GRU: Click Embedding + Slate Embedding (DeepSets)
        # Note: GRU4Rec usually takes just embedding_dim. Here we have embedding_dim + hidden_size because we concat slate history.
        # If we want to load pretrained GRU4Rec weights, we have a dimension mismatch if GRU4Rec was trained only on clicks.
        # GRU4Rec input: embedding_dim.
        # Actor GRU input: embedding_dim + hidden_size.
        # We cannot directly load weights for input-to-hidden weights.
        # However, we CAN load hidden-to-hidden weights.
        
        self.click_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)
        
        if gru_state_dict is not None:
            # Load compatible weights (Hidden-to-Hidden)
            # GRU weights: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
            with torch.no_grad():
                self.click_gru.weight_hh_l0.copy_(gru_state_dict['weight_hh_l0'])
                self.click_gru.bias_hh_l0.copy_(gru_state_dict['bias_hh_l0'])
                self.click_gru.bias_ih_l0.copy_(gru_state_dict['bias_ih_l0'])
                # weight_ih_l0 is mismatched size, so we keep random init for the new input parts
                # or we could copy the part corresponding to click embeddings if we structured it that way.
                # For now, partial loading is better than nothing.
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Auto-Regressive Components
        # Input to Slate GRU: Previous Action Embedding (action_dim)
        # Hidden State: Initialized with User State (hidden_size)
        self.slate_gru = nn.GRUCell(action_dim, hidden_size)
        
        # Output mean and log_std for ONE slot
        self.action_head = nn.Linear(hidden_size, action_dim * 2)
        
        # Start Token (Learnable)
        self.start_token = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, state, deterministic=False):
        # state: (batch, seq_len, 1 + slate_size)
        
        clicked_items = state[:, :, 0] # (batch, seq_len)
        offered_slates = state[:, :, 1:] # (batch, seq_len, slate_size)
        
        # 1. Embed Clicked Items
        embedded_click = self.click_embedding(clicked_items) # (batch, seq_len, embed_dim)
        
        # 2. Embed Offered Slates (DeepSets)
        batch_size, seq_len, slate_size = offered_slates.shape
        offered_slates_flat = offered_slates.reshape(-1, slate_size)
        embedded_slates_flat = self.click_embedding(offered_slates_flat)
        
        deepsets_out_flat = self.deepsets(embedded_slates_flat)
        deepsets_out = deepsets_out_flat.reshape(batch_size, seq_len, -1)
        
        # 3. Concatenate and Feed to User GRU
        gru_input = torch.cat([embedded_click, deepsets_out], dim=2)
        _, hidden_click = self.click_gru(gru_input)
        user_context = hidden_click[-1] # (batch, hidden_size)
        
        # 4. Auto-Regressive Generation
        x = F.relu(self.fc1(user_context))
        x = F.relu(self.fc2(x))
        
        # Initialize Slate GRU hidden state with User Context
        h_slate = x # (batch, hidden_size)
        
        # Initial Input: Start Token
        current_input = self.start_token.repeat(batch_size, 1) # (batch, action_dim)
        
        means = []
        log_stds = []
        actions = []
        
        for i in range(self.slate_size):
            h_slate = self.slate_gru(current_input, h_slate)
            
            out = self.action_head(h_slate) # (batch, action_dim * 2)
            mean, log_std = torch.chunk(out, 2, dim=1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            means.append(mean)
            log_stds.append(log_std)
            
            # Sample action for next step input
            std = log_std.exp()
            normal = Normal(mean, std)
            if deterministic:
                z = mean
            else:
                z = normal.rsample()
            action = torch.tanh(z)
            
            actions.append(action)
            current_input = action # Feed back as input
            
        # Stack outputs
        mean_stack = torch.stack(means, dim=1) # (batch, slate_size, action_dim)
        log_std_stack = torch.stack(log_stds, dim=1)
        action_stack = torch.stack(actions, dim=1)
        
        return mean_stack, log_std_stack, action_stack
    
    def sample(self, state, epsilon=1e-6):
        mean, log_std, action = self.forward(state, deterministic=False)
        std = log_std.exp()
        
        # Re-calculate log_prob (since we already sampled in forward, we can just use the params)
        # Note: The 'action' returned by forward is already tanh(z)
        # We need z to calculate log_prob correctly for tanh transform
        # But forward didn't return z. Let's reconstruct or just use the distribution.
        # Actually, to be precise, we should probably do sampling here or return z from forward.
        # For simplicity, let's assume we can compute log_prob from mean/std and the action (which is tanh(z)).
        # Inverse tanh: z = atanh(action)
        # But action might be clipped or numerically unstable at boundaries.
        # Better to return z from forward if needed.
        
        # Let's modify forward to return z or just do the log_prob calc here if we trust the flow.
        # Wait, forward is doing the sampling loop. If we want log_prob, we need the distribution parameters.
        # We have mean and log_std.
        
        normal = Normal(mean, std)
        
        # We need the z that produced 'action'.
        # Since we used rsample in forward, 'action' corresponds to some z.
        # However, we don't have that z.
        # If we re-sample here, it's a DIFFERENT action sequence than what 'forward' produced internally for the loop.
        # CRITICAL: The loop in forward depends on the sampled action.
        # So 'action' returned by forward IS the trajectory.
        # We just need to compute its log_prob.
        
        # To compute log_prob of tanh(z), we need z.
        # z = atanh(action).
        # To avoid numerical issues:
        action_bound = 0.999999
        action_clipped = torch.clamp(action, -action_bound, action_bound)
        z = 0.5 * torch.log((1 + action_clipped) / (1 - action_clipped))
        
        log_prob = normal.log_prob(z) - torch.log(1 - action_clipped.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=2) # Sum over action_dim -> (batch, slate_size)
        
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, action_dim, item_embeddings=None, gru_state_dict=None):
        super(Critic, self).__init__()
        if item_embeddings is not None:
            self.click_embedding = nn.Embedding.from_pretrained(item_embeddings, freeze=False, padding_idx=0)
        else:
            self.click_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
            
        self.deepsets = DeepSets(embedding_dim, hidden_size)
        self.click_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)
        
        if gru_state_dict is not None:
            with torch.no_grad():
                self.click_gru.weight_hh_l0.copy_(gru_state_dict['weight_hh_l0'])
                self.click_gru.bias_hh_l0.copy_(gru_state_dict['bias_hh_l0'])
                self.click_gru.bias_ih_l0.copy_(gru_state_dict['bias_ih_l0'])
        
        # Q1 architecture (Takes SINGLE item action)
        self.l1 = nn.Linear(hidden_size + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(hidden_size + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        # state: (batch, seq_len, 1 + slate_size)
        # action: (batch, action_dim) OR (batch, slate_size, action_dim)
        
        clicked_items = state[:, :, 0] # (batch, seq_len)
        offered_slates = state[:, :, 1:] # (batch, seq_len, slate_size)
        
        # 1. Embed Clicked Items
        embedded_click = self.click_embedding(clicked_items) # (batch, seq_len, embed_dim)
        
        # 2. Embed Offered Slates (DeepSets)
        batch_size, seq_len, slate_size = offered_slates.shape
        offered_slates_flat = offered_slates.reshape(-1, slate_size)
        embedded_slates_flat = self.click_embedding(offered_slates_flat)
        
        deepsets_out_flat = self.deepsets(embedded_slates_flat)
        deepsets_out = deepsets_out_flat.reshape(batch_size, seq_len, -1)
        
        # 3. Concatenate and Feed to GRU
        gru_input = torch.cat([embedded_click, deepsets_out], dim=2)
        
        _, hidden_click = self.click_gru(gru_input)
        click_context = hidden_click[-1] # (batch, hidden_size)
        
        # If action has slate dimension, we need to handle it
        if action.dim() == 3:
            batch_size, slate_size, action_dim = action.shape
            # Expand context: (batch, slate_size, hidden_size)
            click_context = click_context.unsqueeze(1).expand(-1, slate_size, -1)
            # Flatten: (batch * slate_size, hidden_size)
            click_context = click_context.reshape(-1, click_context.size(-1))
            # Flatten action: (batch * slate_size, action_dim)
            action = action.reshape(-1, action_dim)
            
            xu = torch.cat([click_context, action], 1)
            
            x1 = F.relu(self.l1(xu))
            x1 = F.relu(self.l2(x1))
            q1 = self.l3(x1) # (batch * slate_size, 1)
            
            x2 = F.relu(self.l4(xu))
            x2 = F.relu(self.l5(x2))
            q2 = self.l6(x2)
            
            # Reshape back to (batch, slate_size)
            q1 = q1.view(batch_size, slate_size)
            q2 = q2.view(batch_size, slate_size)
            
        else:
            # Standard single item case (used in KNN evaluation)
            xu = torch.cat([click_context, action], 1)
            
            x1 = F.relu(self.l1(xu))
            x1 = F.relu(self.l2(x1))
            q1 = self.l3(x1)
            
            x2 = F.relu(self.l4(xu))
            x2 = F.relu(self.l5(x2))
            q2 = self.l6(x2)
        
        return q1, q2

class SACAgent:
    def __init__(self, num_items, embedding_dim, hidden_size, action_dim, slate_size=1, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, alpha=0.2, device='cpu', item_embeddings=None, bc_weight=0.0, finetune_embeddings=False, gru_state_dict=None):
        self.slate_size = slate_size
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.num_items = num_items
        self.bc_weight = bc_weight
        
        self.actor = Actor(num_items, embedding_dim, hidden_size, action_dim, slate_size, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict).to(device)
        self.critic = Critic(num_items, embedding_dim, hidden_size, action_dim, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict).to(device)
        self.critic_target = Critic(num_items, embedding_dim, hidden_size, action_dim, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Automatic Entropy Tuning
        self.target_entropy = -action_dim * slate_size # Heuristic: -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        self.alpha = self.log_alpha.exp() # Initial alpha
        
        if item_embeddings is not None:
            # If finetuning, we need to make it a Parameter
            if finetune_embeddings:
                self.item_embeddings = nn.Parameter(F.normalize(item_embeddings, p=2, dim=1).to(device))
                self.item_embeddings.requires_grad = True
                # Add to critic optimizer
                self.critic_optimizer = optim.Adam(list(self.critic.parameters()) + [self.item_embeddings], lr=lr_critic)
            else:
                self.item_embeddings = F.normalize(item_embeddings, p=2, dim=1).to(device)
                self.item_embeddings.requires_grad = False
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        else:
            self.item_embeddings = None
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, state, evaluate=False, slate_size=None, allowed_items=None, k_neighbors=20):
        # state: list (seq_len)
        if slate_size is None:
            slate_size = self.slate_size
            
        state_tensor = torch.LongTensor([state]).to(self.device)
        
        if evaluate:
            self.actor.eval()
            with torch.no_grad():
                _, _, action_emb = self.actor(state_tensor, deterministic=True)
                # action_emb: (1, slate_size, action_dim)
            self.actor.train()
        else:
            with torch.no_grad():
                action_emb, _, _ = self.actor.sample(state_tensor)
            self.actor.train()
            
        # action_emb: (1, slate_size, action_dim)
        
        if self.item_embeddings is not None:
            selected_items = []
            current_allowed = list(allowed_items) if allowed_items is not None else list(range(1, self.num_items + 1))
            
            # For each slot in the slate
            for i in range(min(slate_size, action_emb.shape[1])):
                proto_action = action_emb[0, i] # (action_dim,)
                
                # 1. KNN Search
                action_norm = F.normalize(proto_action.unsqueeze(0), p=2, dim=1)
                scores = torch.mm(action_norm, self.item_embeddings.t()).squeeze(0) # (num_items,)
                
                # Mask padding
                scores[0] = -np.inf
                
                # Mask already selected items
                mask = torch.ones_like(scores, dtype=torch.bool)
                valid_indices = []
                for item_id in current_allowed:
                    idx = item_id - 1
                    if 0 <= idx < scores.shape[0] and item_id not in selected_items:
                        valid_indices.append(idx)
                
                if not valid_indices:
                    break
                    
                mask[valid_indices] = False
                scores[mask] = -np.inf
                
                k = min(k_neighbors, len(valid_indices))
                top_k_scores, top_k_indices = torch.topk(scores, k)
                
                # 2. Critic Evaluation
                # We want to pick the item that maximizes Q(s, item)
                # Note: In SlateQ, we might want to maximize Q(s, slate), but with independent assumption
                # maximizing each slot is equivalent to maximizing the sum.
                
                state_expanded = state_tensor.expand(k, -1, -1)
                candidate_embeddings = self.item_embeddings[top_k_indices]
                
                self.critic.eval()
                with torch.no_grad():
                    q1, q2 = self.critic(state_expanded, candidate_embeddings)
                    q_values = torch.min(q1, q2).squeeze(1)
                self.critic.train()
                
                best_idx_in_k = torch.argmax(q_values)
                best_item_idx = top_k_indices[best_idx_in_k].item()
                best_item_id = best_item_idx + 1
                
                selected_items.append(best_item_id)
                
            return selected_items
            
        else:
            return np.random.choice(range(1, self.num_items + 1), slate_size, replace=False).tolist()


    def learn(self, buffer, batch_size=64):
        if len(buffer) < batch_size:
            return 0.0, 0.0, 0.0
            
        state, action, reward, next_state, done = buffer.sample(batch_size)
        
        state = torch.LongTensor(state).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        if action.dim() == 2:
            action = action.unsqueeze(1)
            
        next_state = torch.LongTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        current_alpha = self.alpha.item()
        
        # 1. Update Critic
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - current_alpha * next_log_prob
            target_q_slate = target_q.mean(dim=1, keepdim=True)
            target_q_total = reward + (1 - done) * self.gamma * target_q_slate
            
        current_q1, current_q2 = self.critic(state, action)
        current_q1_slate = current_q1.mean(dim=1, keepdim=True)
        current_q2_slate = current_q2.mean(dim=1, keepdim=True)
        
        critic_loss = F.huber_loss(current_q1_slate, target_q_total, delta=10.0) + F.huber_loss(current_q2_slate, target_q_total, delta=10.0)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # Added clipping
        self.critic_optimizer.step()
        
        # 2. Update Actor
        new_action, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (current_alpha * log_prob - q_new).mean()
        
        if self.bc_weight > 0:
            bc_loss = F.mse_loss(new_action, action)
            actor_loss += self.bc_weight * bc_loss
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # Added clipping
        self.actor_optimizer.step()
        
        # 3. Update Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # 4. Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()
