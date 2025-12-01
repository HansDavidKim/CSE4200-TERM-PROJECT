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
    
    def push(self, state, action, clicked_item_idx, reward, next_state, done):
        self.buffer.append((state, action, clicked_item_idx, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, clicked_item_idx, reward, next_state, done = zip(*batch)
        return state, action, clicked_item_idx, reward, next_state, done
    
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
        # To avoid numerical issues:
        action_bound = 1.0 - epsilon
        action_clipped = torch.clamp(action, -action_bound, action_bound)
        
        # z = atanh(action)
        z = 0.5 * torch.log((1 + action_clipped) / (1 - action_clipped))
        
        log_prob = normal.log_prob(z) - torch.log(1 - action_clipped.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=2) # Sum over action_dim -> (batch, slate_size)
        
        # Clamp log_prob to avoid explosion
        log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)
        
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, item_embeddings=None, gru_state_dict=None, alpha=0.1):
        super(Critic, self).__init__()
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        
        if item_embeddings is not None:
            self.click_embedding = nn.Embedding.from_pretrained(item_embeddings, freeze=False, padding_idx=0)
        else:
            self.click_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
            
        # Restore DeepSets for Slate History Context
        self.deepsets = DeepSets(embedding_dim, hidden_size)
        
        # Restore GRU for Sequence Context
        # Input to GRU: Item Embedding + Slate Context (DeepSets)
        # Note: Original Actor used (embedding_dim + hidden_size). Let's match that.
        self.click_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)
        
        if gru_state_dict is not None:
            with torch.no_grad():
                self.click_gru.weight_hh_l0.copy_(gru_state_dict['weight_hh_l0'])
                self.click_gru.bias_hh_l0.copy_(gru_state_dict['bias_hh_l0'])
                self.click_gru.bias_ih_l0.copy_(gru_state_dict['bias_ih_l0'])
        
        # Q1 architecture
        # Input to Value MLP: User Profile (EMA) + DeepSets (Slate History) + GRU Context + Item Embedding
        # User Profile dim: embedding_dim
        # DeepSets dim: hidden_size
        # GRU dim: hidden_size
        # Item Embedding dim: embedding_dim
        
        input_dim = embedding_dim + hidden_size + hidden_size + embedding_dim
        
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        # Heads
        self.v1_head = nn.Linear(256, 1) # Value given click
        
        # Scale for Cosine Similarity Logits
        self.p1_scale = nn.Parameter(torch.tensor(10.0))
        
        # Q2 architecture
        self.l4 = nn.Linear(input_dim, 256)
        self.l5 = nn.Linear(256, 256)
        
        self.v2_head = nn.Linear(256, 1)
        self.p2_scale = nn.Parameter(torch.tensor(10.0))
        
    def forward(self, state, item_embeddings):
        # state: (batch, seq_len, 1 + slate_size)
        # item_embeddings: (batch, num_candidates, embedding_dim) OR (batch, embedding_dim)
        
        clicked_items = state[:, :, 0] # (batch, seq_len)
        offered_slates = state[:, :, 1:] # (batch, seq_len, slate_size)
        
        batch_size, seq_len = clicked_items.shape
        
        # 1. Embed Clicked Items
        embedded_click = self.click_embedding(clicked_items) # (batch, seq_len, embed_dim)
        
        # 2. Compute EMA User Profile (Choice Model Context)
        user_profile = torch.zeros(batch_size, self.embedding_dim).to(clicked_items.device)
        
        for t in range(seq_len):
            mask = (clicked_items[:, t] != 0).float().unsqueeze(1) # (batch, 1)
            emb = embedded_click[:, t] # (batch, dim)
            update = self.alpha * emb + (1 - self.alpha) * user_profile
            user_profile = mask * update + (1 - mask) * user_profile
            
        # 3. Compute DeepSets & GRU Context (Value Model Context)
        
        # DeepSets for Slates
        offered_slates_flat = offered_slates.reshape(-1, offered_slates.shape[2])
        embedded_slates_flat = self.click_embedding(offered_slates_flat)
        deepsets_out_flat = self.deepsets(embedded_slates_flat)
        deepsets_out = deepsets_out_flat.reshape(batch_size, seq_len, -1)
        
        # GRU over sequence
        # Input: Clicked Item + Slate Context
        gru_input = torch.cat([embedded_click, deepsets_out], dim=2)
        _, hidden_click = self.click_gru(gru_input)
        gru_context = hidden_click[-1] # (batch, hidden_size)
        
        # Slate Context (Mean of DeepSets)
        slate_context = deepsets_out.mean(dim=1) # (batch, hidden_size)
        
        # 4. Item Scoring
        
        if item_embeddings.dim() == 2:
            item_embeddings = item_embeddings.unsqueeze(1)
            
        num_candidates = item_embeddings.size(1)
        
        # --- P(click): MNL (Cosine Similarity) ---
        # Uses ONLY EMA User Profile + Item
        
        user_profile_norm = F.normalize(user_profile, p=2, dim=1)
        item_embeddings_norm = F.normalize(item_embeddings, p=2, dim=2)
        
        p1_logit = (user_profile_norm.unsqueeze(1) * item_embeddings_norm).sum(dim=2) * self.p1_scale
        p2_logit = (user_profile_norm.unsqueeze(1) * item_embeddings_norm).sum(dim=2) * self.p2_scale
        
        # Numerical Stability: Subtract Max Logit
        # This prevents exp() explosion in Softmax/Sigmoid
        p1_logit = p1_logit - p1_logit.max(dim=1, keepdim=True)[0].detach()
        p2_logit = p2_logit - p2_logit.max(dim=1, keepdim=True)[0].detach()
        
        # --- V(click): MLP ---
        # Input: EMA Profile + Slate Context + GRU Context + Item
        
        user_profile_exp = user_profile.unsqueeze(1).expand(-1, num_candidates, -1)
        slate_context_exp = slate_context.unsqueeze(1).expand(-1, num_candidates, -1)
        gru_context_exp = gru_context.unsqueeze(1).expand(-1, num_candidates, -1)
        
        xu = torch.cat([user_profile_exp, slate_context_exp, gru_context_exp, item_embeddings], dim=2)
        
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        v1 = self.v1_head(x1).squeeze(2)
        
        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        v2 = self.v2_head(x2).squeeze(2)
        
        # Proxy Q
        q1 = v1 * torch.sigmoid(p1_logit)
        q2 = v2 * torch.sigmoid(p2_logit)
        
        return q1, q2, p1_logit, v1, p2_logit, v2

class SACAgent:
    def __init__(self, num_items, embedding_dim, hidden_size, action_dim, slate_size=1, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, alpha=0.2, device='cpu', item_embeddings=None, bc_weight=0.0, ctr_weight=0.0, ema_alpha=0.1, finetune_embeddings=False, gru_state_dict=None):
        self.slate_size = slate_size
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.num_items = num_items
        self.bc_weight = bc_weight
        self.ctr_weight = ctr_weight
        self.embedding_dim = embedding_dim
        
        self.actor = Actor(num_items, embedding_dim, hidden_size, action_dim, slate_size, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict).to(device)
        # Critic now takes embedding_dim instead of action_dim for item input
        self.critic = Critic(num_items, embedding_dim, hidden_size, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict, alpha=ema_alpha).to(device)
        self.critic_target = Critic(num_items, embedding_dim, hidden_size, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict, alpha=ema_alpha).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Position weights for SlateQ
        # w_j(s) - for now, let's make them learnable parameters independent of state, or small network.
        # Prompt says: "w_j(s) = position weight (fixed or learned small network)"
        # Let's use simple learnable weights per position for stability first.
        self.position_weights = nn.Parameter(torch.ones(slate_size, device=device))
        
        # Automatic Entropy Tuning
        # Heuristic: -dim(A). But here Action is embedding.
        # -action_dim * slate_size is too large (-320), drowning reward.
        # Use -slate_size (e.g. -5) or -1.0 * slate_size
        self.target_entropy = -float(slate_size)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        self.alpha = self.log_alpha.exp() # Initial alpha
        
        if item_embeddings is not None:
            # If finetuning, we need to make it a Parameter
            if finetune_embeddings:
                self.item_embeddings = nn.Parameter(F.normalize(item_embeddings, p=2, dim=1).to(device))
                self.item_embeddings.requires_grad = True
                # Add to critic optimizer
                self.critic_optimizer = optim.Adam(list(self.critic.parameters()) + [self.item_embeddings, self.position_weights], lr=lr_critic)
            else:
                self.item_embeddings = F.normalize(item_embeddings, p=2, dim=1).to(device)
                self.item_embeddings.requires_grad = False
                self.critic_optimizer = optim.Adam(list(self.critic.parameters()) + [self.position_weights], lr=lr_critic)
        else:
            self.item_embeddings = None
            self.critic_optimizer = optim.Adam(list(self.critic.parameters()) + [self.position_weights], lr=lr_critic)

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
            
            # SlateQ Construction: Greedy selection
            # For each slot j, pick item that maximizes w_j * q(s, a)
            # But wait, if we pick greedily, we might pick the same item multiple times?
            # Yes, we need to mask selected items.
            
            # Pre-compute Q-values for ALL candidates? Too expensive if num_items is large.
            # We stick to Wolpertinger: KNN -> Candidates -> Score.
            
            # Since w_j is just a scalar scaling, maximizing w_j * q(s, a) is same as maximizing q(s, a) if w_j > 0.
            # If w_j can be negative, it flips. Assuming w_j > 0 or we just care about q(s,a).
            # Actually, if we use separate q per slot, it matters. But here q(s,a) is shared.
            # So for the purpose of "picking the best item for this slot", we just want max q(s,a).
            # The weight w_j mainly affects the Value estimation Q(s, S).
            
            # However, if we want to support "diversity" or "position bias" explicitly in selection...
            # For now, let's just pick top-K items by q(s,a) from the KNN candidates.
            
            # 1. Get Proto-Action for the WHOLE slate? 
            # Wolpertinger usually maps ONE proto-action to k items.
            # Here we have `slate_size` proto-actions.
            
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
                state_expanded = state_tensor.expand(k, -1, -1)
                candidate_embeddings = self.item_embeddings[top_k_indices]
                
                self.critic.eval()
                with torch.no_grad():
                    q1, q2, _, _, _, _ = self.critic(state_expanded, candidate_embeddings)
                    q_values = torch.min(q1, q2).squeeze(1) # (k,)
                    
                    # Apply position weight?
                    # w_j = self.position_weights[i]
                    # score = w_j * q_values
                    # If w_j is positive, argmax is same.
                    # We just use q_values for selection.
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
            
        # Updated sample unpacking
        # buffer stores: (state, slate_items, clicked_item_idx, reward, next_state, done)
        # Wait, utils.py needs to be updated first to store this.
        # Assuming utils.py will return:
        # state, action (slate), clicked_item_idx, reward, next_state, done
        
        batch = buffer.sample(batch_size)
        state, action, clicked_item_idx, reward, next_state, done = batch
        
        state = torch.LongTensor(state).to(self.device)
        action = torch.LongTensor(np.array(action)).to(self.device) # (batch, slate_size)
        clicked_item_idx = torch.LongTensor(clicked_item_idx).to(self.device) # (batch,)
        
        next_state = torch.LongTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        current_alpha = self.alpha.item()
        
        # 1. Update Critic
        # Target = r + gamma * max_S' Q(s', S')
        # Q(s', S') = sum_j w_j * q(s', a'_j)
        # To compute max_S', we need to construct the optimal slate S'.
        # We use the Actor to propose S' (proto-actions) and then find items.
        # But for efficiency in SlateQ, we often assume we can just pick top-K items by q(s', a).
        # Here we follow the Actor-Critic style: Actor gives proto-action -> Items -> Q.
        
        with torch.no_grad():
            # Get next slate items using Actor + Wolpertinger (simplified or full)
            # For batch efficiency, we might skip full KNN if it's too slow, 
            # but we need valid item embeddings.
            # Let's assume we can use the Actor's output directly if we had a way to map to items differentiable?
            # No, we need discrete items for the Critic.
            
            # We'll use the Actor to generate proto-actions, then find nearest items (approximate).
            # OR, since we are doing Q-learning, we can just find max_a q(s', a) for each slot?
            # But we have an Actor. The Actor should guide the policy.
            # So we use Actor to get next_action (slate of items).
            
            # This part is tricky to batch efficiently with KNN.
            # Alternative: Use the items actually in the buffer for next_state? No, that's SARSA.
            # We need off-policy.
            
            # Let's do a simplified approach for Target:
            # 1. Actor -> Proto-actions
            # 2. We need embeddings for these proto-actions.
            #    If we assume the Actor outputs something close to item embeddings, we can pass proto-actions to Critic?
            #    No, Critic expects item embeddings.
            #    We can pass the proto-actions as "approximate" item embeddings.
            #    This avoids KNN in the inner loop.
            
            next_action_proto, next_log_prob, _ = self.actor.sample(next_state) # (batch, slate_size, action_dim)
            next_log_prob_sum = next_log_prob.sum(dim=1, keepdim=True)
            
            # Pass proto-actions to Critic as if they were items
            # Note: Critic takes (state, item_embeddings).
            # next_action_proto is (batch, slate_size, action_dim).
            # We need to reshape.
            
            # SlateQ Aggregation
            # Q(s, S) = sum P(i|S) * V(i|S)
            # P(i|S) = Softmax(logits)
            
            # target_q1, target_q2 are V-values (batch, slate_size)
            # We need logits from Critic to compute P
            # Critic returns q1, q2, p1_logit, v1, p2_logit, v2
            
            _, _, t_p1_logit, t_v1, t_p2_logit, t_v2 = self.critic_target(next_state, next_action_proto)
            
            # Compute P(i|S)
            t_p1 = F.softmax(t_p1_logit, dim=1) # (batch, slate_size)
            t_p2 = F.softmax(t_p2_logit, dim=1)
            
            # Expected Value of Next Slate
            # Q_next = sum_i P(i) * V(i)
            target_q1_slate = (t_p1 * t_v1).sum(dim=1, keepdim=True)
            target_q2_slate = (t_p2 * t_v2).sum(dim=1, keepdim=True)
            
            target_q_total = torch.min(target_q1_slate, target_q2_slate) - current_alpha * next_log_prob_sum
            target_y = reward + (1 - done) * self.gamma * target_q_total
            
        # Current Q
        # 1. Get embeddings for the slate items in the batch
        slate_item_embeddings = self.item_embeddings[action - 1] # (batch, slate_size, embed_dim)
        
        _, _, p1_logit, v1, p2_logit, v2 = self.critic(state, slate_item_embeddings)
        
        # Create mask for clicked item
        mask = F.one_hot(clicked_item_idx, num_classes=self.slate_size).bool() # (batch, slate_size)
        
        # --- Value Loss (Clicked Item Only) ---
        # We want gradients for v ONLY for masked entries.
        # Target for V(clicked) is target_y (Full Return)
        # Because if click happens, V should estimate the return.
        
        v1_clicked = v1[mask].unsqueeze(1) # (batch, 1)
        v2_clicked = v2[mask].unsqueeze(1)
        
        # Huber Loss
        value_loss = F.huber_loss(v1_clicked, target_y, delta=1.0) + F.huber_loss(v2_clicked, target_y, delta=1.0)
        
        # --- CTR Loss (MNL / Choice) ---
        # Labels: clicked_item_idx (batch,) - index in slate (0 to slate_size-1)
        # Logits: p_logits (batch, slate_size)
        
        # CrossEntropyLoss expects class indices
        ctr_loss1 = F.cross_entropy(p1_logit, clicked_item_idx)
        ctr_loss2 = F.cross_entropy(p2_logit, clicked_item_idx)
        ctr_loss = ctr_loss1 + ctr_loss2
        
        # Total Loss
        total_loss = value_loss + (self.ctr_weight if self.ctr_weight > 0 else 1.0) * ctr_loss
        
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        critic_loss = total_loss # For logging
        
        # Debug Prints (Every 100 steps or if loss is huge)
        if total_loss.item() > 1e6 or torch.isnan(total_loss):
            print(f"--- DEBUG EXPLOSION ---")
            print(f"Total Loss: {total_loss.item()}")
            print(f"Value Loss: {value_loss.item()}")
            print(f"CTR Loss: {ctr_loss.item()}")
            print(f"Alpha: {self.alpha.item()}")
            print(f"V1 min/max: {v1.min().item()}/{v1.max().item()}")
            print(f"P1 Logit min/max: {p1_logit.min().item()}/{p1_logit.max().item()}")
            print(f"Target Y min/max: {target_y.min().item()}/{target_y.max().item()}")
            print(f"Target Q Total min/max: {target_q_total.min().item()}/{target_q_total.max().item()}")
            print(f"Next Log Prob Sum min/max: {next_log_prob_sum.min().item()}/{next_log_prob_sum.max().item()}")
            print(f"-----------------------")
        
        # 2. Update Actor
        # Maximize Q(s, S) - alpha * log_prob
        new_action_proto, log_prob, _ = self.actor.sample(state)
        log_prob_sum = log_prob.sum(dim=1, keepdim=True)
        
        # Critic evaluation of new proto-actions
        _, _, p1_logit, v1, p2_logit, v2 = self.critic(state, new_action_proto)
        
        # Compute P(i|S)
        p1 = F.softmax(p1_logit, dim=1)
        p2 = F.softmax(p2_logit, dim=1)
        
        # Q(s, S) = sum P(i) * V(i)
        q1_slate = (p1 * v1).sum(dim=1, keepdim=True)
        q2_slate = (p2 * v2).sum(dim=1, keepdim=True)
        
        q_new_slate = torch.min(q1_slate, q2_slate)
        
        actor_loss = (current_alpha * log_prob_sum - q_new_slate).mean()
        
        # Behavior Cloning / Regularization Loss
        # Force proto-actions to be close to actual item embeddings (manifold constraint)
        if self.bc_weight > 0:
            # We want new_action_proto to be close to the items that were actually in the slate
            # But new_action_proto corresponds to the CURRENT policy's output for the state.
            # The 'action' in buffer corresponds to the OLD policy's output (which mapped to specific items).
            # If we assume the buffer contains valid item embeddings, minimizing MSE(new_action, action_emb)
            # forces the policy to output vectors that lie in the item embedding space.
            
            # We use the 'slate_item_embeddings' (from buffer) as targets.
            # Note: This is standard BC if we assume buffer data is expert.
            # Here it acts as a "validity constraint".
            bc_loss = F.mse_loss(new_action_proto, slate_item_embeddings.detach())
            actor_loss += self.bc_weight * bc_loss
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # 3. Update Alpha
        alpha_loss = -(self.log_alpha * (log_prob_sum + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # 4. Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()
