import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

# Reusing DeepSets and Actor for consistency
class DeepSets(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(DeepSets, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x: (batch, set_size, embedding_dim)
        x = F.relu(self.fc1(x))
        x = x.mean(dim=1) # Mean Pooling
        x = F.relu(self.fc2(x))
        return x

class Actor(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, action_dim, slate_size=1, log_std_min=-20, log_std_max=2, item_embeddings=None, gru_state_dict=None):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.slate_size = slate_size
        self.action_dim = action_dim
        self.hidden_size = hidden_size # Store hidden_size for state embedding
        
        if item_embeddings is not None:
            # GRU4Rec embeddings are (num_items, dim), we need (num_items+1, dim) with padding at index 0
            zero_row = torch.zeros(1, item_embeddings.shape[1]).to(item_embeddings.device)
            item_embeddings_padded = torch.cat([zero_row, item_embeddings], dim=0)
            # CRITICAL FIX: Unfreeze embeddings for fine-tuning!
            self.click_embedding = nn.Embedding.from_pretrained(item_embeddings_padded, freeze=False, padding_idx=0)
        else:
            self.click_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
            
        self.deepsets = DeepSets(embedding_dim, hidden_size)
        self.click_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)
        
        if gru_state_dict is not None:
            with torch.no_grad():
                self.click_gru.weight_hh_l0.copy_(gru_state_dict['weight_hh_l0'])
                self.click_gru.bias_hh_l0.copy_(gru_state_dict['bias_hh_l0'])
                self.click_gru.bias_ih_l0.copy_(gru_state_dict['bias_ih_l0'])
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # CRITICAL FIX: Output item logits directly like GRU4Rec!
        self.item_logits_head = nn.Linear(hidden_size, num_items)
        
    def _get_user_context(self, state):
        # state: (batch, seq_len, 1 + slate_size)
        clicked_items = state[:, :, 0].long()
        offered_slates = state[:, :, 1:].long()
        
        embedded_click = self.click_embedding(clicked_items)
        
        batch_size, seq_len, slate_size = offered_slates.shape
        offered_slates_flat = offered_slates.reshape(-1, slate_size)
        embedded_slates_flat = self.click_embedding(offered_slates_flat)
        deepsets_out_flat = self.deepsets(embedded_slates_flat)
        deepsets_out = deepsets_out_flat.reshape(batch_size, seq_len, -1)
        
        gru_input = torch.cat([embedded_click, deepsets_out], dim=2)
        _, hidden_click = self.click_gru(gru_input)
        user_context = hidden_click[-1]
        
        if torch.isnan(user_context).any():
            user_context = torch.nan_to_num(user_context, nan=0.0)
        return user_context

    def forward(self, state, deterministic=False):
        user_context = self._get_user_context(state)
        
        x = F.relu(self.fc1(user_context))
        x = F.relu(self.fc2(x))
        
        # CRITICAL FIX: Output item logits directly!
        item_logits = self.item_logits_head(x)  # (batch, num_items)
        
        return item_logits

    def get_state_embedding(self, state):
        """
        Returns the user context (GRU hidden state) as the state embedding.
        state: (batch, seq_len, 1 + slate_size)
        Returns: (batch, hidden_size)
        """
        return self._get_user_context(state)

    def evaluate(self, state, action_indices):
        # action_indices: (batch, action_size) - item indices (action_size can be 1 for single-item)
        # Re-evaluate to get log_probs and entropy
        
        item_logits = self.forward(state)  # (batch, num_items)
        
        # Convert to probabilities
        item_probs = F.softmax(item_logits, dim=1)  # (batch, num_items)
        
        # Get log probs for selected items
        log_probs = []
        entropies = []
        
        action_size = action_indices.shape[1]
        for i in range(action_size):
            # Get log prob for i-th item in action
            item_idx = action_indices[:, i]  # (batch,)
            log_prob = torch.log(item_probs.gather(1, item_idx.unsqueeze(1)) + 1e-8)  # (batch, 1)
            log_probs.append(log_prob)
            
            # Entropy
            entropy = -(item_probs * torch.log(item_probs + 1e-8)).sum(dim=1, keepdim=True)
            entropies.append(entropy)
            
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)  # (batch, 1)
        entropies = torch.stack(entropies, dim=1).mean(dim=1)  # (batch, 1)
        
        return log_probs, entropies

class Critic(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, slate_size=1, item_embeddings=None, gru_state_dict=None):
        super(Critic, self).__init__()
        
        if item_embeddings is not None:
            # GRU4Rec embeddings are (num_items, dim), we need (num_items+1, dim) with padding at index 0
            zero_row = torch.zeros(1, item_embeddings.shape[1]).to(item_embeddings.device)
            item_embeddings_padded = torch.cat([zero_row, item_embeddings], dim=0)
            # CRITICAL FIX: Unfreeze embeddings for fine-tuning!
            self.click_embedding = nn.Embedding.from_pretrained(item_embeddings_padded, freeze=False, padding_idx=0)
        else:
            self.click_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
            
        self.deepsets = DeepSets(embedding_dim, hidden_size)
        self.click_gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)
        
        if gru_state_dict is not None:
            with torch.no_grad():
                self.click_gru.weight_hh_l0.copy_(gru_state_dict['weight_hh_l0'])
                self.click_gru.bias_hh_l0.copy_(gru_state_dict['bias_hh_l0'])
                self.click_gru.bias_ih_l0.copy_(gru_state_dict['bias_ih_l0'])
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1) # Output Scalar Value
        
    def forward(self, state):
        # state: (batch, seq_len, 1 + slate_size)
        clicked_items = state[:, :, 0].long() # Cast to Long
        offered_slates = state[:, :, 1:].long() # Cast to Long
        
        embedded_click = self.click_embedding(clicked_items)
        
        batch_size, seq_len, slate_size = offered_slates.shape
        offered_slates_flat = offered_slates.reshape(-1, slate_size)
        embedded_slates_flat = self.click_embedding(offered_slates_flat)
        deepsets_out_flat = self.deepsets(embedded_slates_flat)
        deepsets_out = deepsets_out_flat.reshape(batch_size, seq_len, -1)
        
        gru_input = torch.cat([embedded_click, deepsets_out], dim=2)
        _, hidden_click = self.click_gru(gru_input)
        user_context = hidden_click[-1]
        
        x = F.relu(self.fc1(user_context))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        
        return value

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.teacher_logits = []  # Store teacher logits
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.log_probs[:]
        del self.teacher_logits[:]
    
    def add(self, state, action, reward, is_terminal, log_prob, teacher_logit=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.log_probs.append(log_prob)
        if teacher_logit is not None:
            self.teacher_logits.append(teacher_logit)

class PPOAgent:
    def __init__(self, num_items, embedding_dim, hidden_size, action_dim, slate_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coef, device, item_embeddings=None, gru_state_dict=None):
        self.actor = Actor(num_items, embedding_dim, hidden_size, action_dim, slate_size, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict).to(device)
        self.critic = Critic(num_items, embedding_dim, hidden_size, slate_size, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict).to(device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.device = device
        self.buffer = RolloutBuffer()
        self.MseLoss = nn.MSELoss()
        
        self.num_items = num_items
        self.item_embeddings = self.actor.click_embedding.weight.data # Shared embeddings
        
        # Popularity Tracking
        self.item_counts = torch.zeros(num_items + 1).to(device) # +1 for padding
        
        # Similarity Matrix (Cosine Similarity of Item Embeddings)
        # item_embeddings is (num_items+1, dim) including padding at 0
        # We only care about items 1..num_items
        with torch.no_grad():
            embeddings = self.actor.click_embedding.weight.data[1:] # (num_items, dim)
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            self.similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t()) # (num_items, num_items)
            # Map back to 1-based indexing if needed, but for now 0-based index i corresponds to item i+1
    
    def update_popularity(self, action_ids):
        """
        Update item popularity counts.
        action_ids: list of item IDs
        """
        for item_id in action_ids:
            if 0 <= item_id <= self.num_items:
                self.item_counts[item_id] += 1
    
    def select_action(self, state, allowed_items=None, temperature=1.0, training=False, mask_top_p=0.0, mask_prob=0.0, top_p=1.0, top_k=0, use_gumbel=False):
        # state is list of lists, convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            item_logits = self.actor(state_tensor)
            
            # Masking Logic (Forced Exploration)
            if training and mask_top_p > 0 and random.random() < mask_prob:
                # Identify top-K items
                k = int(self.num_items * mask_top_p)
                if k > 0:
                    # Get top-k indices based on counts
                    # We only care about items 1 to num_items (ignore padding 0)
                    counts = self.item_counts[1:]
                    # Get indices of top-k counts. Note: indices are 0-based relative to counts slice, so add 1 for item ID
                    _, top_indices = torch.topk(counts, k)
                    top_item_ids = top_indices + 1
                    
                    # Mask logits (set to -inf)
                    # item_logits is (1, num_items), corresponding to item IDs 1..num_items
                    # So index i in logits corresponds to item ID i+1
                    # We need to map item ID to logit index: logit_idx = item_id - 1
                    mask_indices = top_item_ids - 1
                    item_logits[0, mask_indices] = -float('inf')

            if allowed_items is not None:
                # Create a mask for allowed items
                # item_logits is (1, num_items) corresponding to items 1 to num_items
                # allowed_items contains item IDs (1-based)
                mask = torch.full_like(item_logits, -float('inf'))
                allowed_indices = [i - 1 for i in allowed_items if 0 < i <= self.num_items]
                mask[0, allowed_indices] = 0
                item_logits = item_logits + mask
            
            # Temperature scaling
            item_logits = item_logits / temperature
            
            # Numerical stability: subtract max logit before softmax
            item_logits = item_logits - item_logits.max(dim=1, keepdim=True)[0]
            
            # Top-K Sampling - ONLY during training
            if training and top_k > 0:
                # Keep only top-k logits, set others to -inf
                v, _ = torch.topk(item_logits, top_k)
                # v[:, [-1]] is the k-th largest value
                item_logits[item_logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) sampling - ONLY during training (if Top-K is not used)
            elif training and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(item_logits, descending=True, dim=1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                # Scatter sorted tensors back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                item_logits[indices_to_remove] = -float('inf')
            
            # Gumbel-Softmax trick for differentiable sampling
            if use_gumbel and training:
                # Add Gumbel noise
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(item_logits) + 1e-10) + 1e-10)
                item_logits = item_logits + gumbel_noise
            
            probs = F.softmax(item_logits, dim=1)
            
            # Sample Slate
            slate_size = self.actor.slate_size
            action_ids = []
            log_probs = 0
            
            if training:
                # Sample without replacement
                # If we have enough items with non-zero prob
                valid_items = (probs > 0).sum().item()
                num_samples = min(slate_size, valid_items)
                
                if num_samples > 0:
                    action_indices = torch.multinomial(probs, num_samples, replacement=False) # (1, num_samples)
                    action_indices = action_indices[0]
                    
                    for idx in action_indices:
                        action_ids.append(idx.item() + 1)
                        log_probs += torch.log(probs[0, idx] + 1e-8)
                
                # Fill remaining if needed (shouldn't happen if top_k >= slate_size)
                while len(action_ids) < slate_size:
                     # Just pick random valid item not in list
                     # Fallback logic
                     pass

            else:
                # Greedy: Top-K items
                _, top_indices = torch.topk(probs, slate_size)
                top_indices = top_indices[0]
                for idx in top_indices:
                    action_ids.append(idx.item() + 1)
                    log_probs += torch.log(probs[0, idx] + 1e-8)
            
            # Get action embedding (mean of items)
            # action_ids is list of ints
            if len(action_ids) > 0:
                action_tensor = torch.tensor(action_ids).to(self.device)
                action_embs = self.actor.click_embedding(action_tensor) # (slate_size, dim)
                action_emb = action_embs.mean(dim=0).detach().cpu().numpy() # (dim,)
            else:
                action_emb = np.zeros(self.actor.click_embedding.embedding_dim)

            return action_ids, action_emb, log_probs.item()

    def update(self, similarity_coef=0.0, similarity_top_k=10, kl_coef=0.0):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if torch.isnan(rewards).any():
            print("Warning: NaNs found in rewards! Replacing with 0.")
            rewards = torch.nan_to_num(rewards, nan=0.0)
            
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        # Convert action lists to tensor - each action is a list of item IDs
        old_actions = torch.LongTensor(self.buffer.actions).to(self.device)  # (batch, slate_size)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        # Teacher logits for Distillation
        has_teacher = len(self.buffer.teacher_logits) > 0
        if has_teacher:
            teacher_logits = torch.stack(self.buffer.teacher_logits).to(self.device)
            teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            state_values = self.critic(old_states)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # PPO Loss
            ppo_loss = -torch.min(surr1, surr2)
            
            loss = ppo_loss + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            # 1. KL Divergence Penalty (Optional)
            if has_teacher and kl_coef > 0:
                # Get current student logits
                student_logits = self.actor(old_states)
                student_log_probs = F.log_softmax(student_logits, dim=1)
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                loss += kl_coef * kl_loss

            # 2. Similarity-based Soft Labeling Loss (New!)
            if similarity_coef > 0:
                # Get current student logits
                student_logits = self.actor(old_states) # (batch, num_items)
                student_log_probs = F.log_softmax(student_logits, dim=1)
                
                # Create Soft Targets based on Clicked Items (Ground Truth)
                # old_states is (batch, window_size, feature_dim) where feature_dim = 1 + slate_size
                # The first element of the state vector is the clicked_item
                # We want the clicked item from the most recent step in the window (index -1)
                clicked_item_ids = old_states[:, -1, 0].long() # (batch,)
                
                # We need 0-based indices for similarity matrix
                clicked_indices = clicked_item_ids - 1
                
                # Handle padding/initial state where clicked_item might be 0
                # If clicked_item is 0, we can't compute similarity. Mask these out later?
                # For now, clamp to 0 to avoid index error, and we'll handle the loss masking if needed.
                # But typically trained on valid clicks.
                clicked_indices = torch.clamp(clicked_indices, min=0)
                
                # Get similarity row for each clicked item
                # self.similarity_matrix is (num_items, num_items)
                # target_sims: (batch, num_items)
                target_sims = self.similarity_matrix[clicked_indices]
                
                # Keep only Top-K similar items
                top_k_vals, top_k_inds = torch.topk(target_sims, similarity_top_k, dim=1)
                
                # Create sparse target distribution
                target_dist = torch.zeros_like(target_sims)
                target_dist.scatter_(1, top_k_inds, top_k_vals)
                
                # Normalize to make it a probability distribution
                target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
                
                # Calculate KL Divergence between Student Policy and Soft Target
                # F.kl_div(input, target) -> input is log-probs, target is probs
                sim_loss = F.kl_div(student_log_probs, target_dist, reduction='batchmean')
                
                loss += similarity_coef * sim_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
        # clear buffer
        self.buffer.clear()
