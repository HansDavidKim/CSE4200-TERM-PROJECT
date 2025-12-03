import os
import json
import torch
import numpy as np
import collections
import random
from tqdm import tqdm
from datetime import datetime
from rec_sim.simulate_api import create_environment

# Wrappers
class AgentWrapper:
    def reset(self):
        pass
    def act(self, obs, slate_size, allowed_items):
        raise NotImplementedError
    def update(self, obs, action, next_obs, reward):
        pass

class GRU4RecWrapper(AgentWrapper):
    def __init__(self, model_path, num_items, device='cpu'):
        self.device = device
        self.num_items = num_items
        
        # Handle directory input
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "gru4rec_model.pth")
            
        # Load Model
        from recommender.gru4rec import GRU4Rec
        
        # Try to detect config from checkpoint if saved, otherwise default
        checkpoint = torch.load(model_path, map_location=device)
        embedding_dim = checkpoint.get('embedding_dim', 64)
        hidden_size = checkpoint.get('hidden_size', 128)
        
        self.model = GRU4Rec(num_items, hidden_size, num_items, embedding_dim=embedding_dim, dropout=0.0).to(device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        smart_load_state_dict(self.model, checkpoint['model_state_dict'], device)
        self.model.eval()
        
        self.history = []
        self.masked_items = set()
        self.consumed_items = set()  # Track consumed items
        
    def reset(self):
        self.history = []
        
    def act(self, obs, slate_size, allowed_items):
        # allowed_items is ignored in favor of internal masking logic for GRU4Rec usually,
        # but we should respect it if provided.
        
        # Prepare Input
        if not self.history:
            # Cold start: Random or Popular?
            # Let's return random from allowed
            return np.random.choice(allowed_items, slate_size, replace=False).tolist()
            
        input_seq = torch.LongTensor([self.history]).to(self.device) # (1, seq_len)
        
        with torch.no_grad():
            logits, _ = self.model(input_seq) # (1, seq_len, num_items)
            scores = logits[0, -1, :].cpu().numpy() # (num_items,)
            
        # Masking
        # 1. Mask padding (0)
        scores[0] = -np.inf
        
        # 2. CRITICAL: Mask consumed items
        for item_id in self.consumed_items:
            if 0 <= item_id < len(scores):
                scores[item_id] = -np.inf
        
        # 3. Mask items not in allowed_items (if provided)
        if allowed_items is not None:
            allowed_set = set(allowed_items)
            for i in range(len(scores)):
                if (i + 1) not in allowed_set:
                    scores[i] = -np.inf
                    
        # Top-K
        top_k_indices = np.argsort(scores)[-slate_size:][::-1]
        action = [int(idx) for idx in top_k_indices]
        
        return action
        
    def update(self, obs, action, next_obs, reward):
        # Check for click
        responses = next_obs['response']
        clicked_item = 0
        for i, resp in enumerate(responses):
            if resp['click'] == 1:
                clicked_item = action[i]
                break
        
        if clicked_item > 0:
            self.history.append(clicked_item)
            self.consumed_items.add(clicked_item)  # Track consumed items

class HybridWrapper(AgentWrapper):
    def __init__(self, gru4rec_path, num_items, alpha=0.1, device='cpu'):
        self.alpha = alpha
        self.num_items = num_items
        
        # Load Embeddings from GRU4Rec
        from recommender.gru4rec import GRU4Rec
        model_path = os.path.join(gru4rec_path, "gru4rec_model.pth")
        checkpoint = torch.load(model_path, map_location=device)
        embedding_dim = checkpoint.get('embedding_dim', 64)
        hidden_size = checkpoint.get('hidden_size', 128)
        
        model = GRU4Rec(num_items, hidden_size, num_items, embedding_dim=embedding_dim).to(device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        smart_load_state_dict(model, checkpoint['model_state_dict'], device)
        
        self.item_embeddings = model.embedding.weight.data.cpu().numpy()
        # Normalize
        norm = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        self.item_embeddings_norm = self.item_embeddings / (norm + 1e-8)
        
        self.user_profile = np.zeros(embedding_dim)
        
    def reset(self):
        self.user_profile = np.zeros_like(self.user_profile)
        
    def act(self, obs, slate_size, allowed_items):
        if np.all(self.user_profile == 0):
            scores = np.random.rand(self.num_items) # Should match embedding size (including padding)
            # If item_embeddings has size 200, scores should be 200.
        else:
            scores = np.dot(self.item_embeddings_norm, self.user_profile)
            # scores is (num_items,) because embedding has padding at 0
            
        scores[0] = -np.inf
        
        # Mask not allowed
        if allowed_items is not None:
            allowed_set = set(allowed_items)
            for i in range(len(scores)):
                if i not in allowed_set: # i is item_id
                    scores[i] = -np.inf
                    
        top_k_indices = np.argsort(scores)[-slate_size:][::-1]
        # indices are item_ids directly because embedding matrix includes padding at 0
        # Wait, usually embedding matrix size is num_items + 1.
        # Index 0 is padding. Index 1 is item 1.
        # So argsort index IS the item_id.
        return top_k_indices.tolist()
        
    def update(self, obs, action, next_obs, reward):
        responses = next_obs['response']
        clicked_item = 0
        for i, resp in enumerate(responses):
            if resp['click'] == 1:
                clicked_item = action[i]
                break
        
        if clicked_item > 0:
            clicked_emb = self.item_embeddings_norm[clicked_item]
            if np.all(self.user_profile == 0):
                self.user_profile = clicked_emb
            else:
                self.user_profile = self.alpha * clicked_emb + (1 - self.alpha) * self.user_profile
                # Normalize
                u_norm = np.linalg.norm(self.user_profile)
                if u_norm > 0:
                    self.user_profile = self.user_profile / u_norm

class PPOWrapper:
    def __init__(self, agent, slate_size, history_length=10):
        self.agent = agent
        self.slate_size = slate_size
        self.history_length = history_length
        self.state = collections.deque([[0] * (slate_size + 1)] * history_length, maxlen=history_length)
        
    def reset(self):
        self.state = collections.deque([[0] * (self.slate_size + 1)] * self.history_length, maxlen=self.history_length)
        
    def act(self, obs, slate_size, allowed_items, top_k=0, temperature=1.0):
        current_state = list(self.state)
        # Enable sampling (training=True) if top_k is specified
        use_sampling = top_k > 0
        # PPOAgent.select_action returns (selected_items, action_emb, log_prob)
        selected_items, _, _ = self.agent.select_action(current_state, allowed_items=allowed_items, top_k=top_k, training=use_sampling, temperature=temperature)
        return selected_items
        
    def update(self, obs, action, next_obs, reward):
        responses = next_obs['response']
        clicked_item = 0
        for i, resp in enumerate(responses):
            if resp['click'] == 1:
                clicked_item = action[i]
                break
        
        # Pad action to slate_size if needed (for single-item actions)
        if len(action) < self.slate_size:
            action = action + [0] * (self.slate_size - len(action))
        
        state_element = [clicked_item] + action
        self.state.append(state_element)

def smart_load_state_dict(model, state_dict, device):
    """
    Loads state_dict into model, handling shape mismatches by padding.
    Useful when model size (num_items) is larger than checkpoint size.
    """
    model_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_dict:
            if param.shape != model_dict[name].shape:
                # Check if mismatch is due to num_items (dim 0)
                if param.shape[0] < model_dict[name].shape[0]:
                    # Pad
                    pad_size = model_dict[name].shape[0] - param.shape[0]
                    if len(param.shape) == 1:
                        # Bias
                        padded = torch.cat([param, torch.zeros(pad_size).to(device)], dim=0)
                    else:
                        # Weight
                        # For embedding: (num_items, dim) -> pad dim 0
                        # For Linear (out_features, in_features):
                        # If mismatch in dim 0 (out_features): pad dim 0
                        # If mismatch in dim 1 (in_features): pad dim 1
                        
                        if param.shape[1] == model_dict[name].shape[1]:
                            # Pad dim 0
                            padded = torch.cat([param, torch.zeros(pad_size, param.shape[1]).to(device)], dim=0)
                        elif param.shape[0] == model_dict[name].shape[0] and param.shape[1] < model_dict[name].shape[1]:
                             # Pad dim 1
                             pad_size_1 = model_dict[name].shape[1] - param.shape[1]
                             padded = torch.cat([param, torch.zeros(param.shape[0], pad_size_1).to(device)], dim=1)
                        else:
                            print(f"Warning: Could not pad {name} {param.shape} to {model_dict[name].shape}")
                            continue
                            
                    state_dict[name] = padded
                    
    model.load_state_dict(state_dict, strict=False)

def evaluate(
    model_type: str,
    model_path: str, # For GRU4Rec: .pth file, For SAC: dir, For Hybrid: GRU4Rec dir
    num_episodes: int = 100,
    drift_scale: float = 0.0,
    seed: int = 42,
    output_dir: str = "evaluation_report",
    # Extra args
    alpha: float = 0.1, # For Hybrid
    gru4rec_path: str = "trained_models/gru4rec", # For SAC/Hybrid
    num_items: int = 2000, # Default
    gamma: float = 0.99, # Discount factor
    device: str = "cpu",
    max_steps: int = 30, # Added max_steps
    top_k: int = 0, # Added top_k
    temperature: float = 1.0 # Added temperature
):
    # Config
    num_candidates = num_items
    slate_size = 5 # Default
    
    # Create Env
    env_config = {
        "num_candidates": num_candidates,
        "slate_size": slate_size,
        "resample_documents": False,
        "seed": seed,
        "drift_scale": drift_scale
    }
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = create_environment(env_config)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Agent
    # Note: item IDs are 1-based (1 to num_candidates).
    # Embedding layers usually need size = num_candidates + 1 (index 0 is padding).
    # So we pass num_candidates + 1 to the wrappers.
    if model_type == 'gru4rec':
        agent = GRU4RecWrapper(model_path, num_candidates + 1, device)
    elif model_type == 'hybrid':
        agent = HybridWrapper(model_path, num_candidates + 1, alpha, device) # model_path here is gru4rec_path

    elif model_type == 'ppo':
        # Load PPO Agent
        from recommender.ppo_agent import PPOAgent
        # Load Embeddings from GRU4Rec
        from recommender.gru4rec import GRU4Rec
        gru_model_path = os.path.join(gru4rec_path, "gru4rec_model.pth")
        checkpoint = torch.load(gru_model_path, map_location=device)
        embedding_dim = checkpoint.get('embedding_dim', 64)
        hidden_size = checkpoint.get('hidden_size', 128)
        
        gru_model = GRU4Rec(num_items, hidden_size, num_items, embedding_dim=embedding_dim).to(device)
        smart_load_state_dict(gru_model, checkpoint['model_state_dict'], device)
        item_embeddings = gru_model.embedding.weight.data
        
        # PPO Agent
        agent_model = PPOAgent(num_items, embedding_dim, hidden_size, embedding_dim, slate_size, 3e-4, 1e-3, 0.99, 4, 0.2, 0.15, device, item_embeddings=item_embeddings)
        
        # Load trained weights
        # model_path is directory or file?
        if os.path.isdir(model_path):
            # Find latest epoch
            files = [f for f in os.listdir(model_path) if f.startswith("ppo_model_epoch_")]
            if files:
                files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                model_file = os.path.join(model_path, files[-1])
            else:
                model_file = os.path.join(model_path, "ppo_model.pth")
        else:
            model_file = model_path
            
        if os.path.exists(model_file):
            print(f"Loading PPO model from {model_file}")
            agent_model.actor.load_state_dict(torch.load(model_file, map_location=device))
        else:
            print(f"Warning: PPO model not found at {model_file}")
            
        agent = PPOWrapper(agent_model, slate_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    print(f"Starting Evaluation for {model_type}...")
    
    # Metrics
    # Metrics
    total_rewards = []
    total_discounted_rewards = []
    total_episode_lengths = [] # Added
    total_clicks = 0
    total_watch_time = 0
    total_steps = 0
    unique_items_recommended = set()
    
    for episode_idx in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        agent.reset()
        
        episode_reward = 0
        episode_discounted_reward = 0
        episode_steps = 0 # Added
        current_gamma = 1.0
        consumed_in_episode = set()
        
        done = False
        while not done:
            if episode_steps >= max_steps:
                done = True
                break
                
            # Masking
            allowed_items = [i for i in range(1, num_candidates + 1) if i not in consumed_in_episode]
            if len(allowed_items) < slate_size:
                allowed_items = range(1, num_candidates + 1)
                
            # Act
            if isinstance(agent, PPOWrapper):
                action = agent.act(obs, slate_size, allowed_items, top_k=top_k, temperature=temperature)
            else:
                action = agent.act(obs, slate_size, allowed_items)
            
            # Step
            # Env expects 0-indexed action
            # Check for invalid actions
            env_action = []
            for x in action:
                if 1 <= x <= num_candidates:
                    env_action.append(x - 1)
                else:
                    # Invalid action, pick random
                    # print(f"Warning: Invalid action {x} for num_candidates {num_candidates}")
                    env_action.append(random.randint(0, num_candidates - 1))
            
            # Debug: Print env_action length and reward
            # print(f"Step: {episode_steps}, Action Len: {len(env_action)}, Reward: {reward}")
            # if len(env_action) != slate_size:
            #     print(f"WARNING: Action length {len(env_action)} != slate_size {slate_size}")
            
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            
            if isinstance(reward, dict):
                reward = sum(reward.values()) if reward else 0.0
                
            # Update Agent
            agent.update(obs, action, next_obs, reward)
            
            # Metrics
            episode_reward += reward
            episode_discounted_reward += current_gamma * reward
            current_gamma *= gamma
            unique_items_recommended.update(action)
            
            responses = next_obs['response']
            clicked_item = 0
            for resp in responses:
                if resp['click']:
                    total_clicks += 1
                    total_watch_time += resp['watch_time']
                    pass
            
            # Find clicked item from action
            for i, resp in enumerate(responses):
                if resp['click'] == 1:
                    clicked_item = action[i]
                    break
            
            if clicked_item > 0:
                consumed_in_episode.add(clicked_item)
            
            total_steps += 1
            episode_steps += 1
            obs = next_obs
            
            if terminated or truncated:
                done = True
        
        total_rewards.append(episode_reward)
        total_discounted_rewards.append(episode_discounted_reward) # Added
        total_episode_lengths.append(episode_steps) # Added
        
    # Calculate Metrics
    avg_reward = np.mean(total_rewards)
    avg_discounted_reward = np.mean(total_discounted_rewards) # Added
    avg_episode_length = np.mean(total_episode_lengths) # Added
    ctr = total_clicks / total_steps if total_steps > 0 else 0
    avg_watch_time = total_watch_time / num_episodes # Changed episodes to num_episodes
    
    # Filter unique items to be within valid range [1, num_candidates]
    valid_unique_items = {i for i in unique_items_recommended if 1 <= i <= num_candidates}
    coverage = len(valid_unique_items) / num_candidates
    
    metrics = {
        "model": model_type,
        "avg_reward": avg_reward,
        "avg_discounted_reward": avg_discounted_reward,
        "avg_episode_length": avg_episode_length, # Added
        "ctr": ctr,
        "avg_watch_time": avg_watch_time,
        "coverage": coverage,
        "unique_items": len(unique_items_recommended),
        "drift_scale": drift_scale,
        "seed": seed
    }
    
    print("\nEvaluation Results:")
    print(json.dumps(metrics, indent=4))
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_metrics_{timestamp}.json"
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
