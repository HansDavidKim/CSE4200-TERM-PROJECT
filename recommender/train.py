import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from recommender.utils import TrajectoryDataset, collate_fn
from recommender.gru4rec import GRU4Rec
from recommender.sac_agent import SACAgent, ReplayBuffer
from rec_sim.simulate_api import simulate_users_csv
from matrix_factorization.embedding import LightGCNEmbedding
from tqdm import tqdm
import collections
import random

def train_gru4rec(
    output_dir="trained_models/gru4rec",
    epochs=10,
    batch_size=256,
    embedding_dim=64,
    hidden_size=128,
    dropout=0.1,
    lr=0.001,

    input_dir="preprocessed_dataset",
    seed=42
):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traj_path = os.path.join(base_dir, input_dir, 'user_trajectories.pkl')
    implicit_csv = os.path.join(base_dir, input_dir, 'implicit_user_item_matrix.csv')
    
    output_full_path = os.path.join(base_dir, output_dir)
    os.makedirs(output_full_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading trajectories from {traj_path}...")
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    dataset = TrajectoryDataset(trajectories)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 2. Embeddings (LightGCN Init)
    print("Generating LightGCN embeddings for initialization...")
    # Determine num_items
    max_item_id = 0
    for seq in dataset.data:
        max_item_id = max(max_item_id, seq.max().item())
    num_items = max_item_id + 1
    
    emb_model = LightGCNEmbedding(implicit_csv, k=embedding_dim, epochs=20)
    emb_model.train()
    _, item_embeddings_np = emb_model.get_embeddings()
    
    pretrained_emb = torch.zeros(num_items, embedding_dim)
    for idx, item_id in emb_model.item_id_map.items():
        if item_id < num_items:
            pretrained_emb[item_id] = torch.tensor(item_embeddings_np[idx], dtype=torch.float)
            
    # 3. Model
    model = GRU4Rec(input_size=num_items, 
                    hidden_size=hidden_size, 
                    output_size=num_items, 
                    num_layers=1, 
                    dropout=dropout,
                    pretrained_embeddings=pretrained_emb,
                    freeze_embeddings=False).to(device)
                    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Train
    print(f"Starting GRU4Rec training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, num_items), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    # 5. Save
    # 5. Save
    model_save_path = os.path.join(output_full_path, 'gru4rec_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'hidden_size': hidden_size,
        'num_items': num_items
    }, model_save_path)
    print(f"Saved model to {model_save_path}")

def train_sac(
    pretrained_actor=None,
    pretrained_critic=None,
    gru4rec_path="trained_models/gru4rec",
    output_dir="trained_models/sac",
    epochs=5,
    batch_size=64,
    buffer_size=10000,
    lr_actor=3e-5,
    lr_critic=3e-5,
    gamma=0.99,
    slate_size=1,
    num_users_per_epoch=100,
    bc_weight=0.0,
    steps=50,
    diversity_weight=0.0,

    finetune_embeddings=False,
    seed=42
):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_full_path = os.path.join(base_dir, output_dir)
    os.makedirs(output_full_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 1. Load Embeddings
    print(f"Loading embeddings from GRU4Rec at {gru4rec_path}...")
    gru_model_path = os.path.join(base_dir, gru4rec_path, 'gru4rec_model.pth')
    checkpoint = torch.load(gru_model_path, map_location=device)
    embedding_dim = checkpoint.get('embedding_dim', 64)
    hidden_size = checkpoint.get('hidden_size', 128)
    
    # Need num_items. GRU4Rec checkpoint might not have it explicitly if we didn't save metadata properly in previous runs.
    # But weight shape has it.
    if 'num_items' in checkpoint:
        num_items = checkpoint['num_items']
    else:
        emb_weight = checkpoint['model_state_dict']['embedding.weight']
        num_items = emb_weight.shape[0] # Includes padding
        
    item_embeddings = checkpoint['model_state_dict']['embedding.weight'].data
    
    # Extract GRU weights
    gru_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('gru.'):
            # Remove 'gru.' prefix
            gru_state_dict[key[4:]] = value
    
    # 2. Agent
    action_dim = embedding_dim
    agent = SACAgent(num_items, embedding_dim, hidden_size, action_dim, slate_size, lr_actor, lr_critic, gamma, 0.2, device, item_embeddings, bc_weight, finetune_embeddings, gru_state_dict=gru_state_dict)
    
    if pretrained_actor:
        agent.actor.load_state_dict(torch.load(pretrained_actor, map_location=device))
    if pretrained_critic:
        agent.critic.load_state_dict(torch.load(pretrained_critic, map_location=device))
        
    replay_buffer = ReplayBuffer(buffer_size)
    
    # 3. Train Loop
    print(f"Starting SAC Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Simulate Users
        # We use simulate_users_csv to generate trajectories using the current agent
        # But simulate_users_csv expects an 'agent' object with 'act' method.
        # Our SACAgent has 'act'.
        
        # We need to wrap agent to handle state management if simulate_users_csv doesn't do it.
        # simulate_users_csv does: state = update_state(...)
        # agent.act(state)
        # So we need to ensure agent.act handles the state format correctly.
        # SACAgent.act expects list/deque.
        
        # We need to collect data.
        # simulate_users_csv returns metrics, but we need the transitions.
        # Actually simulate_users_csv in this project was designed to return metrics.
        # We need to modify it or write a custom collection loop here.
        # Given we have 'steps' and 'num_users_per_epoch', let's write the loop here.
        
        from rec_sim.simulate_api import create_environment
        env = create_environment({
            "num_candidates": num_items - 1, # num_items includes padding
            "slate_size": slate_size,
            "resample_documents": False,
            "seed": 42 # Fixed seed for training stability
        })
        
        epoch_critic_loss = 0
        epoch_actor_loss = 0
        
        pbar = tqdm(range(num_users_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for _ in pbar:
            obs, _ = env.reset()
            # State: History of clicked items + Slate History
            # Shape: (history_len, 1 + slate_size)
            history_len = 10
            state = collections.deque([[0] * (slate_size + 1)] * history_len, maxlen=history_len)
            
            recommended_items = set()
            
            for step in range(steps):
                current_state = list(state)
                
                # Masking
                allowed_items = [i for i in range(1, num_items) if i not in recommended_items]
                if len(allowed_items) < slate_size:
                    allowed_items = range(1, num_items)
                    
                # Act
                action_ids = agent.act(current_state, slate_size=slate_size, allowed_items=allowed_items)
                recommended_items.update(action_ids)
                
                # Get Action Embeddings for Buffer
                # We need continuous action for SAC update
                # action_ids is list of ints.
                # We need (slate_size, action_dim)
                real_action_embs = []
                for item_id in action_ids:
                    idx = item_id # item_id is 1-based index into embedding matrix? 
                    # Wait, embedding matrix has padding at 0.
                    # So item_id 1 is at index 1.
                    if 0 <= idx < len(agent.item_embeddings):
                        emb = agent.item_embeddings[idx].detach().cpu().numpy()
                    else:
                        emb = np.zeros(action_dim)
                    real_action_embs.append(emb)
                real_action_emb = np.stack(real_action_embs, axis=0)
                
                # Step
                env_action = [x - 1 for x in action_ids]
                next_obs, reward, terminated, truncated, _ = env.step(np.array(env_action))
                
                if isinstance(reward, dict):
                    reward = sum(reward.values()) if reward else 0.0
                    
                # Diversity Penalty
                if diversity_weight > 0:
                    # Pairwise cosine similarity
                    if slate_size > 1:
                        sim_sum = 0
                        count = 0
                        for i in range(slate_size):
                            for j in range(i + 1, slate_size):
                                u = real_action_emb[i]
                                v = real_action_emb[j]
                                sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)
                                sim_sum += sim
                                count += 1
                        avg_sim = sim_sum / count if count > 0 else 0
                        reward -= diversity_weight * avg_sim
                
                # Update State
                responses = next_obs['response']
                clicked_item = 0
                for i, resp in enumerate(responses):
                    if resp['click'] == 1:
                        clicked_item = action_ids[i]
                        break
                
                slate_items = action_ids
                state_element = [clicked_item] + slate_items
                
                next_state_deque = state.copy()
                next_state_deque.append(state_element)
                next_state = list(next_state_deque)
                
                done = terminated or truncated
                
                replay_buffer.push(current_state, real_action_emb, reward, next_state, done)
                
                state = next_state_deque
                
                # Train Step
                if len(replay_buffer) > batch_size:
                    c_loss, a_loss, _ = agent.learn(replay_buffer, batch_size)
                    epoch_critic_loss += c_loss
                    epoch_actor_loss += a_loss
                
                if done:
                    break
                    
        print(f"Epoch {epoch+1}: Avg Critic Loss: {epoch_critic_loss/num_users_per_epoch:.4f}, Avg Actor Loss: {epoch_actor_loss/num_users_per_epoch:.4f}")
        
    # Save
    torch.save(agent.actor.state_dict(), os.path.join(output_full_path, 'sac_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(output_full_path, 'sac_critic.pth'))
    torch.save(agent.log_alpha, os.path.join(output_full_path, 'sac_log_alpha.pth'))
    print(f"Saved SAC models to {output_full_path}")

def train(
    model_type,
    **kwargs
):
    if model_type == 'gru4rec':
        train_gru4rec(**kwargs)
    elif model_type == 'sac':
        train_sac(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
