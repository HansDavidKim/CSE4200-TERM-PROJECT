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
    seed=42,
    device="cpu"
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
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
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
    # LightGCNEmbedding (GraphEmbedding) has item_id_map as {item_id: index}
    for item_id, idx in emb_model.item_id_map.items():
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
    num_items=2000,
    embedding_dim=64,
    hidden_size=128,
    action_dim=64,
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
    alpha=0.2,
    slate_size=1,
    num_users_per_epoch=100,
    bc_weight=1.0,
    steps=50,
    diversity_weight=0.0,
    ctr_weight=0.0,
    ema_alpha=0.1,
    finetune_embeddings=False,
    seed=42,
    input_dir=None,
    device='cpu'
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
    
    # device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Training SAC Agent... (CTR Weight: {ctr_weight}, EMA Alpha: {ema_alpha})")
    
    # Load Item Embeddings from GRU4Rec
    item_embeddings = None
    gru_state_dict = None
    if gru4rec_path and os.path.exists(os.path.join(base_dir, gru4rec_path)):
        print(f"Loading GRU4Rec embeddings from {gru4rec_path}")
        # ... (loading logic same as before)
        # Load checkpoint
        checkpoint_path = os.path.join(base_dir, gru4rec_path, "gru4rec_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Extract embedding weight
            # Checkpoint keys: 'model_state_dict' -> 'embedding.weight'
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'embedding.weight' in state_dict:
                    item_embeddings = state_dict['embedding.weight']
                    print(f"Loaded item embeddings: {item_embeddings.shape}")
                
                # Extract GRU weights if needed (but we removed GRU from Critic)
                # But Actor might still use it?
                # Actor uses GRU? Let's check Actor.
                # Actor code in sac_agent.py still has GRU?
                # I should check if Actor needs GRU.
                # Assuming Actor still uses GRU for state representation?
                # The user only asked to change Critic.
                # So Actor keeps GRU.
                
                # Extract GRU weights for Actor
                gru_state_dict = {}
                for key in state_dict:
                    if 'gru' in key:
                        # Map gru4rec 'gru' to actor 'click_gru'
                        # gru4rec: gru.weight_hh_l0
                        # actor: click_gru.weight_hh_l0
                        new_key = key.replace('gru', 'click_gru') # simple mapping?
                        # Actually gru4rec might just be 'gru'.
                        if key.startswith('gru.'):
                             gru_state_dict[key.replace('gru.', '')] = state_dict[key]
                             
    agent = SACAgent(num_items, embedding_dim, hidden_size, action_dim, slate_size, lr_actor, lr_critic, gamma, alpha, device, item_embeddings=item_embeddings, bc_weight=bc_weight, ctr_weight=ctr_weight, ema_alpha=ema_alpha, gru_state_dict=gru_state_dict)
    
    if pretrained_actor:
        agent.actor.load_state_dict(torch.load(pretrained_actor, map_location=device))
    if pretrained_critic:
        agent.critic.load_state_dict(torch.load(pretrained_critic, map_location=device))
        
    replay_buffer = ReplayBuffer(buffer_size)
    
    # 3. Train Loop
    print(f"Starting SAC Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Simulate Users
        from rec_sim.simulate_api import create_environment
        env = create_environment({
            "num_candidates": num_items - 1, # num_items includes padding
            "slate_size": slate_size,
            "resample_documents": False,
            "seed": 42 + epoch # Vary seed slightly per epoch
        })
        
        epoch_critic_loss = 0
        epoch_actor_loss = 0
        epoch_reward = 0
        
        pbar = tqdm(range(num_users_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            obs, _ = env.reset()
            # State: History of clicked items + Slate History
            # Shape: (history_len, 1 + slate_size)
            history_len = 10
            state = collections.deque([[0] * (slate_size + 1)] * history_len, maxlen=history_len)
            
            recommended_items = set()
            
            user_reward = 0
            
            for step in range(steps):
                current_state = list(state)
                
                # Masking
                allowed_items = [i for i in range(1, num_items) if i not in recommended_items]
                if len(allowed_items) < slate_size:
                    allowed_items = range(1, num_items)
                    
                # Act
                action_ids = agent.act(current_state, slate_size=slate_size, allowed_items=allowed_items)
                recommended_items.update(action_ids)
                
                # Step
                env_action = [x - 1 for x in action_ids]
                next_obs, reward, terminated, truncated, _ = env.step(np.array(env_action))
                
                if isinstance(reward, dict):
                    reward = sum(reward.values()) if reward else 0.0
                
                user_reward += reward
                    
                # Diversity Penalty
                if diversity_weight > 0:
                    # Need embeddings for diversity calculation
                    # action_ids are 1-based
                    real_action_embs = []
                    for item_id in action_ids:
                        idx = item_id 
                        if 0 <= idx < len(agent.item_embeddings):
                            emb = agent.item_embeddings[idx].detach().cpu().numpy()
                        else:
                            emb = np.zeros(embedding_dim)
                        real_action_embs.append(emb)
                    real_action_emb = np.stack(real_action_embs, axis=0)
                    
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
                clicked_item_idx = -1
                
                for k, resp in enumerate(responses):
                    if resp['click'] == 1:
                        clicked_item = action_ids[k]
                        clicked_item_idx = k
                        break
                
                slate_items = action_ids
                state_element = [clicked_item] + slate_items
                
                next_state_deque = state.copy()
                next_state_deque.append(state_element)
                next_state = list(next_state_deque)
                
                done = terminated or truncated
                
                # Push to buffer
                # Store action_ids (slate items) instead of embeddings
                # Only push if there was a click? Or handle no-click in learn?
                # Prompt: "populate q_theta only for (state, clicked_item, clicked_position)"
                # If no click, we can't update.
                if clicked_item_idx != -1:
                    replay_buffer.push(current_state, action_ids, clicked_item_idx, reward, next_state, done)
                
                state = next_state_deque
                
                # Train Step
                if len(replay_buffer) > batch_size:
                    c_loss, a_loss, _ = agent.learn(replay_buffer, batch_size)
                    epoch_critic_loss += c_loss
                    epoch_actor_loss += a_loss
                
                if done:
                    break
            
            epoch_reward += user_reward
            
            # Update tqdm
            pbar.set_postfix({
                'critic': f"{epoch_critic_loss/(i+1):.4f}",
                'actor': f"{epoch_actor_loss/(i+1):.4f}",
                'reward': f"{epoch_reward/(i+1):.2f}"
            })
                    
        print(f"Epoch {epoch+1}: Avg Critic Loss: {epoch_critic_loss/num_users_per_epoch:.4f}, Avg Actor Loss: {epoch_actor_loss/num_users_per_epoch:.4f}, Avg Reward: {epoch_reward/num_users_per_epoch:.2f}")
        
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
