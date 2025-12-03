import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from recommender.utils import TrajectoryDataset, collate_fn
from recommender.gru4rec import GRU4Rec

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



def train_ppo(
    num_items=2000,
    embedding_dim=64,
    hidden_size=128,
    action_dim=64,
    pretrained_actor=None,
    pretrained_critic=None,
    gru4rec_path="trained_models/gru4rec",
    output_dir="trained_models/ppo",
    epochs=5,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    K_epochs=4,
    eps_clip=0.2,
    slate_size=1,
    num_users_per_epoch=100,
    diversity_weight=0.5,
    rnd_weight=0.0,
    mask_top_p=0.0, mask_prob=0.0,
    top_p=1.0, top_k=0, use_gumbel=False,
    temp_start=1.0, temp_end=1.0,
    similarity_coef=0.0, similarity_top_k=10, kl_coef=0.0,
    use_bc=True,
    drift_scale=0.1,
    device='cpu',
    entropy_coef=0.01, # Added entropy_coef with a default value
    steps_per_user=30 # Added steps_per_user argument
):
    # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print(f"Training PPO Agent on device: {device}")
    print(f"Drift Scale: {drift_scale}")
    print(f"Entropy Coef: {entropy_coef}")
    print(f"Use BC: {use_bc}")
    print(f"Masking: Top {mask_top_p*100}% with prob {mask_prob}")
    print(f"Top-p: {top_p}, Gumbel: {use_gumbel}")
    print(f"Temperature: {temp_start} -> {temp_end}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_full_path = os.path.join(base_dir, output_dir)
    
    if not os.path.exists(output_full_path):
        os.makedirs(output_full_path)
        
    # Initialize Agent
    # Load Item Embeddings from GRU4Rec (Shared Embeddings)
    item_embeddings = None
    gru_state_dict = None
    
    if gru4rec_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Check if gru4rec_path is absolute or relative
        if not os.path.isabs(gru4rec_path):
             # Try relative to project root
             pass
        
        print(f"Loading GRU4Rec embeddings from {gru4rec_path}")
        checkpoint_path = os.path.join(base_dir, gru4rec_path, "gru4rec_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'embedding.weight' in state_dict:
                    item_embeddings = state_dict['embedding.weight']
                    print(f"Loaded item embeddings: {item_embeddings.shape}")
                
                if use_bc:
                    gru_state_dict = {}
                    for key in state_dict:
                        if key.startswith('gru.'):
                             gru_state_dict[key.replace('gru.', '')] = state_dict[key]
    
    from recommender.ppo_agent import PPOAgent
    agent = PPOAgent(num_items, embedding_dim, hidden_size, action_dim, slate_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coef, device, item_embeddings=item_embeddings, gru_state_dict=gru_state_dict)
    
    # Pretrained actor/critic loading (if any)
    # The original code had these, but the diff removed them.
    # Keeping them commented out for now if they were intended to be preserved.
    # if pretrained_actor:
    #     agent.actor.load_state_dict(torch.load(pretrained_actor, map_location=device))
    # if pretrained_critic:
    #     agent.critic.load_state_dict(torch.load(pretrained_critic, map_location=device))
    
    # PHASE 1: Behavior Cloning from GRU4Rec
    if use_bc and gru4rec_path and os.path.exists(os.path.join(base_dir, gru4rec_path)):
        print("\n=== PHASE 1: Behavior Cloning from GRU4Rec ===")
        from recommender.gru4rec import GRU4Rec
        gru_model_path = os.path.join(base_dir, gru4rec_path, "gru4rec_model.pth")
        checkpoint = torch.load(gru_model_path, map_location=device)
        gru_embedding_dim = checkpoint.get('embedding_dim', 64)
        gru_hidden_size = checkpoint.get('hidden_size', 128)
        
        gru_model = GRU4Rec(num_items, gru_hidden_size, num_items, embedding_dim=gru_embedding_dim).to(device)
        gru_model.load_state_dict(checkpoint['model_state_dict'])
        gru_model.eval()
        
        bc_epochs = 2
        bc_optimizer = optim.Adam(agent.actor.parameters(), lr=lr_actor * 3)
        
        for bc_epoch in range(bc_epochs):
            total_bc_loss = 0.0
            num_bc_steps = 0
            
            # Create environment for BC
            from rec_sim.simulate_api import create_environment
            bc_env = create_environment({
                "num_candidates": num_items,
                "slate_size": slate_size,
                "resample_documents": False,
                "seed": 42 + bc_epoch,
                "drift_scale": drift_scale
            })
            
            for user_idx in tqdm(range(min(50, num_users_per_epoch)), desc=f"BC Epoch {bc_epoch+1}/{bc_epochs}"):
                bc_env.reset()
                history_len = 10
                state = collections.deque([[0] * (slate_size + 1)] * history_len, maxlen=history_len)
                
                for step in range(30):  # Shorter episodes for BC
                    current_state = list(state)
                    
                    # Get GRU4Rec's predictions (teacher)
                    with torch.no_grad():
                        history = [int(s[0]) for s in current_state if s[0] > 0]
                        if len(history) > 0:
                            input_seq = torch.LongTensor([history]).to(device)
                            gru_logits, _ = gru_model(input_seq)
                            teacher_logits = gru_logits[0, -1, :]
                        else:
                            teacher_logits = torch.zeros(num_items).to(device)
                    
                    # Get PPO Actor's predictions (student)
                    state_tensor = torch.FloatTensor([current_state]).to(device)
                    student_logits = agent.actor(state_tensor)[0]
                    
                    # KL divergence loss
                    teacher_probs = torch.softmax(teacher_logits, dim=0)
                    student_log_probs = torch.log_softmax(student_logits, dim=0)
                    bc_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                    
                    bc_optimizer.zero_grad()
                    bc_loss.backward()
                    bc_optimizer.step()
                    
                    total_bc_loss += bc_loss.item()
                    num_bc_steps += 1
                    
                    # Take action
                    scores = teacher_logits.cpu().numpy()
                    scores[0] = -np.inf
                    top_k_indices = np.argsort(scores)[-slate_size:][::-1]
                    action_ids = [int(idx) for idx in top_k_indices]
                    
                    env_action = [x - 1 for x in action_ids]
                    next_obs, reward, terminated, truncated, _ = bc_env.step(np.array(env_action))
                    
                    # Update state
                    responses = next_obs['response']
                    clicked_item = 0
                    for i, resp in enumerate(responses):
                        if resp['click'] == 1:
                            clicked_item = action_ids[i]
                            break
                    
                    state_element = [clicked_item] + action_ids
                    state.append(state_element)
                    
                    if terminated or truncated:
                        break
            
            avg_bc_loss = total_bc_loss / num_bc_steps if num_bc_steps > 0 else 0
            print(f"BC Epoch {bc_epoch+1}: Avg KL Loss = {avg_bc_loss:.4f}")
        
        print("=== PHASE 2: RL Fine-tuning ===\n")
        
    # Train Loop
    print(f"Starting PPO Training for {epochs} epochs...")
    
    # steps_per_user is now passed as argument
    update_timestep = 2000 # Update every 2000 timesteps (approx 20 users)
    timestep = 0
    
    for epoch in range(epochs):
        from rec_sim.simulate_api import create_environment
        env = create_environment({
            "num_candidates": num_items,
            "slate_size": slate_size,
            "resample_documents": False,
            "seed": 42 + epoch,
            "drift_scale": drift_scale
        })
        
        epoch_reward = 0
        
        pbar = tqdm(range(num_users_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for user_idx in pbar:
            obs, _ = env.reset()
            history_len = 10
            state = collections.deque([[0] * (slate_size + 1)] * history_len, maxlen=history_len)
            consumed_items = set()
            user_reward = 0
            
            for step in range(steps_per_user):
                current_state = list(state)
                
                # No consumed items masking for fair comparison with GRU4Rec baseline
                allowed_items = range(1, num_items)
                
                # Temperature scheduling: linear decay from temp_start to temp_end
                progress = (epoch * num_users_per_epoch + user_idx) / (epochs * num_users_per_epoch)
                current_temp = temp_start + (temp_end - temp_start) * progress
                
                # Select Action
                # Pass masking and sampling parameters
                action_ids, action_emb, log_prob = agent.select_action(
                    current_state, 
                    allowed_items=allowed_items, 
                    temperature=current_temp,
                    training=True,
                    mask_top_p=mask_top_p,
                    mask_prob=mask_prob,
                    top_p=top_p,
                    top_k=top_k,
                    use_gumbel=use_gumbel
                )
                
                # Update Popularity
                agent.update_popularity(action_ids)
                
                # Step
                env_action = [x - 1 for x in action_ids]
                next_obs, reward, terminated, truncated, _ = env.step(np.array(env_action))
                
                if isinstance(reward, dict):
                    reward = sum(reward.values()) if reward else 0.0
                
                user_reward += reward
                
                # Diversity Penalty
                if diversity_weight > 0 and slate_size > 1:
                    real_action_embs = []
                    for item_id in action_ids:
                        idx = item_id 
                        if 0 <= idx < len(agent.item_embeddings):
                            emb = agent.item_embeddings[idx].detach().cpu().numpy()
                        else:
                            emb = np.zeros(embedding_dim)
                        real_action_embs.append(emb)
                    real_action_emb = np.stack(real_action_embs, axis=0)
                    
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
                for k, resp in enumerate(responses):
                    if resp['click'] == 1:
                        clicked_item = action_ids[k]
                        break
                
                # Pad action_ids to slate_size for state consistency
                slate_items = action_ids + [0] * (slate_size - len(action_ids))
                state_element = [clicked_item] + slate_items
                
                next_state_deque = state.copy()
                next_state_deque.append(state_element)
                next_state = list(next_state_deque)
                
                done = terminated or truncated
                
                # Get GRU4Rec's predictions (teacher) for distillation
                teacher_logit = None
                if 'gru_model' in locals():
                    with torch.no_grad():
                        history = [int(s[0]) for s in current_state if s[0] > 0]
                        if len(history) > 0:
                            input_seq = torch.LongTensor([history]).to(device)
                            gru_logits, _ = gru_model(input_seq)
                            teacher_logit = gru_logits[0, -1, :]
                        else:
                            teacher_logit = torch.zeros(num_items).to(device)

                # Add to buffer
                agent.buffer.add(current_state, action_ids, reward, done, log_prob, teacher_logit=teacher_logit)
                
                state = next_state_deque
                if clicked_item > 0:
                    consumed_items.add(clicked_item)
                
                timestep += 1
                
                # Update PPO agent
                if timestep % update_timestep == 0:
                    agent.update(similarity_coef=similarity_coef, 
                               similarity_top_k=similarity_top_k, 
                               kl_coef=kl_coef)
                    timestep = 0
                
                if done:
                    break
            
            epoch_reward += user_reward
            pbar.set_postfix({'AvgR': f"{epoch_reward/(user_idx+1):.2f}", 'CurR': f"{user_reward:.2f}"})
        
        # Update at end of epoch if buffer not empty
        if len(agent.buffer.states) > 0:
            agent.update()
            
        print(f"Epoch {epoch+1}: Avg Reward: {epoch_reward/num_users_per_epoch:.2f}")
        
        # Save
        save_path = os.path.join(output_full_path, f"ppo_model_epoch_{epoch+1}.pth")
        torch.save(agent.actor.state_dict(), save_path)
        print(f"Saved model to {save_path}")

def train(
    model_type,
    **kwargs
):
    if model_type == 'gru4rec':
        train_gru4rec(**kwargs)
    elif model_type == 'sac':
        train_sac(**kwargs)
    elif model_type == 'ppo':
        train_ppo(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
