import os
import pickle
import numpy as np
import collections
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.data = []
        for user_id, steps in trajectories.items():
            # Extract sequence of selected items
            # Filter out steps where no item was selected (selected_item == -1)
            item_seq = [step['selected_item'] for step in steps if step['selected_item'] != -1]
            
            if len(item_seq) > 1:
                self.data.append(torch.tensor(item_seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    padded_seqs = pad_sequence(batch, batch_first=True, padding_value=0)
    inputs = padded_seqs[:, :-1]
    targets = padded_seqs[:, 1:]
    return inputs, targets

def prepare_offline_data(
    input_file="dataset/user_trajectories.pkl",
    output_file="recommender/offline_buffer.pkl",
    history_len=10
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, input_file)
    output_path = os.path.join(base_dir, output_file)
    
    print(f"Loading trajectories from {input_path}...")
    try:
        with open(input_path, 'rb') as f:
            trajectories = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return

    buffer_data = []
    
    print("Processing trajectories...")
    for user_id, traj in tqdm(trajectories.items()):
        state = collections.deque([0] * history_len, maxlen=history_len)
        
        for step_data in traj:
            current_state = list(state)
            
            env_action_list = step_data['action_list']
            agent_action_list = [x + 1 for x in env_action_list]
            
            reward = step_data['reward']
            selected_item = step_data['selected_item']
            
            clicked_item_agent_id = 0
            if selected_item != -1:
                clicked_item_agent_id = selected_item + 1
            
            next_state_deque = state.copy()
            next_state_deque.append(clicked_item_agent_id)
            next_state = list(next_state_deque)
            
            done = False 
            
            buffer_data.append((current_state, agent_action_list, reward, next_state, done))
            
            state = next_state_deque
            
    print(f"Created buffer with {len(buffer_data)} transitions.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(buffer_data, f)
    print("Done!")
