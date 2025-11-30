import pandas as pd
import numpy as np
import os
import pickle

def create_trajectories(input_path, output_path):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # User features columns
    user_feat_cols = [f'user_{i}' for i in range(20)]
    
    trajectories = {}
    
    print("Grouping and processing trajectories...")
    grouped = df.groupby('user_id')
    
    for user_id, group in grouped:
        # Sort by step to ensure sequence
        group = group.sort_values('step')
        
        user_traj = []
        
        for _, row in group.iterrows():
            step_data = {
                'step': row['step'],
                'user_features': row[user_feat_cols].values.astype(float),
                'action_list': [int(x) for x in str(row['action']).split(',')],
                'reward': row['reward'],
                'selected_item': -1,
                'response': {}
            }
            
            # Identify selected item and response details
            for i in range(5):
                if row[f'resp_{i}_click'] == 1:
                    # The item at index i in action_list was selected
                    if i < len(step_data['action_list']):
                        step_data['selected_item'] = step_data['action_list'][i]
                    
                    # Store detailed response for the selected item
                    step_data['response'] = {
                        'click': 1,
                        'watch_time': row[f'resp_{i}_watch'],
                        'liked': row[f'resp_{i}_liked'],
                        'quality': row[f'resp_{i}_quality'],
                        'cluster': row[f'resp_{i}_cluster']
                    }
                    break
            
            user_traj.append(step_data)
        
        trajectories[user_id] = user_traj

    print(f"Created trajectories for {len(trajectories)} users.")
    
    # Save to pickle
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print("Done!")
    
    # Verification: Print sample
    first_user = list(trajectories.keys())[0]
    print(f"\nSample trajectory for User {first_user}:")
    for step in trajectories[first_user][:3]: # Print first 3 steps
        print(f"Step {step['step']}: Action={step['action_list']}, Selected={step['selected_item']}, Reward={step['reward']}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'dataset', 'data_200_200_30.csv')
    output_pkl = os.path.join(base_dir, 'dataset', 'user_trajectories.pkl')
    
    create_trajectories(input_csv, output_pkl)
