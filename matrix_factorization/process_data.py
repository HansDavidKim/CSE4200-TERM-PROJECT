import pandas as pd
import os

def process_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    interactions = []

    print("Processing rows...")
    for index, row in df.iterrows():
        user_id = row['user_id']
        # action is a string like "95,15,30,158,128"
        action_list = str(row['action']).split(',')
        
        # Check which item was clicked
        selected_item_index = -1
        for i in range(5):
            if row[f'resp_{i}_click'] == 1:
                selected_item_index = i
                break
        
        if selected_item_index != -1:
            # Get the item ID from the action list using the index
            if selected_item_index < len(action_list):
                item_id = action_list[selected_item_index].strip()
                reward = row['reward']
                
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'reward': reward
                })

    if not interactions:
        print("No interactions found.")
        return

    result_df = pd.DataFrame(interactions)
    
    # Save Explicit Matrix
    print(f"Saving {len(result_df)} interactions to {output_path}...")
    result_df.to_csv(output_path, index=False)
    
    # Save Implicit Matrix
    # Implicit feedback is binary (1 for click)
    implicit_df = result_df[['user_id', 'item_id']].copy()
    implicit_df['value'] = 1
    
    implicit_output_path = output_path.replace('explicit', 'implicit')
    print(f"Saving {len(implicit_df)} interactions to {implicit_output_path}...")
    implicit_df.to_csv(implicit_output_path, index=False)
    
    print("Done!")
    
    # Show sample
    print("\nSample of generated explicit matrix data:")
    print(result_df.head())
    print("\nSample of generated implicit matrix data:")
    print(implicit_df.head())

if __name__ == "__main__":
    # Define paths relative to the project root
    # Assuming script is run from project root or we handle paths carefully
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'dataset', 'data_200_200_30.csv')
    output_csv = os.path.join(base_dir, 'dataset', 'explicit_user_item_matrix.csv')
    
    process_data(input_csv, output_csv)
