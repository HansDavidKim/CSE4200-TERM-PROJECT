import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        sys.exit(1)

def run_experiment(num_users, num_items, steps):
    exp_name = f"{num_users}_{num_items}_{steps}"
    base_dir = os.path.abspath(f"experiments/{exp_name}")
    
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    gru_model_dir = os.path.join(base_dir, "models", "gru4rec")
    sac_model_dir = os.path.join(base_dir, "models", "sac")
    results_dir = os.path.join(base_dir, "results")
    
    print(f"\n=== Starting Experiment: {exp_name} ===")
    
    # 1. Generate Data
    run_command(f"python main.py generate-data --num-users {num_users} --num-items {num_items} --steps {steps} --output-dir {raw_dir}")
    
    # 2. Process Data
    # main.py generate-data saves as data_{num_users}_{num_items}_{steps}.csv
    raw_file = os.path.join(raw_dir, f"data_{num_users}_{num_items}_{steps}.csv")
    run_command(f"python main.py process-dataset --input-file {raw_file} --output-dir {processed_dir}")
    
    # 3. Train GRU4Rec
    # Note: train_gru4rec needs input_dir where processed files are
    run_command(f"python main.py train gru4rec --input-dir {processed_dir} --output-dir {gru_model_dir}")
    
    # 4. Train SAC
    # SAC needs GRU4Rec path to load embeddings/weights
    run_command(f"python main.py train sac --gru4rec-path {gru_model_dir} --output-dir {sac_model_dir}")
    
    # 5. Evaluate GRU4Rec
    run_command(f"python main.py evaluate gru4rec --episodes 100 --model-path {gru_model_dir} --output-dir {results_dir} --num-items {num_items}")
    
    # 6. Evaluate SAC
    run_command(f"python main.py evaluate sac --episodes 100 --model-path {sac_model_dir} --gru4rec-path {gru_model_dir} --output-dir {results_dir} --num-items {num_items}")
    
    # 7. Evaluate Hybrid
    run_command(f"python main.py evaluate hybrid --episodes 100 --model-path {gru_model_dir} --output-dir {results_dir} --num-items {num_items}")
    
    print(f"=== Completed Experiment: {exp_name} ===\n")

if __name__ == "__main__":
    configs = [
        (200, 200, 30),
        (2000, 2000, 100),
        #(10000, 10000, 100)
    ]
    
    for u, i, s in configs:
        run_experiment(u, i, s)
