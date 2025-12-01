import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        sys.exit(1)

def run_experiment(num_users, num_items, steps, epochs=5, ctr_weight=0.0, ema_alpha=0.1, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, bc_weight=1.0, run_gru4rec=True, run_hybrid=True):
    exp_name = f"{num_users}_{num_items}_{steps}"
    base_dir = os.path.abspath(f"experiments/{exp_name}")
    
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    gru_model_dir = os.path.join(base_dir, "models", "gru4rec")
    sac_model_dir = os.path.join(base_dir, "models", "sac")
    results_dir = os.path.join(base_dir, "results")
    
    print(f"\n=== Starting Experiment: {exp_name} ===")
    
    # Detect device
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Detected device: {device}")

    # 1. Generate Data
    run_command(f"python main.py generate-data --num-users {num_users} --num-items {num_items} --steps {steps} --output-dir {raw_dir}")
    
    # 2. Process Data
    # main.py generate-data saves as data_{num_users}_{num_items}_{steps}.csv
    raw_file = os.path.join(raw_dir, f"data_{num_users}_{num_items}_{steps}.csv")
    run_command(f"python main.py process-dataset --input-file {raw_file} --output-dir {processed_dir}")
    
    # 3. Train GRU4Rec
    if run_gru4rec:
        # Note: train_gru4rec needs input_dir where processed files are
        run_command(f"python main.py train gru4rec --epochs {epochs} --input-dir {processed_dir} --output-dir {gru_model_dir} --device {device}")
    else:
        print("Skipping GRU4Rec training...")
    
    # 4. Train SAC
    # SAC needs GRU4Rec path to load embeddings/weights
    run_command(f"python main.py train sac --epochs {epochs} --ctr-weight {ctr_weight} --ema-alpha {ema_alpha} --num-items {num_items} --lr-actor {lr_actor} --lr-critic {lr_critic} --gamma {gamma} --bc-weight {bc_weight} --gru4rec-path {gru_model_dir} --output-dir {sac_model_dir} --device {device}")
    
    # 5. Evaluate GRU4Rec
    if run_gru4rec:
        run_command(f"python main.py evaluate gru4rec --episodes 100 --model-path {gru_model_dir} --output-dir {results_dir} --num-items {num_items} --device {device}")
    
    # 6. Evaluate SAC
    run_command(f"python main.py evaluate sac --episodes 100 --model-path {sac_model_dir} --gru4rec-path {gru_model_dir} --output-dir {results_dir} --num-items {num_items} --device {device}")
    
    # 7. Evaluate Hybrid
    if run_hybrid:
        run_command(f"python main.py evaluate hybrid --episodes 100 --model-path {gru_model_dir} --output-dir {results_dir} --num-items {num_items} --device {device}")
    
    print(f"=== Completed Experiment: {exp_name} ===\n")

if __name__ == "__main__":
    configs = [
        (200, 200, 30),
        (2000, 2000, 100),
        #(10000, 10000, 100)
    ]
    
    for u, i, s in configs:
        run_experiment(u, i, s)
