
import os
import subprocess
import pandas as pd
import json
import time

# Configuration
DRIFT_SCALES = [0.1, 0.5, 1.0]
NUM_ITEMS = 500
NUM_USERS_TRAIN = 500
STEPS_DATASET = 30      # For GRU4Rec training data
STEPS_PPO_TRAIN = 100   # For PPO online training
MAX_STEPS_EVAL = 150    # For evaluation
EPOCHS_GRU = 5
EPOCHS_PPO = 10
DEVICE = "cpu"

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    # 1. Cleanup
    print("=== Cleaning up ===")
    run_command("rm -rf experiments dataset_* preprocessed_* benchmark_results_*.json")

    results = []

    for drift in DRIFT_SCALES:
        print(f"\n\n{'='*50}")
        print(f"Starting Benchmark for Drift Scale: {drift}")
        print(f"{'='*50}\n")

        drift_str = str(drift).replace('.', '_')
        
        # New Directory Structure
        raw_dataset_dir = f"experiments/raw_dataset/{drift_str}"
        preprocessed_dir = f"experiments/dataset/{drift_str}"
        
        # Ensure directories exist
        os.makedirs(raw_dataset_dir, exist_ok=True)
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        gru_model_dir = f"experiments/baseline/gru4rec_{drift_str}"
        ppo_model_dir = f"experiments/ppo/best_{drift_str}"
        
        # --- A. Data Generation & GRU4Rec Training ---
        print(f"--- [Drift {drift}] Generating Data ---")
        run_command(f"python main.py generate-data --num-users {NUM_USERS_TRAIN} --num-items {NUM_ITEMS} --steps {STEPS_DATASET} --drift-scale {drift} --output-dir {raw_dataset_dir}")
        
        print(f"--- [Drift {drift}] Processing Data ---")
        # Find the generated csv file
        csv_file = [f for f in os.listdir(raw_dataset_dir) if f.endswith('.csv')][0]
        run_command(f"python main.py process-dataset --input-file {raw_dataset_dir}/{csv_file} --output-dir {preprocessed_dir}")

        print(f"--- [Drift {drift}] Training GRU4Rec ---")
        run_command(f"python main.py train gru4rec --input-dir {preprocessed_dir} --epochs {EPOCHS_GRU} --output-dir {gru_model_dir} --device {DEVICE}")

        # --- B. PPO Training ---
        print(f"--- [Drift {drift}] Training PPO ---")
        # Note: PPO trains online, so we pass drift-scale directly.
        # We use the same settings as the successful run: Top-K=10, SimCoef=0.01 (User Request), Steps=100
        run_command(f"python main.py train ppo --num-items {NUM_ITEMS} --epochs {EPOCHS_PPO} --top-k 7 --similarity-coef 0.50 --similarity-top-k 5 --drift-scale {drift} --max-steps {STEPS_PPO_TRAIN} --output-dir {ppo_model_dir} --device {DEVICE}")

        # --- C. Evaluation ---
        print(f"--- [Drift {drift}] Evaluating Models ---")
        
        # 1. GRU4Rec
        eval_dir_gru = f"experiments/benchmark/gru4rec_{drift_str}"
        run_command(f"python main.py evaluate gru4rec --num-items {NUM_ITEMS} --max-steps {MAX_STEPS_EVAL} --drift {drift} --model-path {gru_model_dir} --output-dir {eval_dir_gru} --device {DEVICE}")
        
        # 2. PPO
        eval_dir_ppo = f"experiments/benchmark/ppo_{drift_str}"
        ppo_model_path = f"{ppo_model_dir}/ppo_model_epoch_{EPOCHS_PPO}.pth"
        run_command(f"python main.py evaluate ppo --num-items {NUM_ITEMS} --top-k 10 --max-steps {MAX_STEPS_EVAL} --drift {drift} --model-path {ppo_model_path} --gru4rec-path {gru_model_dir} --output-dir {eval_dir_ppo} --device {DEVICE}")

        # 3. Hybrid
        eval_dir_hybrid = f"experiments/benchmark/hybrid_{drift_str}"
        run_command(f"python main.py evaluate hybrid --num-items {NUM_ITEMS} --max-steps {MAX_STEPS_EVAL} --drift {drift} --model-path {gru_model_dir} --output-dir {eval_dir_hybrid} --device {DEVICE}")

        # --- D. Collect Results ---
        def load_result(path, model_name):
            try:
                # Find json file in directory
                json_files = [f for f in os.listdir(path) if f.endswith('.json')]
                if not json_files:
                    print(f"No JSON result found in {path}")
                    return None
                
                # Assume the first one is the correct one (usually only one)
                with open(os.path.join(path, json_files[0]), 'r') as f:
                    data = json.load(f)
                    data['drift'] = drift
                    data['model'] = model_name
                    return data
            except Exception as e:
                print(f"Error loading result for {model_name} at {drift}: {e}")
                return None

        results.append(load_result(eval_dir_gru, "GRU4Rec"))
        results.append(load_result(eval_dir_ppo, "PPO"))
        results.append(load_result(eval_dir_hybrid, "Hybrid"))

    # Filter None results
    results = [r for r in results if r]

    # Save all results as JSON
    benchmark_dir = "experiments/benchmark"
    os.makedirs(benchmark_dir, exist_ok=True)
    
    json_path = os.path.join(benchmark_dir, "benchmark_results_final.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Save all results as CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(benchmark_dir, "benchmark_results_final.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved benchmark results to {csv_path}")
    
    print("\n\n=== Benchmark Completed ===")

if __name__ == "__main__":
    main()
