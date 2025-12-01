import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from run_experiments import run_experiment

if __name__ == "__main__":
    # Small config for fast verification
    # Only run SAC as requested
    # Run a full experiment
    # run_experiment(num_users=100, num_items=200, steps=20, epochs=5, ctr_weight=0.1, ema_alpha=0.1, lr_actor=1e-4, lr_critic=1e-4, gamma=0.95, bc_weight=1.0, run_gru4rec=True, run_hybrid=True)
    run_experiment(
        num_users=2000,
        num_items=2000,
        steps=100,
        epochs=5,
        ctr_weight=0.1,
        ema_alpha=0.1,
        lr_actor=1e-4, 
        lr_critic=1e-4, 
        gamma=0.95, 
        bc_weight=1.0, 
        run_gru4rec=True, 
        run_hybrid=True
    )