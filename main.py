import typer
import os
import tomllib
import sys
from typing import Optional, Any

# Add project root and rec_sim directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'rec_sim'))

from rec_sim.simulate_api import simulate_users_csv
from matrix_factorization.process_data import process_data as process_data_func
from matrix_factorization.create_trajectory import create_trajectories as create_trajectories_func
from recommender.train import train as train_func
from recommender.evaluate import evaluate as eval_func
from recommender.utils import prepare_offline_data as prepare_offline_data_func

# --- Config Loader ---
class Config:
    def __init__(self, data: dict = None):
        self._data = data or {}
    
    def __getattr__(self, name: str) -> Any:
        value = self._data.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

config_path = os.path.join(project_root, "configs", "default.toml")
if os.path.exists(config_path):
    with open(config_path, "rb") as f:
        cfg_data = tomllib.load(f)
    cfg = Config(cfg_data)
else:
    print(f"Warning: Config file not found at {config_path}")
    cfg = Config()

# --- CLI App ---
app = typer.Typer(help="Recommender System CLI")
train_app = typer.Typer(help="Train models")
eval_app = typer.Typer(help="Evaluate models")

app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="evaluate")

@app.callback()
def callback():
    pass

# --- Data Commands ---
@app.command()
def generate_data(
    num_users: int = typer.Option(None, help="Override config: num_users"),
    num_items: int = typer.Option(None, help="Override config: num_items"),
    steps: int = typer.Option(None, help="Override config: steps"),
    drift_scale: float = typer.Option(None, help="Override config: drift_scale"),
    output_dir: str = typer.Option(None, help="Override config: output_dir"),
):
    """Generate user interaction data."""
    # Load defaults from config
    c = cfg.generate_data
    n_users = num_users or c.num_users or 200
    n_items = num_items or c.num_items or 200
    n_steps = steps or c.steps or 30
    d_scale = drift_scale or c.drift_scale or 1.0
    out_dir = output_dir or c.output_dir or "dataset"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    filename = f"data_{n_users}_{n_items}_{n_steps}.csv"
    output_path = os.path.join(out_dir, filename)
    
    print(f"Generating data: Users={n_users}, Items={n_items}, Steps={n_steps}, Drift={d_scale}")
    simulate_users_csv(
        slate_size=5,
        num_candidates=n_items,
        num_users=n_users,
        steps=n_steps,
        file_name=output_path,
        global_seed=cfg.common.seed or 42,
        sim_seed=1,
        drift_scale=d_scale,
    )
    print(f"Saved to {output_path}")

@app.command()
def process_dataset(
    input_file: str = typer.Option(None, help="Override input file"),
    output_dir: str = typer.Option(None, help="Override output directory"),
):
    """Process raw dataset."""
    c = cfg.process_dataset
    inp = input_file or c.input_file or "dataset/data_200_200_30.csv"
    out_dir = output_dir or c.output_dir or "preprocessed_dataset"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print(f"Processing {inp} -> {out_dir}")
    process_data_func(inp, os.path.join(out_dir, "explicit_user_item_matrix.csv"))
    create_trajectories_func(inp, os.path.join(out_dir, "user_trajectories.pkl"))
    print("Done.")

@app.command()
def prepare_offline_data():
    """Convert trajectories to offline buffer (Config only)."""
    c = cfg.prepare_offline_data
    prepare_offline_data_func(
        c.input_file or "dataset/user_trajectories.pkl",
        c.output_file or "recommender/offline_buffer.pkl",
        c.history_len or 10
    )

# --- Train Commands ---
@train_app.command("gru4rec")
def train_gru4rec(
    epochs: int = typer.Option(None, help="Override epochs"),
    lr: float = typer.Option(None, help="Override learning rate"),
    input_dir: str = typer.Option(None, help="Override input directory"),
    output_dir: str = typer.Option(None, help="Override output directory"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)")
):
    """Train GRU4Rec model."""
    c = cfg.train.gru4rec
    train_func(
        'gru4rec',
        output_dir=output_dir or c.output_dir or "trained_models/gru4rec",
        epochs=epochs or c.epochs or 10,
        batch_size=c.batch_size or 256,
        embedding_dim=c.embedding_dim or 64,
        hidden_size=c.hidden_size or 128,
        dropout=c.dropout or 0.1,
        lr=lr or c.lr or 0.001,
        input_dir=input_dir or "preprocessed_dataset",
        seed=c.seed or cfg.common.seed or 42,
        device=device
    )

@train_app.command("ppo")
def train_ppo(
    epochs: int = typer.Option(5, help="Number of epochs"),
    batch_size: int = typer.Option(64, help="Batch size"),
    lr_actor: float = typer.Option(3e-4, help="Actor learning rate"),
    lr_critic: float = typer.Option(1e-3, help="Critic learning rate"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    K_epochs: int = typer.Option(4, help="PPO update epochs"),
    eps_clip: float = typer.Option(0.2, help="PPO clip parameter"),
    entropy_coef: float = typer.Option(0.15, help="Entropy regularization coefficient"),
    diversity_weight: float = typer.Option(0.5, help="Diversity Penalty weight"),
    rnd_weight: float = typer.Option(0.0, help="RND Intrinsic Reward weight"),
    mask_top_p: float = typer.Option(0.0, help="Percentage of top items to mask (0.0-1.0)"),
    mask_prob: float = typer.Option(0.0, help="Probability of applying mask (0.0-1.0)"),
    top_p: float = typer.Option(1.0, help="Nucleus sampling threshold (0.0-1.0, 1.0=disabled)"),
    top_k: int = typer.Option(0, help="Top-K sampling (0=disabled)"),
    use_gumbel: bool = typer.Option(False, help="Use Gumbel-Softmax trick for differentiable sampling"),
    temp_start: float = typer.Option(1.0, help="Starting temperature for exploration"),
    temp_end: float = typer.Option(1.0, help="Ending temperature for exploitation"),
    similarity_coef: float = typer.Option(0.0, help="Coefficient for similarity-based soft labeling loss"),
    similarity_top_k: int = typer.Option(10, help="Top-K similar items for soft labeling"),
    kl_coef: float = typer.Option(0.0, help="Coefficient for KL divergence penalty (BC)"),
    use_bc: bool = typer.Option(True, help="Use Behavior Cloning initialization"),
    drift_scale: float = typer.Option(0.1, help="Drift scale for environment"),
    num_items: int = typer.Option(2000, help="Number of items"),
    max_steps: int = typer.Option(30, help="Max steps per episode"),
    output_dir: str = typer.Option("experiments/default/models/ppo", help="Directory to save models"),
    gru4rec_path: str = typer.Option(None, help="Path to trained GRU4Rec model directory for initialization"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)")
):
    """Train PPO agent."""
    from recommender.train import train_ppo as train_ppo_func
    
    train_ppo_func(
        num_items=num_items,
        embedding_dim=64,
        hidden_size=128,
        action_dim=64,
        slate_size=5,
        epochs=epochs,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,

        eps_clip=eps_clip,
        entropy_coef=entropy_coef,
        diversity_weight=diversity_weight,
        rnd_weight=rnd_weight,
        mask_top_p=mask_top_p,
        mask_prob=mask_prob,
        top_p=top_p,
        top_k=top_k,
        use_gumbel=use_gumbel,
        temp_start=temp_start,
        temp_end=temp_end,
        similarity_coef=similarity_coef,
        similarity_top_k=similarity_top_k,
        kl_coef=kl_coef,
        use_bc=use_bc,
        drift_scale=drift_scale,
        output_dir=output_dir,
        gru4rec_path=gru4rec_path,
        device=device,
        steps_per_user=max_steps # Pass max_steps as steps_per_user
    )

# --- Evaluate Commands ---
@eval_app.command("gru4rec")
def evaluate_gru4rec(
    episodes: int = typer.Option(None, help="Override num_episodes"),
    drift: float = typer.Option(None, help="Override drift_scale"),
    model_path: str = typer.Option(None, help="Override model path"),
    output_dir: str = typer.Option(None, help="Override output directory"),
    num_items: int = typer.Option(None, help="Override num_items"),
    max_steps: int = typer.Option(30, help="Max steps per episode"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)")
):
    """Evaluate GRU4Rec."""
    c = cfg.evaluate
    eval_func(
        'gru4rec',
        model_path or c.gru4rec_path or "trained_models/gru4rec",
        episodes or c.num_episodes or 100,
        drift if drift is not None else (c.drift_scale or 0.0),
        c.seed or 42,
        output_dir or c.output_dir or "evaluation_report",
        num_items=num_items or c.num_items or 2000,
        max_steps=max_steps,
        device=device
    )

@eval_app.command("ppo")
def evaluate_ppo(
    episodes: int = typer.Option(None, help="Override num_episodes"),
    drift: float = typer.Option(None, help="Override drift_scale"),
    model_path: str = typer.Option(None, help="Override model path"),
    gru4rec_path: str = typer.Option(None, help="Override GRU4Rec path"),
    output_dir: str = typer.Option(None, help="Override output directory"),
    num_items: int = typer.Option(None, help="Override num_items"),
    max_steps: int = typer.Option(30, help="Max steps per episode"),
    top_k: int = typer.Option(0, help="Top-K sampling for evaluation (0=disabled, uses greedy)"),
    temperature: float = typer.Option(1.0, help="Sampling temperature"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)")
):
    """Evaluate PPO."""
    c = cfg.evaluate
    eval_func(
        'ppo',
        model_path or cfg.train.ppo.output_dir or "trained_models/ppo",
        episodes or c.num_episodes or 100,
        drift if drift is not None else (c.drift_scale or 0.0),
        c.seed or 42,
        output_dir or c.output_dir or "evaluation_report",
        gru4rec_path=gru4rec_path or c.gru4rec_path or "trained_models/gru4rec",
        num_items=num_items or c.num_items or 2000,
        max_steps=max_steps,
        top_k=top_k,
        temperature=temperature,
        device=device
    )

@eval_app.command("hybrid")
def evaluate_hybrid(
    episodes: int = typer.Option(None, help="Override num_episodes"),
    alpha: float = typer.Option(None, help="Override alpha"),
    drift: float = typer.Option(None, help="Override drift_scale"),
    model_path: str = typer.Option(None, help="Override model path (GRU4Rec path)"),
    output_dir: str = typer.Option(None, help="Override output directory"),
    num_items: int = typer.Option(None, help="Override num_items"),
    max_steps:  int = typer.Option(30, help="Max steps per episode"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)")
):
    """Evaluate Hybrid."""
    c = cfg.evaluate
    eval_func(
        'hybrid',
        model_path or c.gru4rec_path or "trained_models/gru4rec",
        episodes or c.num_episodes or 100,
        drift if drift is not None else (c.drift_scale or 0.0),
        c.seed or 42,
        output_dir or c.output_dir or "evaluation_report",
        alpha=alpha if alpha is not None else (c.alpha or 0.1),
        num_items=num_items or c.num_items or 2000,
        max_steps=max_steps,
        device=device
    )

if __name__ == "__main__":
    app()
