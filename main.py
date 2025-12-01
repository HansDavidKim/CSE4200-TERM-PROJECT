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
    output_dir: str = typer.Option(None, help="Override config: output_dir"),
):
    """Generate user interaction data."""
    # Load defaults from config
    c = cfg.generate_data
    n_users = num_users or c.num_users or 200
    n_items = num_items or c.num_items or 200
    n_steps = steps or c.steps or 30
    out_dir = output_dir or c.output_dir or "dataset"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    filename = f"data_{n_users}_{n_items}_{n_steps}.csv"
    output_path = os.path.join(out_dir, filename)
    
    print(f"Generating data: Users={n_users}, Items={n_items}, Steps={n_steps}")
    simulate_users_csv(
        slate_size=5,
        num_candidates=n_items,
        num_users=n_users,
        steps=n_steps,
        file_name=output_path,
        global_seed=cfg.common.seed or 42,
        sim_seed=1,
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

@train_app.command("sac")
def train_sac(
    epochs: int = typer.Option(10, help="Number of epochs"),
    batch_size: int = typer.Option(64, help="Batch size"),
    buffer_size: int = typer.Option(100000, help="Replay buffer size"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    lr_actor: float = typer.Option(3e-4, help="Actor learning rate"),
    lr_critic: float = typer.Option(3e-4, help="Critic learning rate"),
    alpha: float = typer.Option(0.2, help="Entropy regularization coefficient"),
    bc_weight: float = typer.Option(1.0, help="Behavior Cloning weight"),
    ctr_weight: float = typer.Option(0.0, help="CTR Supervision weight"),
    ema_alpha: float = typer.Option(0.1, help="EMA decay rate for user profile"),
    num_items: int = typer.Option(2000, help="Number of items"),
    input_dir: str = typer.Option("experiments/default/processed", help="Directory containing processed data"),
    output_dir: str = typer.Option("experiments/default/models/sac", help="Directory to save models"),
    gru4rec_path: str = typer.Option(None, help="Path to trained GRU4Rec model directory for initialization"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)")
):
    """Train SAC agent."""
    # We need to import train_sac from recommender.train
    from recommender.train import train_sac as train_sac_func
    
    train_sac_func(
        num_items=num_items,
        embedding_dim=64,
        hidden_size=128,
        action_dim=64,
        slate_size=5,
        epochs=epochs,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        alpha=alpha,
        bc_weight=bc_weight,
        ctr_weight=ctr_weight,
        ema_alpha=ema_alpha,
        input_dir=input_dir,
        output_dir=output_dir,
        gru4rec_path=gru4rec_path,
        device=device
    )

# --- Evaluate Commands ---
@eval_app.command("gru4rec")
def evaluate_gru4rec(
    episodes: int = typer.Option(None, help="Override num_episodes"),
    drift: float = typer.Option(None, help="Override drift_scale"),
    model_path: str = typer.Option(None, help="Override model path"),
    output_dir: str = typer.Option(None, help="Override output directory"),
    num_items: int = typer.Option(None, help="Override num_items"),
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
        device=device
    )

@eval_app.command("sac")
def evaluate_sac(
    episodes: int = typer.Option(None, help="Override num_episodes"),
    drift: float = typer.Option(None, help="Override drift_scale"),
    model_path: str = typer.Option(None, help="Override model path"),
    gru4rec_path: str = typer.Option(None, help="Override GRU4Rec path"),
    output_dir: str = typer.Option(None, help="Override output directory"),
    num_items: int = typer.Option(None, help="Override num_items"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)")
):
    """Evaluate SAC."""
    c = cfg.evaluate
    # SAC uses its own output_dir as model path, and GRU4Rec path for embeddings
    eval_func(
        'sac',
        model_path or cfg.train.sac.output_dir or "trained_models/sac",
        episodes or c.num_episodes or 100,
        drift if drift is not None else (c.drift_scale or 0.0),
        c.seed or 42,
        output_dir or c.output_dir or "evaluation_report",
        gru4rec_path=gru4rec_path or c.gru4rec_path or "trained_models/gru4rec",
        num_items=num_items or c.num_items or 2000,
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
        device=device
    )

if __name__ == "__main__":
    app()
