# CSE4200 Term Project: SlateQ & GRU4Rec Recommender System

## Overview
This project implements a comprehensive recommender system framework comparing **SlateQ** (Reinforcement Learning based), **GRU4Rec** (Sequential Recommendation), and a **Hybrid** approach. It utilizes a simulation environment (`rec_sim`) to generate user interaction data, train models, and evaluate their performance in dynamic environments with interest drift.

## Prerequisites
- **Python 3.11+** (Required for `tomllib` support)
- PyTorch
- Typer
- Pandas, Numpy, Scipy, Tqdm

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install torch pandas numpy scipy tqdm typer
   ```

## Configuration
All default hyperparameters and settings are managed in `configs/default.toml`.
You can modify this file to change global settings like:
- **Common**: Random seed.
- **Data Generation**: Number of users, items, steps.
- **Training**: Epochs, batch size, learning rates, model architecture (embedding dim, hidden size).
- **Evaluation**: Number of episodes, drift scale.

## Usage
The project uses a unified CLI `main.py` with sub-commands.

### 1. Data Generation
Generate synthetic user interaction data using the simulation environment.
```bash
python main.py generate-data --num-users 1000 --num-items 2000
```
*Overrides defaults in `configs/default.toml`.*

### 2. Data Processing
Process the generated data into explicit/implicit user-item matrices and user trajectories.
```bash
python main.py process-dataset
```

### 3. Training
Train the recommendation models. Key parameters can be overridden via CLI, others are read from config.

**GRU4Rec (Sequential Model):**
```bash
python main.py train gru4rec --epochs 20 --batch-size 256
```

**SAC (SlateQ / RL Agent):**
```bash
python main.py train sac --epochs 10 --finetune
```
*Note: SAC training typically requires a pre-trained GRU4Rec model for state representation.*

### 4. Evaluation
Evaluate the trained models. Results are saved to `evaluation_report/`.

**GRU4Rec:**
```bash
python main.py evaluate gru4rec --episodes 100
```

**SAC:**
```bash
python main.py evaluate sac --drift 0.1
```

**Hybrid (Content-Based + Collaborative):**
```bash
python main.py evaluate hybrid --alpha 0.5
```

## Project Structure
- **`main.py`**: Main CLI entry point.
- **`configs/`**: Configuration files (`default.toml`).
- **`recommender/`**: Core model implementations and logic.
    - `gru4rec.py`, `sac_agent.py`: Model architectures.
    - `train.py`, `evaluate.py`: Unified training and evaluation logic.
    - `utils.py`: Data utilities.
- **`rec_sim/`**: User simulation environment (based on Google RecSim).
- **`matrix_factorization/`**: Data processing and matrix generation utilities.
- **`dataset/`**: Directory for generated raw and processed data.
- **`trained_models/`**: Directory where trained models are saved.
- **`evaluation_report/`**: Directory for evaluation metrics (JSON).
