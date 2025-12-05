# System Identification Package

RNN-based identification of nonlinear dynamical systems from data using PyTorch.

## Features

- **Flexible RNN architectures**: SimpleRNN, LSTM, GRU, and Constrained RNN (CRNN) with Lyapunov certificates
- **Custom regularization**: Support for parameter constraints (Lipschitz bounds, stability, sector bounds)
- **Lyapunov stability**: SimpleLure models with certified stability regions and post-processing optimization
- **Data preprocessing**: Automatic normalization (min-max, standard) with direct CSV folder loading
- **Training utilities**: Early stopping, learning rate scheduling, gradient clipping, adaptive regularization
- **MLflow integration**: Automatic experiment tracking and model versioning
- **Comprehensive logging**: Timestamped log files for all operations (see [LOGGING.md](LOGGING.md))
- **Evaluation tools**: Comprehensive metrics and visualization including Lyapunov certificate plots
- **Configurable metrics**: Choose which evaluation metrics to compute and log (see [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md))
- **Model comparison**: Compare multiple trained models with trajectory and error analysis
- **GPU support**: Automatic CUDA/MPS detection
- **Testing**: Comprehensive pytest suite

> **Note**: This package was developed with assistance from GitHub Copilot and Claude (Anthropic) AI coding assistants.

## Installation

### From source

```bash
cd python
pip install -e .
```

### Development installation

```bash
cd python
pip install -e ".[dev]"
```

### Additional dependencies for Constrained RNN

For SimpleLure models with Lyapunov certificate optimization (post-processing):

```bash
pip install cvxpy mosek
```

MOSEK requires a license (free for academia). See https://www.mosek.com/products/academic-licenses/

## Quick Start

### 1. Organize your data

**Recommended**: Organize CSV files in train/test/validation folders:

```
data/prepared/
├── train/
│   ├── sequence_001.csv
│   ├── sequence_002.csv
│   └── ...
├── test/
│   └── ...
└── validation/
    └── ...
```

Each CSV should have columns for input (e.g., `d`) and output (e.g., `e`).

**No preprocessing needed!** The package loads directly from CSV files.

### 2. Configure and train

Use the direct loading config (recommended for simplicity):

```bash
# Train directly from CSV folder structure
python scripts/train.py --config configs/rnn_direct.yaml
```

**Example configuration (Constrained RNN with Lyapunov certificate):**

```yaml
# configs/crnn_gen-sec.yaml
data:
  train_path: "~/genSecSysId-Data/data/prepared"  # Folder with train/test/validation
  input_col: "d"        # Input column name
  output_col: "e"       # Output column name
  state_col: ["x_1", "x_2"]  # State column names
  pattern: "*.csv"
  batch_size: 64
  normalize: false

model:
  model_type: "crnn"    # Constrained RNN with Lyapunov certificate
  nw: 2                 # Hidden state dimension
  nx: 2                 # State dimension
  activation: "dzn"     # Dead-zone nonlinearity
  custom_params:
    learn_L: true       # Learn sector bound matrix L

optimizer:
  optimizer_type: "adam"
  learning_rate: 0.002
  use_scheduler: true
  scheduler_type: "reduce_on_plateau"

training:
  max_epochs: 4000
  use_custom_regularization: true
  regularization_weight: 0.1
  decay_regularization_weight: true  # Interior point method
  min_regularization_weight: 0       # Set to 0 to disable early stopping
  device: "cpu"

mlflow:
  experiment_name: "crnn-generalized-sector"
  log_models: true

evaluation:
  metrics:
    - rmse
    - nrmse
```

**That's it!** The script will:
- ✅ Load all CSV files from folders
- ✅ Normalize data during training
- ✅ Track experiments with MLflow
- ✅ Save best model automatically
- ✅ Train with stability constraints (for CRNN models)

### 3. Evaluate and visualize

Evaluate on test data:

```bash
python scripts/evaluate.py \
    --config configs/crnn_gen-sec.yaml \
    --run-id <run_id_from_mlflow> \
    --test-data ~/genSecSysId-Data/data/prepared/test
```

For SimpleLure models (CRNN with nx=2), this automatically generates:
- Performance metrics (RMSE, NRMSE, etc.)
- Lyapunov ellipse and sector bound polytope visualization

### 4. Post-process for optimal Lyapunov certificate (optional)

For SimpleLure models, optimize P and L matrices via SDP:

```bash
# Feasibility: find valid P and L
python scripts/post_process.py \
    --run-id <run_id> \
    --config configs/crnn_gen-sec.yaml

# Optimization: minimize s for tightest certificate
python scripts/post_process.py \
    --run-id <run_id> \
    --config configs/crnn_gen-sec.yaml \
    --optimize-s
```

This solves a semidefinite program to find the optimal Lyapunov certificate while keeping the trained dynamics (A, B, C, D) fixed.

### 5. Compare multiple models

```bash
python scripts/compare.py \
    --run-ids <run_id_1> <run_id_2> <run_id_3> \
    --test-data ~/genSecSysId-Data/data/prepared/test \
    --output-dir comparisons/my_comparison
```

This generates:
- Summary table with parameters and metrics
- Evaluation metrics comparison
- Training and validation loss curves
- Trajectory comparison (ground truth vs predictions)
- Absolute error plots over time

### 6. Additional utilities

**Export models to MATLAB:**
```bash
python scripts/export_for_matlab.py --run-id <run_id> --output models/model.mat
```

**Generate pedagogical plots:**
```bash
# Local stability motivation (damped pendulum)
python scripts/plot_local_stability_motivation.py --c 1.0 --output-dir figures
```

## Data Loading

The package uses **direct CSV folder loading** (recommended):

```yaml
data:
  train_path: "~/genSecSysId-Data/data/prepared"  # Folder with train/test/validation subfolders
  input_col: "d"               # Column name for input
  output_col: "e"              # Column name for output
  state_col: ["x_1", "x_2"]    # Optional: state column names (for evaluation)
  pattern: "*.csv"             # File pattern
```

✅ **No preprocessing** - loads directly from original CSV files  
✅ **No duplication** - uses original files only  
✅ **Simpler workflow** - one step  
✅ **Easy evaluation** - same folder structure for training and testing

See [docs/DIRECT_LOADING.md](docs/DIRECT_LOADING.md) for details.

## Project Structure

```
python/
├── src/sysid/              # Main package
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions (visualization, Lyapunov plots)
│   ├── data/               # Data loading and preprocessing
│   │   ├── dataset.py      # PyTorch Dataset
│   │   ├── direct_loader.py # Direct CSV folder loading (recommended)
│   │   └── normalizer.py   # Data normalization
│   ├── models/             # Model architectures
│   │   ├── base.py         # Base RNN class
│   │   ├── rnn.py          # RNN implementations (Simple/LSTM/GRU)
│   │   ├── constrained_rnn.py  # SimpleLure with Lyapunov certificates
│   │   └── regularization.py  # Custom regularization
│   ├── training/           # Training utilities
│   │   ├── trainer.py      # Main trainer class with adaptive mechanisms
│   │   ├── losses.py       # Loss functions
│   │   └── optimizers.py   # Optimizer setup
│   └── evaluation/         # Evaluation utilities
│       ├── evaluator.py    # Main evaluator class
│       └── metrics.py      # Evaluation metrics
├── scripts/                # Main scripts
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script with Lyapunov visualization
│   ├── compare.py         # Compare multiple MLflow runs
│   ├── post_process.py    # Post-process models (SDP optimization for P/L)
│   ├── export_for_matlab.py  # Export models to MATLAB .mat format
│   ├── plot_local_stability_motivation.py  # Pedagogical phase portrait
│   └── generate_sample_data.py  # Generate sample data for testing
├── tests/                 # Unit tests
├── configs/               # Example configurations
│   ├── rnn_baseline.yaml  # Simple RNN baseline
│   ├── lstm_baseline.yaml # LSTM baseline
│   ├── constrained_rnn_lmi.yaml  # CRNN with LMI constraints
│   └── constrained_rnn_dual.yaml # CRNN with dual regularization
├── docs/                  # Documentation
│   ├── README.md          # Documentation index
│   ├── DIRECT_LOADING.md  # Guide for direct CSV loading
│   ├── REGULARIZATION_QUICK_START.md  # Regularization guide
│   └── EVALUATION_METRICS.md  # Metrics documentation
├── setup.py              # Package setup
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Advanced Usage

### Custom Models

Extend the `BaseRNN` class to create custom models:

```python
from sysid.models.base import BaseRNN
import torch.nn as nn

class CustomRNN(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(input_size, hidden_size, output_size)
        # Define your architecture
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        return self.fc(out)
    
    def get_regularization_loss(self):
        # Add custom regularization
        return super().get_regularization_loss()
```

### Custom Regularization

```python
from sysid.models.regularization import (
    lipschitz_regularization,
    stability_regularization,
    bounded_parameters_regularization,
)

# In your model's get_regularization_loss method:
def get_regularization_loss(self):
    reg_loss = 0.0
    
    # Lipschitz constraint
    for name, param in self.named_parameters():
        if "weight" in name:
            reg_loss += lipschitz_regularization(param, target_lipschitz=1.0)
    
    # Stability constraint
    if hasattr(self, "rnn"):
        reg_loss += stability_regularization(
            self.rnn.weight_hh_l0,
            target_spectral_radius=0.9
        )
    
    return reg_loss
```

### Hyperparameter Tuning

Modify configuration files or use a hyperparameter search library:

```python
import optuna
from sysid.config import Config

def objective(trial):
    config = Config.from_yaml("configs/example_config.yaml")
    
    # Tune hyperparameters
    config.model.hidden_size = trial.suggest_int("hidden_size", 32, 256)
    config.model.num_layers = trial.suggest_int("num_layers", 1, 4)
    config.optimizer.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    
    # Train and evaluate
    # ... (training code)
    
    return val_loss

study = optuna.create_study()
study.optimize(objective, n_trials=50)
```

## Debugging

Enable debug mode for detailed logging:

```bash
python scripts/train.py --config configs/example_config.yaml --debug
```

## GPU/Cluster Usage

The package automatically detects CUDA and MPS devices. For multi-GPU training, set:

```yaml
training:
  device: "cuda:0"  # Specify GPU
```

For cluster usage with SLURM:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

python scripts/train.py --config configs/example_config.yaml
```

## Testing

Run all tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=sysid --cov-report=html
```

## MLflow Tracking

The package uses **local file-based MLflow tracking** by default (no server needed).

View experiments:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser.

All experiment runs are stored in the `mlruns/` directory.

## Logging and File Organization

All scripts create **detailed timestamped log files** for complete traceability:
- Training logs: `outputs/<experiment>/training_*.log`
- Evaluation logs: `evaluation_results/evaluation_*.log`
- Analysis logs: `analysis_results/analysis_*.log`

See [LOGGING.md](LOGGING.md) for complete logging documentation and [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) for where all files are stored.

## License

See LICENSE file.
