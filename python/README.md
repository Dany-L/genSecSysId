# System Identification Package

RNN-based identification of nonlinear dynamical systems from data using PyTorch.

## Features

- **Flexible RNN architectures**: SimpleRNN, LSTM, GRU with customizable layers
- **Custom regularization**: Support for parameter constraints (Lipschitz bounds, stability, bounded parameters)
- **Data preprocessing**: Automatic normalization (min-max, standard)
- **Training utilities**: Early stopping, learning rate scheduling, gradient clipping
- **MLflow integration**: Automatic experiment tracking and model versioning
- **Comprehensive logging**: Timestamped log files for all operations (see [LOGGING.md](LOGGING.md))
- **Evaluation tools**: Comprehensive metrics and visualization
- **Configurable metrics**: Choose which evaluation metrics to compute and log (see [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md))
- **Model analysis**: Parameter inspection, stability checks, bound verification
- **GPU support**: Automatic CUDA/MPS detection
- **Testing**: Comprehensive pytest suite

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

**That's it!** The script will:
- ✅ Load all CSV files from folders
- ✅ Normalize data during training
- ✅ Track experiments with MLflow
- ✅ Save best model automatically

### 3. Alternative: Single CSV files

If you prefer single-file datasets:

```yaml
# configs/your_config.yaml
data:
  train_path: "data/prepared/train.csv"
  val_path: "data/prepared/val.csv"
  test_path: "data/prepared/test.csv"
  batch_size: 32
  normalize: true

model:
  model_type: "lstm"
  input_size: 1
  hidden_size: 64
  output_size: 1
  num_layers: 2

training:
  max_epochs: 500
  early_stopping_patience: 30
  learning_rate: 0.001
```

Then train:

```bash
python scripts/train.py --config configs/your_config.yaml
```

### 4. Evaluate the model

**Folder path (recommended):**
```bash
python scripts/evaluate.py \
    --config configs/rnn_baseline.yaml \
    --model models/rnn_baseline/best_model.pt \
    --test-data data/prepared/test
```

**Or single CSV file:**
```bash
python scripts/evaluate.py \
    --config configs/rnn_baseline.yaml \
    --model models/rnn_baseline/best_model.pt \
    --test-data data/test.csv
```

### 5. Compare multiple models

```bash
python scripts/compare.py \
    --run-ids <run_id_1> <run_id_2> <run_id_3> \
    --test-data data/prepared/test \
    --output-dir comparisons/my_comparison
```

This generates:
- Summary table with parameters and metrics
- Evaluation metrics comparison
- Training curves for each run
- Validation loss comparison plot

### 6. Post-process a model (Optional)

For SimpleLure models, you can solve an SDP to optimize P and L while keeping A, B, C, D fixed:

**Using the script:**
```bash
# Feasibility: find valid P and L
python scripts/post_process.py --run-id <run_id>

# Optimization: minimize s for tightest certificate
python scripts/post_process.py --run-id <run_id> --optimize-s
```

**Or call the method directly:**
```python
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
result = model.post_process(optimize_s=True)
# Model is automatically updated with optimized P, L
```

## Data Loading Options

The package supports two flexible loading methods:

### 1. Direct Folder Loading (Recommended!)
```yaml
data:
  train_path: "data/prepared"  # Folder with train/test/validation subfolders
  input_col: "d"               # Column name for input
  output_col: "e"              # Column name for output
  pattern: "*.csv"             # File pattern
```
✅ **No preprocessing** - loads directly from original CSV files  
✅ **No duplication** - uses original files only  
✅ **Simpler workflow** - one step  
✅ **Easy evaluation** - same folder structure for training and testing

See [docs/DIRECT_LOADING.md](docs/DIRECT_LOADING.md) for details.

### 2. Single CSV Files
```yaml
data:
  train_path: "data/prepared/train.csv"
  val_path: "data/prepared/val.csv"
  test_path: "data/prepared/test.csv"
```
✅ **Simple** - one file per split  
⚠️ **Manual splitting** - need to combine sequences first  
⚠️ **Manual paths** - need to specify train/val/test separately  

## Project Structure

```
python/
├── src/sysid/              # Main package
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions
│   ├── data/               # Data loading and preprocessing
│   │   ├── dataset.py      # PyTorch Dataset
│   │   ├── direct_loader.py # Direct CSV folder loading (recommended)
│   │   ├── loader.py       # Legacy single-file CSV loader
│   │   └── normalizer.py   # Data normalization
│   ├── models/             # Model architectures
│   │   ├── base.py         # Base RNN class
│   │   ├── rnn.py          # RNN implementations (Simple/LSTM/GRU)
│   │   └── regularization.py  # Custom regularization
│   ├── training/           # Training utilities
│   │   ├── trainer.py      # Main trainer class
│   │   ├── losses.py       # Loss functions
│   │   └── optimizers.py   # Optimizer setup
│   └── evaluation/         # Evaluation utilities
│       ├── evaluator.py    # Main evaluator class
│       └── metrics.py      # Evaluation metrics
├── scripts/                # Main scripts
│   ├── train.py           # Training script (auto-detects folder/CSV)
│   ├── evaluate.py        # Evaluation script (supports folder/CSV)
│   ├── compare.py         # Compare multiple MLflow runs
│   ├── post_process.py    # Post-process models (optimize P/L only)
│   ├── export_for_matlab.py  # Export models to MATLAB .mat format
│   └── generate_sample_data.py  # Generate sample data for testing
├── tests/                 # Unit tests
├── configs/               # Example configurations
│   ├── rnn_baseline.yaml  # Baseline config with direct folder loading
│   ├── rnn_direct.yaml    # Alternative direct loading config
│   └── lstm_baseline.yaml # LSTM example config
├── docs/                  # Documentation
│   ├── README.md          # Documentation index
│   ├── DIRECT_LOADING.md  # Guide for direct CSV loading
│   ├── CSV_VS_NPY.md      # Format comparison
│   └── QUICKSTART_DIRECT.md # Quick start guide
└── setup.py              # Package setup
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

## Citation

If you use this package, please cite:

```bibtex
@software{sysid2025,
  title = {System Identification with RNNs},
  author = {Your Name},
  year = {2025},
}
```
