# Quick Reference - System Identification Package

## Installation
```bash
cd python
pip install -e .                # Install package
pip install -e ".[dev]"         # Install with dev dependencies
```

## Generate Sample Data
```bash
python scripts/generate_sample_data.py --system pendulum --n-train 800
```

## Training
```bash
# Basic training
python scripts/train.py --config configs/example_config.yaml

# Debug mode (small model, CPU)
python scripts/train.py --config configs/debug_config.yaml --debug
```

## Evaluation
```bash
python scripts/evaluate.py \
    --config configs/example_config.yaml \
    --model models/best_model.pt \
    --test-data data/test.csv \
    --output-dir evaluation_results
```

## Model Analysis
```bash
# Basic analysis
python scripts/analyze.py \
    --config configs/example_config.yaml \
    --model models/best_model.pt

# With stability and bounds checking
python scripts/analyze.py \
    --config configs/example_config.yaml \
    --model models/best_model.pt \
    --check-stability \
    --lower-bound -10 \
    --upper-bound 10
```

## Testing
```bash
pytest tests/                           # Run all tests
pytest tests/test_models.py -v         # Run specific test file
pytest tests/ --cov=sysid              # With coverage
pytest tests/ --cov=sysid --cov-report=html  # HTML coverage report
```

## MLflow
```bash
mlflow server --host 0.0.0.0 --port 5000     # Start MLflow server
# Open browser: http://localhost:5000
```

## TensorBoard
```bash
tensorboard --logdir logs                     # Start TensorBoard
# Open browser: http://localhost:6006
```

## Configuration Quick Edit

Key parameters in `configs/example_config.yaml`:

### Model Architecture
```yaml
model:
  model_type: "lstm"        # "rnn", "lstm", "gru"
  hidden_size: 64          # 16, 32, 64, 128, 256
  num_layers: 2            # 1, 2, 3, 4
  dropout: 0.1             # 0.0 to 0.5
```

### Training
```yaml
training:
  max_epochs: 1000
  learning_rate: 0.001     # 1e-5 to 1e-2
  batch_size: 32           # 8, 16, 32, 64, 128
  early_stopping_patience: 50
  device: "cuda"           # "cuda", "cpu", "mps"
```

### Data
```yaml
data:
  normalize: true
  normalization_method: "minmax"  # "minmax", "standard"
  sequence_length: 100      # null for full sequences
```

## Common Workflows

### 1. Quick Start (Sample Data)
```bash
# Generate data
python scripts/generate_sample_data.py --system pendulum

# Train
python scripts/train.py --config configs/debug_config.yaml

# Evaluate
python scripts/evaluate.py \
    --config configs/debug_config.yaml \
    --model debug_models/best_model.pt \
    --test-data data/test.csv
```

### 2. Hyperparameter Tuning
```bash
# Copy and modify config
cp configs/example_config.yaml configs/experiment1.yaml
# Edit experiment1.yaml: change hidden_size, learning_rate, etc.

# Train
python scripts/train.py --config configs/experiment1.yaml

# Compare in MLflow UI
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Production Pipeline
```bash
# 1. Train on full data
python scripts/train.py --config configs/production_config.yaml

# 2. Evaluate
python scripts/evaluate.py \
    --config configs/production_config.yaml \
    --model models/best_model.pt \
    --test-data data/test.csv

# 3. Analyze constraints
python scripts/analyze.py \
    --config configs/production_config.yaml \
    --model models/best_model.pt \
    --check-stability
```

## Python API Examples

### Load and Use Trained Model
```python
import torch
from sysid.models import LSTM
from sysid.data import DataNormalizer

# Load normalizer
normalizer = DataNormalizer.load("models/normalizer.json")

# Load model
checkpoint = torch.load("models/best_model.pt")
model = LSTM(input_size=2, hidden_size=64, output_size=1)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Make prediction
import numpy as np
input_data = np.random.randn(1, 100, 2)
input_norm = normalizer.transform_inputs(input_data)
input_tensor = torch.FloatTensor(input_norm)

with torch.no_grad():
    output = model(input_tensor)
    output_denorm = normalizer.inverse_transform_outputs(output.numpy())
```

### Custom Training Loop
```python
from sysid.config import Config
from sysid.data import create_dataloaders, DataLoader
from sysid.models import LSTM
from sysid.training import Trainer, get_loss_function, get_optimizer

# Load config
config = Config.from_yaml("configs/example_config.yaml")

# Load data
train_inputs, train_outputs = DataLoader.load_from_csv("data/train.csv")
val_inputs, val_outputs = DataLoader.load_from_csv("data/val.csv")

# Create dataloaders
train_loader, val_loader, _, normalizer = create_dataloaders(
    train_inputs, train_outputs,
    val_inputs, val_outputs,
    batch_size=config.data.batch_size,
    normalize=True,
)

# Create model, loss, optimizer
model = LSTM(input_size=2, hidden_size=64, output_size=1)
loss_fn = get_loss_function("mse")
optimizer = get_optimizer(model.parameters(), learning_rate=0.001)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
)

history = trainer.train(max_epochs=100)
```

## Troubleshooting

**Import errors**: `pip install -e .` in python/ directory

**CUDA out of memory**: Reduce `batch_size` or `hidden_size`

**Slow training**: Enable CUDA (`device: "cuda"`) or reduce data size

**NaN loss**: Lower learning rate, add gradient clipping

**Overfitting**: Increase `dropout`, add regularization

**Underfitting**: Increase `hidden_size` or `num_layers`

## File Locations

- **Models**: `models/best_model.pt`, `models/final_model.pt`
- **Normalizer**: `models/normalizer.json`
- **Outputs**: `outputs/training_history.json`
- **Logs**: `logs/` (TensorBoard events)
- **MLflow**: `mlruns/` (local) or remote server
- **Evaluation**: `evaluation_results/`
- **Analysis**: `analysis_results/`
