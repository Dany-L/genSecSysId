# System Identification Package - Project Structure

## Complete Directory Tree

```
python/
├── setup.py                    # Package installation configuration
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Development dependencies
├── README.md                   # Main documentation
├── GETTING_STARTED.md         # Quick start guide
├── .gitignore                 # Git ignore patterns
│
├── configs/                   # Configuration files
│   ├── example_config.yaml    # Full configuration example (YAML)
│   ├── example_config.json    # Full configuration example (JSON)
│   └── debug_config.yaml      # Small model for debugging
│
├── src/sysid/                 # Main package source
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Configuration management (dataclasses)
│   ├── utils.py               # Utility functions (seed, device)
│   │
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch Dataset for time series
│   │   ├── loader.py          # Data loading from CSV/NPY/MAT
│   │   └── normalizer.py      # Data normalization (minmax, standard)
│   │
│   ├── models/                # Neural network models
│   │   ├── __init__.py
│   │   ├── base.py            # Base RNN class with common methods
│   │   ├── rnn.py             # SimpleRNN, LSTM, GRU implementations
│   │   └── regularization.py  # Custom regularization functions
│   │
│   ├── training/              # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py         # Main Trainer class
│   │   ├── losses.py          # Loss functions
│   │   └── optimizers.py      # Optimizer and scheduler setup
│   │
│   └── evaluation/            # Evaluation utilities
│       ├── __init__.py
│       ├── evaluator.py       # Main Evaluator class
│       └── metrics.py         # Evaluation metrics
│
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── analyze.py            # Model analysis script
│
└── tests/                     # Unit tests
    ├── __init__.py
    ├── conftest.py           # Pytest configuration
    ├── test_data.py          # Tests for data module
    ├── test_models.py        # Tests for models
    └── test_training.py      # Tests for training utilities
```

## Key Features by Module

### 1. Configuration (`config.py`)
- **DataConfig**: Data paths, normalization, batch size
- **ModelConfig**: Model architecture parameters
- **OptimizerConfig**: Learning rate, scheduler settings
- **TrainingConfig**: Training parameters, regularization
- **MLflowConfig**: Experiment tracking settings
- Load/save from YAML/JSON

### 2. Data (`data/`)
- **TimeSeriesDataset**: Handle sequences with optional sliding window
- **DataLoader**: Load from CSV, NPY, MAT files
- **DataNormalizer**: Min-max and standard normalization
- **create_dataloaders**: One-function setup for train/val/test loaders

### 3. Models (`models/`)
- **BaseRNN**: Abstract base class with common methods
  - `get_regularization_loss()`: For custom constraints
  - `get_parameter_dict()`: Parameter statistics
  - `save()` / `load()`: Checkpoint management
- **SimpleRNN, LSTM, GRU**: Standard implementations
- **Regularization functions**:
  - L1/L2/Elastic net
  - Lipschitz constraints
  - Stability (spectral radius)
  - Bounded parameters

### 4. Training (`training/`)
- **Trainer**: Main training loop
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping
  - MLflow & TensorBoard logging
  - Checkpoint saving
- **Loss functions**: MSE, MAE, Huber, Smooth L1
- **Optimizers**: Adam, AdamW, SGD, RMSprop
- **Schedulers**: Step, Exponential, ReduceLROnPlateau, Cosine

### 5. Evaluation (`evaluation/`)
- **Evaluator**: Model evaluation
  - Compute metrics (MSE, RMSE, MAE, R², NRMSE)
  - Multi-step prediction metrics
  - Visualization (predictions, errors)
- **Metrics**: Comprehensive evaluation functions

### 6. Scripts (`scripts/`)
- **train.py**: Full training pipeline with MLflow
- **evaluate.py**: Test set evaluation with plots
- **analyze.py**: Parameter analysis and constraint checking

### 7. Tests (`tests/`)
- Comprehensive pytest suite
- Tests for all modules
- Fixtures for reproducibility

## Usage Examples

### Basic Training
```bash
python scripts/train.py --config configs/example_config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py \
    --config configs/example_config.yaml \
    --model models/best_model.pt \
    --test-data data/test.csv
```

### Model Analysis
```bash
python scripts/analyze.py \
    --config configs/example_config.yaml \
    --model models/best_model.pt \
    --check-stability \
    --lower-bound -10 \
    --upper-bound 10
```

### Running Tests
```bash
pytest tests/ -v
pytest tests/ --cov=sysid --cov-report=html
```

### Debug Mode
```bash
python scripts/train.py --config configs/debug_config.yaml --debug
```

## Customization Points

### 1. Custom Models
Extend `BaseRNN` in `models/base.py`:
```python
class CustomRNN(BaseRNN):
    def __init__(self, ...):
        super().__init__(...)
        # Your architecture
    
    def forward(self, x, hidden=None):
        # Your forward pass
        pass
    
    def get_regularization_loss(self):
        # Your constraints
        pass
```

### 2. Custom Regularization
Add functions to `models/regularization.py` or override `get_regularization_loss()`.

### 3. Custom Loss Functions
Add to `training/losses.py` and modify `get_loss_function()`.

### 4. Hyperparameter Tuning
- Modify config files
- Use Optuna or similar for automated search
- Simple: change `hidden_size`, `num_layers`, `learning_rate` in config

## MLflow Integration

All training runs are automatically logged to MLflow:
- Hyperparameters
- Training/validation metrics
- Model artifacts
- Configuration files

Start MLflow UI:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

## GPU/Cluster Support

- Automatic CUDA/MPS detection
- Set device in config: `device: "cuda"`
- Compatible with SLURM job schedulers
- Multi-GPU: specify device index

## Best Practices Implemented

1. **Modularity**: Clean separation of concerns
2. **Configurability**: YAML/JSON configs, not hardcoded
3. **Reproducibility**: Seed setting, deterministic operations
4. **Logging**: MLflow + TensorBoard integration
5. **Testing**: Comprehensive pytest suite
6. **Documentation**: README, docstrings, examples
7. **Error handling**: Validation, assertions
8. **Type hints**: For better IDE support
9. **Standard PyTorch patterns**: DataLoader, Module, etc.
10. **Extensibility**: Easy to add custom models/losses/regularizers

## Next Steps

1. **Install**: `pip install -e .`
2. **Prepare data**: CSV format with inputs/outputs
3. **Configure**: Edit `configs/example_config.yaml`
4. **Train**: Run `train.py`
5. **Evaluate**: Run `evaluate.py`
6. **Analyze**: Run `analyze.py`
7. **Iterate**: Tune hyperparameters, try different models

## Support

- Check `GETTING_STARTED.md` for detailed setup
- Review `README.md` for advanced usage
- Inspect example configs in `configs/`
- Run tests to verify installation: `pytest tests/`
