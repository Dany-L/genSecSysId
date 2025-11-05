# System Identification Package - Getting Started

## Installation

1. **Install the package**:
   ```bash
   cd python
   pip install -e .
   ```

2. **Install development dependencies** (for testing):
   ```bash
   pip install -e ".[dev]"
   ```

## Prepare Your Data

Your data should be CSV files with the following format:

```csv
input1,input2,...,output1,output2,...
0.1,0.2,...,0.3,0.4,...
0.5,0.6,...,0.7,0.8,...
...
```

Save your data in three files:
- `train.csv`: Training data
- `val.csv`: Validation data
- `test.csv`: Test data

## Basic Workflow

### 1. Train a Model

```bash
python scripts/train.py --config configs/example_config.yaml
```

This will:
- Load and normalize your data
- Create the model
- Train with early stopping
- Save checkpoints and the best model
- Log to MLflow and TensorBoard

### 2. Evaluate the Model

```bash
python scripts/evaluate.py \
    --config configs/example_config.yaml \
    --model models/best_model.pt \
    --test-data data/test.csv
```

This will:
- Load the trained model
- Evaluate on test data
- Generate metrics (MSE, RMSE, MAE, R², etc.)
- Create visualization plots

### 3. Analyze the Model

```bash
python scripts/analyze.py \
    --config configs/example_config.yaml \
    --model models/best_model.pt \
    --check-stability
```

This will:
- Load the model
- Analyze parameter statistics
- Check stability conditions (spectral radius)
- Verify parameter bounds (if specified)

## Configuration

Edit `configs/example_config.yaml` to customize:

- **Data settings**: Paths, batch size, normalization
- **Model architecture**: Type (RNN/LSTM/GRU), layers, hidden size
- **Training settings**: Learning rate, epochs, early stopping
- **Regularization**: Custom parameter constraints
- **Logging**: MLflow, TensorBoard

## Debugging

For quick debugging with a small model:

```bash
python scripts/train.py --config configs/debug_config.yaml --debug
```

## Hyperparameter Tuning

1. Copy `configs/example_config.yaml` to a new file
2. Modify the hyperparameters
3. Train with the new config
4. Compare results in MLflow

Common hyperparameters to tune:
- `hidden_size`: Size of RNN hidden state
- `num_layers`: Number of RNN layers
- `learning_rate`: Optimizer learning rate
- `dropout`: Dropout probability
- `batch_size`: Training batch size

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=sysid

# Run specific test file
pytest tests/test_models.py -v
```

## MLflow Tracking

1. Start MLflow server:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

2. Open browser: `http://localhost:5000`

3. View experiments, compare runs, and download models

## Next Steps

- **Custom models**: See `src/sysid/models/` for examples
- **Custom regularization**: See `src/sysid/models/regularization.py`
- **Advanced training**: Modify `src/sysid/training/trainer.py`
- **GPU training**: Set `device: "cuda"` in config

## Common Issues

**Import errors**: Make sure you installed the package with `pip install -e .`

**CUDA out of memory**: Reduce `batch_size` in config

**Poor performance**: Try different architectures, increase `hidden_size` or `num_layers`

**Overfitting**: Increase `dropout`, add regularization, or reduce model size
