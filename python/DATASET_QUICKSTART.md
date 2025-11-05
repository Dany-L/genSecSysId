# Quick Start Guide for Your Dataset

## 1. Prepare Your Data

Your CSV files have columns: `time`, `d`, `e`, `x_1`, `x_2`
- **Input**: Column `d`
- **Output**: Column `e`
- **Data is already split** into `train/`, `test/`, and `validation/` folders

**Good Practice**: Keep your data separate from the package code!

Run the preparation script:

```bash
cd python

# Process data from your data directory
python scripts/prepare_data.py \
    --data-dir data/prepared

# Or use a different dataset
python scripts/prepare_data.py \
    --data-dir /path/to/other/dataset \
    --output-dir data/experiment1

# Specify custom columns if needed
python scripts/prepare_data.py \
    --data-dir data/prepared \
    --input-col d \
    --output-col e \
    --pattern "*.csv"
```

This will:
- Load all CSV files from `data/prepared/train/`, `test/`, `validation/`
- Extract columns `d` (input) and `e` (output) from each file
- Process each split separately (preserving your existing split)
- Save as NPY files in the output directory (train_inputs.npy, train_outputs.npy, etc.)
- Create flattened CSV files for reference (train.csv, val.csv, test.csv)
- Generate metadata.json with dataset statistics
- **Create a log file** (`data_preparation_YYYYMMDD_HHMMSS.log`) with all processing details

## 2. Train RNN Baseline

```bash
python scripts/train.py --config configs/rnn_baseline.yaml
```

This trains a simple RNN with:
- 1 layer
- 32 hidden units
- tanh activation
- No dropout

## 3. Train LSTM Baseline

```bash
python scripts/train.py --config configs/lstm_baseline.yaml
```

This trains an LSTM with:
- 2 layers
- 64 hidden units
- 0.1 dropout

## 4. Evaluate Models

### RNN Baseline
```bash
python scripts/evaluate.py \
    --config configs/rnn_baseline.yaml \
    --model models/rnn_baseline/best_model.pt \
    --test-data data/prepared/test.csv \
    --output-dir evaluation_results/rnn_baseline
```

### LSTM Baseline
```bash
python scripts/evaluate.py \
    --config configs/lstm_baseline.yaml \
    --model models/lstm_baseline/best_model.pt \
    --test-data data/prepared/test.csv \
    --output-dir evaluation_results/lstm_baseline
```

## 5. Compare Results

Start MLflow to compare both models:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

Open browser: `http://localhost:5000`

You'll see:
- Training curves for both models
- Validation losses
- Final metrics
- Model artifacts

## File Structure After Preparation

```
python/
├── data/
│   ├── raw/                          # Your original CSV files
│   │   ├── init_rand-output_noise_1.csv
│   │   ├── init_rand-output_noise_2.csv
│   │   └── ...
│   └── prepared/                     # Prepared data
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── train_inputs.npy
│       ├── train_outputs.npy
│       ├── val_inputs.npy
│       ├── val_outputs.npy
│       ├── test_inputs.npy
│       └── test_outputs.npy
├── models/
│   ├── rnn_baseline/
│   │   ├── best_model.pt
│   │   ├── final_model.pt
│   │   └── normalizer.json
│   └── lstm_baseline/
│       ├── best_model.pt
│       ├── final_model.pt
│       └── normalizer.json
├── outputs/
│   ├── rnn_baseline/
│   │   └── training_history.json
│   └── lstm_baseline/
│       └── training_history.json
└── evaluation_results/
    ├── rnn_baseline/
    │   ├── evaluation_results.json
    │   ├── predictions.npy
    │   ├── targets.npy
    │   ├── predictions_plot.png
    │   └── error_analysis.png
    └── lstm_baseline/
        └── ...
```

## Tips

### Adjusting Hyperparameters

Edit the config files:

**For RNN** (`configs/rnn_baseline.yaml`):
```yaml
model:
  hidden_size: 32    # Try: 16, 32, 64, 128
  num_layers: 1      # Try: 1, 2, 3
  dropout: 0.0       # Try: 0.0, 0.1, 0.2
```

**For LSTM** (`configs/lstm_baseline.yaml`):
```yaml
model:
  hidden_size: 64    # Try: 32, 64, 128, 256
  num_layers: 2      # Try: 1, 2, 3, 4
  dropout: 0.1       # Try: 0.0, 0.1, 0.2, 0.3
```

### If Training is Slow

```yaml
training:
  device: "cpu"      # Use CPU if GPU issues
```

Or reduce batch size:
```yaml
data:
  batch_size: 16     # Default is 32
```

### If Model Overfits

Increase dropout:
```yaml
model:
  dropout: 0.2       # Higher = more regularization
```

Or add weight decay:
```yaml
optimizer:
  weight_decay: 0.0001  # L2 regularization
```

### If Model Underfits

Increase model capacity:
```yaml
model:
  hidden_size: 128   # Larger hidden size
  num_layers: 3      # More layers
```

## Next Steps

1. **Compare RNN vs LSTM**: Check which works better for your data
2. **Tune hyperparameters**: Adjust hidden size, layers, learning rate
3. **Add regularization**: If overfitting, increase dropout or weight decay
4. **Analyze models**: Use `scripts/analyze.py` to check parameter statistics
5. **Custom models**: Later, you can extend the base classes for custom architectures

## Quick Commands Summary

```bash
# 1. Prepare data (specify your data directory)
python scripts/prepare_data.py --data-dir data/prepared

# 2. Train RNN
python scripts/train.py --config configs/rnn_baseline.yaml

# 3. Train LSTM
python scripts/train.py --config configs/lstm_baseline.yaml

# 4. Evaluate RNN
python scripts/evaluate.py --config configs/rnn_baseline.yaml \
    --model models/rnn_baseline/best_model.pt \
    --test-data data/prepared/test_inputs.npy

# 5. Evaluate LSTM
python scripts/evaluate.py --config configs/lstm_baseline.yaml \
    --model models/lstm_baseline/best_model.pt \
    --test-data data/prepared/test_inputs.npy

# 6. Start MLflow (optional - for experiment tracking)
mlflow server --host 0.0.0.0 --port 5000

# 7. Start TensorBoard (optional - for real-time monitoring)
tensorboard --logdir=logs
```

## Where Are My Files?

See [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) for detailed information about:
- Where all files are stored
- What each file contains
- How MLflow and TensorBoard organize data
- Log file locations and formats
- Best practices for file management
