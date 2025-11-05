# File Organization and Storage Guide

This document explains where all files are stored during data preparation, training, evaluation, and analysis.

## Directory Structure Overview

```
python/
├── data/                           # Your datasets (separate from package)
│   ├── prepared/                   # Pre-split CSV files
│   │   ├── train/
│   │   ├── test/
│   │   ├── validation/
│   │   ├── train_inputs.npy       # Processed data (created by prepare_data.py)
│   │   ├── train_outputs.npy
│   │   ├── val_inputs.npy
│   │   ├── val_outputs.npy
│   │   ├── test_inputs.npy
│   │   ├── test_outputs.npy
│   │   ├── metadata.json          # Dataset info
│   │   └── data_preparation_*.log # Data prep logs
│   └── raw/                        # Original data (if any)
│
├── models/                         # Saved model checkpoints
│   ├── rnn_baseline/
│   │   ├── best_model.pt          # Best model based on validation loss
│   │   ├── final_model.pt         # Model at end of training
│   │   ├── normalizer.json        # Data normalization parameters
│   │   └── checkpoint_epoch_*.pt  # Periodic checkpoints
│   └── lstm_baseline/
│       └── ...
│
├── outputs/                        # Training outputs
│   ├── rnn_baseline/
│   │   ├── training_history.json  # Loss curves, metrics per epoch
│   │   └── training_*.log         # Training logs
│   └── lstm_baseline/
│       └── ...
│
├── logs/                           # TensorBoard logs
│   ├── rnn_baseline/
│   │   └── events.out.tfevents.*  # TensorBoard event files
│   └── lstm_baseline/
│       └── ...
│
├── mlruns/                         # MLflow tracking (auto-created)
│   ├── 0/                          # Default experiment
│   ├── 1/                          # Your experiments
│   │   ├── <run_id>/
│   │   │   ├── artifacts/         # Model files, plots
│   │   │   ├── metrics/           # Metric files
│   │   │   ├── params/            # Hyperparameter files
│   │   │   └── tags/              # Run metadata
│   └── ...
│
├── evaluation_results/             # Evaluation outputs
│   ├── rnn_baseline/
│   │   ├── evaluation_results.json # Metrics (MSE, MAE, R², etc.)
│   │   ├── predictions.npy        # Model predictions
│   │   ├── targets.npy            # Ground truth
│   │   ├── predictions_plot.png   # Prediction vs actual plots
│   │   ├── error_analysis.png     # Error distribution plots
│   │   └── evaluation_*.log       # Evaluation logs
│   └── lstm_baseline/
│       └── ...
│
├── analysis/                       # Model analysis outputs
│   ├── rnn_baseline/
│   │   ├── parameter_stats.json   # Weight statistics
│   │   ├── stability_check.json   # Stability metrics
│   │   └── analysis_*.log         # Analysis logs
│   └── lstm_baseline/
│       └── ...
│
└── configs/                        # Configuration files
    ├── rnn_baseline.yaml
    └── lstm_baseline.yaml
```

## File Types and Purposes

### 1. Data Files

**Location**: `data/prepared/`

| File | Format | Purpose | Created By |
|------|--------|---------|------------|
| `train_inputs.npy` | NumPy binary | Training input sequences | `prepare_data.py` |
| `train_outputs.npy` | NumPy binary | Training output sequences | `prepare_data.py` |
| `val_inputs.npy` | NumPy binary | Validation inputs | `prepare_data.py` |
| `val_outputs.npy` | NumPy binary | Validation outputs | `prepare_data.py` |
| `test_inputs.npy` | NumPy binary | Test inputs | `prepare_data.py` |
| `test_outputs.npy` | NumPy binary | Test outputs | `prepare_data.py` |
| `train.csv` | CSV | Flattened training data (human-readable) | `prepare_data.py` |
| `val.csv` | CSV | Flattened validation data | `prepare_data.py` |
| `test.csv` | CSV | Flattened test data | `prepare_data.py` |
| `metadata.json` | JSON | Dataset info (shapes, columns, etc.) | `prepare_data.py` |

### 2. Model Files

**Location**: `models/<experiment_name>/`

| File | Format | Purpose | Created By |
|------|--------|---------|------------|
| `best_model.pt` | PyTorch | Best model (lowest val loss) | `train.py` |
| `final_model.pt` | PyTorch | Model at end of training | `train.py` |
| `normalizer.json` | JSON | Data normalization params (mean, std, min, max) | `train.py` |
| `checkpoint_epoch_*.pt` | PyTorch | Periodic checkpoints | `train.py` |

**Model files contain**:
- Model architecture state
- All learned parameters (weights, biases)
- Training state (epoch, optimizer state)
- Configuration used

### 3. Training Outputs

**Location**: `outputs/<experiment_name>/`

| File | Format | Purpose | Created By |
|------|--------|---------|------------|
| `training_history.json` | JSON | Loss and metrics per epoch | `train.py` |
| `training_*.log` | Text | Detailed training logs | `train.py` |

**Training history contains**:
```json
{
  "train_loss": [0.5, 0.4, 0.3, ...],
  "val_loss": [0.6, 0.5, 0.4, ...],
  "epochs": [1, 2, 3, ...],
  "learning_rates": [0.001, 0.001, 0.0005, ...],
  "best_epoch": 50,
  "best_val_loss": 0.123
}
```

### 4. TensorBoard Logs

**Location**: `logs/<experiment_name>/`

| File | Format | Purpose | Created By |
|------|--------|---------|------------|
| `events.out.tfevents.*` | Binary | TensorBoard event data | `train.py` |

**View with**: `tensorboard --logdir=logs`

**Contains**:
- Scalar metrics (loss curves)
- Histograms (weight distributions)
- Graphs (model architecture)
- Custom visualizations

### 5. MLflow Tracking

**Location**: `mlruns/`

**Structure**:
```
mlruns/
├── 0/                      # Default experiment
├── <experiment_id>/        # Your experiment (e.g., "1")
│   ├── <run_id>/          # Unique run ID (e.g., "a1b2c3d4...")
│   │   ├── artifacts/
│   │   │   ├── model/     # Saved model
│   │   │   ├── plots/     # Training curves, etc.
│   │   │   └── config.yaml
│   │   ├── metrics/       # One file per metric
│   │   │   ├── train_loss
│   │   │   ├── val_loss
│   │   │   └── ...
│   │   ├── params/        # One file per hyperparameter
│   │   │   ├── learning_rate
│   │   │   ├── hidden_size
│   │   │   └── ...
│   │   └── tags/          # Metadata
│   │       ├── mlflow.runName
│   │       └── ...
│   └── meta.yaml          # Experiment metadata
```

**View with**: `mlflow ui` or `mlflow server`

**Contains**:
- All hyperparameters
- All metrics over time
- Model artifacts
- Code version/git commit
- System info (Python version, etc.)

### 6. Evaluation Results

**Location**: `evaluation_results/<experiment_name>/`

| File | Format | Purpose | Created By |
|------|--------|---------|------------|
| `evaluation_results.json` | JSON | Test metrics (MSE, MAE, R², etc.) | `evaluate.py` |
| `predictions.npy` | NumPy | Model predictions on test set | `evaluate.py` |
| `targets.npy` | NumPy | Ground truth test values | `evaluate.py` |
| `predictions_plot.png` | PNG | Prediction vs actual visualization | `evaluate.py` |
| `error_analysis.png` | PNG | Error distribution plots | `evaluate.py` |
| `evaluation_*.log` | Text | Evaluation logs | `evaluate.py` |

### 7. Analysis Results

**Location**: `analysis/<experiment_name>/`

| File | Format | Purpose | Created By |
|------|--------|---------|------------|
| `parameter_stats.json` | JSON | Weight statistics (mean, std, norm) | `analyze.py` |
| `stability_check.json` | JSON | Spectral radius, eigenvalues | `analyze.py` |
| `analysis_*.log` | Text | Analysis logs | `analyze.py` |

### 8. Log Files

**All scripts now generate timestamped log files!**

**Naming Convention**: `<script_name>_YYYYMMDD_HHMMSS.log`

**Location & Examples**:
- **Data preparation**: `data/prepared/data_preparation_20251105_143022.log`
- **Training**: `outputs/<experiment>/training_20251105_150000.log`
- **Evaluation**: `evaluation_results/<experiment>/evaluation_20251105_160000.log`
- **Analysis**: `analysis/<experiment>/analysis_20251105_170000.log`

**Format**: Plain text with timestamps (same format across all scripts)
```
2025-11-05 14:30:22 - INFO - Data preparation started
2025-11-05 14:30:23 - INFO - Loaded 273 training sequences
2025-11-05 14:30:24 - INFO - Processing complete
2025-11-05 14:30:24 - ERROR - Failed to load file X
2025-11-05 14:30:25 - WARNING - Normalizer not found
```

**Log Levels**:
- `INFO`: Normal operations, progress updates
- `WARNING`: Non-critical issues (e.g., missing normalizer, instability warnings)
- `ERROR`: Critical errors that prevent completion
- `EXCEPTION`: Full stack traces for debugging

**Features**:
- Logs to **both** file and console simultaneously
- Timestamps for every message
- Full error tracebacks saved to file
- Includes experiment/model metadata
- Preserves complete history of all runs

## What Each Tool Manages

### Our Scripts (All with consistent logging!)
- **Data preparation**: NPY files, CSV files, metadata, **timestamped logs**
- **Training**: Model checkpoints, training history, normalizer, **timestamped logs**
- **Evaluation**: Metrics JSON, predictions, plots, **timestamped logs**
- **Analysis**: Parameter stats, stability checks, **timestamped logs**

**Every script now creates detailed log files** with:
- Timestamps for all operations
- Progress updates
- Error messages and stack traces
- Configuration details
- Results summary

### MLflow (Automatic tracking)
- **Experiments**: Organize multiple runs
- **Runs**: Track individual training runs with unique IDs
- **Parameters**: All hyperparameters (model, optimizer, training)
- **Metrics**: Time-series metrics (loss, accuracy, etc.)
- **Artifacts**: Files (models, plots, configs)
- **Tags**: Metadata (git commit, run name, notes, etc.)

### TensorBoard (Real-time visualization)
- **Scalars**: Real-time loss curves, learning rate
- **Histograms**: Weight distributions over time
- **Graphs**: Model architecture visualization
- **Images**: Training visualizations (if added)

## Best Practices

### 1. Keep Data Separate
✅ Store datasets in `data/` (not in `src/`)
✅ Use `--data-dir` argument to point to different datasets
✅ This allows easy dataset switching without code changes

### 2. Organize by Experiment
✅ Use meaningful experiment names (e.g., `rnn_baseline`, `lstm_large`, `gru_regularized`)
✅ All files for one experiment go in folders with the same name
✅ Easy to compare across experiments

### 3. Archive Old Experiments
```bash
# Archive completed experiments
mkdir -p archive/2025-11
mv models/old_experiment archive/2025-11/
mv outputs/old_experiment archive/2025-11/
mv logs/old_experiment archive/2025-11/
```

### 4. Version Control
✅ **DO commit**: Code, configs, scripts, documentation
✅ **DON'T commit**: Data files, model checkpoints, logs, MLflow runs
✅ **Use .gitignore**: Already configured for you

### 5. Backup Important Results
```bash
# Backup best models and results
tar -czf experiment_backup.tar.gz \
    models/rnn_baseline/best_model.pt \
    evaluation_results/rnn_baseline/ \
    outputs/rnn_baseline/training_history.json
```

## Quick Reference

### Find and view logs:
```bash
# Data preparation logs
ls data/prepared/*.log
cat data/prepared/data_preparation_*.log

# Training logs (most recent)
ls -t outputs/rnn_baseline/*.log | head -1
tail -f outputs/rnn_baseline/training_*.log  # Follow in real-time

# Evaluation logs
cat evaluation_results/rnn_baseline/evaluation_*.log

# Analysis logs
cat analysis/rnn_baseline/analysis_*.log

# Search for errors in logs
grep -i "error\|warning" outputs/rnn_baseline/*.log
```

### Find best model:
```bash
ls models/rnn_baseline/best_model.pt
```

### View TensorBoard:
```bash
tensorboard --logdir=logs
# Open http://localhost:6006
```

### View MLflow:
```bash
mlflow ui
# Open http://localhost:5000
```

### Check data preparation:
```bash
cat data/prepared/metadata.json
cat data/prepared/data_preparation_*.log
```

### Compare experiments:
```bash
# Compare training history
python -c "
import json
with open('outputs/rnn_baseline/training_history.json') as f:
    rnn = json.load(f)
with open('outputs/lstm_baseline/training_history.json') as f:
    lstm = json.load(f)
print(f'RNN best val loss: {min(rnn[\"val_loss\"])}')
print(f'LSTM best val loss: {min(lstm[\"val_loss\"])}')
"
```

## Clean Up

### Remove all generated files (careful!):
```bash
rm -rf models/ outputs/ logs/ mlruns/ evaluation_results/ analysis/
rm -f data/prepared/*.npy data/prepared/*.log
```

### Keep only best models:
```bash
# Remove checkpoints but keep best/final models
find models/ -name "checkpoint_epoch_*.pt" -delete
```

### Clean old logs:
```bash
# Remove logs older than 30 days
find . -name "*.log" -mtime +30 -delete
```
