# Logging System

All scripts in the package now have **consistent logging** to both files and console.

## Features

✅ **Timestamped log files** for every operation  
✅ **Dual output**: Logs to both file and console  
✅ **Consistent format** across all scripts  
✅ **Automatic error tracking** with full stack traces  
✅ **Progress monitoring** for long-running operations  
✅ **Experiment metadata** included in logs  

## Log File Locations

| Script | Log Location | Filename Pattern |
|--------|-------------|------------------|
| `prepare_data.py` | `data/prepared/` | `data_preparation_YYYYMMDD_HHMMSS.log` |
| `train.py` | `outputs/<experiment>/` | `training_YYYYMMDD_HHMMSS.log` |
| `evaluate.py` | `evaluation_results/<experiment>/` | `evaluation_YYYYMMDD_HHMMSS.log` |
| `analyze.py` | `analysis/<experiment>/` | `analysis_YYYYMMDD_HHMMSS.log` |

## Log Format

All logs use the same format:

```
YYYY-MM-DD HH:MM:SS - LEVEL - Message
```

**Example**:
```
2025-11-05 14:30:22 - INFO - ======================================================================
2025-11-05 14:30:22 - INFO - Training RNN for System Identification
2025-11-05 14:30:22 - INFO - ======================================================================
2025-11-05 14:30:22 - INFO - Config file: configs/rnn_baseline.yaml
2025-11-05 14:30:22 - INFO - Model type: rnn
2025-11-05 14:30:22 - INFO - Using device: cuda
2025-11-05 14:30:23 - INFO - Loading data...
2025-11-05 14:30:24 - INFO - Train data loaded: inputs=(273, 101, 1), outputs=(273, 101, 1)
2025-11-05 14:30:24 - INFO - Validation data loaded: inputs=(45, 101, 1), outputs=(45, 101, 1)
2025-11-05 14:30:25 - INFO - Creating data loaders...
2025-11-05 14:30:25 - INFO - Train batches: 9, Val batches: 2
2025-11-05 14:30:25 - INFO - Creating model...
2025-11-05 14:30:25 - INFO - Total parameters: 3,234
2025-11-05 14:30:25 - INFO - Loss function: mse
2025-11-05 14:30:25 - INFO - Optimizer: adam
2025-11-05 14:30:25 - INFO - Learning rate: 0.001
2025-11-05 14:30:26 - INFO - ======================================================================
2025-11-05 14:30:26 - INFO - Starting training...
2025-11-05 14:30:26 - INFO - ======================================================================
2025-11-05 14:32:15 - INFO - Epoch 50/500 - Train Loss: 0.0234 - Val Loss: 0.0256
2025-11-05 14:35:42 - INFO - Early stopping triggered at epoch 125
2025-11-05 14:35:42 - INFO - ======================================================================
2025-11-05 14:35:42 - INFO - Training completed successfully!
2025-11-05 14:35:42 - INFO - ======================================================================
2025-11-05 14:35:42 - INFO - Best validation loss: 0.015234
2025-11-05 14:35:42 - INFO - Best epoch: 95
2025-11-05 14:35:42 - WARNING - Model shows signs of instability (spectral radius > 1.0)
2025-11-05 14:35:43 - ERROR - Failed to save checkpoint: disk full
```

## Log Levels

| Level | Purpose | Examples |
|-------|---------|----------|
| `INFO` | Normal operations | "Loading data...", "Training started", "Model saved" |
| `WARNING` | Non-critical issues | "Normalizer not found", "Potential instability", "Using CPU" |
| `ERROR` | Critical errors | "Failed to load data", "Training crashed", "Invalid config" |
| `EXCEPTION` | With full traceback | Catches all exceptions with stack traces |

## What Gets Logged

### Data Preparation (`prepare_data.py`)
- Input/output directories
- Column names used
- Number of files found
- Data shapes for each split
- Processing time
- Errors during file loading
- Saved file locations

### Training (`train.py`)
- Configuration details
- Model architecture and parameters
- Data shapes and batch sizes
- Optimizer settings
- Learning rate schedule
- Epoch progress with losses
- Best model checkpoints
- Early stopping events
- MLflow run ID
- Total training time
- Errors/exceptions

### Evaluation (`evaluate.py`)
- Model being evaluated
- Test data information
- Normalizer status
- Evaluation metrics (MSE, MAE, R², etc.)
- Plot generation status
- Errors during evaluation

### Analysis (`analyze.py`)
- Model parameters statistics
- Stability analysis results (spectral radius)
- Parameter bound violations
- Saved analysis files
- Warnings about instability

## Usage Examples

### View logs in real-time (during training)
```bash
# Follow training progress
tail -f outputs/rnn_baseline/training_*.log

# Watch for errors
tail -f outputs/rnn_baseline/training_*.log | grep -i "error\|warning"
```

### Review completed operations
```bash
# Data preparation
cat data/prepared/data_preparation_20251105_143022.log

# Training
cat outputs/rnn_baseline/training_20251105_150000.log

# Evaluation
cat evaluation_results/rnn_baseline/evaluation_20251105_160000.log

# Analysis
cat analysis/rnn_baseline/analysis_20251105_170000.log
```

### Search for specific information
```bash
# Find all errors
grep -r "ERROR" outputs/

# Find warnings
grep -r "WARNING" outputs/

# Find specific model info
grep "Total parameters" outputs/*/training_*.log

# Find best validation loss
grep "Best validation loss" outputs/*/training_*.log

# Check stability
grep "spectral radius" analysis/*/analysis_*.log
```

### Compare multiple runs
```bash
# Compare training times
grep "Total training time" outputs/rnn_baseline/*.log
grep "Total training time" outputs/lstm_baseline/*.log

# Compare best losses
grep "Best validation loss" outputs/*/*.log

# Find all runs with errors
find outputs -name "*.log" -exec grep -l "ERROR" {} \;
```

### Clean old logs
```bash
# Remove logs older than 30 days
find . -name "*.log" -mtime +30 -delete

# Archive old logs
mkdir -p archive/logs_2025-11
mv outputs/*/training_202511*.log archive/logs_2025-11/
```

## Log Rotation

Logs are **never automatically deleted**. Each run creates a new log file with a timestamp.

**To manage log files**:

```bash
# Keep only the 10 most recent logs per experiment
cd outputs/rnn_baseline
ls -t training_*.log | tail -n +11 | xargs rm

# Archive old logs by month
mkdir -p ../../archive/2025-11
mv training_202510*.log ../../archive/2025-11/

# Compress old logs
gzip archive/2025-11/*.log
```

## Integration with MLflow and TensorBoard

### Logs are complementary:

**Text Logs (.log files)**:
- Complete operation details
- Error messages and tracebacks
- Configuration info
- Timing information
- Human-readable format

**MLflow**:
- Hyperparameters
- Metrics over time (plots)
- Model artifacts
- Run comparison
- Web UI

**TensorBoard**:
- Real-time training curves
- Weight histograms
- Model graph
- Interactive visualization

### Use all three together:
1. **During training**: Watch TensorBoard for real-time progress
2. **For debugging**: Check text logs for detailed error messages
3. **For comparison**: Use MLflow to compare experiments
4. **For reporting**: Export metrics from MLflow, details from logs

## Debugging Workflow

When something goes wrong:

1. **Check the log file** for error messages:
   ```bash
   grep -A 10 "ERROR" outputs/rnn_baseline/training_*.log
   ```

2. **Look for warnings** that might indicate issues:
   ```bash
   grep "WARNING" outputs/rnn_baseline/training_*.log
   ```

3. **Check the full traceback** (automatically captured):
   ```bash
   grep -A 50 "Traceback" outputs/rnn_baseline/training_*.log
   ```

4. **Verify configuration** was loaded correctly:
   ```bash
   grep "Config\|Model type\|Learning rate" outputs/rnn_baseline/training_*.log
   ```

5. **Check data loading**:
   ```bash
   grep "data loaded\|shape" outputs/rnn_baseline/training_*.log
   ```

6. **Review MLflow** for metrics:
   ```bash
   mlflow ui
   ```

## Best Practices

✅ **Keep logs for completed experiments** - They're your audit trail  
✅ **Archive logs by date/experiment** - Organize for long-term storage  
✅ **Check logs after every run** - Catch issues early  
✅ **Use grep to search logs** - Find patterns across experiments  
✅ **Compress old logs** - Save disk space with `gzip`  
✅ **Include log paths in your reports** - Provide full traceability  

## Log File Contents Checklist

Every log file includes:

- ✅ Timestamp for creation
- ✅ Script name and version
- ✅ All input arguments
- ✅ Configuration details
- ✅ System info (device, Python version)
- ✅ Progress updates
- ✅ Error messages with context
- ✅ Success/completion status
- ✅ Output file locations
- ✅ Timing information

## Troubleshooting

### Log file not created
- Check directory permissions
- Ensure output directory exists (created automatically)
- Check disk space

### Log file empty
- Script might have crashed immediately
- Check console output for early errors

### Can't find recent log
- Use `ls -t` to sort by time:
  ```bash
  ls -t outputs/rnn_baseline/*.log | head -1
  ```

### Logs too large
- Consider log rotation
- Archive old experiments
- Compress with gzip:
  ```bash
  gzip outputs/rnn_baseline/training_*.log
  ```
