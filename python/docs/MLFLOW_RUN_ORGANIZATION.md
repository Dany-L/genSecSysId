# MLflow Run Organization

## Overview

The system now automatically organizes all training, evaluation, and analysis results by MLflow run ID. This provides:

1. **Easy file organization** - All files for a single experiment are grouped together
2. **Complete traceability** - Link models, metrics, and analysis back to training runs
3. **Unified MLflow tracking** - All results (train/eval/analysis) logged to the same run
4. **Query-friendly** - Use MLflow UI to compare models across all metrics

## Directory Structure

### Before (Old Structure)
```
models/
├── rnn_baseline/
│   ├── best_model.pt
│   ├── final_model.pt
│   └── normalizer.json
outputs/
├── rnn_baseline/
│   ├── config.yaml
│   └── training_*.log
evaluation_results/
├── evaluation_*.log
├── predictions_plot.png
└── evaluation_results.json
```

### After (New Structure)
```
models/
├── <run_id_1>/
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── normalizer.json
│   └── run_info.json          # NEW: Contains run_id and experiment info
├── <run_id_2>/
│   └── ...

outputs/
├── <run_id_1>/
│   ├── config.yaml
│   └── training_*.log
├── <run_id_2>/
│   └── ...

evaluation_results/
├── <run_id_1>/                 # NEW: Organized by run_id
│   ├── evaluation_*.log
│   ├── predictions_plot.png
│   └── evaluation_results.json
├── <run_id_2>/
│   └── ...

analysis_results/
├── <run_id_1>/                 # NEW: Organized by run_id
│   ├── analysis_*.log
│   ├── parameter_statistics.json
│   ├── stability_check.json
│   └── bound_violations.json
├── <run_id_2>/
│   └── ...
```

## Workflow

### 1. Training

```bash
python scripts/train.py --config configs/rnn_baseline.yaml
```

**What happens:**
- Creates a new MLflow run with unique `run_id`
- Saves model to `models/<run_id>/`
- Saves outputs to `outputs/<run_id>/`
- Creates `run_info.json` with run metadata
- Logs all artifacts to MLflow run

**Console output:**
```
MLflow run ID: abc123def456...
Model directory: models/abc123def456
Output directory: outputs/abc123def456
...
Run ID: abc123def456 - Use this for evaluation/analysis
```

### 2. Evaluation

**Automatic run detection (recommended):**
```bash
python scripts/evaluate.py \
    --config configs/rnn_baseline.yaml \
    --model models/abc123def456/best_model.pt \
    --test-data data/prepared/test
```

**What happens:**
- Detects `run_info.json` in model directory
- Automatically reopens the training MLflow run
- Saves results to `evaluation_results/abc123def456/`
- **Adds evaluation metrics to the same MLflow run**:
  - Logs metrics with `eval_` prefix: `eval_mse`, `eval_rmse`, `eval_mae`, `eval_r2`, etc.
  - All metrics (training + evaluation) in one place!
- Logs plots and artifacts to the run

**Without run_info.json:**
If `run_info.json` doesn't exist (legacy models), evaluation still works but results aren't logged to MLflow.

### 3. Analysis

**Automatic run detection (recommended):**
```bash
python scripts/analyze.py \
    --config configs/rnn_baseline.yaml \
    --model models/abc123def456/best_model.pt \
    --check-stability \
    --lower-bound -10 \
    --upper-bound 10
```

**What happens:**
- Detects `run_info.json` in model directory
- Automatically reopens the training MLflow run
- Saves results to `analysis_results/abc123def456/`
- **Adds analysis metrics to the same MLflow run**:
  - Logs metrics: `analysis_total_params`, `analysis_bound_violations`
  - Logs stability metrics: `stability_<layer>_spectral_radius`, `stability_<layer>_is_stable`
  - All metrics (training + evaluation + analysis) in one place!
- Logs analysis artifacts to the run

## MLflow UI View

After training, evaluation, and analysis, you can view all results in one place:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000`

### Metrics Available

**All metrics in one run!**

**Training metrics:**
- `best_val_loss`, `best_epoch`, `final_epoch`
- Per-epoch: `train_loss`, `val_loss`

**Evaluation metrics (added after training):**
- `eval_mse`, `eval_rmse`, `eval_mae`, `eval_r2`
- `eval_nrmse`, `eval_max_error`
- Per-sequence averages: `eval_mse_avg`, `eval_rmse_avg`, `eval_mae_avg`
- Final step: `eval_mse_final`, `eval_rmse_final`, `eval_mae_final`

**Analysis metrics (added after training):**
- `analysis_total_params`
- `analysis_bound_violations`
- `stability_<layer>_spectral_radius`
- `stability_<layer>_is_stable`

### Artifacts Available

**In MLflow UI, you'll see:**
- `models/` - Model checkpoints and normalizer
- `outputs/` - Training logs and config
- `evaluation/` - Evaluation plots and results
- `analysis/` - Parameter statistics and stability checks

## Querying Results

### Using MLflow Python API

```python
import mlflow

# Get all runs from an experiment
experiment = mlflow.get_experiment_by_name("rnn_baseline")
all_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Get only training runs
training_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.run_type != 'evaluation' AND tags.run_type != 'analysis'"
)

# Get evaluation runs
eval_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.run_type = 'evaluation'"
)

# Join training and evaluation data
merged = training_runs.merge(
    eval_runs[['tags.training_run_id', 'metrics.eval_r2', 'metrics.eval_rmse']],
    left_on='run_id',
    right_on='tags.training_run_id',
    how='left'
)

# Filter by evaluation performance
good_models = merged[merged['metrics.eval_r2'] > 0.8]

# Sort by evaluation RMSE
best_models = merged.sort_values('metrics.eval_rmse', ascending=True)

print(best_models[['run_id', 'metrics.best_val_loss', 'metrics.eval_r2', 'metrics.eval_rmse']])
```

### Using MLflow CLI

```bash
# List all runs
mlflow runs list --experiment-id 0

# Search for runs with good performance
mlflow runs search --filter "metrics.eval_r2 > 0.8"

# Compare specific metrics
mlflow runs compare run1_id run2_id run3_id
```

## Benefits

### 1. Easy Model Management
- All files related to one training run are in one place
- Use run_id to identify everything about a model
- No confusion about which evaluation belongs to which model

### 2. Complete Experiment Tracking
- Training, evaluation, and analysis all logged together
- Single source of truth in MLflow
- Easy to compare different hyperparameter configurations

### 3. Reproducibility
- `run_info.json` contains all metadata
- Config files saved with outputs
- Full artifact history in MLflow

### 4. Efficient Workflows
- Automatic detection - just pass model path
- No manual tracking of run IDs needed
- Scripts handle everything automatically

## Migration from Old Structure

If you have models trained with the old structure (without run_id directories):

1. **Evaluation/Analysis still works** - Scripts will detect missing `run_info.json` and continue without MLflow logging
2. **No breaking changes** - Old models remain functional
3. **Future runs** - All new training runs will use the new structure automatically

To manually migrate an old model:
1. Create a new MLflow run: `mlflow.start_run()`
2. Log the model as an artifact
3. Save the `run_info.json` file
4. Close the run: `mlflow.end_run()`

## Best Practices

1. **Always save run_id** - The training script does this automatically
2. **Use descriptive run names** - Set in config: `mlflow.run_name`
3. **Group related experiments** - Use consistent experiment names
4. **Clean up old runs** - Periodically remove failed/test runs
5. **Back up mlruns/** - This directory contains your experiment history

## Troubleshooting

### "No run_info.json found" warning
- Model was trained before this update
- Evaluation/analysis will work but won't log to MLflow
- Re-train the model to get full tracking

### Can't find model files
- Check `models/<run_id>/` directory
- Run ID is printed during training
- Check MLflow UI for the correct run_id

### MLflow UI doesn't show evaluation metrics
- Ensure `run_info.json` exists in model directory
- Check that evaluation completed successfully
- Look for `eval_*` metrics in the run details

### Disk space concerns
- Each run creates new directories
- Use `.gitignore` to exclude from version control
- Periodically clean up failed/test runs
- Consider using MLflow's artifact storage options
