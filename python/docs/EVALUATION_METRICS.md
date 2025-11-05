# Configurable Evaluation Metrics

This guide explains how to configure which evaluation metrics are computed and logged during model evaluation.

## Overview

The evaluation script computes various metrics to assess model performance. You can control which metrics are computed and logged to MLflow by configuring the `evaluation` section in your config file.

## Configuration

Add an `evaluation` section to your YAML config file:

```yaml
evaluation:
  metrics:
    - rmse      # Root Mean Squared Error
    - mae       # Mean Absolute Error
    - r2        # R-squared score
    - nrmse     # Normalized RMSE
    - max_error # Maximum absolute error
```

## Available Metrics

### Base Metrics

These metrics compare predictions to targets across all samples:

- **`mse`** - Mean Squared Error
  - Formula: `mean((y_pred - y_true)²)`
  - Range: [0, ∞), lower is better
  - Penalizes large errors more heavily

- **`rmse`** - Root Mean Squared Error
  - Formula: `sqrt(mean((y_pred - y_true)²))`
  - Range: [0, ∞), lower is better
  - Same units as the target variable

- **`mae`** - Mean Absolute Error
  - Formula: `mean(|y_pred - y_true|)`
  - Range: [0, ∞), lower is better
  - More robust to outliers than MSE

- **`r2`** - R-squared (coefficient of determination)
  - Formula: `1 - (SS_residual / SS_total)`
  - Range: (-∞, 1], higher is better
  - Proportion of variance explained by the model
  - Perfect prediction = 1.0, baseline (mean) = 0.0

- **`nrmse`** - Normalized RMSE
  - Formula: `rmse / (max(y_true) - min(y_true))`
  - Range: [0, ∞), lower is better
  - Scale-independent, normalized by data range

- **`max_error`** - Maximum Absolute Error
  - Formula: `max(|y_pred - y_true|)`
  - Range: [0, ∞), lower is better
  - Worst-case error across all predictions

### Sequence Metrics

For multi-step sequence predictions, each base metric has two additional variants:

- **`<metric>_avg`** - Average metric over all time steps
  - Example: `rmse_avg` = average RMSE across sequence length
  - Useful for understanding average per-step performance

- **`<metric>_final`** - Metric at the final time step only
  - Example: `rmse_final` = RMSE at the last prediction step
  - Useful for long-term prediction quality

Example: If you enable `rmse`, you will also get `rmse_avg` and `rmse_final` for sequence data.

## Examples

### Minimal Configuration (Only Key Metrics)

```yaml
evaluation:
  metrics:
    - rmse  # Most common metric
    - r2    # Model quality indicator
```

This logs only:
- `rmse`, `rmse_avg`, `rmse_final`
- `r2`, `r2_avg`, `r2_final`

### Standard Configuration (Recommended)

```yaml
evaluation:
  metrics:
    - rmse      # Root Mean Squared Error
    - mae       # Mean Absolute Error
    - r2        # R-squared score
    - nrmse     # Normalized RMSE
    - max_error # Maximum absolute error
```

This logs 5 base metrics + their `_avg` and `_final` variants (15 total for sequences).

### Full Configuration (All Metrics)

```yaml
evaluation:
  metrics:
    - mse
    - rmse
    - mae
    - r2
    - nrmse
    - max_error
```

Or simply omit the `evaluation` section entirely to use all metrics by default.

### No Configuration (Default)

If you don't include an `evaluation` section, all metrics are enabled by default:

```yaml
# No evaluation section = all metrics enabled
data:
  train_path: "data/prepared"
  ...
model:
  model_type: "rnn"
  ...
```

## Usage in Scripts

### Training Script

The training script is not affected by the evaluation configuration. It always logs:
- `best_val_loss`
- `best_epoch`
- `final_epoch`
- Per-epoch: `train_loss`, `val_loss`

### Evaluation Script

```bash
python scripts/evaluate.py \
  --config configs/rnn_baseline.yaml \
  --model models/rnn_baseline/<run_id>/best_model.pt \
  --test-data data/prepared/test
```

Only the metrics configured in `configs/rnn_baseline.yaml` will be computed and logged.

### MLflow Tracking

All configured metrics are logged to MLflow with the `eval_` prefix:
- `eval_rmse`, `eval_rmse_avg`, `eval_rmse_final`
- `eval_mae`, `eval_mae_avg`, `eval_mae_final`
- etc.

View them in the MLflow UI or query programmatically:

```python
import mlflow

mlflow.set_experiment("rnn_baseline")
runs = mlflow.search_runs()

# Show configured metrics
print(runs[["metrics.eval_rmse", "metrics.eval_r2", "metrics.eval_mae"]])
```

## Best Practices

### For Experimentation
- Use minimal metrics (`rmse`, `r2`) to reduce clutter
- Focus on metrics relevant to your specific problem

### For Production/Reporting
- Use standard configuration with 4-5 key metrics
- Include both error metrics (RMSE, MAE) and quality metrics (R², NRMSE)

### For Research/Analysis
- Use all metrics to fully characterize model performance
- Compare different metrics to understand model behavior

## Metric Selection Guide

**Choose metrics based on your use case:**

| Use Case | Recommended Metrics | Reason |
|----------|-------------------|--------|
| **General purpose** | `rmse`, `r2` | Most common, well-understood |
| **Outlier sensitivity** | `mae`, `max_error` | Robust to outliers, shows worst case |
| **Scale-independent comparison** | `nrmse`, `r2` | Compare across different datasets |
| **Long-term prediction** | `*_final` variants | Focus on prediction horizon |
| **Average behavior** | `*_avg` variants | Overall sequence performance |

## Notes

- All configured metrics are computed during evaluation (no performance difference)
- Only configured metrics are logged to MLflow (reduces storage and UI clutter)
- The `per_step` detailed metrics are computed but never logged to MLflow
- Console output shows all computed metrics (even filtered ones)
- Evaluation results JSON file contains all computed metrics

## See Also

- [MLflow Run Organization](MLFLOW_RUN_ORGANIZATION.md) - How evaluation results are organized
- [Config Schema](../configs/rnn_baseline.yaml) - Example configuration
- [Metrics Implementation](../src/sysid/evaluation/metrics.py) - Metric calculation details
