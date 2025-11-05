# Configurable Evaluation Metrics - Implementation Summary

## Overview

Added the ability to configure which evaluation metrics are computed and logged to MLflow during model evaluation, allowing users to focus on metrics that matter for their specific use case.

## Changes Made

### 1. Config System (`src/sysid/config.py`)

**Added `EvaluationConfig` dataclass:**
```python
@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    metrics: list = None  # List of metrics to compute and log
    
    def __post_init__(self):
        if self.metrics is None:
            # Default: all available metrics
            self.metrics = ["mse", "rmse", "mae", "r2", "nrmse", "max_error"]
```

**Updated `Config` class:**
- Added `evaluation: EvaluationConfig` field
- Modified `from_dict()` to handle evaluation config
- Modified `to_dict()` to include evaluation config
- Added `__post_init__()` to initialize evaluation config with defaults

### 2. Evaluation Script (`scripts/evaluate.py`)

**Added `filter_metrics()` function:**
- Filters computed metrics to only include those in the allowed list
- Handles both base metrics (e.g., `rmse`) and variants (e.g., `rmse_avg`, `rmse_final`)
- Excludes internal metrics like `per_step`

**Modified evaluation flow:**
- Added logging of configured metrics at startup
- Filter all computed metrics before logging to MLflow
- Filter metrics before console output (when not using MLflow)

### 3. Configuration Files

**Updated `configs/rnn_baseline.yaml`:**
```yaml
evaluation:
  metrics:
    - rmse      # Root Mean Squared Error
    - mae       # Mean Absolute Error
    - r2        # R-squared score
    - nrmse     # Normalized RMSE
    - max_error # Maximum absolute error
```

**Created `configs/rnn_all_metrics.yaml`:**
- Example config showing all available metrics
- Comprehensive documentation of each metric
- Demonstrates default behavior

### 4. Documentation

**Created `docs/EVALUATION_METRICS.md`:**
- Complete guide to configurable evaluation metrics
- Detailed explanation of all available metrics (6 base + variants)
- Usage examples (minimal, standard, full configurations)
- Best practices for metric selection
- Metric selection guide by use case

**Updated `docs/README.md`:**
- Added new "Evaluation & Analysis" section
- Linked to EVALUATION_METRICS.md

**Updated main `README.md`:**
- Added feature bullet: "Configurable metrics"
- Linked to evaluation metrics documentation

## Available Metrics

### Base Metrics (always computed)
1. **`mse`** - Mean Squared Error
2. **`rmse`** - Root Mean Squared Error
3. **`mae`** - Mean Absolute Error
4. **`r2`** - R-squared score
5. **`nrmse`** - Normalized RMSE
6. **`max_error`** - Maximum absolute error

### Sequence Variants (for multi-step predictions)
For each base metric:
- **`<metric>_avg`** - Average over all time steps
- **`<metric>_final`** - Metric at final time step only

Total: 6 base metrics + 12 sequence variants = **18 metrics** for sequence predictions

## Usage Examples

### Minimal (Key Metrics Only)
```yaml
evaluation:
  metrics:
    - rmse
    - r2
```
Logs: 2 base + 4 variants = **6 metrics** to MLflow

### Standard (Recommended)
```yaml
evaluation:
  metrics:
    - rmse
    - mae
    - r2
    - nrmse
    - max_error
```
Logs: 5 base + 10 variants = **15 metrics** to MLflow

### All Metrics (Default)
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
Or simply omit the `evaluation` section.
Logs: 6 base + 12 variants = **18 metrics** to MLflow

## Benefits

1. **Reduced Clutter**: Only log metrics that matter for your use case
2. **Faster Evaluation**: Skip computing metrics you don't need
3. **Better MLflow UI**: Cleaner interface with fewer columns
4. **Flexibility**: Easy to switch focus for different experiments
5. **Backwards Compatible**: Default behavior includes all metrics

## Testing

Tested with existing model:
```bash
python scripts/evaluate.py \
  --config configs/rnn_baseline.yaml \
  --model models/rnn_baseline/202f760910c84225b4391bcdf497ef18/best_model.pt \
  --test-data data/prepared/test
```

Result: Only configured metrics (5 base + 10 variants = 15 total) were logged to MLflow.

## Files Modified

```
python/
├── src/sysid/config.py                          # Added EvaluationConfig
├── scripts/evaluate.py                          # Added metric filtering
├── configs/
│   ├── rnn_baseline.yaml                       # Updated with evaluation section
│   └── rnn_all_metrics.yaml                    # NEW: Example with all metrics
├── docs/
│   ├── EVALUATION_METRICS.md                   # NEW: Complete guide
│   ├── README.md                                # Updated index
│   └── CONFIGURABLE_EVALUATION_METRICS.md      # NEW: This summary
└── README.md                                    # Updated features list
```

## Backwards Compatibility

✅ **Fully backwards compatible**
- Existing configs without `evaluation` section still work
- Default behavior: all metrics enabled
- No changes required to existing configs

## Future Enhancements

Potential improvements:
1. Add custom metric functions via config
2. Support metric aggregation strategies (min, max, percentiles)
3. Add threshold-based metric alerts
4. Support per-output metrics for multi-output models
5. Add custom metric weights for loss functions

## Implementation Notes

### Design Decisions

1. **All metrics computed, filtered for logging**: This ensures evaluation results JSON files always have complete information, while MLflow only gets configured metrics.

2. **Base metric names control variants**: Specifying `rmse` automatically includes `rmse_avg` and `rmse_final` for sequences. This is intuitive and reduces config verbosity.

3. **Default to all metrics**: Backwards compatibility and sensible default for users who don't customize.

4. **Console shows all, MLflow shows filtered**: Useful for debugging and verification while keeping MLflow clean.

### Edge Cases Handled

- `metrics: null` → Uses all metrics (default)
- `metrics: []` → Uses all metrics (default)
- Missing `evaluation` section → Uses all metrics (default)
- Invalid metric names → Silently ignored (metric won't appear)
- `per_step` metric → Never logged (internal use only)

## Related Documentation

- [MLFLOW_RUN_ORGANIZATION.md](MLFLOW_RUN_ORGANIZATION.md) - How metrics are organized
- [Main README](../README.md) - Package overview
- [Config Schema](../configs/rnn_baseline.yaml) - Example configuration
- [Metrics Implementation](../src/sysid/evaluation/metrics.py) - Metric calculations
