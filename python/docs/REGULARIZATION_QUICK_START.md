# Regularization Weight Decay - Quick Start

## What Was Implemented

Added **automatic regularization weight decay** that follows the learning rate schedule, implementing the **Interior Point Method** for LMI-constrained RNN training.

## Key Changes

### 1. Config (`src/sysid/config.py`)

Added new fields to `TrainingConfig`:

```python
# Interior Point Method for LMI constraints
use_custom_regularization: bool = False
regularization_weight: float = 0.01  # Initial barrier weight μ₀
decay_regularization_weight: bool = True  # Decay with LR
regularization_decay_factor: float = 0.5  # Decay factor
```

### 2. Trainer (`src/sysid/training/trainer.py`)

- **New method**: `decay_regularization()` - reduces regularization weight
- **Updated `__init__`**: accepts decay parameters
- **Updated training loop**: detects LR changes and decays regularization
- **MLflow logging**: tracks `regularization_weight` over epochs

### 3. Train Script (`scripts/train.py`)

- Passes decay parameters to Trainer
- Logs regularization settings at startup
- Shows decay events during training

## How to Use

### In your config YAML:

```yaml
training:
  # Enable custom regularization (for LMI constraints)
  use_custom_regularization: true
  regularization_weight: 1.0  # Start with μ = 1.0
  
  # Enable decay (Interior Point Method)
  decay_regularization_weight: true
  regularization_decay_factor: 0.5  # Decay by 50% when LR decays
```

### Example Config

See `configs/constrained_rnn_lmi.yaml` for a complete example.

## How It Works

```
Epoch 0-50:
  LR = 1e-3, μ = 1.0
  (Conservative phase - far from constraints)

Epoch 50: Validation loss plateau detected
  LR → 5e-4 (×0.5)
  μ → 0.5 (×0.5)  ← Automatically decayed!
  Console: "Regularization weight decayed: 1.000000e+00 → 5.000000e-01"

Epoch 100: Another plateau
  LR → 2.5e-4 (×0.5)
  μ → 0.25 (×0.5)
  (Solution approaching optimal point)
```

## Training Logs

```
2025-11-06 16:27:21 - INFO - ======================================================================
2025-11-06 16:27:21 - INFO - Starting training...
2025-11-06 16:27:21 - INFO - Custom regularization: enabled
2025-11-06 16:27:21 - INFO -   Initial weight: 1.0
2025-11-06 16:27:21 - INFO -   Decay with LR: enabled (Interior Point Method)
2025-11-06 16:27:21 - INFO -   Decay factor: 0.5
2025-11-06 16:27:21 - INFO - ======================================================================
```

During training when LR is reduced:
```
Regularization weight decayed: 1.000000e+00 → 5.000000e-01
```

## MLflow Tracking

The regularization weight is logged as a metric, allowing you to visualize:
- How regularization changes over training
- Correlation with learning rate decay
- Impact on loss and constraint satisfaction

## Benefits

1. **Automatic**: No manual schedule tuning needed
2. **Synchronized**: Decays with learning rate naturally
3. **Theory-based**: Implements interior point method correctly
4. **Flexible**: Works with any LR scheduler (ReduceLROnPlateau, StepLR, etc.)
5. **Observable**: Logged to console and MLflow

## Files Created

- `docs/REGULARIZATION_DECAY.md` - Detailed documentation
- `configs/constrained_rnn_lmi.yaml` - Example config
- `docs/REGULARIZATION_QUICK_START.md` - This file

## Model Requirements

Your model must implement:

```python
def get_regularization_loss(self) -> torch.Tensor:
    """Return LMI barrier term: -Σ log(det(F_i))"""
    reg_loss = torch.tensor(0.0)
    for F_i in self.get_lmis():
        reg_loss += -torch.logdet(F_i())
    return reg_loss
```

See `src/sysid/models/constrained_rnn.py` (`SimpleLure` class) for reference implementation.

## Configuration Presets

### For LMI-Constrained Models (Recommended)

```yaml
training:
  use_custom_regularization: true
  regularization_weight: 1.0
  decay_regularization_weight: true
  regularization_decay_factor: 0.5

optimizer:
  scheduler_type: "reduce_on_plateau"
  scheduler_patience: 10
  scheduler_factor: 0.5  # Match regularization decay
```

### For Simple Regularization (No Decay)

```yaml
training:
  use_custom_regularization: true
  regularization_weight: 0.01
  decay_regularization_weight: false
```

### Disabled

```yaml
training:
  use_custom_regularization: false  # No custom regularization
```

## Testing

Verified:
✅ Config fields load correctly  
✅ Trainer accepts new parameters  
✅ Decay triggers when LR changes  
✅ Logging to console and MLflow  
✅ Backward compatible (old configs work)  

## Next Steps

1. Update your config to enable regularization decay
2. Train your constrained RNN model
3. Monitor regularization weight in MLflow
4. Adjust `regularization_weight` and `regularization_decay_factor` as needed

See `docs/REGULARIZATION_DECAY.md` for detailed mathematical background and troubleshooting.
