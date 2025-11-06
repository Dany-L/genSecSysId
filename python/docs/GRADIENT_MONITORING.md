# Gradient Monitoring

## Overview

The training pipeline now includes **comprehensive gradient monitoring** to help debug and optimize training, especially important for constrained optimization with LMI constraints.

## Gradient Statistics Tracked

### Global Metrics

1. **`grad_norm_total`** - L2 norm of all gradients combined
   - **Purpose**: Overall magnitude of the gradient
   - **Interpretation**: 
     - Too large (>100): Risk of exploding gradients
     - Too small (<1e-6): Vanishing gradients, learning stalled
     - Typical range: 0.01 - 10.0

2. **`grad_value_max`** - Largest absolute gradient value
   - **Purpose**: Detect individual exploding gradients
   - **Watch for**: Sudden spikes indicating instability

3. **`grad_value_min`** - Smallest absolute gradient value
   - **Purpose**: Detect vanishing gradients
   - **Watch for**: Very small values (<1e-10)

4. **`grad_mean`** - Mean gradient value
   - **Purpose**: Check for bias in gradients
   - **Interpretation**: Should oscillate around 0

5. **`grad_std`** - Standard deviation of gradients
   - **Purpose**: Measure gradient variance
   - **Interpretation**: High variance may indicate instability

### Per-Parameter Metrics

6. **`grad_norm_max`** - Maximum gradient norm across parameters
   - **Purpose**: Identify which parameter groups have largest gradients
   - **Use**: Find bottlenecks in learning

7. **`grad_norm_min`** - Minimum gradient norm across parameters  
   - **Purpose**: Identify parameters that aren't learning
   - **Use**: Detect dead neurons or frozen layers

8. **`dead_params_ratio`** - Ratio of parameters with zero gradient
   - **Purpose**: Detect dead neurons/layers
   - **Interpretation**:
     - 0.0: All parameters learning (good)
     - >0.1: Some parameters stuck (investigate)
     - >0.5: Major problem - many parameters dead

### Update Size Metrics

9. **`grad_param_ratio_mean`** - Mean ratio of gradient norm to parameter norm
   - **Purpose**: Measure relative size of updates
   - **Interpretation**:
     - <0.001: Updates too small, slow learning
     - 0.001-0.1: Healthy range
     - >0.1: Updates too large, may overshoot

10. **`grad_param_ratio_max`** - Maximum gradient-to-parameter ratio
    - **Purpose**: Detect parameters with extreme updates
    - **Use**: Identify unstable parameters

## Configuration

### Enable/Disable Gradient Logging

In your config YAML:

```yaml
training:
  log_gradients: true  # Enable gradient monitoring (default)
  # or
  log_gradients: false  # Disable for faster training
```

**When to disable:**
- Production training with known stable configuration
- When every millisecond counts
- Already debugged and optimized

**When to enable (recommended):**
- Debugging training issues
- Developing new models
- Training constrained models (LMI, etc.)
- First time with a new dataset

## Viewing in MLflow

All gradient statistics are logged to MLflow under the `grad/` namespace:

```
grad/grad_norm_total
grad/grad_value_max
grad/grad_value_min
grad/grad_mean
grad/grad_std
grad/grad_norm_max
grad/grad_norm_min
grad/dead_params_ratio
grad/grad_param_ratio_mean
grad/grad_param_ratio_max
```

### In MLflow UI:

1. Start MLflow: `mlflow ui`
2. Navigate to your run
3. Click "Metrics" tab
4. Filter for `grad/` to see all gradient metrics
5. Plot multiple metrics together to see correlations

## Progress Bar Display

During training, the global gradient norm is shown in the progress bar:

```
Training Progress:  25%|██▌       | 250/1000 [02:15<06:45, 1.85it/s, 
    train_loss=0.0234, 
    val_loss=0.0289, 
    best_val=0.0245, 
    patience=5/50,
    grad_norm=1.23e-02]  ← Gradient norm
```

## Common Patterns & Diagnostics

### Pattern 1: Healthy Training

```
grad_norm_total: Gradually decreasing (1.0 → 0.01 → 0.001)
dead_params_ratio: 0.0
grad_param_ratio_mean: 0.01 - 0.1
```

**Interpretation**: Normal convergence, all parameters learning

### Pattern 2: Exploding Gradients

```
grad_norm_total: Suddenly spikes (0.1 → 100 → NaN)
grad_value_max: Extremely large (>1000)
```

**Solutions:**
- Add/reduce gradient clipping: `gradient_clip_value: 1.0`
- Reduce learning rate: `learning_rate: 1e-4`
- Check for numerical instability in model

### Pattern 3: Vanishing Gradients

```
grad_norm_total: Very small (<1e-6) and not decreasing
dead_params_ratio: Increasing over time
grad_value_min: Nearly zero
```

**Solutions:**
- Increase learning rate
- Check activation functions (use ReLU instead of sigmoid)
- Reduce model depth
- Use skip connections or normalization

### Pattern 4: Dead Neurons

```
dead_params_ratio: >0.2
grad_norm_min: 0.0
```

**Solutions:**
- Different initialization (Xavier, He)
- Different activation function
- Add batch normalization
- Reduce dropout

### Pattern 5: LMI Constraint Violations (Your Use Case)

```
grad_norm_total: Spikes when regularization_weight is low
grad_param_ratio_max: Very large for LMI parameters
```

**Expected behavior:**
- Early epochs: High `grad_norm_total` (regularization dominates)
- As `regularization_weight` decays: Gradients become smaller
- Parameters approach feasibility boundary

**Watch for:**
- Sudden gradient spikes when reg weight decays
- `dead_params_ratio` increasing for non-LMI parameters
- Very high `grad_param_ratio` for P, L, or A matrices

## For LMI-Constrained Models

### Special Considerations

Your `SimpleLure` model has:
- **LMI parameters**: P (Lyapunov), L (Locality), A, B, C matrices
- **Regularization**: Log-determinant barrier

### What to Monitor:

1. **Gradient norm vs Regularization weight**
   - Plot: `grad/grad_norm_total` vs `regularization_weight`
   - Expected: Inversely correlated
   - As μ decreases, gradients should decrease

2. **Dead parameters**
   - LMI matrices should never be dead
   - If `dead_params_ratio` > 0, investigate which parameters

3. **Gradient-to-parameter ratio**
   - LMI parameters may have different update scales
   - Monitor `grad_param_ratio_max` for extreme values

### Debugging LMI Training:

**Problem**: Training unstable after regularization decay

**Diagnosis**:
```python
# In MLflow, compare these metrics:
# - grad/grad_norm_total (should increase when reg weight decays)
# - regularization_weight (should decrease over time)
# - train_loss (should not spike)
```

**Solution**: Slow down regularization decay or use smaller decay factor

## Performance Impact

Gradient logging adds minimal overhead:
- **Computation**: ~0.5% per batch (negligible)
- **Memory**: No additional memory (uses existing gradients)
- **MLflow**: One metric per statistic per epoch (~10 metrics)

**Disable only if:**
- Training on extremely large models (billions of parameters)
- Need every bit of performance for production
- Already fully optimized and debugged

## Example: Analyzing a Training Run

### Scenario: Training Unstable

1. **Check gradient norm:**
   ```
   mlflow ui
   → grad/grad_norm_total
   → Look for spikes or NaN
   ```

2. **Check dead neurons:**
   ```
   → grad/dead_params_ratio
   → If >0.1, some parameters not learning
   ```

3. **Check update size:**
   ```
   → grad/grad_param_ratio_mean
   → If >1.0, updates too large
   ```

4. **Correlate with learning rate:**
   ```
   → Plot lr vs grad/grad_norm_total
   → Should decrease together
   ```

## API Reference

### Trainer

```python
trainer = Trainer(
    model=model,
    log_gradients=True,  # Enable gradient logging
    ...
)
```

### Gradient Statistics Method

```python
stats = trainer.compute_gradient_stats()
# Returns:
# {
#     'grad_norm_total': float,
#     'grad_value_max': float,
#     'grad_value_min': float,
#     'grad_mean': float,
#     'grad_std': float,
#     'grad_norm_max': float,
#     'grad_norm_min': float,
#     'dead_params_ratio': float,
#     'grad_param_ratio_mean': float,
#     'grad_param_ratio_max': float,
# }
```

## Best Practices

### 1. Always Monitor Initially

Start with `log_gradients: true` for new:
- Models
- Datasets
- Hyperparameter combinations
- Constraint formulations

### 2. Look for Trends

Don't just look at individual values, track:
- Changes over epochs
- Correlation with learning rate
- Behavior after regularization decay

### 3. Set Alerts

In MLflow, you can set alerts for:
- `grad_norm_total > 10.0` (exploding)
- `grad_norm_total < 1e-6` (vanishing)
- `dead_params_ratio > 0.2` (dead neurons)

### 4. Compare Runs

Use MLflow to compare gradient stats across:
- Different learning rates
- Different regularization schedules
- Different initializations

## Summary

Gradient monitoring provides invaluable insights into:
✅ Training stability  
✅ Convergence behavior  
✅ Parameter learning rates  
✅ Dead neurons detection  
✅ Update size analysis  
✅ LMI constraint satisfaction (for your models)  

Enable it during development, disable only when fully optimized!
