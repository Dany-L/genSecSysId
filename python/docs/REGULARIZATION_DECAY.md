# Regularization Weight Decay for Interior Point Method

## Overview

When training constrained RNN models with LMI (Linear Matrix Inequality) constraints, the regularization term uses a **barrier method** similar to interior point optimization. This document explains how the regularization weight decay works.

## Interior Point Method Background

In convex optimization, interior point methods solve constrained problems by:
1. Adding a **barrier term** to the objective that penalizes constraint violations
2. Gradually **reducing the barrier weight** as the solution approaches optimality
3. This allows the solution to get closer to the constraint boundaries

For LMI constraints, we use the **log-determinant barrier**:
```
minimize: loss(θ) - μ * Σ log(det(F_i(θ)))
```

where:
- `F_i(θ)` are LMI matrices that must be positive definite
- `μ` is the barrier parameter (regularization weight)
- `μ → 0` as optimization progresses

## Implementation

### Configuration

In `config.yaml`:

```yaml
training:
  # Enable custom regularization (LMI constraints)
  use_custom_regularization: true
  regularization_weight: 1.0  # Initial barrier weight μ_0
  
  # Decay regularization weight with learning rate (Interior Point Method)
  decay_regularization_weight: true
  regularization_decay_factor: 0.5  # Reduce by 50% each time
```

### How It Works

1. **Initial Phase**: High regularization weight (`μ = 1.0`) keeps solution far from constraint boundaries
2. **Learning Rate Decay**: When validation loss plateaus, LR scheduler reduces learning rate
3. **Barrier Decay**: Whenever LR is reduced, regularization weight is also reduced
4. **Convergence**: Lower barrier weight allows solution to approach optimal point

### Example Training Progression

```
Epoch 0-50:
  LR = 1e-3
  μ = 1.0
  (Finding feasible region)

Epoch 50: LR plateau detected
  LR → 5e-4 (×0.5)
  μ → 0.5 (×0.5)
  (Getting closer to constraints)

Epoch 100: Another plateau
  LR → 2.5e-4 (×0.5)
  μ → 0.25 (×0.5)
  (Approaching optimal solution)

...
```

## Benefits

### 1. Stable Convergence
- Start with conservative constraints (high μ)
- Gradually relax as model learns
- Avoids constraint violations during early training

### 2. Synchronized Decay
- Regularization decays with learning rate
- Both parameters reduce when progress stalls
- Natural coupling between optimization phases

### 3. Automatic Scheduling
- No need to manually tune decay schedule
- Adapts to actual training dynamics
- Works with any LR scheduler

## Model Requirements

Your model must implement `get_regularization_loss()`:

```python
class ConstrainedRNN(BaseRNN):
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute LMI barrier term: -Σ log(det(F_i))
        
        Returns:
            Regularization loss (barrier penalty)
        """
        reg_loss = torch.tensor(0.0)
        
        # For each LMI constraint F_i(θ) > 0
        for F_i in self.get_lmis():
            # Add log-determinant barrier
            reg_loss += -torch.logdet(F_i())
        
        return reg_loss
```

## Training Logs

The trainer logs regularization weight changes:

```
Epoch 50: Validation loss plateaued
  Learning rate: 1e-3 → 5e-4
  Regularization weight decayed: 1.000000e+00 → 5.000000e-01

Epoch 100: Validation loss plateaued
  Learning rate: 5e-4 → 2.5e-4
  Regularization weight decayed: 5.000000e-01 → 2.500000e-01
```

MLflow tracks `regularization_weight` as a metric over epochs.

## Configuration Options

### Decay Enabled (Recommended for LMI)

```yaml
training:
  use_custom_regularization: true
  regularization_weight: 1.0
  decay_regularization_weight: true  # Enable decay
  regularization_decay_factor: 0.5   # Match LR decay
```

**Use when:**
- Training constrained models (LMI, convex constraints)
- Using barrier/penalty methods
- Want solution to approach constraint boundaries

### Decay Disabled (Constant Weight)

```yaml
training:
  use_custom_regularization: true
  regularization_weight: 0.01
  decay_regularization_weight: false  # Keep constant
```

**Use when:**
- Simple L2/L1 regularization
- Want fixed penalty throughout training
- Regularization for preventing overfitting only

## Scheduler Compatibility

Works with all PyTorch LR schedulers:

### ReduceLROnPlateau (Most Common)
```yaml
optimizer:
  scheduler_type: "reduce_on_plateau"
  scheduler_patience: 10
  scheduler_factor: 0.5
```
- Decays when validation loss stops improving
- Adaptive to training dynamics

### StepLR
```yaml
optimizer:
  scheduler_type: "step"
  step_size: 50
  gamma: 0.5
```
- Decays every N epochs
- Predictable schedule

### ExponentialLR
```yaml
optimizer:
  scheduler_type: "exponential"
  gamma: 0.95
```
- Continuous decay
- Smooth reduction

## Mathematical Details

### Total Loss Function

```
L_total(θ) = L_task(θ) + μ * L_reg(θ)
```

where:
- `L_task`: Task loss (MSE, MAE, etc.)
- `L_reg`: Regularization term (LMI barrier)
- `μ`: Regularization weight (decays over time)

### LMI Barrier Term

For stability constraints in the form `F(θ) > 0`:

```
L_reg(θ) = -Σ_i log(det(F_i(θ)))
```

This approaches infinity as `F_i` approaches singularity (constraint violation).

### Decay Rule

When learning rate scheduler triggers:

```
μ_{t+1} = γ * μ_t
```

where `γ` is `regularization_decay_factor` (typically 0.5).

## Example: SimpleLure Model

See `src/sysid/models/constrained_rnn.py` for full implementation:

```python
class SimpleLure(nn.Module):
    def __init__(self, ...):
        # Define LMI matrices based on Lyapunov theory
        self.P = nn.Parameter(torch.eye(nx))  # Lyapunov matrix
        self.L = nn.Parameter(torch.zeros(nz, nx))  # Locality
        # ... other parameters
    
    def get_lmis(self):
        """Construct LMI constraints for stability."""
        lmi_list = []
        
        # Stability LMI
        def stability_lmi() -> torch.Tensor:
            F = torch_bmat([
                [-self.alpha**2 * self.P, ..., self.P @ self.A.T],
                [..., -self.P]
            ])
            return -F  # Must be positive definite
        
        lmi_list.append(stability_lmi)
        # ... more LMIs
        
        return lmi_list
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Barrier method for LMI constraints."""
        reg_loss = torch.tensor(0.0)
        for f_i in self.get_lmis():
            reg_loss += -torch.logdet(f_i())
        return reg_loss
```

## Tips

### Initial Weight Selection

- **High initial weight** (μ₀ = 1.0 to 10.0): More conservative, slower but stable
- **Low initial weight** (μ₀ = 0.01 to 0.1): Faster but may violate constraints early

### Decay Factor

- **Aggressive** (γ = 0.3 to 0.5): Fast convergence, may become unstable
- **Conservative** (γ = 0.7 to 0.9): Slow but stable, good for tight constraints

### Matching LR Schedule

Generally best to use **same factor** for both:
```yaml
optimizer:
  scheduler_factor: 0.5

training:
  regularization_decay_factor: 0.5  # Match!
```

## Troubleshooting

### Constraint Violations

**Symptom**: Model violates LMI constraints during training

**Solution**:
- Increase `regularization_weight` (start higher)
- Use more conservative `regularization_decay_factor` (0.7-0.9)
- Increase `scheduler_patience` (decay less frequently)

### Slow Convergence

**Symptom**: Training takes very long, loss decreases slowly

**Solution**:
- Decrease `regularization_weight` (less conservative)
- Use more aggressive `regularization_decay_factor` (0.3-0.5)
- Check if regularization weight becomes too small (monitor logs)

### NaN/Inf Losses

**Symptom**: Loss becomes NaN or Inf

**Solution**:
- Regularization weight decayed too fast (constraints violated)
- LMI matrices became singular
- Increase initial weight and use conservative decay

## References

- **Interior Point Methods**: Boyd & Vandenberghe, "Convex Optimization", Ch. 11
- **LMI Constraints**: "Linear Matrix Inequalities in System and Control Theory"
- **Lure Systems**: Your literature folder has relevant papers on stability analysis

## Summary

This implementation provides:
✅ Automatic regularization weight decay tied to learning rate  
✅ Interior point method for LMI constraints  
✅ Configurable decay schedule  
✅ MLflow tracking of all parameters  
✅ Compatible with existing LR schedulers  

Just enable in config and the trainer handles the rest!
