# Dual Regularization for Constrained RNN

## Overview

The `SimpleLure` model now supports two regularization methods for enforcing LMI constraints:

1. **Interior Point Method** (default) - Original implementation
2. **Dual Method** (new) - Robust to infeasible parameters

## Problem with Interior Point Method

The interior point method uses a log-determinant barrier function:

```python
reg_loss = -Σ log(det(F_i))
```

**Limitations:**
- Requires **strictly feasible** parameters (all LMI eigenvalues > 0)
- Gradients **explode** when constraints are violated (log(det) → -∞)
- Training can fail if parameters become even slightly infeasible

## Dual Method Solution

The dual method penalizes constraint violations instead of using a barrier:

```python
reg_loss = ρ * Σ Σ max(0, -λ_i)²
```

where:
- `ρ` is an adaptive dual penalty coefficient
- `λ_i` are eigenvalues of LMI matrices `F_i`
- Only negative eigenvalues contribute to the penalty

**Advantages:**
- ✅ Allows **infeasible parameters** during training
- ✅ **Smooth gradients** even when constraints are violated
- ✅ Adaptive penalty coefficient `ρ` grows/shrinks based on constraint satisfaction
- ✅ More robust optimization, especially early in training

## Configuration

Add the following to your model's `custom_params` in the config file:

```yaml
model:
  type: SimpleLure
  custom_params:
    regularization_method: "dual"  # or "interior_point" (default)
    dual_penalty_init: 1.0         # Initial penalty coefficient (default: 1.0)
    dual_penalty_growth: 1.1       # Growth factor when violated (default: 1.1)
    dual_penalty_shrink: 0.9       # Shrink factor when satisfied (default: 0.9)
```

### Parameter Guidelines

**`regularization_method`:**
- `"interior_point"`: Use when you have a good initialization and expect to stay feasible
- `"dual"`: Use when starting from random/infeasible initialization

**`dual_penalty_init`:**
- Start small (e.g., 0.1-1.0) if you expect significant violations early
- Start larger (e.g., 10-100) if you're close to feasible

**`dual_penalty_growth`:**
- Larger values (e.g., 1.2-2.0) = more aggressive penalty increase
- Smaller values (e.g., 1.05-1.1) = gentler adaptation

**`dual_penalty_shrink`:**
- Smaller values (e.g., 0.5-0.8) = more aggressive penalty reduction
- Larger values (e.g., 0.9-0.95) = gentler adaptation

## Monitoring

When using the dual method, MLflow automatically logs:

- `dual_penalty`: Current value of the penalty coefficient ρ
- `constraint_violation`: Total sum of negative eigenvalues
- `constraints`: Boolean indicating if all constraints are satisfied

### Expected Behavior

During training, you should see:

1. **Early epochs**: 
   - `constraint_violation` may be large
   - `dual_penalty` increases
   - Model learns to reduce violations

2. **Mid training**:
   - `constraint_violation` decreases
   - `dual_penalty` may fluctuate
   - Model balances accuracy vs constraints

3. **Late training**:
   - `constraint_violation` → 0
   - `constraints` → True
   - `dual_penalty` decreases
   - Model converges to feasible solution

## API Reference

### New Methods in `SimpleLure`

```python
def get_regularization_loss(method: Optional[str] = None) -> torch.Tensor:
    """
    Compute regularization loss.
    
    Args:
        method: 'interior_point' or 'dual'. If None, uses self.regularization_method
        
    Returns:
        Regularization loss tensor
    """

def update_dual_penalty(constraints_satisfied: bool):
    """
    Update the dual penalty coefficient.
    Called automatically during training after each epoch.
    
    Args:
        constraints_satisfied: True if all LMI constraints satisfied
    """

def get_constraint_violation() -> float:
    """
    Compute total constraint violation (sum of negative eigenvalues).
    
    Returns:
        Total violation (0 if all constraints satisfied)
    """
```

## Example Configuration

### Conservative Dual Method
```yaml
model:
  type: SimpleLure
  custom_params:
    regularization_method: "dual"
    dual_penalty_init: 10.0
    dual_penalty_growth: 1.05
    dual_penalty_shrink: 0.95

training:
  regularization_weight: 1.0
```

### Aggressive Dual Method
```yaml
model:
  type: SimpleLure
  custom_params:
    regularization_method: "dual"
    dual_penalty_init: 0.1
    dual_penalty_growth: 1.5
    dual_penalty_shrink: 0.7

training:
  regularization_weight: 0.1  # Lower weight since penalty adapts
```

## Implementation Details

### Eigenvalue Computation

The dual method uses `torch.linalg.eigvalsh()` which:
- Assumes symmetric/Hermitian matrices (valid for LMIs)
- Returns real eigenvalues in ascending order
- More efficient than general eigenvalue decomposition

### Penalty Update Rule

After each epoch:
```python
if all_constraints_satisfied:
    dual_penalty *= dual_penalty_shrink  # Reduce penalty
else:
    dual_penalty *= dual_penalty_growth  # Increase penalty
```

This creates a **feedback loop** that automatically tunes the penalty strength.

### Gradient Behavior

Unlike the interior point method, the dual method has bounded gradients:

```
∂reg_loss/∂θ = 2ρ Σ max(0, -λ_i) * ∂λ_i/∂θ
```

Even with large violations, gradients remain finite and usable.

## Switching Between Methods

You can switch methods mid-training by:

1. Load checkpoint with `interior_point` method
2. Change config to `dual` method
3. Continue training

The model will automatically use the new regularization in subsequent epochs.

## Troubleshooting

### Dual penalty grows unbounded
- Reduce `dual_penalty_growth` (e.g., 1.05 instead of 1.5)
- Increase `regularization_weight` globally
- Check if constraints are actually satisfiable

### Constraints never satisfied
- Model may need more capacity
- Try reducing `regularization_weight` initially
- Consider using warmup: start with low weight, increase gradually

### Training unstable
- Reduce `dual_penalty_init` (start smaller)
- Use gradient clipping
- Reduce learning rate

## References

- Interior Point Methods: Boyd & Vandenberghe, "Convex Optimization" (2004), Chapter 11
- Augmented Lagrangian Methods: Nocedal & Wright, "Numerical Optimization" (2006), Chapter 17
- LMI Constraints: Boyd et al., "Linear Matrix Inequalities in System and Control Theory" (1994)
