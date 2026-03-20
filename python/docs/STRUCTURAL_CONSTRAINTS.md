# Structural Constraints for SimpleLure Models

This document describes the structural constraints feature for SimpleLure models, which allows encoding domain-specific knowledge (e.g., physics-based constraints) directly into model parameters.

## Overview

Structural constraints enable you to:
- **Fix parameters** to specific values (e.g., output matrix C = [1, 0] to observe only position)
- **Partially constrain parameters** by making only certain rows/columns learnable
- **Encode physical laws** into the model structure (e.g., Duffing oscillator dynamics)

All constraints are:
- ✅ **Fully backward compatible** (existing configs work unchanged)
- ✅ **Compatible with all initialization methods** (ESN, N4SID, Identity)
- ✅ **Compatible with LMI regularization** (interior point/dual methods)
- ✅ **Tracked in MLflow** for reproducibility

## Quick Start: Duffing Oscillator Example

The Duffing oscillator has dynamics:
```
x1_dot = x2                                  (velocity)
x2_dot = -delta*x2 - alpha*x1 - beta*x1^3 + gamma*u(t)
y = x1                                       (observe position only)
```

Key structural properties:
- Only x2 (velocity) is affected by input and nonlinearity
- Output observes only x1 (position): C = [1, 0]
- No direct feedthrough: D = 0, D12 = 0

### Configuration

```yaml
model:
  model_type: "simple_lure"
  nd: 1   # Input dimension (forcing)
  ne: 1   # Output dimension (position)
  nx: 2   # State dimension (position + velocity)
  nw: 10  # Nonlinearity dimension
  activation: "dzn"
  
  custom_params:
    structural_constraints:
      # Only x2_dot depends on input
      B:
        learnable_rows: [1]  # Only row 1 (x2) is learnable
        fixed_value: 0.0     # Row 0 (x1) fixed to zero
      
      # Only x2_dot has nonlinear term
      B2:
        learnable_rows: [1]
        fixed_value: 0.0
      
      # Observe position only
      C:
        fixed: true
        value: [[1.0, 0.0]]  # y = x1
      
      # No direct feedthrough
      D:
        fixed: true
        value: 0.0
      
      D12:
        fixed: true
        value: 0.0
```

See `configs/example_duffing_oscillator.yaml` for a complete working example.

## Constraint Types

### 1. Fully Fixed Parameters

Fix a parameter to a specific value (scalar or array). The parameter will have `requires_grad=False`.

```yaml
structural_constraints:
  # Scalar parameter
  D:
    fixed: true
    value: 0.0
  
  # Matrix parameter
  C:
    fixed: true
    value: [[1.0, 0.0]]  # Shape must match parameter dimensions
```

**Use cases:**
- No direct feedthrough: `D: {fixed: true, value: 0.0}`
- Observe specific states: `C: {fixed: true, value: [[1, 0]]}`
- Known parameter values from physics

### 2. Learnable Rows

Make only specific rows of a parameter learnable. Other rows are fixed to `fixed_value`.

```yaml
structural_constraints:
  B:
    learnable_rows: [1, 2]  # Only rows 1 and 2 are learnable
    fixed_value: 0.0         # Rows 0, 3, 4, ... fixed to zero
```

**Use cases:**
- Only certain states affected by input (e.g., only velocity in Duffing oscillator)
- Block diagonal structures
- Sparse input matrices

**Implementation:** Uses gradient masking - gradients for non-learnable rows are zeroed during backpropagation.

### 3. Learnable Columns

Make only specific columns of a parameter learnable. Other columns are fixed to `fixed_value`.

```yaml
structural_constraints:
  C:
    learnable_cols: [0]  # Only first column learnable
    fixed_value: 0.0     # Other columns fixed to zero
```

**Use cases:**
- Observe only certain states
- Sparse output matrices
- Structured observability

**Implementation:** Uses gradient masking - gradients for non-learnable columns are zeroed during backpropagation.

## Supported Parameters

Constraints can be applied to any of these SimpleLure parameters:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `A` | (nx, nx) | State transition matrix |
| `B` | (nx, nd) | Input matrix |
| `B2` | (nx, nw) | Nonlinearity input matrix |
| `C` | (ne, nx) | Output matrix |
| `D` | (ne, nd) | Direct feedthrough |
| `D12` | (ne, nw) | Nonlinearity feedthrough |
| `C2` | (nz, nx) | Nonlinearity observation |
| `D21` | (nz, nd) | Nonlinearity input feedthrough |
| `D22` | (nz, nw) | Nonlinearity self-feedthrough |

## Interaction with Initialization

All three initialization methods respect structural constraints:

### Identity Initialization
- Fixed parameters keep their constrained values
- Learnable parameters initialized to identity/zero/random as usual
- B2 initialized to zeros (per default behavior)

### ESN (Echo State Network) Initialization
- Random reservoir generation skips fixed parameters
- Only learnable parameters updated during reservoir trials
- Output matrix refitting respects constraints

### N4SID Initialization
- Loads A, B, C, D from MATLAB file
- Only updates learnable parameters with loaded values
- Fixed parameters keep their constrained values
- Useful when you have partial system identification data

Example:
```yaml
model:
  custom_params:
    structural_constraints:
      C: {fixed: true, value: [[1.0, 0.0]]}
  
  initialization:
    method: "n4sid"  # Will load A, B but C stays fixed
```

## Interaction with Training

### Gradient Computation
- **Fully fixed parameters**: `requires_grad=False`, no gradients computed
- **Partially learnable parameters**: Gradients computed, then masked to zero non-learnable elements

### Optimizer Updates
- Fixed parameters never updated (not in optimizer parameter list)
- Partially learnable parameters only update learnable elements

### LMI Regularization
- Structural constraints and LMI constraints work together
- LMI barrier function enforces stability/performance
- Structural constraints enforce physics/domain knowledge
- Both constraints must be satisfied simultaneously

Example:
```yaml
training:
  use_custom_regularization: true  # Enable LMI constraints
  regularization_weight: 1.0
  decay_regularization_weight: true

model:
  custom_params:
    structural_constraints:  # Plus structural constraints
      B: {learnable_rows: [1], fixed_value: 0.0}
```

## MLflow Logging

Structural constraints are automatically logged to MLflow:

```python
# Logged parameters:
"has_structural_constraints": True/False
"constrained_parameters": "B,B2,C,D,D12"  # Comma-separated list

# Per-parameter details:
"constraint_B": "learnable_rows_1"
"constraint_C": "fully_fixed"
"constraint_D": "fully_fixed"
```

This allows filtering and comparing experiments with different constraints.

## Validation and Error Handling

The implementation performs comprehensive validation:

### Parameter Name Validation
```python
# ❌ Error: Unknown parameter name
structural_constraints:
  INVALID: {fixed: true, value: 0.0}
# Raises: ValueError("Unknown parameter name: INVALID")
```

### Required Field Validation
```python
# ❌ Error: Missing required field
structural_constraints:
  B:
    learnable_rows: [1]
    # Missing: fixed_value
# Raises: ValueError("learnable_rows requires fixed_value")
```

### Conflict Detection
```python
# ❌ Error: Conflicting constraint types
structural_constraints:
  B:
    fixed: true
    value: 0.0
    learnable_rows: [1]  # Can't be both fixed and partial
# Raises: ValueError("Cannot specify both fixed and learnable_rows")
```

### Shape Validation
```python
# ❌ Error: Shape mismatch
structural_constraints:
  C:
    fixed: true
    value: [[1.0, 0.0, 0.0]]  # Wrong shape for ne=1, nx=2
# Raises: ValueError("Shape mismatch for C")
```

## Advanced Usage

### Combining Multiple Constraints

```yaml
structural_constraints:
  # State transition: partially learnable
  A:
    learnable_rows: [1]
    fixed_value: 0.0
  
  # Input: partially learnable
  B:
    learnable_rows: [1]
    fixed_value: 0.0
  
  # Output: fully fixed
  C:
    fixed: true
    value: [[1.0, 0.0]]
  
  # No feedthrough
  D:
    fixed: true
    value: 0.0
  
  D12:
    fixed: true
    value: 0.0
```

### Sparse Structures

```yaml
# Block diagonal output matrix
structural_constraints:
  C:
    learnable_cols: [0, 3, 6]  # Only diagonal blocks learnable
    fixed_value: 0.0
```

### Time-Varying Systems

For systems where structure changes over time:
1. Train separate models with different constraints for different regimes
2. Use the learned parameters as initialization for the next regime
3. Gradually relax constraints as you gather more data

## Testing

A comprehensive test suite is provided in `tests/test_structural_constraints.py`:

```bash
# Run all structural constraint tests
pytest tests/test_structural_constraints.py -v

# Run specific test class
pytest tests/test_structural_constraints.py::TestStructuralConstraintsDuffing -v

# Run with coverage
pytest tests/test_structural_constraints.py --cov=sysid.models.constrained_rnn
```

Test coverage includes:
- Backward compatibility (no constraints)
- Fully fixed parameters
- Partially learnable parameters (rows/cols)
- Gradient masking correctness
- All initialization methods
- Validation and error handling
- Constraint persistence through operations

## Implementation Details

### Architecture

The implementation adds 8 helper methods to `SimpleLure`:

1. **`_parse_structural_constraints()`** - Parse and validate constraints from config
2. **`_create_constrained_parameter()`** - Create parameters with constraints
3. **`_create_gradient_mask()`** - Create mask tensors for partial constraints
4. **`_register_gradient_masks()`** - Register gradient hooks
5. **`_is_parameter_fixed()`** - Check if parameter is fully fixed
6. **`_should_skip_initialization()`** - Check if init should skip parameter
7. **`_apply_partial_initialization()`** - Apply init data to learnable portions
8. **`_log_structural_constraints()`** - Log constraint info

### Gradient Masking

For partially learnable parameters, we use PyTorch's `register_hook` mechanism:

```python
def gradient_mask_hook(grad):
    return grad * mask  # Element-wise multiplication zeros non-learnable grads

parameter.register_hook(gradient_mask_hook)
```

This ensures that:
- Only learnable elements receive gradient updates
- Fixed elements remain exactly at their fixed values
- No numerical drift over training iterations

### Memory Efficiency

- Gradient masks are stored once and reused (not recreated each iteration)
- Masks are registered as buffers (moved to GPU if model is on GPU)
- Fixed parameters have `requires_grad=False` (no gradient computation)

## Troubleshooting

### Issue: Constraints not respected after training

**Cause:** Manual parameter updates after initialization

**Solution:** Always use the constraint-aware methods, never directly assign to `.data`

```python
# ❌ Don't do this
model.B.data = torch.randn(nx, nd)

# ✅ Use constraints during initialization
model._init_identity(train_inputs, train_states, train_outputs)
```

### Issue: Gradients still computed for fixed parameters

**Cause:** Fixed parameter incorrectly set to learnable

**Solution:** Check constraint specification in config

```yaml
# ❌ Wrong: missing "fixed: true"
C:
  value: [[1.0, 0.0]]

# ✅ Correct
C:
  fixed: true
  value: [[1.0, 0.0]]
```

### Issue: Validation error on parameter shape

**Cause:** Fixed value shape doesn't match parameter dimensions

**Solution:** Ensure fixed value has correct shape

```python
# For C with shape (ne, nx) where ne=1, nx=2:
C:
  fixed: true
  value: [[1.0, 0.0]]  # Shape (1, 2) ✅

# Not:
C:
  fixed: true
  value: [1.0, 0.0]  # Shape (2,) ❌
```

## References

1. **SimpleLure Model**: Constrained recurrent neural network with Lure structure
2. **LMI Theory**: Linear Matrix Inequalities for stability guarantees
3. **Interior Point Method**: Barrier function optimization for constraint satisfaction
4. **Duffing Oscillator**: Classic nonlinear dynamical system example

## Citation

If you use structural constraints in your research, please cite:

```bibtex
@software{structural_constraints_sysid,
  title = {Structural Constraints for System Identification with SimpleLure Models},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/genSecSysId}
}
```

## Contributing

To add new constraint types or improve the implementation:

1. Add constraint parsing logic to `_parse_structural_constraints()`
2. Update `_create_constrained_parameter()` to handle new type
3. Add validation in constraint parsing
4. Update documentation and tests
5. Add example config demonstrating the new constraint type

See `src/sysid/models/constrained_rnn.py` lines 131-520 for implementation details.
