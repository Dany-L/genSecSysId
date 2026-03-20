# Structural Constraints Usage Guide

This guide provides practical examples of using structural constraints with SimpleLure models.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Common Patterns](#common-patterns)
3. [Real-World Examples](#real-world-examples)
4. [Training Workflow](#training-workflow)
5. [Debugging Tips](#debugging-tips)

## Basic Usage

### Example 1: No Direct Feedthrough

Many physical systems have no direct feedthrough from input to output (i.e., D = 0).

```yaml
# config_no_feedthrough.yaml
model:
  model_type: "simple_lure"
  nd: 1
  ne: 1
  nx: 10
  nw: 10
  
  custom_params:
    structural_constraints:
      D:
        fixed: true
        value: 0.0
```

Train with:
```bash
python scripts/train.py --config configs/config_no_feedthrough.yaml
```

### Example 2: Observe Only First State

Output observes only the first state variable (e.g., position in a mechanical system).

```yaml
model:
  custom_params:
    structural_constraints:
      C:
        fixed: true
        value: [[1.0, 0.0, 0.0]]  # For nx=3, observe only first state
      
      D:
        fixed: true
        value: 0.0
```

### Example 3: Block Input Structure

Only certain states are directly affected by input (e.g., control acts only on acceleration).

```yaml
model:
  nx: 3  # position, velocity, acceleration
  nd: 1  # single control input
  
  custom_params:
    structural_constraints:
      B:
        learnable_rows: [2]  # Only acceleration (third state) affected by input
        fixed_value: 0.0
```

## Common Patterns

### Pattern 1: Mechanical Systems (Position-Velocity)

For systems with position-velocity state representation:
- x1 = position, x2 = velocity
- x1_dot = x2 (kinematic relationship)
- x2_dot = dynamics (affected by input and nonlinearity)

```yaml
model:
  nx: 2
  nd: 1
  ne: 1
  nw: 10
  
  custom_params:
    structural_constraints:
      # First equation: x1_dot = x2
      A:
        learnable_rows: [1]  # Only second equation is learnable
        fixed_value: 0.0
        # Note: You might want to manually set A[0,1] = 1 after initialization
      
      # Only x2_dot depends on input
      B:
        learnable_rows: [1]
        fixed_value: 0.0
      
      # Only x2_dot has nonlinear terms
      B2:
        learnable_rows: [1]
        fixed_value: 0.0
      
      # Observe position
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

### Pattern 2: MIMO Systems with Decoupled Outputs

For multi-input multi-output systems where outputs observe different states:

```yaml
model:
  nx: 4
  nd: 2
  ne: 2
  
  custom_params:
    structural_constraints:
      # Output 1 observes states 1-2, Output 2 observes states 3-4
      C:
        fixed: true
        value: [[1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]]
```

### Pattern 3: Known Linear Dynamics + Unknown Nonlinearity

When you know the linear part but want to learn nonlinear corrections:

```yaml
model:
  custom_params:
    structural_constraints:
      # Fix linear matrices to known values
      A:
        fixed: true
        value: [[0.9, 0.1], [0.0, 0.8]]  # Known from linearization
      
      B:
        fixed: true
        value: [[0.0], [1.0]]  # Known input gain
      
      # Learn nonlinear part (B2, C2, D21, D22 remain learnable)
```

## Real-World Examples

### Example 1: Duffing Oscillator

Complete configuration for a Duffing oscillator (see `configs/example_duffing_oscillator.yaml`):

```bash
# Generate synthetic Duffing data (you'll need to create this script)
python scripts/generate_duffing_data.py --samples 1000 --noise 0.01

# Train with structural constraints
python scripts/train.py --config configs/example_duffing_oscillator.yaml

# Evaluate
python scripts/evaluate.py --model models/duffing/<run_id>/model.pt \
                           --data data/duffing/test
```

Key features:
- State dimension: 2 (position, velocity)
- Nonlinearity dimension: 10 (approximate x^3 term with dead-zone activation)
- Constraints encode physics: only velocity equation has input and nonlinearity

### Example 2: Van der Pol Oscillator

Similar to Duffing, but with different nonlinearity:

```yaml
model:
  model_type: "simple_lure"
  nd: 1
  ne: 1
  nx: 2
  nw: 8
  activation: "tanh"  # Van der Pol has smooth nonlinearity
  
  custom_params:
    structural_constraints:
      # x1_dot = x2
      # x2_dot = mu*(1-x1^2)*x2 - x1 + u
      
      B:
        learnable_rows: [1]
        fixed_value: 0.0
      
      B2:
        learnable_rows: [1]
        fixed_value: 0.0
      
      C:
        fixed: true
        value: [[1.0, 0.0]]  # Observe position
      
      D:
        fixed: true
        value: 0.0
```

### Example 3: Inverted Pendulum on Cart

State: [cart_position, cart_velocity, pendulum_angle, pendulum_angular_velocity]

```yaml
model:
  nx: 4
  nd: 1  # Force on cart
  ne: 2  # Observe cart position and pendulum angle
  nw: 12
  
  custom_params:
    structural_constraints:
      # Input affects cart acceleration (state 1) and angular acceleration (state 3)
      B:
        learnable_rows: [1, 3]
        fixed_value: 0.0
      
      # Nonlinearity affects both accelerations
      B2:
        learnable_rows: [1, 3]
        fixed_value: 0.0
      
      # Observe cart position and pendulum angle
      C:
        fixed: true
        value: [[1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]]
      
      D:
        fixed: true
        value: 0.0
      
      D12:
        fixed: true
        value: 0.0
```

### Example 4: Electrical Circuit (RLC)

For an RLC circuit with state [inductor_current, capacitor_voltage]:

```yaml
model:
  nx: 2
  nd: 1  # Voltage source
  ne: 1  # Measure capacitor voltage
  nw: 5
  
  custom_params:
    structural_constraints:
      # Observe capacitor voltage (second state)
      C:
        fixed: true
        value: [[0.0, 1.0]]
      
      # No direct feedthrough in RLC
      D:
        fixed: true
        value: 0.0
      
      D12:
        fixed: true
        value: 0.0
```

## Training Workflow

### Step 1: Prepare Configuration

Create a YAML config with your structural constraints:

```yaml
# my_experiment.yaml
data:
  train_path: "data/my_system/train"
  input_col: ["u"]
  output_col: ["y"]
  # ... other data config

model:
  model_type: "simple_lure"
  nd: 1
  ne: 1
  nx: 2
  nw: 10
  
  custom_params:
    structural_constraints:
      # Your constraints here
      B:
        learnable_rows: [1]
        fixed_value: 0.0
  
  initialization:
    method: "esn"  # or "n4sid" or "identity"
    esn_n_restarts: 5

training:
  max_epochs: 500
  # ... other training config

mlflow:
  experiment_name: "my_system_structured"
```

### Step 2: Verify Constraints

Before full training, create a quick test to verify constraints:

```python
# test_constraints.py
import yaml
from sysid.config import load_config
from sysid.models import create_model

# Load config
with open("configs/my_experiment.yaml") as f:
    config_dict = yaml.safe_load(f)
config = load_config(config_dict)

# Create model
model = create_model(config.model)

# Check constraints
print("B requires_grad:", model.B.requires_grad)
print("B[0,:]:", model.B[0, :])  # Should be all zeros if row 0 is fixed
print("C requires_grad:", model.C.requires_grad)  # Should be False if fixed

# Check gradient masks
if hasattr(model, '_gradient_masks'):
    print("Gradient masks:", list(model._gradient_masks.keys()))
```

Run with:
```bash
python test_constraints.py
```

### Step 3: Train Model

```bash
# Train with your config
python scripts/train.py --config configs/my_experiment.yaml

# Monitor with MLflow UI
mlflow ui --port 5000
# Open browser to http://localhost:5000
```

### Step 4: Check MLflow Logs

In MLflow UI, verify that:
- `has_structural_constraints = True`
- `constrained_parameters` lists your constrained params
- Individual constraint specs are logged (e.g., `constraint_B = learnable_rows_1`)

### Step 5: Validate Results

After training, check that constraints held:

```python
# validate_constraints.py
import torch
from sysid.models import create_model

# Load trained model
model = torch.load("models/my_system/<run_id>/model.pt")

# Check that fixed values didn't change
print("B[0,:] should be zero:", model.B[0, :])
print("C should be [1, 0]:", model.C)

# Check that learnable parameters did change
# (compare to initialization values if you saved them)
```

## Debugging Tips

### Issue: Model Not Learning

**Symptoms:** Training loss doesn't decrease, or decreases very slowly.

**Possible causes:**
1. Too many constraints → insufficient model capacity
2. Constraints conflict with data
3. Learning rate too low for constrained parameters

**Solutions:**
```yaml
# Try relaxing some constraints
structural_constraints:
  B:
    learnable_rows: [1, 2]  # Allow more rows instead of just [1]
    
# Or increase model capacity
model:
  nw: 20  # Increase nonlinearity dimension

# Or increase learning rate
optimizer:
  learning_rate: 5e-3  # Higher than default 1e-3
```

### Issue: Constraints Not Respected

**Symptoms:** Fixed parameters change during training.

**Check:**
```python
# Before training
initial_C = model.C.clone()

# After training
assert torch.allclose(model.C, initial_C), "C changed!"
```

**Solutions:**
1. Verify config syntax is correct (check indentation in YAML)
2. Ensure `fixed: true` is specified for fully fixed params
3. Check that you're not manually updating `.data` after initialization

### Issue: Gradient Masking Not Working

**Symptoms:** Non-learnable elements of partially constrained params change slightly.

**Debug:**
```python
# Check gradient mask exists
print("Gradient masks:", model._gradient_masks.keys())

# Check mask shape
if 'B' in model._gradient_masks:
    print("B mask shape:", model._gradient_masks['B'].shape)
    print("B mask values:", model._gradient_masks['B'])
    # Should be 0 for non-learnable rows, 1 for learnable rows
```

**Solutions:**
1. Ensure gradient hooks are registered (happens in `__init__`)
2. Check that learnable_rows indices are correct (0-indexed)

### Issue: Initialization Fails

**Symptoms:** Error during model initialization with constraints.

**Common errors:**
- Shape mismatch: Fixed value shape doesn't match parameter
- Invalid indices: learnable_rows/cols out of bounds
- Conflicting specs: Both `fixed` and `learnable_rows` specified

**Debug:**
```python
# Check parameter shapes
print(f"B shape should be ({model.nx}, {model.nd})")
print(f"C shape should be ({model.ne}, {model.nx})")

# Check constraint specification
print("Constraints:", model.structural_constraints)
```

### Issue: Poor Performance vs. Unconstrained Model

**Symptoms:** Constrained model performs worse than unconstrained.

**Analysis:**
1. Is the constraint physically justified? Double-check your domain knowledge.
2. Is the constraint too restrictive? Try partial constraints instead of fully fixed.
3. Does the data actually follow the constraint? Validate on known equations.

**Experiment:**
```yaml
# Start with minimal constraints
structural_constraints:
  D: {fixed: true, value: 0.0}  # Just no feedthrough

# Gradually add more as you verify they help
```

## Performance Tips

### 1. Start Simple

Begin with only the most certain constraints (e.g., no feedthrough, known output matrices):

```yaml
# Iteration 1: Minimal constraints
structural_constraints:
  D: {fixed: true, value: 0.0}
  D12: {fixed: true, value: 0.0}
```

Then add more as you validate:

```yaml
# Iteration 2: Add output constraints
structural_constraints:
  D: {fixed: true, value: 0.0}
  D12: {fixed: true, value: 0.0}
  C: {fixed: true, value: [[1.0, 0.0]]}
```

### 2. Use Constraints with Regularization

Combine structural constraints with LMI regularization for best results:

```yaml
model:
  custom_params:
    structural_constraints:
      B: {learnable_rows: [1], fixed_value: 0.0}

training:
  use_custom_regularization: true
  regularization_weight: 1.0
  decay_regularization_weight: true
```

This gives you:
- **Structural constraints** → Encode physics/domain knowledge
- **LMI constraints** → Ensure stability and performance bounds

### 3. Tune Initialization Method

Different initialization methods work better with different constraints:

- **ESN**: Good for sparse/structured matrices (many zero elements)
- **Identity**: Good when you have strong priors on structure
- **N4SID**: Good when you have partial system identification

Experiment:
```bash
# Try ESN
python scripts/train.py --config config_esn.yaml

# Try Identity
python scripts/train.py --config config_identity.yaml

# Compare in MLflow
```

### 4. Monitor Gradient Norms

Check that learnable parameters are receiving gradients:

```python
# In training loop
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name} grad norm: {grad_norm}")
```

If a learnable parameter has very small gradients, you might need to:
- Increase learning rate for that parameter
- Check if constraint is too restrictive
- Verify data has enough information to learn that parameter

## Next Steps

- See `docs/STRUCTURAL_CONSTRAINTS.md` for complete API reference
- Run tests with `pytest tests/test_structural_constraints.py`
- Try the Duffing oscillator example in `configs/example_duffing_oscillator.yaml`
- Adapt examples to your specific system

For questions or issues, please open a GitHub issue.
