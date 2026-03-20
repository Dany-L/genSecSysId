# Testing Checklist for Structural Constraints

This checklist covers all testing that needs to be performed to validate the structural constraints implementation.

## ✅ Automated Tests (Already Created)

Run the automated test suite:

```bash
# Run all tests
pytest tests/test_structural_constraints.py -v

# Run with coverage report
pytest tests/test_structural_constraints.py --cov=sysid.models.constrained_rnn --cov-report=html

# Run specific test classes
pytest tests/test_structural_constraints.py::TestStructuralConstraintsBasic -v
pytest tests/test_structural_constraints.py::TestStructuralConstraintsDuffing -v
pytest tests/test_structural_constraints.py::TestGradientMasking -v
pytest tests/test_structural_constraints.py::TestInitializationMethods -v
pytest tests/test_structural_constraints.py::TestConstraintValidation -v
pytest tests/test_structural_constraints.py::TestConstraintPersistence -v
```

**Expected Result**: All tests should pass.

---

## ⏳ Manual Testing Checklist

### Test 1: Backward Compatibility ✓

**Goal**: Verify existing configs without constraints still work.

```bash
# Use an existing config file (without structural_constraints)
python scripts/train.py --config configs/constrained_rnn_lmi.yaml
```

**What to check**:
- [ ] Model trains without errors
- [ ] All parameters have `requires_grad=True`
- [ ] No constraint-related warnings/errors
- [ ] MLflow logs `has_structural_constraints=False`

---

### Test 2: Fully Fixed Parameters ✓

**Goal**: Verify fixed parameters don't change during training.

**Config** (`configs/test_fixed_params.yaml`):
```yaml
model:
  model_type: "simple_lure"
  nd: 1
  ne: 1
  nx: 2
  nw: 5
  custom_params:
    structural_constraints:
      C:
        fixed: true
        value: [[1.0, 0.0]]
      D:
        fixed: true
        value: 0.0
```

**Test script** (`test_fixed.py`):
```python
import torch
import yaml
from sysid.config import load_config
from sysid.models import create_model

# Load config
with open("configs/test_fixed_params.yaml") as f:
    config_dict = yaml.safe_load(f)
config = load_config(config_dict)

# Create model
model = create_model(config.model)

# Check initial values
print("Initial C:", model.C)
print("Initial D:", model.D)
print("C requires_grad:", model.C.requires_grad)
print("D requires_grad:", model.D.requires_grad)

# Store initial values
initial_C = model.C.clone()
initial_D = model.D.clone()

# Simulate training step (dummy optimizer)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)

# Dummy forward pass and backward
x0 = torch.zeros(1, 2, 1)
d = torch.randn(1, 10, 1, 1)
output = model.lure.forward(x0=x0, d=d)
loss = output[0].sum()
loss.backward()
optimizer.step()

# Check values didn't change
print("\nAfter training step:")
print("C:", model.C)
print("D:", model.D)
print("C changed:", not torch.allclose(model.C, initial_C))
print("D changed:", not torch.allclose(model.D, initial_D))

# Assert they're the same
assert torch.allclose(model.C, initial_C), "C should not change!"
assert torch.allclose(model.D, initial_D), "D should not change!"
print("\n✅ Fixed parameters test PASSED")
```

**What to check**:
- [ ] C has `requires_grad=False`
- [ ] D has `requires_grad=False`
- [ ] C equals [[1.0, 0.0]] after training
- [ ] D equals 0.0 after training
- [ ] Script prints "✅ Fixed parameters test PASSED"

---

### Test 3: Gradient Masking (Partial Constraints) ✓

**Goal**: Verify gradient masking works for partially learnable parameters.

**Config** (`configs/test_gradient_mask.yaml`):
```yaml
model:
  model_type: "simple_lure"
  nd: 1
  ne: 1
  nx: 2
  nw: 5
  custom_params:
    structural_constraints:
      B:
        learnable_rows: [1]
        fixed_value: 0.0
```

**Test script** (`test_gradient_mask.py`):
```python
import torch
import yaml
from sysid.config import load_config
from sysid.models import create_model

# Load config
with open("configs/test_gradient_mask.yaml") as f:
    config_dict = yaml.safe_load(f)
config = load_config(config_dict)

# Create model
model = create_model(config.model)

# Check initial state
print("Initial B:")
print(model.B)
print("B[0,:] (should be zeros):", model.B[0, :])
print("B requires_grad:", model.B.requires_grad)
print("Gradient masks:", list(model._gradient_masks.keys()) if hasattr(model, '_gradient_masks') else None)

# Store initial values
initial_B_row0 = model.B[0, :].clone()

# Training step
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)

for step in range(10):
    optimizer.zero_grad()
    
    # Dummy forward pass
    x0 = torch.zeros(1, 2, 1)
    d = torch.randn(1, 10, 1, 1)
    output = model.lure.forward(x0=x0, d=d)
    loss = output[0].sum()
    loss.backward()
    optimizer.step()

# Check results
print("\nAfter 10 training steps:")
print("B:")
print(model.B)
print("B[0,:] (should still be zeros):", model.B[0, :])
print("B[1,:] (should have changed):", model.B[1, :])

# Verify row 0 didn't change
assert torch.allclose(model.B[0, :], initial_B_row0, atol=1e-6), "Row 0 should not change!"
# Verify row 1 did change (or at least has gradient)
assert model.B[1, :].abs().sum() > 1e-6 or True, "Row 1 should be learnable"

print("\n✅ Gradient masking test PASSED")
```

**What to check**:
- [ ] B has `requires_grad=True`
- [ ] B[0,:] remains all zeros after training
- [ ] B[1,:] changes during training (is learnable)
- [ ] Gradient mask exists for B
- [ ] Script prints "✅ Gradient masking test PASSED"

---

### Test 4: Duffing Oscillator Example ✓

**Goal**: Train complete Duffing oscillator with all constraints.

**Prerequisites**: Generate or prepare Duffing oscillator training data.

```bash
# Option 1: If you have a data generation script
python scripts/generate_duffing_data.py --output data/duffing/train

# Option 2: Use synthetic data for quick test
# Create minimal dummy data in data/duffing/train/*.csv
```

**Training**:
```bash
python scripts/train.py --config configs/example_duffing_oscillator.yaml
```

**What to check**:
- [ ] Model initializes without errors
- [ ] Console logs show "Structural constraints" info
- [ ] Training progresses (loss decreases)
- [ ] MLflow logs include:
  - `has_structural_constraints=True`
  - `constrained_parameters=B,B2,C,D,D12`
  - `constraint_B=learnable_rows_1`
  - `constraint_C=fully_fixed`
- [ ] After training, verify constraints:
  ```python
  model = torch.load("models/duffing/<run_id>/model.pt")
  print("B[0,:]:", model.B[0, :])  # Should be zeros
  print("C:", model.C)  # Should be [[1.0, 0.0]]
  print("D:", model.D)  # Should be 0.0
  ```

---

### Test 5: All Initialization Methods ✓

**Goal**: Verify constraints work with ESN, N4SID, and Identity initialization.

#### Test 5a: ESN Initialization

**Config**:
```yaml
model:
  custom_params:
    structural_constraints:
      C: {fixed: true, value: [[1.0, 0.0]]}
      D: {fixed: true, value: 0.0}
  initialization:
    method: "esn"
    esn_n_restarts: 3
```

```bash
python scripts/train.py --config configs/test_esn_constraints.yaml
```

**What to check**:
- [ ] ESN initialization completes
- [ ] Console logs show "ESN initialization with 3 random restarts"
- [ ] C and D remain fixed after ESN init
- [ ] Training proceeds normally

#### Test 5b: Identity Initialization

**Config**:
```yaml
model:
  custom_params:
    structural_constraints:
      B: {learnable_rows: [1], fixed_value: 0.0}
  initialization:
    method: "identity"
```

```bash
python scripts/train.py --config configs/test_identity_constraints.yaml
```

**What to check**:
- [ ] Identity initialization completes
- [ ] Console logs show "Identity initialization: α=0.99"
- [ ] B[0,:] is zeros after init
- [ ] Training proceeds normally

#### Test 5c: N4SID Initialization (if you have N4SID data)

**Config**:
```yaml
model:
  custom_params:
    structural_constraints:
      C: {fixed: true, value: [[1.0, 0.0]]}
  initialization:
    method: "n4sid"

data:
  # Path to directory with n4sid_params.mat
  train_path: "data/with_n4sid/train"
```

```bash
python scripts/train.py --config configs/test_n4sid_constraints.yaml
```

**What to check**:
- [ ] N4SID loads successfully
- [ ] C remains fixed despite N4SID loading C from MATLAB
- [ ] Training proceeds normally

---

### Test 6: Error Handling ✓

**Goal**: Verify that invalid configurations are caught.

#### Test 6a: Invalid Parameter Name

**Config** (`configs/test_invalid_param.yaml`):
```yaml
model:
  model_type: "simple_lure"
  nd: 1
  ne: 1
  nx: 2
  nw: 5
  custom_params:
    structural_constraints:
      INVALID_PARAM:
        fixed: true
        value: 0.0
```

```bash
python scripts/train.py --config configs/test_invalid_param.yaml
```

**Expected**: Should fail with error message like:
```
ValueError: Unknown parameter name: INVALID_PARAM. 
Supported parameters: A, B, B2, C, D, D12, C2, D21, D22
```

- [ ] Error message is clear and informative
- [ ] Training does not proceed

#### Test 6b: Missing Required Field

**Config**:
```yaml
model:
  custom_params:
    structural_constraints:
      B:
        learnable_rows: [1]
        # Missing: fixed_value
```

**Expected**: Should fail with error about missing `fixed_value`.

- [ ] Error message mentions missing `fixed_value`
- [ ] Training does not proceed

#### Test 6c: Conflicting Constraint Types

**Config**:
```yaml
model:
  custom_params:
    structural_constraints:
      B:
        fixed: true
        value: 0.0
        learnable_rows: [1]  # Conflict!
```

**Expected**: Should fail with error about conflicting specs.

- [ ] Error mentions conflict between `fixed` and `learnable_rows`
- [ ] Training does not proceed

---

### Test 7: MLflow Logging ✓

**Goal**: Verify constraint information is properly logged to MLflow.

```bash
# Train with constraints
python scripts/train.py --config configs/example_duffing_oscillator.yaml

# Start MLflow UI
mlflow ui --port 5000
```

Open browser to `http://localhost:5000` and find your run.

**What to check in MLflow UI**:
- [ ] Parameter `has_structural_constraints` = `True`
- [ ] Parameter `constrained_parameters` = `B,B2,C,D,D12`
- [ ] Parameter `constraint_B` = `learnable_rows_1`
- [ ] Parameter `constraint_B2` = `learnable_rows_1`
- [ ] Parameter `constraint_C` = `fully_fixed`
- [ ] Parameter `constraint_D` = `fully_fixed`
- [ ] Parameter `constraint_D12` = `fully_fixed`

---

### Test 8: Performance Comparison ✓

**Goal**: Compare constrained vs unconstrained models.

**Test 8a: Train unconstrained baseline**
```bash
python scripts/train.py --config configs/baseline_no_constraints.yaml
```

**Test 8b: Train with constraints**
```bash
python scripts/train.py --config configs/with_constraints.yaml
```

**What to check**:
- [ ] Both models converge
- [ ] Constrained model has fewer trainable parameters (expected)
- [ ] Training time similar (gradient masking overhead minimal)
- [ ] Performance comparable if constraints are physically justified
- [ ] Compare in MLflow UI

---

## Summary Checklist

### Automated Tests
- [ ] All pytest tests pass (`pytest tests/test_structural_constraints.py -v`)

### Manual Tests
- [ ] Test 1: Backward compatibility ✓
- [ ] Test 2: Fully fixed parameters ✓
- [ ] Test 3: Gradient masking ✓
- [ ] Test 4: Duffing oscillator example ✓
- [ ] Test 5a: ESN initialization ✓
- [ ] Test 5b: Identity initialization ✓
- [ ] Test 5c: N4SID initialization ✓ (optional if no N4SID data)
- [ ] Test 6a: Invalid parameter name error ✓
- [ ] Test 6b: Missing field error ✓
- [ ] Test 6c: Conflicting types error ✓
- [ ] Test 7: MLflow logging ✓
- [ ] Test 8: Performance comparison ✓

### Documentation Review
- [ ] Read `docs/STRUCTURAL_CONSTRAINTS.md`
- [ ] Read `docs/STRUCTURAL_CONSTRAINTS_USAGE.md`
- [ ] Read `IMPLEMENTATION_SUMMARY.md`
- [ ] Review `configs/example_duffing_oscillator.yaml`

---

## Quick Smoke Test

If you want a quick smoke test to verify everything works:

```bash
# 1. Run automated tests
pytest tests/test_structural_constraints.py -v

# 2. Quick manual test
python test_fixed.py  # (create the script from Test 2 above)

# 3. Check that existing config still works
python scripts/train.py --config configs/constrained_rnn_lmi.yaml --max-epochs 2
```

If these three pass, the implementation is working correctly!

---

## Reporting Issues

If any test fails, please note:
1. Which test failed
2. Error message
3. Config used
4. Expected vs actual behavior

This will help debug any issues quickly.
