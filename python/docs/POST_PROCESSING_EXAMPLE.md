# Post-Processing Example

This example demonstrates how to use the `post_process()` method of SimpleLure models.

## Quick Example

```python
import mlflow
from sysid.models import SimpleLure

# Load a trained SimpleLure model from MLflow
run_id = "abc123def456"  # Replace with your run ID
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# Post-process: optimize P and L while keeping A, B, C, D fixed
result = model.post_process(optimize_s=True, eps=1e-3)

# Check if successful
if result['success']:
    print("✓ Post-processing successful!")
    
    # Print summary
    summary = result['summary']
    print(f"\nOriginal s:    {summary['original']['s']:.6f}")
    print(f"Optimized s:   {summary['optimized']['s']:.6f}")
    print(f"Improvement:   {-summary['changes']['s_relative']:.2f}%")
    print(f"Max eig(F):    {summary['optimized']['max_eig_F']:.6e}")
    print(f"Constraints:   {'✓ Satisfied' if result['constraints_satisfied'] else '✗ Violated'}")
    
    # Model parameters are automatically updated
    print(f"\nModel updated with optimized P and L")
    
    # You can now save the updated model or continue using it
    import torch
    torch.save(model.state_dict(), 'post_processed_model.pt')
    
else:
    print(f"✗ Post-processing failed: {result.get('error', 'unknown')}")
```

## Using in a Script

If you want MLflow tracking and automatic saving:

```bash
python scripts/post_process.py --run-id abc123def456 --optimize-s
```

This will:
- Load the model from the specified run
- Call `model.post_process(optimize_s=True)`
- Create a new MLflow run with the results
- Save the post-processed model and matrices
- Log all metrics to MLflow

## Advanced: Custom Workflow

You can integrate post-processing into your own workflow:

```python
import mlflow
import numpy as np
from sysid.models import SimpleLure

# Load model
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# Check if constraints are satisfied
constraints_ok = model.check_constraints()
print(f"Initial constraints satisfied: {constraints_ok}")

if not constraints_ok:
    # Post-process to satisfy constraints
    result = model.post_process(optimize_s=False)  # Feasibility only
    
    if result['success']:
        print("✓ Constraints now satisfied!")
    else:
        print("✗ Could not satisfy constraints")
        # Try with different eps
        result = model.post_process(optimize_s=False, eps=1e-2)

# If constraints are satisfied, optimize for tighter certificate
if model.check_constraints():
    result = model.post_process(optimize_s=True)
    
    # Access optimized matrices
    P_new = result['P_opt']
    L_new = result['L_opt']
    s_new = result['s_opt']
    
    # Export to MATLAB if needed
    np.savez('optimized_certificate.npz',
             P=P_new, L=L_new, s=s_new,
             max_eig_F=result['max_eig_F'])
```

## Understanding the Results

The `post_process()` method returns a dictionary with:

```python
{
    'success': True,                    # Whether SDP solved successfully
    'P_opt': array([[...]]),           # Optimized Lyapunov matrix
    'L_opt': array([[...]]),           # Optimized coupling matrix
    'm_opt': array([[...]]),           # Optimized multipliers
    's_opt': 1.234,                    # Optimized sector bound
    'S_hat_opt': 0.656,                # S_hat = 1/s²
    'max_eig_F': -0.001234,            # Max eigenvalue of F (should be < 0)
    'constraints_satisfied': True,      # Verification
    'summary': {
        'original': {
            's': 1.456,
            'max_eig_P': 2.345,
            'min_eig_P': 0.123,
            'cond_P': 19.08,
        },
        'optimized': {
            's': 1.234,
            'max_eig_P': 2.012,
            'min_eig_P': 0.156,
            'cond_P': 12.89,
            'max_eig_F': -0.001234,
        },
        'changes': {
            'P_frobenius_norm': 0.456,
            'L_frobenius_norm': 0.234,
            's_absolute': 0.222,
            's_relative': -15.24,      # Negative = improvement
        }
    }
}
```

**Key metrics to check:**
- `success`: Must be True
- `constraints_satisfied`: Should be True after post-processing
- `max_eig_F`: Should be negative (more negative = less conservative)
- `s_relative`: Negative values indicate tighter certificate (better)
- `cond_P`: Lower is better for numerical stability

## Comparison with MATLAB

The Python `post_process()` method solves the same SDP as your MATLAB code:

**MATLAB:**
```matlab
% Your existing code
L = sdpvar(nz,nx);
P = sdpvar(nx,nx);
m = sdpvar(nz,1);
S_hat = sdpvar(1,1);
% ... constraints ...
sol = optimize(lmis, S_hat, sdpsettings('solver','mosek','verbose', 0))
```

**Python:**
```python
# Equivalent
result = model.post_process(optimize_s=True, eps=1e-3)
```

Both use:
- Same decision variables (P, L, m, S_hat)
- Same LMI constraints
- Same solver (MOSEK)
- Same objective (minimize S_hat)

You can verify consistency by comparing results from both implementations.
