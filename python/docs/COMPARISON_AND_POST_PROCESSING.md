# Model Comparison and Post-Processing

This guide covers the new comparison and post-processing capabilities for analyzing and refining trained models.

## Table of Contents
1. [Model Comparison](#model-comparison)
2. [Post-Processing](#post-processing)
3. [Use Cases](#use-cases)
4. [Examples](#examples)

---

## Model Comparison

The `compare.py` script allows you to compare multiple trained models from MLflow runs side-by-side.

### Features

- **Evaluation Metrics Table**: Compare test MSE, RMSE, MAE, R², NRMSE, and FIT scores
- **Training Curves**: Visualize training/validation losses for each run
- **Validation Comparison**: Combined plot showing all validation curves
- **Summary Table**: Model parameters, hyperparameters, and final metrics
- **Constraint Analysis**: For SimpleLure models, check constraint satisfaction

### Basic Usage

```bash
python scripts/compare.py \
    --run-ids abc123def456 xyz789ghi012 \
    --test-data data/prepared/test \
    --output-dir comparisons/lstm_vs_rnn
```

### Arguments

- `--run-ids`: Space-separated list of MLflow run IDs (required)
- `--test-data`: Path to test data (folder or CSV file)
- `--output-dir`: Directory to save comparison results (default: `comparisons`)
- `--mlflow-tracking-uri`: MLflow tracking URI (optional)

### Output Files

The comparison generates the following files in the output directory:

```
comparisons/my_comparison/
├── summary.csv                    # Model parameters and final metrics
├── evaluation_metrics.csv         # Detailed test metrics for each run
├── training_curves.png           # Individual training curves
└── validation_comparison.png     # Combined validation curve comparison
```

### Example Output

**Summary Table:**
```
Run ID   | Name          | Model Type | Hidden Size | Learning Rate | Final Val Loss | MSE      | RMSE     | FIT
---------|---------------|------------|-------------|---------------|----------------|----------|----------|--------
abc123   | rnn_baseline  | SimpleRNN  | 32          | 0.001         | 0.000123       | 0.000145 | 0.012042 | 95.23%
def456   | lstm_model    | LSTM       | 64          | 0.001         | 0.000098       | 0.000112 | 0.010583 | 96.45%
```

**Console Output:**
The script prints formatted tables directly to the console for quick inspection.

---

## Post-Processing

The `post_process.py` script allows you to refine a trained SimpleLure model by solving a semidefinite program (SDP) to find the optimal Lyapunov certificate (P and L matrices) while keeping the learned dynamics (A, B, C, D) fixed.

### Why Post-Process?

After training, you may want to:
1. **Improve constraint satisfaction** by solving an SDP with frozen dynamics
2. **Optimize the Lyapunov certificate** for better stability guarantees
3. **Minimize s** (sector bound) to get a tighter certificate
4. **Verify learned dynamics** satisfy theoretical constraints

### How It Works

The post-processor solves the following SDP (matching your MATLAB implementation):

**Decision variables:**
- P (Lyapunov matrix)
- L (coupling matrix for locality)
- m (multipliers for sector condition)
- S_hat (optional, for minimizing s)

**Constraints:**
1. Main LMI (stability):
   ```
   F = [-α²P    0      P*C2'+L'  P*A'  ]
       [0       -I     D21'      B'    ] ≤ -εI
       [C2*P+L  D21    -2M       M*B2' ]
       [A*P     B      B2*M      -P    ]
   ```

2. Locality constraints (for each row i of L):
   ```
   [S_hat  li ] ≥ εI
   [li'    P  ]
   ```

3. Positive definiteness: P ≥ εI
4. Multiplier constraints: m_i ≥ ε

**Objective:**
- If `--optimize-s`: minimize S_hat (which minimizes s = 1/√S_hat)
- Otherwise: feasibility problem (find any valid P, L)

### Basic Usage

**Option 1: Using the script (with MLflow tracking)**
```bash
# Post-process with fixed s
python scripts/post_process.py --run-id abc123def456

# Post-process and optimize for minimum s
python scripts/post_process.py --run-id abc123def456 --optimize-s
```

**Option 2: Direct method call (in your code)**
```python
import mlflow
from sysid.models import SimpleLure

# Load model
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# Post-process
result = model.post_process(optimize_s=True, eps=1e-3)

if result['success']:
    print(f"s optimized: {result['s_opt']:.6f}")
    print(f"Constraints satisfied: {result['constraints_satisfied']}")
```

### Arguments

- `--run-id`: MLflow run ID of the trained model (required)
- `--optimize-s`: Optimize for minimum s (default: False, s kept fixed)
- `--eps`: Small positive constant for strict inequalities (default: 1e-3)
- `--output-dir`: Directory to save post-processed results (default: `post_processed`)
- `--mlflow-tracking-uri`: MLflow tracking URI (optional)

### What Gets Frozen/Optimized

**Frozen (from trained model):**
- System matrices: A, B, B2, C, D, D12, C2, D21, D22
- Stability parameter: α (alpha)
- Sector bound: s (unless `--optimize-s` is set)

**Optimized (via SDP):**
- Lyapunov matrix: P
- Coupling matrix: L
- Multipliers: m (for M = diag(m))

### Output Files

The post-processor generates:

```
post_processed/
├── post_processed_<run_id>.npz    # Matrices and results
└── model_<run_id>.pt              # Updated model state dict
```

The `.npz` file contains:
- `P_original`, `P_opt`: Original and optimized Lyapunov matrices
- `L_original`, `L_opt`: Original and optimized coupling matrices
- `m_opt`: Optimized multipliers
- `s_original`, `s_opt`: Original and optimized sector bounds
- `S_hat_opt`: Optimized S_hat value
- `max_eig_F`: Maximum eigenvalue of F (should be < 0)
- Frozen system matrices: A, B, B2, C2, D21, alpha

### MLflow Integration

Post-processing automatically:
- Creates a new MLflow run linked to the original experiment
- Logs SDP solution metrics (s values, eigenvalues, changes)
- Saves the post-processed model and results to MLflow
- Records constraint satisfaction status

### Interpreting Results

The script outputs:
```
POST-PROCESSING SUMMARY
================================================================================
Original s: 1.234567
Optimized s: 1.123456
Change: -9.00%
Max eigenvalue of F: -1.234e-03
Constraints satisfied: True
P change (Frobenius): 0.123456
L change (Frobenius): 0.234567
================================================================================
```

**Key metrics:**
- **s change**: Negative means tighter certificate (better)
- **Max eig(F)**: Should be negative; closer to 0 means tighter bound
- **Constraints satisfied**: Verifies the solution is valid
- **P/L changes**: Shows how much the certificate changed

### Using the Method Directly

The `post_process()` method can be called directly on any SimpleLure model instance:

```python
import mlflow
from sysid.models import SimpleLure

# Load trained model
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# Post-process (optimize for minimum s)
result = model.post_process(optimize_s=True, eps=1e-3)

# Check results
if result['success']:
    print(f"✓ Post-processing successful!")
    print(f"  Original s:    {result['summary']['original']['s']:.6f}")
    print(f"  Optimized s:   {result['s_opt']:.6f}")
    print(f"  Change:        {result['summary']['changes']['s_relative']:.2f}%")
    print(f"  Constraints:   {result['constraints_satisfied']}")
    
    # Model parameters are already updated
    # Access optimized matrices
    P_new = model.P
    L_new = model.L
else:
    print(f"✗ Post-processing failed: {result.get('error', 'unknown')}")
```

**Return value** is a dictionary with:
- `success`: bool
- `P_opt`, `L_opt`, `m_opt`: Optimized matrices
- `s_opt`, `S_hat_opt`: Optimized sector bound
- `max_eig_F`: Maximum eigenvalue of F (verification)
- `constraints_satisfied`: bool
- `summary`: Dictionary with detailed comparison metrics

---

## Use Cases

### 1. Comparing Different Architectures

Compare RNN, LSTM, and GRU models:

```bash
# Train models
python scripts/train.py --config configs/rnn_baseline.yaml
python scripts/train.py --config configs/lstm_baseline.yaml

# Get run IDs from MLflow UI or logs
python scripts/compare.py \
    --run-ids <rnn_run_id> <lstm_run_id> \
    --test-data data/prepared/test
```

### 2. Hyperparameter Comparison

Compare models with different learning rates:

```bash
# Train with different LRs
python scripts/train.py --config configs/rnn_baseline.yaml --learning-rate 0.001
python scripts/train.py --config configs/rnn_baseline.yaml --learning-rate 0.0001

# Compare
python scripts/compare.py \
    --run-ids <lr_001_run_id> <lr_0001_run_id> \
    --output-dir comparisons/learning_rate_study
```

### 3. Constrained vs Unconstrained

Compare SimpleLure models with different regularization:

```bash
python scripts/compare.py \
    --run-ids <interior_point_run_id> <dual_run_id> <no_reg_run_id> \
    --output-dir comparisons/regularization_comparison
```

### 4. Improving Constraint Satisfaction

Post-process a model that violates constraints:

```bash
# Train model
python scripts/train.py --config configs/constrained_rnn_lmi.yaml

# Post-process to satisfy constraints (feasibility)
python scripts/post_process.py --run-id <run_id>

# Or optimize for tighter certificate
python scripts/post_process.py --run-id <run_id> --optimize-s
```

### 5. Integrating with MATLAB

Since the post-processor uses the same SDP formulation as your MATLAB code, you can:

**Option 1: Use Python post-processing, export to MATLAB**
```bash
# Post-process in Python
python scripts/post_process.py --run-id <run_id> --optimize-s

# Load results in MATLAB
data = load('post_processed/post_processed_<run_id>.npz');
P = data.P_opt;
L = data.L_opt;
% Continue with your MATLAB analysis...
```

**Option 2: Export model, post-process in MATLAB**
```python
import mlflow
import numpy as np

# Load model
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# Extract matrices for MATLAB
matrices = {
    'A': model.A.detach().cpu().numpy(),
    'B': model.B.detach().cpu().numpy(),
    'B2': model.B2.detach().cpu().numpy(),
    'C2': model.C2.detach().cpu().numpy(),
    'D21': model.D21.detach().cpu().numpy(),
    'alpha': model.alpha.item(),
    's': model.s.item(),
    'P': model.P.detach().cpu().numpy(),
    'L': model.L.detach().cpu().numpy() if model.learn_L else None,
}

# Save for MATLAB
np.savez('model_for_matlab.npz', **matrices)
```

Then in MATLAB, use your existing SDP code with the frozen A, B, C2, D21, alpha values.

---

## Examples

### Example 1: Quick Comparison

```bash
python scripts/compare.py \
    --run-ids 1a2b3c4d 5e6f7g8h \
    --test-data data/prepared/test
```

### Example 2: Comprehensive Analysis

```bash
# Train multiple models
python scripts/train.py --config configs/rnn_baseline.yaml
python scripts/train.py --config configs/lstm_baseline.yaml
python scripts/train.py --config configs/constrained_rnn_lmi.yaml

# Compare (replace with actual run IDs)
python scripts/compare.py \
    --run-ids <rnn_id> <lstm_id> <constrained_id> \
    --test-data data/prepared/test \
    --output-dir comparisons/full_comparison

# Post-process the constrained model
python scripts/post_process.py --run-id <constrained_id> --optimize-s
```

### Example 3: Batch Comparison

If you have many runs, you can use MLflow search:

```python
import mlflow

# Search for runs
experiment = mlflow.get_experiment_by_name("my_experiment")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.model_type = 'SimpleLure'"
)

# Get run IDs
run_ids = runs['run_id'].tolist()

# Compare (from command line)
# python scripts/compare.py --run-ids {' '.join(run_ids)}
```

---

## Tips

1. **Large Comparisons**: When comparing many runs (>5), the plots can become crowded. Consider comparing in smaller groups.

2. **MLflow Tracking**: Make sure your MLflow tracking URI is set correctly:
   ```bash
   export MLFLOW_TRACKING_URI=file:./mlruns
   ```

3. **Test Data**: Use the same test data for all comparisons to ensure fair evaluation.

4. **SDP Solver**: Post-processing requires MOSEK solver. Make sure it's installed and licensed:
   ```bash
   pip install mosek
   # Add your MOSEK license file
   ```

5. **Optimize s**: Use `--optimize-s` when you want the tightest possible certificate. Without it, the script finds a feasible solution faster.

6. **Numerical Issues**: If the SDP fails, try adjusting `--eps` (smaller values like 1e-4 for tighter constraints, larger like 1e-2 for easier feasibility).

7. **Model Compatibility**: Post-processing only works with SimpleLure models. Standard RNN/LSTM models don't have the Lure structure.

---

## Integration with MATLAB

The post-processing script uses the same SDP formulation as your MATLAB code, making integration seamless.

### Workflow 1: Python Post-Processing → MATLAB Analysis

1. **Post-process in Python:**
   ```bash
   python scripts/post_process.py --run-id <run_id> --optimize-s
   ```

2. **Load results in MATLAB:**
   ```matlab
   % Load post-processed results
   data = load('post_processed/post_processed_<run_id>.npz');
   
   % Extract matrices
   P = data.P_opt;
   L = data.L_opt;
   A = data.A;
   B = data.B;
   % ... etc
   
   % Verify in MATLAB
   Pinv = inv(P);
   X = Pinv;
   H = L * Pinv;
   
   % Continue with your analysis
   ```

### Workflow 2: Python Training → MATLAB Post-Processing

1. **Train in Python, export matrices:**
   ```python
   import mlflow
   import numpy as np
   
   model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
   
   matrices = {
       'A': model.A.detach().cpu().numpy(),
       'B': model.B.detach().cpu().numpy(),
       'B2': model.B2.detach().cpu().numpy(),
       'C2': model.C2.detach().cpu().numpy(),
       'D21': model.D21.detach().cpu().numpy(),
       'alpha': model.alpha.item(),
       's': model.s.item(),
       'nx': model.nx,
       'nd': model.nd,
       'nw': model.nw,
       'nz': model.nz,
   }
   
   np.savez('model_for_matlab.npz', **matrices)
   ```

2. **Post-process in MATLAB using your existing code:**
   ```matlab
   % Load exported model
   data = load('model_for_matlab.npz');
   A = data.A;
   B = data.B;
   B2 = data.B2;
   C2 = data.C2;
   D21 = data.D21;
   alpha = data.alpha;
   
   % Run your SDP code (as shown in your example)
   % ... your MATLAB SDP code here ...
   ```

### Consistency

Both implementations solve the same problem:
- Same decision variables (P, L, m, S_hat)
- Same LMI constraints
- Same solver (MOSEK)
- Same objective (minimize S_hat or feasibility)

You can verify results match by comparing:
- Eigenvalues of F
- Values of s
- Constraint satisfaction

---

For more information, see:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Training Guide](README.md)
- [Evaluation Metrics](EVALUATION_METRICS.md)
