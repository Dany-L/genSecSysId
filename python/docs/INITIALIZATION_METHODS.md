# Model Initialization Methods

The SimpleLure model supports three different initialization strategies, configurable via the YAML config file:

## 1. ESN (Echo State Network) — Default

Random reservoir approach: samples multiple random system matrices (A, B, C2), simulates with each, and selects the one with lowest training error.

**Configuration:**
```yaml
model:
  model_type: crnn
  nw: 64
  nx: 64
  initialization:
    method: esn
    esn_n_restarts: 5  # Number of random reservoirs to try
```

**Initialization Process:**
1. For each of `esn_n_restarts` trials:
   - Sample random A with spectral radius ≈ 0.9
   - Sample random B, C2, D21
   - Simulate to get internal states x, w
   - Fit C, D, D12 via least squares (masking NaN-padded rows)
   - Record training MSE on valid entries only
2. Keep the reservoir with lowest MSE
3. Run SDP to find feasible B, D21, P, L matrices
4. Refit C, D, D12 with SDP-updated B, D21

**Pros:**
- Explores diverse random reservoirs
- Robust to NaN-padded trajectories (row masking in least squares)
- SDP ensures feasibility of constraints
- Works well with limited training data

**Cons:**
- Slower than other methods (requires multiple simulations + SDP)
- Best for exploratory training

**Example with fewer restarts (faster):**
```yaml
initialization:
  method: esn
  esn_n_restarts: 2  # Faster initialization
```

---

## 2. N4SID (Numerical algorithms for Subspace State-Space System IDentification)

Loads pre-computed linear system matrices from MATLAB identification, pads to model dimension if needed, then runs SDP for constraint feasibility.

**Configuration:**
```yaml
model:
  model_type: crnn
  nw: 64
  nx: 64
  initialization:
    method: n4sid
```

**Requirements:**
- File `n4sid_params.mat` must exist in the data directory
- MATLAB structure must contain: `A`, `B`, `C`, `D` matrices
- Format: standard MATLAB `.mat` format (scipy.io.loadmat compatible)

**Initialization Process:**
1. Load A, B, C, D from `n4sid_params.mat`
2. Validate dimensions and detect MATLAB transposition if needed
3. If N4SID state order < model nx, zero-pad matrices:
   - A: top-left block
   - B: top rows
   - C: left columns
4. Set B2 = 0 (keep fully linear initially)
5. Sample random C2 and D21
6. Run SDP to find feasible P, L, M matrices
7. Refit C, D, D12 with new internal states

**Pros:**
- Warm-starts from physics-based or data-driven identification
- Allows smaller initial orders (automatic padding)
- Fast when N4SID file exists
- Leverages domain knowledge (pre-identified A, B, C, D)

**Cons:**
- Requires pre-existing N4SID results
- May be overspecialized if N4SID quality is poor
- Falls back to ESN if file not found

**Example with ESN fallback (safe):**
```yaml
initialization:
  method: n4sid  # Tries N4SID; falls back to ESN if file missing
  esn_n_restarts: 5  # Fallback ESN runs
```

---

## 3. Identity (Simple Default)

Predefined stable system with identity-like structure: diagonal A = 0.9I, identity output matrix C = [I, 0], all feedthrough channels zero.

**Configuration:**
```yaml
model:
  model_type: crnn
  nw: 64
  nx: 64
  initialization:
    method: identity
```

**Initialization Result:**
- α = 0.99 (spectral radius)
- A = 0.9 I (diagonal, stable)
- B = 0 (learned during training)
- B2 = 0 (no nonlinearity feedthrough initially)
- C = [I, 0] (identity on first min(ne, nx) outputs, rest zero)
- D = 0 × (ne, nd)
- D12 = 0 × (ne, nw)
- C2 = Rand(-1, 1) (random measurement matrix)
- D21 = small random (0.01 scale)

**Pros:**
- Deterministic and reproducible
- Fastest initialization (no simulation or SDP)
- Simple baseline for debugging/prototyping
- Works with any NaN-padded data

**Cons:**
- Very simple initial model
- May require more training epochs to learn dynamics
- Less suitable for complex systems

**Example:**
```yaml
initialization:
  method: identity  # Simplest possible initialization
```

---

## Comparison Table

| Aspect | ESN | N4SID | Identity |
|--------|-----|-------|----------|
| Speed | Slow (multiple simulations + SDP) | Medium (SDP only) | Very fast |
| Training time | Comparable | Slightly less | Slightly more |
| Requires file | No | Yes (n4sid_params.mat) | No |
| Reproducibility | High (fixed seed) | High | Guaranteed |
| Robustness | Very good | Good | Good |
| Best use case | General-purpose | Pre-identified physics | Prototyping/debug |
| NaN handling | Excellent (masked lstsq) | Good | Excellent |

---

## Migration Notes

### From Old API

Previous code used positional arguments:
```python
# OLD (not supported anymore)
model.initialize_parameters(train_inputs, train_states, train_outputs, 
                            n_restarts=5, data_dir=data_path)
```

New code uses config:
```python
# NEW (current API)
model.initialize_parameters(train_inputs, train_states, train_outputs,
                            init_config=config.model.initialization,
                            data_dir=data_path)
```

Config in YAML:
```yaml
model:
  initialization:
    method: esn              # or "n4sid", "identity"
    esn_n_restarts: 5        # ESN-specific parameter
```

---

## Examples

### Quick experiment (identity)
```yaml
model:
  initialization:
    method: identity
```

### Production with multiple trials
```yaml
model:
  initialization:
    method: esn
    esn_n_restarts: 10
```

### Physics-informed (N4SID + ESN fallback)
```yaml
model:
  initialization:
    method: n4sid           # Primary: load from n4sid_params.mat
    esn_n_restarts: 5       # Fallback if file missing
```

### Lightweight hyperparameter tuning
```python
# In optimize_hyperparameters.py: automatically uses esn_n_restarts=1
trial_init_config = InitializationConfig(method="esn", esn_n_restarts=1)
```

---

## Troubleshooting

**Problem:** "ESN initialization failed: no trial had enough finite rows"
- **Cause:** Data has too many NaN values relative to valid rows
- **Solution:** Review NaN padding strategy; increase sequence length

**Problem:** "N4SID file not found, falling back to ESN"
- **Log Message:** Normal if you don't have pre-computed N4SID
- **Action:** Either provide n4sid_params.mat or use method="esn" directly

**Problem:** "ESN initialization is slow"
- **Cause:** Large nx, nw, or high esn_n_restarts
- **Solution:** Reduce esn_n_restarts for trials; use identity for prototyping

**Problem:** "Model doesn't satisfy constraints after initialization"
- **Log:** "Initial parameters satisfy constraints: False"
- **Cause:** SDP solver infeasible or numerical issues
- **Action:** Check constraint parameters (alpha, delta, s); run with different random seed
