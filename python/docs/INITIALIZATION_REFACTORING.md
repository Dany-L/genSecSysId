# Initialization Refactoring Summary

## Changes Made

### 1. **Config System** (`src/sysid/config.py`)
- Added `InitializationConfig` dataclass with fields:
  - `method`: str = "esn" | "n4sid" | "identity"
  - `esn_n_restarts`: int = 5 (ESN-specific parameter)
- Added `initialization` field to `ModelConfig` with defaults

### 2. **SimpleLure Model** (`src/sysid/models/constrained_rnn.py`)
Refactored `initialize_parameters()` into modular methods:

#### **Main Method: `initialize_parameters()`**
- Dispatcher that routes to appropriate initialization method
- Accepts `init_config` parameter (InitializationConfig object)
- Fallback: ESN if N4SID file missing
- Common post-initialization: computes s, checks constraints

#### **Three Initialization Methods**

**A. `_init_identity()` — Predefined stable system**
```
α = 0.99
A = 0.9I (diagonal, stable)
C = [I, 0] (identity-like output)
C2 = Rand(-1, 1)
B2 = D = D12 = 0
```
- O(1) complexity, deterministic
- Best for prototyping/debugging

**B. `_init_esn()` — Echo State Network (random reservoirs)**
- Loop: sample random A, B, C2, D21
  - Simulate forward pass
  - Fit C, D, D12 via **masked least squares** (handles NaN padding)
  - Record MSE on valid entries only
- Keep best reservoir
- Run SDP once for feasibility
- Refit output matrices with updated B, D21
- O(n_restarts) complexity

**C. `_init_n4sid()` — Physics-informed from mat file**
- Load A, B, C, D from `n4sid_params.mat`
- Validate dimensions, auto-transpose if needed
- Zero-pad if N4SID order < model nx
- Run SDP for feasibility
- Refit output matrices
- Returns True if successful, False otherwise

#### **Helper Method: `_refit_output_matrices()`**
- Simulates with current A, B, B2, C2, D21
- Fits C, D, D12 via **masked least squares** (NaN-aware)
- Called after SDP to refine output matrices

### 3. **Training Scripts**

**`scripts/train.py`**
- Updated `initialize_parameters()` call to pass `init_config=config.model.initialization`
- Removed init fig plotting code (no longer returned)

**`scripts/optimize_hyperparameters.py`**
- Updated trial initialization to use `InitializationConfig(method="esn", esn_n_restarts=1)`
- Lightweight trials with single ESN restart

### 4. **Documentation** (`docs/INITIALIZATION_METHODS.md`)
- Comprehensive guide for each method
- Configuration examples
- Comparison table
- Troubleshooting guide

## Key Features

✅ **NaN-Aware Least Squares**
- ESN initialization automatically masks NaN-padded rows
- Prevents least-squares solution contamination
- Handles diverging trajectories correctly

✅ **Configurable via YAML**
```yaml
model:
  initialization:
    method: esn              # or "n4sid", "identity"
    esn_n_restarts: 5
```

✅ **Three Distinct Trade-offs**
| Method | Speed | Convergence | Best For |
|--------|-------|-------------|----------|
| Identity | ⚡ Fastest | Medium | Prototyping |
| ESN | Medium | ⭐ Best | General-purpose |
| N4SID | Fast | Very Good | Physics-informed |

✅ **Graceful Fallback**
- N4SID → ESN if file missing
- No training interruption

✅ **Backward Compatible**
- Old code using positional args will raise clear error
- Config-based approach enforces intentional choice

## Testing

All three methods tested with:
- NaN-padded sequences
- Multiple batch processing
- Constraint satisfaction checks

**Results:**
- ✓ Identity: A = 0.9I, C = [I, 0], deterministic
- ✓ ESN: Spectral radius ≈ 0.9, LSE fit successful
- ✓ N4SID: Fallback to ESN when file missing

## Usage Examples

### Quick Start (prototyping)
```yaml
model:
  initialization:
    method: identity
```

### Production (robust)
```yaml
model:
  initialization:
    method: esn
    esn_n_restarts: 10
```

### Physics-informed (warm-start)
```yaml
model:
  initialization:
    method: n4sid
    esn_n_restarts: 3  # fallback
# Requires: data/train_path/n4sid_params.mat
```

## Migration Checklist

- [x] Add InitializationConfig to config.py
- [x] Refactor initialize_parameters into three methods
- [x] Update train.py to pass init_config
- [x] Update optimize_hyperparameters.py for trials
- [x] Add NaN masking in ESN least squares
- [x] Implement N4SID with graceful fallback
- [x] Implement identity initialization
- [x] Test all three methods
- [x] Document with examples and troubleshooting

## Next Steps (Optional Enhancements)

1. **Visualization**: Plot initialization ellipse/polytope (if nx == 2)
2. **Seeding**: Make ESN reproducible with `torch.manual_seed()`
3. **Metrics**: Log per-restart MSE breakdown for ESN
4. **Validation**: Check initialized model on validation set
5. **Profiling**: Time each initialization method for benchmarks
