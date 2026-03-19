# Initialization Methods — Quick Reference

## Overview

The SimpleLure model now supports three distinct initialization strategies, configurable in your YAML config file:

### 1️⃣ **Identity** (⚡ Fastest - for prototyping)
```yaml
model:
  initialization:
    method: identity
```
**What it does:** Sets A=0.9I, C=[I,0], all other matrices deterministic
- **Speed:** ~0.1 seconds (no simulation)
- **Use case:** Model debugging, architecture testing
- **Config file:** `example_identity_init.yaml`

### 2️⃣ **ESN** (🔄 Flexible - general purpose)
```yaml
model:
  initialization:
    method: esn
    esn_n_restarts: 5  # or 1, 2, 3, 10, etc.
```
**What it does:** Tries N different random reservoirs, keeps the best
- **Speed:** ~5-30 seconds (depends on esn_n_restarts)
- **Use case:** Most projects, robust initialization
- **Config file:** `example_esn_init.yaml`
- **For HPO:** Use esn_n_restarts=1 or 2 for speed

### 3️⃣ **N4SID** (📊 Physics-informed - warm-start)
```yaml
model:
  initialization:
    method: n4sid
    esn_n_restarts: 3  # fallback if file missing
```
**What it does:** Loads A,B,C,D from n4sid_params.mat file
- **Speed:** ~2-5 seconds (SDP only)
- **Requirements:** File `data/train_path/n4sid_params.mat`
- **Use case:** When you have pre-identified linear system
- **Config file:** `example_n4sid_init.yaml`

---

## Configuration Examples

### Quick experiment (identity - fastest)
```bash
python scripts/train.py --config configs/example_identity_init.yaml
```

### Production training (ESN - robust)
```bash
python scripts/train.py --config configs/example_esn_init.yaml
```

### Physics-informed (N4SID - warm-start)
```bash
python scripts/train.py --config configs/example_n4sid_init.yaml
```

### Hyperparameter optimization (lightweight ESN)
```bash
python scripts/optimize_hyperparameters.py --config configs/example_hpo_lightweight.yaml
```

---

## Side-by-Side Comparison

| Feature | Identity | ESN | N4SID |
|---------|----------|-----|-------|
| **Speed** | ⚡ 0.1s | 🔄 5-30s | 📊 2-5s |
| **Reproducible** | ✅ Yes | ✅ Yes (fixed seed) | ✅ Yes |
| **Requires file** | ❌ No | ❌ No | ✅ Yes (n4sid_params.mat) |
| **NaN masking** | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| **Training epochs** | Higher | Moderate | Lower |
| **Best for** | Debug | General | Physics-informed |

---

## Practical Advice

### 🎯 What should I use?

**Starting a new project?** → **ESN** with `esn_n_restarts=5`
- Robust, handles any data, good warm-start

**Rapid prototyping?** → **Identity**
- Fast iteration, focus on architecture not init

**Have system ID results?** → **N4SID** (with ESN fallback)
- Leverages domain knowledge, faster convergence

**Tuning hyperparameters?** → **ESN** with `esn_n_restarts=1`
- Fast trials, deterministic, good enough

### ⚠️ Common pitfalls

- ❌ Using Identity for final training (may need many extra epochs)
- ❌ N4SID without checking n4sid_params.mat exists (will fallback silently)
- ❌ ESN with very large nx and nw (slow initialization)

### 💡 Tips

✅ Monitor logs for "Initial parameters satisfy constraints" → should be True
✅ ESN picks best from N trials, so higher `esn_n_restarts` = better init
✅ If N4SID fails, it automatically tries ESN — no manual intervention needed
✅ All methods handle NaN-padded sequences correctly

---

## Configuration Details

### ESN Parameters
```yaml
initialization:
  method: esn
  esn_n_restarts: 5  # Number of random reservoirs to try
```

**Recommended values:**
- `1`: Fast (hyperparameter tuning)
- `2-3`: Balanced speed and quality
- `5`: Good (default)
- `10+`: Very robust (slow)

### N4SID Parameters
```yaml
initialization:
  method: n4sid
  esn_n_restarts: 3  # Used if N4SID file not found
```

Must have file: `{data.train_path}/n4sid_params.mat`

Format:
```matlab
% MATLAB: save your system ID results
save('n4sid_params.mat', 'A', 'B', 'C', 'D');
```

### Identity (no parameters)
```yaml
initialization:
  method: identity  # No additional parameters needed
```

---

## Example YAML Snippets

### Copy-paste: ESN (recommended for most users)
```yaml
model:
  model_type: crnn
  nw: 64
  nx: 8
  activation: sat
  initialization:
    method: esn
    esn_n_restarts: 5
```

### Copy-paste: Fast prototyping
```yaml
model:
  model_type: crnn
  nw: 64
  nx: 8
  activation: sat
  initialization:
    method: identity
```

### Copy-paste: Hyperparameter tuning
```yaml
model:
  model_type: crnn
  nw: 64
  nx: 8
  activation: sat
  initialization:
    method: esn
    esn_n_restarts: 1  # Minimal for speed
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Init takes too long | Large nx, high esn_n_restarts | Reduce esn_n_restarts; use Identity |
| "N4SID file not found, falling back" | File doesn't exist | Create n4sid_params.mat or use ESN |
| "constraints not satisfied" | Numerical issues | Try different seed or method |
| Training doesn't converge | Poor initialization | Try ESN with esn_n_restarts=10 |

---

## Files Changed

For full refactoring details, see:
- 📄 [`docs/INITIALIZATION_METHODS.md`](INITIALIZATION_METHODS.md) — Comprehensive guide
- 📄 [`docs/INITIALIZATION_REFACTORING.md`](INITIALIZATION_REFACTORING.md) — Technical changes
- ⚙️ `src/sysid/config.py` — InitializationConfig class
- 🧠 `src/sysid/models/constrained_rnn.py` — Three initialization methods
- 🚀 `scripts/train.py`, `optimize_hyperparameters.py` — Updated calls

---

## Next Steps

1. ✅ Choose an initialization method (recommend: **ESN**)
2. ✅ Update your config YAML (see examples above)
3. ✅ Run training as usual
4. ✅ Monitor logs for "Initial parameters satisfy constraints: True"

Happy training! 🚀
