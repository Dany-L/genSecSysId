# Random Seed Configuration Guide

## Overview

The system now fully supports controlling randomness for reproducibility and variance estimation:
- **With seed**: Get identical results across runs (reproducible research)
- **Without seed**: Get different results each run (estimate standard deviation, understand variance)

---

## Configuration Methods

There are two ways to control seeding:

### 1. In Config File (YAML/JSON)

**For reproducible runs (with seed):**
```yaml
seed: 42  # Any integer
```

**For variance estimation (no seed):**
```yaml
seed: null  # or simply omit the seed line
```

### 2. Command-Line Arguments (Overrides Config)

**Train with specific seed (reproducible):**
```bash
python scripts/train.py \
    --config config.yaml \
    --seed 42
```

**Train without seed (get variance):**
```bash
python scripts/train.py \
    --config config.yaml \
    --seed -1
```

**Optimize hyperparameters reproducibly:**
```bash
python scripts/optimize_hyperparameters.py \
    --config config.yaml \
    --seed 42 \
    --n-trials 50
```

**Optimize hyperparameters for variance estimation:**
```bash
python scripts/optimize_hyperparameters.py \
    --config config.yaml \
    --no-seed \
    --n-trials 50
```

---

## Priority Order

When multiple sources specify seed, priority is:

1. **Command-line argument** (highest priority)
   - `--seed -1` or `--no-seed` disables seeding
   - `--seed 42` sets seed to 42

2. **Config file** (intermediate priority)
   - `seed: 42` or `seed: null`

3. **Default** (lowest priority)
   - Default is `None` (no seeding)

**Example:** Config has `seed: 42` but you run with `--seed -1`:
```bash
# This will use NO seed despite config having seed: 42
python scripts/train.py --config config.yaml --seed -1
```

---

## Use Cases

### Case 1: Reproducible Hyperparameter Optimization

**Goal:** Find best hyperparameters, get exact same results every time

```bash
# Config file (example.yaml)
seed: 42

# Command
python scripts/optimize_hyperparameters.py \
    --config example.yaml \
    --n-trials 100 \
    --study-name final-optimization

# Result: Exact same trials, same order, same losses every run
```

**Next time:**
```bash
# Running again with same seed = same results
python scripts/optimize_hyperparameters.py \
    --config example.yaml \
    --n-trials 100 \
    --study-name final-optimization-v2
```

### Case 2: Estimate Standard Deviation

**Goal:** Run same optimization multiple times to measure variance

```bash
# Config file - no seed specified, or seed: null
# optimization.yaml (without seed)

# Run 1
python scripts/optimize_hyperparameters.py \
    --config optimization.yaml \
    --n-trials 50 \
    --study-name variance-test-run1

# Run 2
python scripts/optimize_hyperparameters.py \
    --config optimization.yaml \
    --n-trials 50 \
    --study-name variance-test-run2

# Run 3
python scripts/optimize_hyperparameters.py \
    --config optimization.yaml \
    --n-trials 50 \
    --study-name variance-test-run3

# Compare results across runs to calculate standard deviation
```

### Case 3: Training with Known Good Hyperparameters (Reproducible)

**Goal:** Train final model with known hyperparameters, want reproducible results

```bash
# Use config with seed
seed: 42

# Train multiple times for baseline
for i in {1..3}; do
    python scripts/train.py \
        --config best_config.yaml \
        --seed 42
done
# All three runs will produce identical models
```

### Case 4: Training Multiple Times for Variance Estimation

**Goal:** Understand how much variance training introduces

```bash
# Config without seed (or seed: null)

# Train multiple times with same hyperparameters but different random initializations
for i in {1..10}; do
    python scripts/train.py \
        --config config.yaml \
        --seed -1  # Explicitly disable seeding
done

# Compare 10 trained models to measure variance in:
# - Final validation loss
# - Model weights
# - Generalization performance
```

---

## SLURM Cluster Usage

### Reproducible Large-Scale Optimization

```bash
# scripts/slurm_optimize.sh with seed in config.yaml

config.yaml:
seed: 42

# Submit job array
sbatch scripts/slurm_optimize.sh config.yaml study-name 20
# All 20 workers will get same trials in same order (reproducible)
```

### Variance Estimation on Cluster

```bash
# scripts/slurm_optimize.sh with no seed

config.yaml:
seed: null  # or omitted

# Run multiple times
for i in {1..3}; do
    sbatch scripts/slurm_optimize.sh config.yaml variance-run-$i 20
done

# Pool results from all three runs to estimate hyperparameter variance
```

---

## What Gets Seeded?

When a seed is set, the system ensures reproducibility of:

✅ **Seeded (deterministic):**
- PyTorch random operations (`torch.randn`, weight initialization, dropout)
- CUDA operations (random number generation on GPU)
- NumPy operations (used in data loading and preprocessing)
- Python built-in `random` module

✅ **Also seeded (via PyTorch):**
- CUDA cuDNN benchmarking (disabled for determinism)

❌ **Not affected by seed:**
- Floating-point operation order (may differ on different hardware)
- Parallelization effects (thread scheduling)
- External data loading variability

---

## Checking Current Seed

When you start training/optimization, the log will show:

**With seed:**
```
Random seed: 42
```

**Without seed:**
```
Random seed: Disabled (different results each run)
```

or in optimize_hyperparameters.py:
```
Random seed: None
```

---

## FAQ

**Q: I want reproducible results. What should I do?**

A: Set `seed: 42` (or any int) in your config file. That's it. No special CLI args needed.

```yaml
# config.yaml
seed: 42
```

**Q: I want to estimate standard deviation. What should I do?**

A: Either:
1. Remove the `seed:` line from config (or set `seed: null`)
2. Add `--seed -1` to your command

Then run the experiment multiple times.

**Q: Can I mix reproducibility and variance estimation?**

A: Yes! 
- Use `seed: null` in base config for reproducible hyperparameter search
- Then train final model with `--seed 42` to fix weights
- Or run final training multiple times with `--seed -1` to estimate final variance

**Q: What if I forget to set a seed?**

A: Default is `None` (no seeding), so you'll get different results each time. If you need reproducibility, specify a seed.

**Q: Does seed affect hyperparameter optimization?**

A: Yes, it controls:
- Data shuffling order
- Weight initialization
- Dropout randomness

But Optuna's trial suggestions are deterministic (based on its algorithm), not affected by PyTorch seed.

**Q: On SLURM, will all workers get the same seed?**

A: Yes, they load the same config file. To get different results across workers, use `seed: null` in config, then each worker's random operations will differ (due to process IDs, timing, etc.).

**Q: How do I reproduce results from a specific run?**

A: 1. Check the run's logged config (in MLflow or saved config file)
   2. Use same config with same seed
   3. Your results should be identical

---

## Examples Summary

| Goal | Config | CLI Command |
|------|--------|-------------|
| Reproducible training | `seed: 42` | `python train.py --config config.yaml` |
| Variance estimate (train) | `seed: null` | `python train.py --config config.yaml --seed -1` |
| Reproducible optimization | `seed: 42` | `python optimize.py --config config.yaml` |
| Variance estimate (optimize) | `seed: null` | `python optimize.py --config config.yaml --no-seed` |
| Override config seed | (ignored) | `python train.py --config config.yaml --seed 123` |
| Disable seed override | (ignored) | `python train.py --config config.yaml --seed -1` |
