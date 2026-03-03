# SLURM Hyperparameter Optimization - Quick Reference

## ✅ **Yes, Fully Compatible with SLURM!**

Your Optuna-based optimization works perfectly with SLURM using a **shared database** approach where multiple SLURM jobs coordinate automatically.

---

## TL;DR - Quick Start

```bash
# 1. Copy code to cluster
rsync -avz python/ user@cluster:~/genSecSysId/python/

# 2. Setup on cluster
ssh cluster
cd ~/genSecSysId/python
source venv/bin/activate
pip install optuna psycopg2-binary

# 3. Submit 10 parallel workers (100 total trials)
sbatch scripts/slurm_optimize.sh \
    ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    my-study \
    10
```

---

## How It Works

```
┌─────────────────────────────────┐
│  Shared Database (PostgreSQL)   │  ← Single source of truth
│        Optuna Study             │
└─────────────────────────────────┘
    ▲       ▲       ▲       ▲
    │       │       │       │
┌───┴───┬───┴───┬───┴───┬───┴───┐
│Worker │Worker │Worker │Worker │  ← SLURM jobs
│ GPU:0 │ GPU:1 │ GPU:2 │ GPU:3 │
└───────┴───────┴───────┴───────┘
```

**Each SLURM job:**
1. Pulls a trial from shared Optuna study
2. Trains model with those hyperparameters  
3. Reports validation loss back
4. Requests next trial
5. Repeat until done

**Key benefits:**
- ✅ No manual job management
- ✅ Automatic load balancing
- ✅ Handles failures gracefully
- ✅ Works with SLURM job arrays
- ✅ Scales to 100+ workers

---

## Files Created

```
scripts/
├── slurm_optimize.sh          # Main optimization script (job array)
└── slurm_single_trial.sh      # Single training run

docs/
└── SLURM_OPTIMIZATION.md      # Complete documentation
```

---

## Usage Examples

### Example 1: Small Test (SQLite)

```bash
# Edit scripts/slurm_optimize.sh:
#SBATCH --array=0-2           # 3 workers
#SBATCH --time=02:00:00       # 2 hours

# Submit
sbatch scripts/slurm_optimize.sh config.yaml test-run 5
# → 3 workers × 5 trials = 15 total trials
```

### Example 2: Production (PostgreSQL)

```bash
# Setup PostgreSQL database once (see docs)
# Edit scripts/slurm_optimize.sh:
#SBATCH --array=0-19          # 20 workers
DB_STORAGE="postgresql://user:pass@db-host:5432/optuna"

# Submit
sbatch scripts/slurm_optimize.sh config.yaml production 25
# → 20 workers × 25 trials = 500 total trials
```

### Example 3: Interactive Test

```bash
# Get interactive node
srun --pty --gres=gpu:1 bash

# Run manually
python scripts/optimize_hyperparameters.py \
    --config config.yaml \
    --n-trials 5 \
    --storage sqlite:///test.db
```

---

## Database Options

### SQLite (Simple, for ≤10 workers)

```bash
# Built-in, no setup needed
DB_STORAGE="sqlite:////scratch/${USER}/optuna/study.db"
```

**Pros:** No setup, works immediately  
**Cons:** Slow with many workers (locking issues)

### PostgreSQL (Recommended for >10 workers)

```bash
# One-time setup on DB server:
CREATE DATABASE optuna_studies;
CREATE USER optuna WITH PASSWORD 'pass';
GRANT ALL PRIVILEGES ON DATABASE optuna_studies TO optuna;

# In script:
DB_STORAGE="postgresql://optuna:pass@db-host:5432/optuna_studies"
```

**Pros:** Fast, scales to 100+ workers, production-ready  
**Cons:** Requires DB setup

---

## Monitoring

### Check Job Status

```bash
# All your jobs
squeue -u $USER

# Specific job array
squeue -j <job_id>

# Live output
tail -f logs/slurm/hpo_<job_id>_<task_id>.out
```

### View Optimization Progress

```bash
# SSH tunnel from local machine
ssh -L 8080:localhost:8080 user@cluster

# On cluster, start dashboard
optuna-dashboard sqlite:////scratch/${USER}/optuna/study.db

# Open browser: http://localhost:8080
```

### Count Completed Trials

```bash
# SQLite
sqlite3 study.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"

# PostgreSQL
psql -h db-host -U optuna -d optuna_studies \
    -c "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"
```

---

## Configuration

Edit [scripts/slurm_optimize.sh](file:///Users/jack/Documents/01_Git/01_promotion/genSecSysId/python/scripts/slurm_optimize.sh):

```bash
# Resources per worker
#SBATCH --cpus-per-task=4     # CPU cores
#SBATCH --mem=16G              # RAM
#SBATCH --gres=gpu:1           # GPUs (0 for CPU-only)
#SBATCH --time=24:00:00        # Max time

# Number of parallel workers
#SBATCH --array=0-9            # 10 workers (0-9 inclusive)

# Module loads (cluster-specific)
module load python/3.10
module load cuda/11.8
module load gcc/11.2.0

# Storage (choose one)
DB_STORAGE="sqlite:///..."     # Simple
DB_STORAGE="postgresql://..."  # Production
```

---

## Typical Resource Allocation

| Model Size | CPUs | RAM | GPUs | Time/Trial |
|------------|------|-----|------|------------|
| Small (nw<10) | 2 | 8G | 1 | ~2 min |
| Medium (nw 10-20) | 4 | 16G | 1 | ~3-5 min |
| Large (nw>20) | 8 | 32G | 1 | ~8-10 min |

For your SimpleLure with SDP solver: **~3 min/trial on GPU**

---

## Common Patterns

### Pattern 1: Job Array (Recommended)

```bash
# Submit once, creates N workers automatically
#SBATCH --array=0-19           # Creates 20 jobs

sbatch scripts/slurm_optimize.sh config.yaml study 10
# → 20 workers × 10 trials = 200 total
```

### Pattern 2: Sequential Submission

```bash
# Submit multiple independent jobs
for i in {1..10}; do
    sbatch scripts/slurm_optimize.sh config.yaml study 20
done
# → 10 workers × 20 trials = 200 total
```

### Pattern 3: Job Dependencies

```bash
# Optimize first
OPT_JOB=$(sbatch --parsable scripts/slurm_optimize.sh ...)

# Train best config after optimization completes
sbatch --dependency=afterok:${OPT_JOB} \
    scripts/slurm_single_trial.sh best_config.yaml
```

---

## Troubleshooting

### "Database is locked" (SQLite)

**Solution:** Use PostgreSQL or reduce workers to ≤5

```bash
# Enable WAL mode for better concurrency:
sqlite3 study.db "PRAGMA journal_mode=WAL;"
```

### "No GPU detected"

```bash
# Check loaded modules
module list

# Check CUDA
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Load CUDA module
module load cuda/11.8  # Adjust version
```

### "Out of memory"

```bash
# Reduce batch size in optimization
# Edit scripts/optimize_hyperparameters.py line ~45:
batch_size = trial.suggest_categorical("batch_size", [8, 16])
```

### "Worker timeout"

```bash
# Increase time limit
#SBATCH --time=48:00:00        # 48 hours

# Or reduce max_epochs in optimization (line ~73):
config.training.max_epochs = min(50, config.training.max_epochs)
```

---

## Performance Expectations

**Your SimpleLure Model:**

| Setup | Workers | Trials/Worker | Total Trials | Est. Time |
|-------|---------|---------------|--------------|-----------|
| **Test** | 3 | 5 | 15 | ~15 min |
| **Small** | 5 | 20 | 100 | ~60 min |
| **Medium** | 10 | 25 | 250 | ~75 min |
| **Large** | 20 | 50 | 1000 | ~150 min |

**Speedup vs Sequential:**
- Sequential (1 worker): 500 trials × 3 min = **25 hours**
- Parallel (20 workers): 500 trials / 20 = **75 minutes** ⚡

---

## Complete Workflow Example

```bash
# 1. Copy to cluster  
rsync -avz python/ cluster:~/genSecSysId/python/

# 2. SSH to cluster
ssh cluster

# 3. Setup environment
cd ~/genSecSysId/python
source venv/bin/activate
pip install optuna psycopg2-binary

# 4. Edit SLURM script for your cluster
# - Module loads
# - Partition name
# - Database storage

# 5. Submit optimization
sbatch scripts/slurm_optimize.sh \
    ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    silverbox-final \
    20

# 6. Monitor from local machine
ssh -L 8080:localhost:8080 cluster
# On cluster: optuna-dashboard sqlite:///...
# Browser: http://localhost:8080

# 7. Download results
scp cluster:~/genSecSysId/python/outputs/best_config_*.yaml \
    ~/genSecSysId-Data/configs/
```

---

## Key Advantages for SLURM

1. **Native Integration**: Each worker is a standard SLURM job
2. **Automatic Coordination**: Optuna handles all synchronization
3. **Fault Tolerance**: Failed workers don't affect others
4. **Resource Efficiency**: SLURM manages GPU allocation
5. **Scalability**: Easily scale from 5 to 100+ workers
6. **Job Arrays**: Built-in SLURM support for parallel jobs
7. **Priority/QoS**: Works with SLURM accounting and priorities

---

## Next Steps

1. **Test locally** (already done ✅)
2. **Copy to cluster**: `rsync -avz python/ cluster:~/genSecSysId/`
3. **Edit** [scripts/slurm_optimize.sh](file:///Users/jack/Documents/01_Git/01_promotion/genSecSysId/python/scripts/slurm_optimize.sh) for your cluster
4. **Submit test run**: 2-3 workers, 2-3 trials each
5. **Monitor**: `squeue -u $USER` and dashboard
6. **Scale up**: Increase workers and trials per worker

---

## Full Documentation

See [docs/SLURM_OPTIMIZATION.md](file:///Users/jack/Documents/01_Git/01_promotion/genSecSysId/python/docs/SLURM_OPTIMIZATION.md) for:
- Detailed database setup
- Advanced configurations
- Cluster-specific examples
- Troubleshooting guide
- Performance tuning tips
