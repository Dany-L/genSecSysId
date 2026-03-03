# Hyperparameter Optimization with SLURM

This guide explains how to run distributed hyperparameter optimization on HPC clusters using SLURM.

## Overview

The Optuna-based optimization works perfectly with SLURM by using a **shared database** where multiple SLURM jobs (workers) coordinate automatically:

```
┌─────────────────────────────────────────────┐
│         Shared Database (PostgreSQL)        │
│              Optuna Study                   │
└─────────────────────────────────────────────┘
         ▲         ▲         ▲         ▲
         │         │         │         │
    ┌────┴────┬────┴────┬────┴────┬────┴────┐
    │ Worker  │ Worker  │ Worker  │ Worker  │
    │ GPU:0   │ GPU:1   │ GPU:2   │ GPU:3   │
    │ (SLURM) │ (SLURM) │ (SLURM) │ (SLURM) │
    └─────────┴─────────┴─────────┴─────────┘
```

Each worker:
1. Requests a trial from the shared Optuna study
2. Trains the model with those hyperparameters
3. Reports the validation loss back to the study
4. Requests the next trial

## Setup

### 1. Setup Shared Database (One-Time)

**Option A: SQLite on Shared Filesystem** (Simple, but slower)
```bash
# Works if your cluster has a shared filesystem
# Database will be at: /cluster/scratch/${USER}/optuna_studies/study_name.db
# No additional setup needed!
```

**Option B: PostgreSQL Database** (Recommended for production)
```bash
# On your cluster's database node or external server:
# 1. Install PostgreSQL
sudo apt-get install postgresql

# 2. Create database and user
sudo -u postgres psql
CREATE DATABASE optuna_studies;
CREATE USER optuna_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE optuna_studies TO optuna_user;
\q

# 3. Configure PostgreSQL to accept connections
# Edit /etc/postgresql/*/main/postgresql.conf:
listen_addresses = '*'

# Edit /etc/postgresql/*/main/pg_hba.conf:
host    optuna_studies    optuna_user    10.0.0.0/8    md5

# 4. Restart PostgreSQL
sudo systemctl restart postgresql
```

### 2. Install Optuna on Cluster

```bash
# SSH to cluster
ssh your-cluster

# Create virtual environment
cd ~/genSecSysId/python
python3 -m venv ~/venv/genSecSysId
source ~/venv/genSecSysId/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install optuna

# For PostgreSQL support
pip install psycopg2-binary

# For MySQL support (if using MySQL)
pip install pymysql
```

### 3. Copy Code to Cluster

```bash
# From your local machine
rsync -avz --exclude='mlruns/' --exclude='*.pt' --exclude='__pycache__/' \
    ~/Documents/01_Git/01_promotion/genSecSysId/python/ \
    user@cluster:~/genSecSysId/python/

rsync -avz \
    ~/genSecSysId-Data/configs/ \
    user@cluster:~/genSecSysId-Data/configs/
```

## Usage

### Method 1: SLURM Job Array (Recommended)

Run 10 parallel workers, each doing 10 trials = 100 total trials:

```bash
# Edit slurm_optimize.sh to configure:
# - Database storage URL
# - SLURM partition/resources
# - Module loads for your cluster

# Submit job array
sbatch scripts/slurm_optimize.sh \
    ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    silverbox-optimization \
    10

# Job array submits 10 workers (--array=0-9)
# Each worker runs 10 trials
# Total: 100 trials
```

### Method 2: Multiple Sequential Submissions

Submit workers one at a time (useful for testing):

```bash
# Edit slurm_optimize.sh: Comment out #SBATCH --array line

# Submit multiple jobs
for i in {1..5}; do
    sbatch scripts/slurm_optimize.sh \
        ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
        silverbox-test \
        20
    sleep 1
done
```

### Method 3: Interactive Job (Testing)

Test on a single GPU interactively:

```bash
# Request interactive GPU node
srun --pty --gres=gpu:1 --mem=16G --time=2:00:00 bash

# Load environment
source ~/venv/genSecSysId/bin/activate
cd ~/genSecSysId/python

# Run optimization
python scripts/optimize_hyperparameters.py \
    --config ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    --n-trials 5 \
    --study-name test-interactive \
    --storage sqlite:////scratch/${USER}/test.db
```

## Configuration Examples

### Example 1: Small-Scale (SQLite)

```bash
# scripts/slurm_optimize.sh modifications:
#SBATCH --array=0-4           # 5 workers
#SBATCH --gres=gpu:1          # 1 GPU per worker
#SBATCH --time=04:00:00       # 4 hours

DB_STORAGE="sqlite:////cluster/scratch/${USER}/optuna/${STUDY_NAME}.db"
TRIALS_PER_WORKER=20          # 5 workers × 20 trials = 100 total
```

### Example 2: Large-Scale (PostgreSQL)

```bash
# scripts/slurm_optimize.sh modifications:
#SBATCH --array=0-19          # 20 workers
#SBATCH --gres=gpu:1          # 1 GPU per worker  
#SBATCH --time=48:00:00       # 48 hours

DB_STORAGE="postgresql://optuna_user:password@db.cluster.edu:5432/optuna_studies"
TRIALS_PER_WORKER=25          # 20 workers × 25 trials = 500 total
```

### Example 3: Multi-GPU per Worker

```bash
# For models that benefit from multiple GPUs:
#SBATCH --array=0-7           # 8 workers
#SBATCH --gres=gpu:2          # 2 GPUs per worker
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# In config.yaml:
# training:
#   device: "cuda"  # Will use all available GPUs
```

## Monitoring

### Check Job Status

```bash
# View all jobs
squeue -u $USER

# View specific job array
squeue -j <job_id>

# View detailed job info
scontrol show job <job_id>
```

### Monitor Progress

```bash
# View live output
tail -f logs/slurm/hpo_<job_id>_<array_task_id>.out

# Count completed trials
sqlite3 /scratch/${USER}/optuna/${STUDY_NAME}.db \
    "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"

# For PostgreSQL:
psql -h db.cluster.edu -U optuna_user -d optuna_studies \
    -c "SELECT COUNT(*) FROM trials WHERE study_id=1 AND state='COMPLETE';"
```

### View Results (After Completion)

```bash
# SSH tunnel to cluster for dashboard
ssh -L 8080:localhost:8080 user@cluster

# On cluster:
source ~/venv/genSecSysId/bin/activate
optuna-dashboard sqlite:////scratch/${USER}/optuna/${STUDY_NAME}.db

# Open browser on local machine:
# http://localhost:8080
```

## Best Practices

### 1. Start Small

```bash
# Test with 2 workers, 2 trials each
#SBATCH --array=0-1
TRIALS_PER_WORKER=2
```

### 2. Use Job Dependencies

```bash
# Run optimization first
OPT_JOB=$(sbatch --parsable scripts/slurm_optimize.sh ...)

# Then train with best config after optimization completes
sbatch --dependency=afterok:${OPT_JOB} \
    scripts/slurm_single_trial.sh best_config.yaml
```

### 3. Resource Allocation

```bash
# Match resources to your model size:
# Small model (nw < 10):  --mem=8G  --cpus-per-task=2
# Medium model (nw 10-20): --mem=16G --cpus-per-task=4
# Large model (nw > 20):   --mem=32G --cpus-per-task=8
```

### 4. Handle Node Failures

Optuna automatically handles worker failures! If a SLURM job fails:
- The trial is marked as FAIL
- Other workers continue unaffected
- You can resubmit failed workers

### 5. Checkpointing

```bash
# For long runs, enable periodic saves:
# In optimize_hyperparameters.py, add checkpoint_interval:
study.optimize(
    objective,
    n_trials=n_trials,
    timeout=timeout,
    callbacks=[checkpoint_callback],  # Save every N trials
)
```

## Troubleshooting

### Database Locked (SQLite)

```bash
# SQLite doesn't handle many concurrent writers well
# Solution: Use PostgreSQL or reduce workers

# Or use WAL mode for SQLite:
sqlite3 database.db "PRAGMA journal_mode=WAL;"
```

### GPU Not Detected

```bash
# Check CUDA setup
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Load correct CUDA module
module load cuda/11.8  # Adjust version
```

### Out of Memory

```bash
# Reduce batch size in optimization script
# Line ~45 in optimize_hyperparameters.py:
batch_size = trial.suggest_categorical("batch_size", [8, 16])  # Remove 32, 64
```

### Jobs Stuck in Queue

```bash
# Check partition limits
sinfo -p gpu

# Check your allocation
sshare -u $USER

# Request different partition
#SBATCH --partition=gpu-short  # For quick jobs
```

## Advanced: Custom SLURM Wrapper

For more control, create a Python wrapper that submits SLURM jobs:

```python
# scripts/slurm_submit_optuna.py
import subprocess
import optuna

def submit_trial(trial_params, study_name):
    """Submit a SLURM job for a single trial."""
    cmd = f"""sbatch --export=ALL,TRIAL_PARAMS='{trial_params}' \
        scripts/slurm_single_trial.sh"""
    subprocess.run(cmd, shell=True)

# Use this with Optuna's ask-and-tell interface
study = optuna.create_study(storage="postgresql://...")
for _ in range(100):
    trial = study.ask()
    params = trial.params
    submit_trial(params, study.study_name)
```

## Performance Expectations

Based on your SimpleLure model:

| Setup | Workers | Trials/Worker | Training Time/Trial | Total Time |
|-------|---------|---------------|---------------------|------------|
| Small | 5 | 20 | ~3 min | ~60 min |
| Medium | 10 | 25 | ~3 min | ~75 min |
| Large | 20 | 50 | ~3 min | ~150 min |

With GPU vs CPU:
- **GPU**: ~3 min/trial (with SDP overhead)
- **CPU**: ~12 min/trial

## Complete Example Workflow

```bash
# 1. Setup on cluster
ssh cluster
cd ~/genSecSysId/python
source ~/venv/genSecSysId/bin/activate

# 2. Submit optimization (10 workers × 20 trials = 200 total)
sbatch scripts/slurm_optimize.sh \
    ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    silverbox-final \
    20

# 3. Monitor (from local machine)
ssh -L 8080:localhost:8080 cluster \
    "cd ~/genSecSysId/python && optuna-dashboard sqlite:////scratch/\${USER}/optuna/silverbox-final.db"

# 4. After completion, download best config
scp cluster:~/genSecSysId/python/outputs/best_config_silverbox-final.yaml \
    ~/genSecSysId-Data/configs/

# 5. Train final model with best config
sbatch scripts/slurm_single_trial.sh \
    ~/genSecSysId-Data/configs/best_config_silverbox-final.yaml
```

## Summary

✅ **Key Points:**
- Each SLURM job is an independent Optuna worker
- All workers share the same study via database
- Perfect for HPC clusters with job schedulers
- Scales to hundreds of parallel workers
- Automatically handles failures and restarts
- Compatible with job arrays and dependencies

For questions or issues with your specific SLURM cluster configuration, check with your cluster documentation or system administrators.
