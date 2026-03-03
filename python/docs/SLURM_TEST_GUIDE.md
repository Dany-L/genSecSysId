# SLURM Testing Guide

Step-by-step guide to test your cluster setup before running full hyperparameter optimization.

---

## Prerequisites

1. **Copy code to cluster:**
   ```bash
   rsync -avz --exclude='mlruns' --exclude='*.pyc' \
       ~/Documents/01_Git/01_promotion/genSecSysId/python/ \
       ac137967@cluster:/home/ac137967/genSecSysId/
   ```

2. **Setup virtual environment on cluster:**
   ```bash
   ssh ac137967@cluster
   cd /home/ac137967/genSecSysId
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create log directory:**
   ```bash
   mkdir -p logs/slurm
   ```

---

## Test 1: Single Training Run (Most Important)

This test verifies:
- ✓ GPU access works
- ✓ MLflow logging works
- ✓ Artifacts are saved properly
- ✓ Dependencies are installed

### Step 1: Configure MLflow Connection

Edit [scripts/slurm_test_single_run.sh](../scripts/slurm_test_single_run.sh) line 69:

```bash
# Option A: Direct connection (if MLflow server is accessible from compute nodes)
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

# Option B: SSH tunnel (if MLflow only accessible from login node)
# On your local machine or login node:
# ssh -L 5000:localhost:5000 mlflow-server
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Option C: File-based (no server needed, for testing only)
export MLFLOW_TRACKING_URI="file:///home/ac137967/genSecSysId/mlruns"
```

### Step 2: Submit Test Job

```bash
# Copy your config to cluster first
scp ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    ac137967@cluster:/home/ac137967/configs/

# SSH to cluster
ssh ac137967@cluster
cd /home/ac137967/genSecSysId

# Submit test job (will run for ~5-10 minutes with 10 epochs)
sbatch scripts/slurm_test_single_run.sh /home/ac137967/configs/crnn_gen-sec_silverbox.yaml
```

### Step 3: Monitor Job

```bash
# Check job status
squeue -u ac137967

# Watch live output (replace JOBID with your job ID from squeue)
tail -f logs/slurm/test_run_<JOBID>.out

# Check for errors
tail -f logs/slurm/test_run_<JOBID>.err
```

### Step 4: Verify Results

**Check SLURM output:**
```bash
# Should see:
# - GPU detected: ✓
# - CUDA available: True
# - MLflow connection: ✓
# - Training completed: ✓
# - models/ directory exists: ✓

grep "CUDA available" logs/slurm/test_run_*.out
grep "Training completed" logs/slurm/test_run_*.out
```

**Check MLflow dashboard:**
1. Open your MLflow server in browser
2. Look for experiment "slurm-test"
3. Verify you see:
   - ✓ New run with job ID in name
   - ✓ Parameters logged (learning_rate, batch_size, etc.)
   - ✓ Metrics logged (train_loss, val_loss)
   - ✓ Artifacts saved (config, model checkpoints)

**Check files on cluster:**
```bash
# List saved models
ls -lh models/

# List output configs
ls -lh outputs/

# Check disk usage
du -sh models/ outputs/ mlruns/
```

---

## Test 2: Optuna Test (After Test 1 Works)

Once single training works, test Optuna with 2-3 trials.

### Option A: Without Dashboard (Simpler)

Just run optimization and check MLflow for individual training runs:

```bash
# Create test optimization script
cat > scripts/slurm_test_optuna.sh << 'EOF'
#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --output=./logs/slurm/test_optuna_%j.out
#SBATCH --job-name=test_optuna

source /home/ac137967/genSecSysId/venv/bin/activate

export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

python scripts/optimize_hyperparameters.py \
    --config /home/ac137967/configs/crnn_gen-sec_silverbox.yaml \
    --n-trials 3 \
    --study-name test-slurm-optuna \
    --storage sqlite:////home/ac137967/genSecSysId/optuna_test.db \
    --device cuda

exit $?
EOF

# Submit
sbatch scripts/slurm_test_optuna.sh
```

**Verify:** Check MLflow dashboard for 3 training runs under experiment name.

### Option B: With Dashboard (Full Setup)

If you want live Optuna visualization like your MLflow setup:

**1. Install Optuna dashboard:**
```bash
pip install optuna-dashboard
```

**2. Start dashboard on cluster (separate session):**
```bash
# SSH to cluster in separate terminal
ssh ac137967@cluster

# Start dashboard
optuna-dashboard sqlite:////home/ac137967/genSecSysId/optuna_test.db \
    --host 0.0.0.0 \
    --port 8080
```

**3. SSH tunnel from local machine:**
```bash
ssh -L 8080:cluster-node:8080 ac137967@cluster
```

**4. Open browser:**
```
http://localhost:8080
```

**5. Submit optimization job:**
```bash
sbatch scripts/slurm_test_optuna.sh
```

**6. Watch live in dashboard:**
- Trials appear as they complete
- See hyperparameter importance plots
- Compare trial performance

---

## Test 3: Multiple Workers (Final Test Before Production)

Test parallel optimization with 2-3 workers sharing a database.

### Setup

Edit [scripts/slurm_optimize.sh](../scripts/slurm_optimize.sh):

```bash
# Change to small job array
#SBATCH --array=0-2           # Only 3 workers

# Update partition
#SBATCH --partition=dgx

# Update module loads (lines 24-26) - comment out if not needed

# Update database storage (line 39)
DB_STORAGE="sqlite:////home/ac137967/genSecSysId/optuna_multi_test.db"
```

### Run Test

```bash
# Submit 3 workers, each doing 2 trials (6 total)
sbatch scripts/slurm_optimize.sh \
    /home/ac137967/configs/crnn_gen-sec_silverbox.yaml \
    multi-worker-test \
    2

# Monitor all workers
squeue -u ac137967

# Check output from each worker
tail -f logs/slurm/hpo_*_0.out  # Worker 0
tail -f logs/slurm/hpo_*_1.out  # Worker 1
tail -f logs/slurm/hpo_*_2.out  # Worker 2
```

### Verify

```bash
# Count trials in database
sqlite3 /home/ac137967/genSecSysId/optuna_multi_test.db \
    "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"
# Should show 6

# Check MLflow
# Should see 6 training runs with different hyperparameters
```

---

## Troubleshooting

### Issue: "Module not found"

**Solution:** Check module loads in SLURM script
```bash
# Try without modules first
# Comment out lines 49-51 in slurm_test_single_run.sh

# Or find what's available
module avail python
module avail cuda
```

### Issue: "CUDA not available"

**Solution:** Verify GPU allocation and CUDA setup
```bash
# In job output, check:
nvidia-smi  # Should show GPU
echo $CUDA_VISIBLE_DEVICES  # Should be "0" or similar

# May need to load CUDA module
module load cuda/11.8
```

### Issue: "MLflow connection failed"

**Solution:** Check network and URI
```bash
# Test from login node
curl http://your-mlflow-server:5000

# Test from compute node (in SLURM job)
srun --partition=dgx --gres=gpu:1 --pty bash
curl http://your-mlflow-server:5000

# If only login node can access, use file-based tracking:
export MLFLOW_TRACKING_URI="file:///home/ac137967/genSecSysId/mlruns"
```

### Issue: "Permission denied"

**Solution:** Check paths and permissions
```bash
# Ensure directories exist and are writable
mkdir -p /home/ac137967/genSecSysId/{logs/slurm,outputs,models}
chmod -R u+w /home/ac137967/genSecSysId/
```

### Issue: "Database is locked" (SQLite with multiple workers)

**Solution:** Use PostgreSQL or reduce workers
```bash
# For testing with SQLite, enable WAL mode:
sqlite3 optuna_test.db "PRAGMA journal_mode=WAL;"

# For production with >5 workers, use PostgreSQL (see SLURM_OPTIMIZATION.md)
```

---

## Quick Checklist

Before running full optimization, verify:

- [ ] **Test 1 passed:** Single training run completes on GPU
- [ ] **MLflow logs visible:** Parameters, metrics, and artifacts in dashboard
- [ ] **Test 2 passed:** Optuna single worker completes 2-3 trials
- [ ] **Test 3 passed:** Multiple workers coordinate via shared database
- [ ] **Disk space sufficient:** Check with `df -h /home/ac137967/` and `df -h /data/work/ac137967/`
- [ ] **Paths configured:** All paths in scripts match your cluster setup

---

## MLflow vs Optuna Dashboard

**You asked: Should I run Optuna dashboard like MLflow?**

### Differences

| Feature | MLflow | Optuna Dashboard |
|---------|--------|------------------|
| **Purpose** | Tracks individual training runs | Tracks optimization studies |
| **Shows** | Parameters, metrics over epochs, artifacts | Hyperparameter importance, trial comparison |
| **Need it?** | ✅ Essential for viewing training results | ⚠️ Optional, nice for live monitoring |
| **Data source** | MLflow tracking server/files | Optuna database (SQLite/PostgreSQL) |

### Recommendations

**Option 1: MLflow Only (Recommended for now)**
- ✅ You already have MLflow running
- ✅ Shows all training runs with their hyperparameters
- ✅ Sufficient for seeing optimization progress
- ✅ No additional setup needed
- ❌ No hyperparameter importance plots
- ❌ No live trial comparison

**Option 2: MLflow + Optuna Dashboard**
- ✅ Best visualization for optimization
- ✅ See hyperparameter importance live
- ✅ Compare trials side-by-side
- ❌ Need to run separate dashboard service
- ❌ Need another SSH tunnel/port

**My suggestion:** Start with MLflow only (Option 1). You can:
1. Check optimization progress by viewing training runs in MLflow
2. Add Optuna dashboard later if you want better visualization
3. Use `optuna` CLI commands to analyze studies without dashboard:
   ```bash
   # Best trial
   python -c "import optuna; study = optuna.load_study(study_name='...', storage='sqlite:///...'); print(study.best_trial)"
   
   # Hyperparameter importance
   optuna importance study_name --storage sqlite:///path/to/db.sqlite
   ```

---

## Next Steps

1. **Now:** Run Test 1 (single training run)
2. **After Test 1 works:** Run Test 2 (single worker optimization)
3. **After Test 2 works:** Run Test 3 (multi-worker optimization)
4. **After all pass:** Run production optimization with 10-20 workers

**Each test should take 5-15 minutes.** If any test fails, check the troubleshooting section before proceeding.
