# Hyperparameter Optimization on Remote GPU

This guide explains how to run hyperparameter optimization on a remote server with GPU access.

## Quick Start (Local GPU)

```bash
# 1. Enable GPU in your config
# Edit ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml:
# training:
#   device: "cuda"  # or "auto"

# 2. Run optimization locally
python scripts/optimize_hyperparameters.py \
    --config ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    --n-trials 50 \
    --study-name silverbox-gpu-opt
```

## Remote GPU Setup

### Option 1: SSH + tmux/screen (Simple)

```bash
# 1. Copy code to remote server
rsync -avz --exclude='mlruns' --exclude='*.pt' --exclude='__pycache__' \
    ~/Documents/01_Git/01_promotion/genSecSysId/python/ \
    user@remote-gpu-server:~/genSecSysId/

# Copy data
rsync -avz ~/genSecSysId-Data/ \
    user@remote-gpu-server:~/genSecSysId-Data/

# 2. SSH into remote server
ssh user@remote-gpu-server

# 3. Setup environment
cd ~/genSecSysId/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install optuna optuna-dashboard

# 4. Start tmux session (persists after disconnect)
tmux new -s optuna

# 5. Run optimization
python scripts/optimize_hyperparameters.py \
    --config ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    --n-trials 100 \
    --study-name silverbox-gpu-opt \
    --storage sqlite:///~/genSecSysId-Data/optuna.db

# 6. Detach tmux: Ctrl+B, then D
# 7. Reattach anytime: tmux attach -t optuna
```

### Option 2: Optuna Dashboard (Recommended)

```bash
# On remote server - Terminal 1: Run optimization with database storage
python scripts/optimize_hyperparameters.py \
    --config ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    --n-trials 100 \
    --storage sqlite:////home/user/genSecSysId-Data/optuna.db \
    --study-name silverbox-gpu-opt

# On remote server - Terminal 2: Start dashboard
optuna-dashboard sqlite:////home/user/genSecSysId-Data/optuna.db \
    --host 0.0.0.0 --port 8080

# On local machine: Forward port
ssh -L 8080:localhost:8080 user@remote-gpu-server

# Open browser: http://localhost:8080
```

### Option 3: Distributed Optimization (Multiple GPUs/Servers)

```bash
# Setup PostgreSQL or MySQL as shared storage
# Example with PostgreSQL:

# 1. Start database (can be on remote server or cloud)
# Create database: optuna_studies

# 2. On each worker (can be different servers with GPUs):
python scripts/optimize_hyperparameters.py \
    --config ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml \
    --n-trials 100 \
    --storage postgresql://user:password@db-host:5432/optuna_studies \
    --study-name silverbox-distributed

# Optuna automatically parallelizes across workers!
```

## Monitoring GPU Usage

```bash
# Check GPU availability
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check if PyTorch sees GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Cloud GPU Services

### AWS EC2 (p2.xlarge or p3.2xlarge)

```bash
# 1. Launch EC2 instance with Deep Learning AMI
# 2. SSH with key: ssh -i key.pem ubuntu@ec2-x-x-x-x.compute.amazonaws.com
# 3. Environment is pre-configured, just install package:
pip install -e .
pip install optuna

# 4. Run optimization (use tmux!)
tmux new -s optuna
python scripts/optimize_hyperparameters.py --config ... --n-trials 100
```

### Google Colab (Free GPU)

```python
# In Colab notebook:
!git clone https://github.com/your-repo/genSecSysId.git
%cd genSecSysId/python
!pip install -r requirements.txt
!pip install optuna

# Upload config and data to Colab or mount Google Drive

# Run optimization
!python scripts/optimize_hyperparameters.py \
    --config /content/drive/MyDrive/configs/crnn_gen-sec_silverbox.yaml \
    --n-trials 50
```

### Vast.ai / Lambda Labs (Cheap GPU rental)

```bash
# Similar to SSH method above
# These provide bare GPU instances - follow Option 1 setup
```

## Syncing Results Back

```bash
# After optimization completes, sync results back
rsync -avz user@remote-gpu-server:~/genSecSysId-Data/optuna.db \
    ~/genSecSysId-Data/

rsync -avz user@remote-gpu-server:~/genSecSysId/python/results/ \
    ~/Documents/01_Git/01_promotion/genSecSysId/python/results/
```

## Performance Tips

1. **Batch size**: Increase on GPU (16-64 vs 4 on CPU)
2. **Parallel trials**: Run multiple workers on same GPU:
   ```bash
   # Terminal 1
   python scripts/optimize_hyperparameters.py --config ... --storage sqlite:///optuna.db
   # Terminal 2 (shares same database)
   python scripts/optimize_hyperparameters.py --config ... --storage sqlite:///optuna.db
   ```

3. **Pruning**: Enabled by default - stops bad trials early

4. **Resume interrupted optimization**:
   ```bash
   # Same command - Optuna resumes from database
   python scripts/optimize_hyperparameters.py \
       --config ... \
       --storage sqlite:///optuna.db \
       --study-name silverbox-gpu-opt  # Same study name
   ```

## Expected Speedup

Based on your model (SimpleLure with constrained RNN):

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Forward pass | 100ms | 15ms | ~6-7x |
| Backward pass | 150ms | 20ms | ~7-8x |
| SDP solver | 500ms | 500ms | 1x (CPU-bound) |
| **Total per epoch** | ~750ms | ~535ms | ~1.4x |

With larger batches on GPU (32 vs 4):
- **Total speedup: ~3-4x** (considering SDP overhead)

If SDP is disabled or infrequent:
- **Total speedup: ~7-10x**

## Hyperparameters Being Optimized

The script optimizes:
- `learning_rate`: 1e-4 to 1e-2 (log scale)
- `weight_decay`: 1e-5 to 1e-2 (log scale)
- `batch_size`: [8, 16, 32, 64]
- `nw` (state dimension): 5 to 20
- `nx` (state dimension): 5 to 20
- `regularization_weight`: 1e-4 to 1e-1 (log scale)
- `learn_L`: True/False (for CRNN)

To add more hyperparameters, edit the `objective()` function in `scripts/optimize_hyperparameters.py`.

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size in trial suggestions:
batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])  # Remove 32, 64
```

### Slow trials

```python
# Reduce max_epochs in objective function:
config.training.max_epochs = 50  # Instead of 100
```

### Database locked (SQLite)

```bash
# Use PostgreSQL for multiple workers instead of SQLite
```
