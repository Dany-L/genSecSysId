#!/bin/bash
#SBATCH --partition dgx
#SBATCH --time 02:00:00
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --mem 16GB
#SBATCH --gres gpu:1
#SBATCH --gpus 1
#SBATCH --job-name test_train
#SBATCH --chdir /home/ac137967/genSecSysId
#SBATCH --output /home/ac137967/genSecSysId/logs/slurm/test_run_%j.out
#SBATCH --error /home/ac137967/genSecSysId/logs/slurm/test_run_%j.err

# ============================================================================
# Test Single Training Run on SLURM
# ============================================================================
# Purpose: Verify GPU access, MLflow logging, and artifact storage
# 
# Usage:
#   sbatch scripts/slurm_test_single_run.sh <config_path>
#
# Example:
#   sbatch scripts/slurm_test_single_run.sh \
#       ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml
# ============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
CONFIG_PATH="$1"

if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: No config file provided"
    echo "Usage: sbatch $0 <config_path>"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================

echo "Setting up environment..."

# Load modules (adjust for your cluster)
# module purge
# module load python/3.10  # Uncomment and adjust if needed
# module load cuda/11.8     # Uncomment and adjust if needed

# Activate virtual environment
# First, try the standard location
if [ -f "/home/ac137967/genSecSysId/venv/bin/activate" ]; then
    echo "Using venv in /home/ac137967/genSecSysId/venv"
    source /home/ac137967/genSecSysId/venv/bin/activate
elif [ -f "/home/ac137967/venv/genSecSysId/bin/activate" ]; then
    echo "Using venv in /home/ac137967/venv/genSecSysId"
    source /home/ac137967/venv/genSecSysId/bin/activate
else
    echo "ERROR: Could not find virtual environment"
    echo "Checked:"
    echo "  - /home/ac137967/genSecSysId/venv/bin/activate"
    echo "  - /home/ac137967/venv/genSecSysId/bin/activate"
    exit 1
fi

# Verify Python and GPU
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# Check CUDA availability in PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

echo ""

# ============================================================================
# MLflow Configuration
# ============================================================================

# Set MLflow tracking URI to your remote server
# Replace with your actual MLflow server address
export MLFLOW_TRACKING_URI="http://mlflowui.informatik.uni-stuttgart.de/"
# Or if using SSH tunnel:
# export MLFLOW_TRACKING_URI="http://localhost:5000"

# Optionally set experiment name
export MLFLOW_EXPERIMENT_NAME="slurm-test"

echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
echo "MLflow Experiment: $MLFLOW_EXPERIMENT_NAME"

# Test MLflow connection
echo "Testing MLflow connection..."
python -c "import mlflow; mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI'); print(f'MLflow tracking URI: {mlflow.get_tracking_uri()}'); exps = mlflow.search_experiments(); print(f'Found {len(exps)} experiments')"

echo ""

# ============================================================================
# Training
# ============================================================================

echo "Starting training..."
echo "Config: $CONFIG_PATH"
echo ""

# Run training with GPU and reduced epochs for quick test
python scripts/train.py \
    --config "$CONFIG_PATH" \
    --device cuda \
    --max-epochs 10 \
    --log-mlflow

TRAIN_EXIT_CODE=$?

echo ""
echo "Training completed with exit code: $TRAIN_EXIT_CODE"

# ============================================================================
# Post-Training Verification
# ============================================================================

echo ""
echo "=========================================="
echo "Verifying outputs..."
echo "=========================================="

# Check if artifacts were created
if [ -d "outputs" ]; then
    echo "✓ outputs/ directory exists"
    ls -lh outputs/ | tail -n 10
else
    echo "✗ outputs/ directory not found"
fi

echo ""

if [ -d "models" ]; then
    echo "✓ models/ directory exists"
    ls -lh models/ | tail -n 10
else
    echo "✗ models/ directory not found"
fi

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Duration: $SECONDS seconds"
echo "=========================================="

exit $TRAIN_EXIT_CODE
