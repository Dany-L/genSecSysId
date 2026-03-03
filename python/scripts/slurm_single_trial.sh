#!/bin/bash
#SBATCH --job-name=crnn_trial
#SBATCH --output=logs/slurm/trial_%A_%a.out
#SBATCH --error=logs/slurm/trial_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# Single Training Run with SLURM
# ============================================================================
# This script runs a single training run with specific hyperparameters.
# Useful for testing or running the best configuration from optimization.
#
# Usage:
#   sbatch slurm_single_trial.sh <config.yaml>
# ============================================================================

# Load modules
module purge
module load python/3.10
module load cuda/11.8
module load gcc/11.2.0

# Configuration
CONFIG_FILE=${1:-"~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml"}

# Activate virtual environment
source ${HOME}/venv/genSecSysId/bin/activate

# Print job info
echo "=============================================="
echo "SLURM Training Run"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Config: ${CONFIG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Check GPU
nvidia-smi
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Change to project directory
cd ${HOME}/genSecSysId/python

# Run training
python scripts/train.py --config ${CONFIG_FILE}

EXIT_CODE=$?

echo "=============================================="
echo "Training completed with exit code ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
