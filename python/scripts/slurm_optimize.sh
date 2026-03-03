#!/bin/bash
#SBATCH --job-name=crnn_hpo
#SBATCH --output=logs/slurm/hpo_%A_%a.out
#SBATCH --error=logs/slurm/hpo_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-9  # Run 10 parallel workers

# ============================================================================
# SLURM Hyperparameter Optimization with Optuna
# ============================================================================
# This script runs multiple parallel SLURM jobs that all work on the same
# Optuna study via a shared database. Each job is a worker that pulls trials
# from the study, trains the model, and reports results back.
#
# Usage:
#   sbatch slurm_optimize.sh <config.yaml> <study-name> <trials-per-worker>
#
# Example:
#   sbatch slurm_optimize.sh ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml silverbox-gpu 10
# ============================================================================

# Load modules (adjust for your cluster)
module purge
module load python/3.10
module load cuda/11.8
module load gcc/11.2.0

# Configuration from command line arguments
CONFIG_FILE=${1:-"~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml"}
STUDY_NAME=${2:-"silverbox-slurm-${SLURM_JOB_ID}"}
TRIALS_PER_WORKER=${3:-10}

# Database configuration (shared across all workers)
# Option 1: SQLite on shared filesystem (simple but slower)
DB_STORAGE="sqlite:////cluster/scratch/${USER}/optuna_studies/${STUDY_NAME}.db"

# Option 2: PostgreSQL/MySQL (recommended for production)
# DB_STORAGE="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:5432/optuna"
# DB_STORAGE="mysql://${DB_USER}:${DB_PASS}@${DB_HOST}:3306/optuna"

# Create directories
mkdir -p logs/slurm
mkdir -p /cluster/scratch/${USER}/optuna_studies

# Activate virtual environment
source ${HOME}/venv/genSecSysId/bin/activate

# Print job info
echo "=============================================="
echo "SLURM Hyperparameter Optimization"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Config: ${CONFIG_FILE}"
echo "Study name: ${STUDY_NAME}"
echo "Trials per worker: ${TRIALS_PER_WORKER}"
echo "Database: ${DB_STORAGE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Check GPU availability
nvidia-smi || echo "No GPU detected"
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"

# Change to project directory
cd ${HOME}/genSecSysId/python

# Run optimization
# Each SLURM array task runs as an independent worker
# All workers share the same study via database
python scripts/optimize_hyperparameters.py \
    --config ${CONFIG_FILE} \
    --n-trials ${TRIALS_PER_WORKER} \
    --study-name ${STUDY_NAME} \
    --storage ${DB_STORAGE}

EXIT_CODE=$?

echo "=============================================="
echo "Worker ${SLURM_ARRAY_TASK_ID} completed with exit code ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
