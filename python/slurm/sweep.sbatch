#!/bin/bash
#SBATCH --job-name=genSecSysId-sweep
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --output=/home/ac137967/Report/genSecSysId/sweep_%A_%a.out

# Usage (via submit_sweep.sh — sets --array automatically):
#   bash slurm/submit_sweep.sh configs/sweep_duffing.yaml
#
# The array range is computed by submit_sweep.sh from the sweep config so you
# never have to update it here manually.

set -euo pipefail

REPO_DIR="${HOME}/01_git/genSecSysId/python"
VENV_DIR="${HOME}/venv/genSecSysId"

# Resolve to absolute path *before* cd, so relative paths from the caller work.
SWEEP_CONFIG="$(realpath "${1:-${REPO_DIR}/configs/sweep_duffing.yaml}")"

export TMPDIR="/data/work/${USER}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
cd "${REPO_DIR}"

echo "=== Sweep job info ==="
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID:      ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "Sweep config: ${SWEEP_CONFIG}"
echo

nvidia-smi
echo

python scripts/sweep.py \
    --sweep-config "${SWEEP_CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --device cuda
