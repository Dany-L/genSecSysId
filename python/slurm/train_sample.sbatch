#!/bin/bash
#SBATCH --job-name=genSecSysId-sample
#SBATCH --partition=gpu-dev
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --output=/home/ac137967/Report/genSecSysId/output_%j.out

# -----------------------------------------------------------------------------
# Paths — adjust if your cluster layout differs.
# Matches the SysIdBench template: code under $HOME/01_git, data under /data/work.
# -----------------------------------------------------------------------------
REPO_DIR="${HOME}/01_git/genSecSysId/python"
VENV_DIR="${HOME}/venv/genSecSysId"
DATA_ROOT="/data/work/${USER}/Duffing"
CONFIG_FILE="${DATA_ROOT}/config/crnn_gen-sec_duffing.yaml"

# Make sure the output directory exists (sbatch does not create it).
mkdir -p ./Report/genSecSysId

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
source "${VENV_DIR}/bin/activate"

# Help PyTorch find the installed sysid package.
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"

# Run from the repo so relative paths (./Report, ./mlruns) land in REPO_DIR.
cd "${REPO_DIR}"

# -----------------------------------------------------------------------------
# Diagnostics — confirm GPU visibility before launching the run.
# -----------------------------------------------------------------------------
echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Venv:      ${VENV_DIR}"
echo "Config:    ${CONFIG_FILE}"
echo

echo "=== nvidia-smi ==="
nvidia-smi
echo

echo "=== Torch / CUDA ==="
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
echo

# -----------------------------------------------------------------------------
# Training — --device cuda overrides the YAML so the same config works on CPU + GPU.
# -----------------------------------------------------------------------------
python scripts/train.py --config "${CONFIG_FILE}" --device cuda

exit 0
