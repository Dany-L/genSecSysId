#!/bin/bash
#SBATCH --job-name=genSecSysId-bench-gpu
#SBATCH --partition=gpu-dev
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --output=/home/ac137967/Report/genSecSysId/bench_gpu_%j.out

# GPU half of the CPU-vs-GPU benchmark. Submit alongside benchmark_cpu.sbatch
# and compare the "=== Timing summary ===" blocks at the bottom of each output
# file, or filter MLflow by run name ("bench_cpu_*" / "bench_gpu_*").
#
# Keep the YAML config identical between the two scripts — only --device differs.

REPO_DIR="${HOME}/01_git/genSecSysId/python"
VENV_DIR="${HOME}/venv/genSecSysId"
DATA_ROOT="/data/work/${USER}/Duffing"
CONFIG_FILE="${DATA_ROOT}/config/crnn_gen-sec_duffing.yaml"

mkdir -p ./Report/genSecSysId

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"
cd "${REPO_DIR}"

echo "=== Job info ==="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Config:  ${CONFIG_FILE}"
echo

echo "=== nvidia-smi ==="
nvidia-smi
echo

echo "=== Torch / CUDA ==="
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
echo

STAMP=$(date +%Y%m%d_%H%M%S)

START=$SECONDS
python scripts/train.py --config "${CONFIG_FILE}" --device cuda --run-name "bench_gpu_${STAMP}"
ELAPSED=$((SECONDS - START))

echo
echo "############################################################"
echo "# GPU run finished in ${ELAPSED}s                           "
echo "# Run name: bench_gpu_${STAMP}                              "
echo "############################################################"

exit 0
