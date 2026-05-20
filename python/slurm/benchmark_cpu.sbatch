#!/bin/bash
#SBATCH --job-name=genSecSysId-bench-cpu
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --output=/home/ac137967/Report/genSecSysId/bench_cpu_%j.out

# CPU half of the CPU-vs-GPU benchmark. Submit alongside benchmark_gpu.sbatch
# and compare the "=== Timing summary ===" blocks at the bottom of each output
# file, or filter MLflow by run name ("bench_cpu_*" / "bench_gpu_*").
#
# Keep the YAML config identical between the two scripts — only --device differs.

REPO_DIR="${HOME}/01_git/genSecSysId/python"
VENV_DIR="${HOME}/venv/py311"
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

echo "=== CPU info ==="
lscpu | grep -E "Model name|Socket|Core|Thread|CPU MHz" || true
echo

STAMP=$(date +%Y%m%d_%H%M%S)

START=$SECONDS
python scripts/train.py --config "${CONFIG_FILE}" --device cpu --run-name "bench_cpu_${STAMP}"
ELAPSED=$((SECONDS - START))

echo
echo "############################################################"
echo "# CPU run finished in ${ELAPSED}s                           "
echo "# Run name: bench_cpu_${STAMP}                              "
echo "############################################################"

exit 0
