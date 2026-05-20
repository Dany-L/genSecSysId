#!/bin/bash
#SBATCH --job-name=genSecSysId-pipeline
#SBATCH --partition=gpu-dev
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --output=/home/ac137967/Report/genSecSysId/pipeline_%j.out

# End-to-end pipeline: train.py -> evaluate.py -> post_process.py, with the
# MLflow run_id from training threaded through to the downstream scripts.
#
# Bail on any step's failure so we don't, e.g., run post_process against a
# model that never trained.
set -euo pipefail

# -----------------------------------------------------------------------------
# Paths — adjust if your cluster layout differs.
# -----------------------------------------------------------------------------
REPO_DIR="${HOME}/01_git/genSecSysId/python"
VENV_DIR="${HOME}/venv/genSecSysId"
DATA_ROOT="/data/work/${USER}/Duffing"
CONFIG_FILE="${DATA_ROOT}/config/crnn_gen-sec_duffing.yaml"

# Must match `root_dir` and `model.model_type` inside ${CONFIG_FILE}, since
# train.py derives the checkpoint directory from those.
RUN_ROOT_DIR="${DATA_ROOT}"
MODEL_TYPE="crnn"

# Test/train data directories handed to evaluate.py and post_process.py.
# Evaluate accepts either a parent folder containing test/ or the test/ folder
# directly; post_process needs both train and test split folders.
DATA_DIR="${DATA_ROOT}/data"
TEST_DATA="${DATA_DIR}"
POST_TEST_DATA="${DATA_DIR}/test"
POST_TRAIN_DATA="${DATA_DIR}/train"

# Pass `duffing` (or another registered name) to enable ground-truth comparison
# in post_process.py's regional-verification step. Leave empty to skip.
TRUE_DYNAMICS="duffing"

mkdir -p ./Report/genSecSysId

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
cd "${REPO_DIR}"

echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Config:    ${CONFIG_FILE}"
echo "Run root:  ${RUN_ROOT_DIR}"
echo

echo "=== nvidia-smi ==="
nvidia-smi
echo

# A scratch file the training script writes the run_id into. Scoped to the job
# so concurrent pipeline jobs don't clobber each other.
RUN_ID_FILE="${TMPDIR:-/tmp}/genSecSysId_run_id_${SLURM_JOB_ID}.txt"
trap 'rm -f "${RUN_ID_FILE}"' EXIT

# -----------------------------------------------------------------------------
# Step 1 — train
# -----------------------------------------------------------------------------
echo
echo "############################################################"
echo "# Step 1/3: train.py                                        "
echo "############################################################"
python scripts/train.py \
    --config "${CONFIG_FILE}" \
    --device cuda \
    --run-id-out "${RUN_ID_FILE}"

if [[ ! -s "${RUN_ID_FILE}" ]]; then
    echo "ERROR: train.py did not write a run_id to ${RUN_ID_FILE}" >&2
    exit 1
fi
RUN_ID=$(cat "${RUN_ID_FILE}")
echo "Captured RUN_ID=${RUN_ID}"

# Derive the checkpoint path from the same layout train.py uses:
# <root_dir>/models/<model_type>/<run_id>/best_model.pt
BEST_MODEL="${RUN_ROOT_DIR}/models/${MODEL_TYPE}/${RUN_ID}/best_model.pt"
if [[ ! -f "${BEST_MODEL}" ]]; then
    echo "ERROR: expected checkpoint not found at ${BEST_MODEL}" >&2
    echo "Check that RUN_ROOT_DIR and MODEL_TYPE above match your YAML." >&2
    exit 1
fi
echo "Best model: ${BEST_MODEL}"

# -----------------------------------------------------------------------------
# Step 2 — evaluate
# -----------------------------------------------------------------------------
echo
echo "############################################################"
echo "# Step 2/3: evaluate.py                                     "
echo "############################################################"
python scripts/evaluate.py \
    --run-id "${RUN_ID}" \
    --test-data "${TEST_DATA}" \
    --data-root "${DATA_ROOT}"

# -----------------------------------------------------------------------------
# Step 3 — post_process
# -----------------------------------------------------------------------------
echo
echo "############################################################"
echo "# Step 3/3: post_process.py                                 "
echo "############################################################"
POST_ARGS=(
    --run-id "${RUN_ID}"
    --data-root "${DATA_ROOT}"
)
if [[ -n "${TRUE_DYNAMICS}" ]]; then
    POST_ARGS+=(--true-dynamics "${TRUE_DYNAMICS}")
fi
python scripts/post_process.py "${POST_ARGS[@]}"

echo
echo "############################################################"
echo "# Pipeline complete                                         "
echo "# RUN_ID: ${RUN_ID}                                         "
echo "############################################################"

exit 0
