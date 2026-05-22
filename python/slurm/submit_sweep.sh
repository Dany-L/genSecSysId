#!/bin/bash
# Compute the total number of sweep tasks from the sweep config, then submit
# sweep.sbatch as a SLURM array job with the correct range.
#
# Usage (from ~/01_git/genSecSysId/python):
#   bash slurm/submit_sweep.sh [configs/sweep_duffing.yaml]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Resolve to absolute path so the heredoc and sbatch call both work regardless
# of the caller's working directory.
SWEEP_CONFIG="$(realpath "${1:-${REPO_DIR}/configs/sweep_duffing.yaml}")"

# Activate the venv only if not already in one (so the script is safe to call
# from both interactive shells and from inside another job).
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "${HOME}/venv/genSecSysId/bin/activate"
fi

# Compute total number of tasks (product of all param list lengths × n_seeds).
N=$(python - <<EOF
import yaml, math
with open("${SWEEP_CONFIG}") as f:
    cfg = yaml.safe_load(f)
ss = cfg["search_space"]
n = math.prod(len(v) for v in ss.values())
print(n * cfg.get("n_seeds", 1) - 1)
EOF
)

echo "Sweep config : ${SWEEP_CONFIG}"
echo "Total tasks  : $((N + 1))  (array 0–${N})"
echo

sbatch \
    --array="0-${N}" \
    "${REPO_DIR}/slurm/sweep.sbatch" \
    "${SWEEP_CONFIG}"
