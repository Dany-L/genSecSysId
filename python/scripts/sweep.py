"""Run one (param-combo, seed) task from a hyperparameter sweep.

Intended to be called by slurm/sweep.sbatch via a SLURM array job.  Each
array task maps to a unique (parameter combination, seed) pair; the full
train → evaluate → post_process pipeline is executed for that pair.

Usage (direct):
    python scripts/sweep.py \\
        --sweep-config configs/sweep_duffing.yaml \\
        --task-id 0 \\
        --device cuda

Usage (SLURM array, via submit_sweep.sh):
    bash slurm/submit_sweep.sh configs/sweep_duffing.yaml
"""

import argparse
import copy
import itertools
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def enumerate_tasks(sweep_cfg: dict) -> list:
    """Return ordered list of (overrides, seed) for every array task."""
    search_space = sweep_cfg["search_space"]
    n_seeds = sweep_cfg.get("n_seeds", 1)
    keys = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))
    tasks = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        for seed in range(n_seeds):
            tasks.append((overrides, seed))
    return tasks


def n_tasks(sweep_cfg: dict) -> int:
    search_space = sweep_cfg["search_space"]
    n = 1
    for v in search_space.values():
        n *= len(v)
    return n * sweep_cfg.get("n_seeds", 1)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Apply dot-notation overrides to a nested dict, returning a deep copy."""
    result = copy.deepcopy(base)
    for dotkey, value in overrides.items():
        keys = dotkey.split(".")
        node = result
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return result


def _fmt_value(v) -> str:
    """Format a param value for use in a run name (no spaces, short floats)."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def make_run_name(sweep_name: str, overrides: dict, seed: int) -> str:
    parts = [sweep_name]
    for dotkey, value in overrides.items():
        short_key = dotkey.split(".")[-1]
        parts.append(f"{short_key}{_fmt_value(value)}")
    parts.append(f"s{seed}")
    return "-".join(parts)


def run(cmd: list, label: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)
    print(" ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", required=True,
                        help="Path to sweep YAML (e.g. configs/sweep_duffing.yaml)")
    parser.add_argument("--task-id", type=int, required=True,
                        help="0-based task index (set to $SLURM_ARRAY_TASK_ID by sbatch)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps", "auto"],
                        help="Training device (default: cuda)")
    args = parser.parse_args()

    sweep_cfg = _load_yaml(args.sweep_config)
    tasks = enumerate_tasks(sweep_cfg)
    total = len(tasks)

    if args.task_id >= total:
        print(
            f"ERROR: --task-id {args.task_id} is out of range "
            f"(total tasks: {total})",
            file=sys.stderr,
        )
        sys.exit(1)

    overrides, seed = tasks[args.task_id]
    run_name = make_run_name(sweep_cfg.get("sweep_name", "sweep"), overrides, seed)

    print(f"Sweep task {args.task_id}/{total - 1}")
    print(f"  overrides : {overrides}")
    print(f"  seed      : {seed}")
    print(f"  run_name  : {run_name}")

    # Merge overrides into base config and write to a per-task temp file.
    base_cfg = _load_yaml(os.path.expanduser(sweep_cfg["base_config"]))
    merged_cfg = deep_merge(base_cfg, overrides)

    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_config = tmp_dir / f"sweep_task{args.task_id}_seed{seed}.yaml"
    with open(tmp_config, "w") as f:
        yaml.safe_dump(merged_cfg, f, default_flow_style=False, sort_keys=False)

    run_id_file = tmp_dir / f"sweep_run_id_task{args.task_id}.txt"

    repo_dir = Path(__file__).resolve().parent.parent
    py = sys.executable
    data_root = merged_cfg.get("root_dir", ".")

    # --- Step 1: train ---
    run(
        [
            py, str(repo_dir / "scripts" / "train.py"),
            "--config", str(tmp_config),
            "--seed", str(seed),
            "--device", args.device,
            "--run-name", run_name,
            "--run-id-out", str(run_id_file),
        ],
        "Step 1/3: train.py",
    )

    if not run_id_file.exists() or not run_id_file.read_text().strip():
        print("ERROR: train.py did not write a run_id.", file=sys.stderr)
        sys.exit(1)
    run_id = run_id_file.read_text().strip()
    print(f"  run_id: {run_id}")

    # Tag the MLflow run with sweep metadata so runs can be grouped/filtered.
    try:
        import mlflow
        mlflow.set_tracking_uri(merged_cfg.get("mlflow", {}).get("tracking_uri"))
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags({
                "sweep_name": sweep_cfg.get("sweep_name", "sweep"),
                "sweep_task_id": str(args.task_id),
                "sweep_seed": str(seed),
                **{k: _fmt_value(v) for k, v in overrides.items()},
            })
    except Exception as e:
        print(f"Warning: could not set MLflow sweep tags: {e}", flush=True)

    # --- Step 2: evaluate ---
    run(
        [
            py, str(repo_dir / "scripts" / "evaluate.py"),
            "--run-id", run_id,
            "--data-root", data_root,
        ],
        "Step 2/3: evaluate.py",
    )

    # --- Step 3: post_process ---
    post_cmd = [
        py, str(repo_dir / "scripts" / "post_process.py"),
        "--run-id", run_id,
        "--data-root", data_root,
    ]
    true_dynamics = sweep_cfg.get("true_dynamics")
    if true_dynamics:
        post_cmd += ["--true-dynamics", true_dynamics]
    run(post_cmd, "Step 3/3: post_process.py")

    print(f"\nTask {args.task_id} complete — run_id: {run_id}", flush=True)


if __name__ == "__main__":
    main()
