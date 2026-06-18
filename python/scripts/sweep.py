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


def _search_space_groups(sweep_cfg: dict) -> list:
    """Return ``search_space`` as a list of group dicts.

    Two YAML forms are accepted:
      - dict: ``search_space: {key: [...], ...}`` -> single group (legacy).
      - list of dicts: ``search_space: [{...}, {...}]`` -> tasks from each
        group are concatenated, not multiplied. Use this when some keys are
        only meaningful for a subset of runs (e.g. LMI-only params).
    """
    ss = sweep_cfg["search_space"]
    if isinstance(ss, dict):
        return [ss]
    if isinstance(ss, list):
        for i, g in enumerate(ss):
            if not isinstance(g, dict):
                raise ValueError(
                    f"search_space[{i}] must be a dict of param -> [values], "
                    f"got {type(g).__name__}"
                )
        return ss
    raise ValueError(
        f"search_space must be a dict or list of dicts, got {type(ss).__name__}"
    )


def enumerate_tasks(sweep_cfg: dict) -> list:
    """Return ordered list of (overrides, seed) for every array task."""
    groups = _search_space_groups(sweep_cfg)
    n_seeds = sweep_cfg.get("n_seeds", 1)
    tasks = []
    for group in groups:
        keys = list(group.keys())
        combos = list(itertools.product(*[group[k] for k in keys]))
        for combo in combos:
            overrides = dict(zip(keys, combo))
            for seed in range(n_seeds):
                tasks.append((overrides, seed))
    return tasks


def n_tasks(sweep_cfg: dict) -> int:
    groups = _search_space_groups(sweep_cfg)
    total = 0
    for group in groups:
        n = 1
        for v in group.values():
            n *= len(v)
        total += n
    return total * sweep_cfg.get("n_seeds", 1)


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


def build_sweep_tags(sweep_cfg: dict, task_id: int, seed: int, overrides: dict) -> dict:
    """MLflow tags identifying a sweep task.

    Set on the run *before* training starts so the metadata is visible even if
    the run fails (e.g. an infeasible initial parameter set). Values are
    stringified because MLflow tags must be strings.
    """
    return {
        "sweep_name": sweep_cfg.get("sweep_name", "sweep"),
        "sweep_task_id": str(task_id),
        "sweep_seed": str(seed),
        **{k: _fmt_value(v) for k, v in overrides.items()},
    }


def find_ood_sibling(train_path: str) -> "Path | None":
    """Return the OOD sibling dir if an 'id' component exists in *train_path*.

    E.g. .../Duffing/id  ->  .../Duffing/ood  (returned only if that dir exists).
    """
    p = Path(os.path.expanduser(train_path)).resolve()
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.lower() == "id":
            ood_parts = parts[:]
            ood_parts[i] = "ood"
            ood_path = Path(*ood_parts)
            if ood_path.is_dir():
                return ood_path
    return None


def run(cmd: list, label: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)
    print(" ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_task(task_id: int, total: int, sweep_cfg: dict, device: str,
             repo_dir: Path, py: str) -> None:
    """Execute the full train → evaluate → post_process pipeline for one task."""
    tasks = enumerate_tasks(sweep_cfg)
    overrides, seed = tasks[task_id]
    run_name = make_run_name(sweep_cfg.get("sweep_name", "sweep"), overrides, seed)

    print(f"Sweep task {task_id}/{total - 1}")
    print(f"  overrides : {overrides}")
    print(f"  seed      : {seed}")
    print(f"  run_name  : {run_name}")

    # Merge overrides into base config and write to a per-task temp file.
    base_cfg = _load_yaml(os.path.expanduser(sweep_cfg["base_config"]))
    merged_cfg = deep_merge(base_cfg, overrides)

    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_config = tmp_dir / f"sweep_task{task_id}_seed{seed}.yaml"
    with open(tmp_config, "w") as f:
        yaml.safe_dump(merged_cfg, f, default_flow_style=False, sort_keys=False)

    run_id_file = tmp_dir / f"sweep_run_id_task{task_id}.txt"

    data_root = merged_cfg.get("root_dir", ".")
    mlflow_cfg = merged_cfg.get("mlflow", {}) or {}

    # Pre-create the MLflow run and tag it with sweep metadata BEFORE training,
    # so the tags are visible even if train.py fails before it would have
    # created the run (e.g. an infeasible initial parameter set). train.py
    # attaches to this run via --run-id and marks it FAILED on error, keeping
    # the tags. If pre-creation fails (e.g. tracking server unreachable), we
    # fall back to letting train.py create the run and tag it afterwards.
    tags = build_sweep_tags(sweep_cfg, task_id, seed, overrides)
    run_id = None
    try:
        import mlflow
        if mlflow_cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        if mlflow_cfg.get("experiment_name"):
            mlflow.set_experiment(mlflow_cfg["experiment_name"])
        # Use a context manager so the run is always ended, even if set_tags()
        # raises. Otherwise a run left active in this process would make the
        # fallback start_run(run_id=...) below fail with "run already active",
        # and would also leave the run dangling as RUNNING on the server.
        with mlflow.start_run(run_name=run_name) as active_run:
            run_id = active_run.info.run_id
            mlflow.set_tags(tags)
        run_id_file.write_text(run_id)
        print(f"  pre-created and tagged run_id: {run_id}")
    except Exception as e:
        print(f"Warning: could not pre-create/tag MLflow run: {e}", flush=True)
        run_id = None

    # --- Step 1: train ---
    train_cmd = [
        py, str(repo_dir / "scripts" / "train.py"),
        "--config", str(tmp_config),
        "--seed", str(seed),
        "--device", device,
        "--run-name", run_name,
    ]
    if run_id is not None:
        # Attach to the run we already created and tagged above.
        train_cmd += ["--run-id", run_id]
    else:
        # Fallback: let train.py create the run; recover the id afterwards.
        train_cmd += ["--run-id-out", str(run_id_file)]
    run(train_cmd, "Step 1/3: train.py")

    if run_id is None:
        # Fallback path: train.py owns the run. Recover the id and tag now
        # (best-effort — tags are missing if train.py crashed before writing).
        if not run_id_file.exists() or not run_id_file.read_text().strip():
            print("ERROR: train.py did not write a run_id.", file=sys.stderr)
            sys.exit(1)
        run_id = run_id_file.read_text().strip()
        try:
            import mlflow
            if mlflow_cfg.get("tracking_uri"):
                mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tags(tags)
        except Exception as e:
            print(f"Warning: could not set MLflow sweep tags: {e}", flush=True)
    print(f"  run_id: {run_id}")

    # --- Step 2: evaluate (ID) ---
    run(
        [
            py, str(repo_dir / "scripts" / "evaluate.py"),
            "--run-id", run_id,
            "--data-root", data_root,
        ],
        "Step 2/3: evaluate.py (ID)",
    )

    # --- Step 2b: evaluate (OOD) — only if an ood/ sibling exists ---
    train_path = merged_cfg.get("data", {}).get("train_path", "")
    ood_path = find_ood_sibling(train_path) if train_path else None
    if ood_path is not None:
        print(f"  OOD data found at {ood_path} — running OOD evaluation", flush=True)
        run(
            [
                py, str(repo_dir / "scripts" / "evaluate.py"),
                "--run-id", run_id,
                "--data-root", data_root,
                "--test-data", str(ood_path),
            ],
            "Step 2b/3: evaluate.py (OOD)",
        )
    else:
        print(f"  No ood/ sibling found for '{train_path}' — skipping OOD evaluation",
              flush=True)

    # --- Step 3: post_process ---
    post_cmd = [
        py, str(repo_dir / "scripts" / "post_process.py"),
        "--run-id", run_id,
        "--data-root", data_root,
    ]
    true_dynamics = sweep_cfg.get("true_dynamics")
    if true_dynamics:
        post_cmd += ["--true-dynamics", true_dynamics]
    # Optional extra args forwarded verbatim to post_process.py (e.g. to shrink
    # the regional-verification workload in debug/smoke runs):
    #   post_process_args: ["--rv-num-trajectories", "2", "--rv-horizon", "50"]
    for extra in sweep_cfg.get("post_process_args", []):
        post_cmd.append(str(extra))
    run(post_cmd, "Step 3/3: post_process.py")

    print(f"\nTask {task_id} complete — run_id: {run_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", required=True,
                        help="Path to sweep YAML (e.g. configs/sweep_duffing.yaml)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="0-based task index to run (set to $SLURM_ARRAY_TASK_ID "
                             "by sbatch). Omit to run all tasks sequentially on this "
                             "machine — useful for local debugging or non-SLURM clusters.")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps", "auto"],
                        help="Training device (default: cuda)")
    parser.add_argument("--count", action="store_true",
                        help="Print total task count and exit (used by submit_sweep.sh)")
    args = parser.parse_args()

    sweep_cfg = _load_yaml(args.sweep_config)

    if args.count:
        print(n_tasks(sweep_cfg))
        return

    tasks = enumerate_tasks(sweep_cfg)
    total = len(tasks)
    repo_dir = Path(__file__).resolve().parent.parent
    py = sys.executable

    if args.task_id is None:
        # No SLURM array task id — run all tasks sequentially on this machine.
        print(f"No --task-id given: running all {total} task(s) sequentially "
              f"on device={args.device}", flush=True)
        for task_id in range(total):
            run_task(task_id, total, sweep_cfg, args.device, repo_dir, py)
        print(f"\nSweep complete ({total} task(s))", flush=True)
        return

    if args.task_id >= total:
        print(
            f"ERROR: --task-id {args.task_id} is out of range "
            f"(total tasks: {total})",
            file=sys.stderr,
        )
        sys.exit(1)

    run_task(args.task_id, total, sweep_cfg, args.device, repo_dir, py)


if __name__ == "__main__":
    main()
