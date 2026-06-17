"""Smoke tests for the sweep pipeline.

Schema tests (fast, no subprocesses):
  - Single-group dict and multi-group list counting / enumeration
  - Tag building
  - find_ood_sibling path resolution
  - Error surface

Integration test (slow, full subprocess pipeline):
  - test_sweep_full_pipeline_single_task: runs sweep.py --task-id 0 against
    synthetic data and asserts the model + evaluation artefacts exist.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_PY / "scripts"


def _load_script_module(name: str, path: Path):
    """Import a scripts/ module by file path without mutating sys.path.

    sys.path.insert would persist for the whole pytest session and give
    scripts/ priority over every other import (its modules have generic names
    like train/compare/evaluate), so we load by spec instead.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_script_module("sweep", SCRIPTS / "sweep.py")


def _write_sweep(tmp_path: Path, search_space, n_seeds: int = 1) -> Path:
    cfg = {
        "sweep_name": "test",
        "base_config": str(tmp_path / "dummy_base.yaml"),
        "n_seeds": n_seeds,
        "search_space": search_space,
    }
    path = tmp_path / "sweep.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _count_via_cli(sweep_path: Path) -> int:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "sweep.py"),
         "--sweep-config", str(sweep_path), "--count"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


# --- legacy single-group (dict) form ---------------------------------------

def test_count_single_group_matches_cartesian(tmp_path):
    """Dict form: count == product(list lengths) * n_seeds (legacy behavior)."""
    sweep_path = _write_sweep(
        tmp_path,
        search_space={"model.nw": [4, 8], "optimizer.learning_rate": [1e-3, 5e-3, 1e-2]},
        n_seeds=2,
    )
    assert _count_via_cli(sweep_path) == 2 * 3 * 2


def test_enumerate_single_group_is_cartesian(tmp_path):
    sweep_path = _write_sweep(
        tmp_path,
        search_space={"a": [1, 2], "b": ["x", "y"]},
        n_seeds=1,
    )
    cfg = sweep._load_yaml(str(sweep_path))
    tasks = sweep.enumerate_tasks(cfg)
    overrides = [ov for ov, _seed in tasks]
    assert {(o["a"], o["b"]) for o in overrides} == {(1, "x"), (1, "y"), (2, "x"), (2, "y")}


# --- new multi-group (list) form -------------------------------------------

def test_count_multi_group_is_sum_not_product(tmp_path):
    """List form: counts from each group are summed, not multiplied —
    that's the whole point of supporting multi-group."""
    sweep_path = _write_sweep(
        tmp_path,
        search_space=[
            {"training.use_custom_regularization": [True],
             "model.custom_params.learn_L": [True, False]},  # 2 combos
            {"training.use_custom_regularization": [False]},  # 1 combo
        ],
        n_seeds=3,
    )
    assert _count_via_cli(sweep_path) == (2 + 1) * 3


def test_enumerate_multi_group_dedups_irrelevant_keys(tmp_path):
    """The no-regularization group must NOT carry the LMI-only key, which is
    the actual user-visible dedup behavior (no duplicate runs that differ
    only in an ignored param)."""
    sweep_path = _write_sweep(
        tmp_path,
        search_space=[
            {"training.use_custom_regularization": [True],
             "model.custom_params.learn_L": [True, False]},
            {"training.use_custom_regularization": [False]},
        ],
        n_seeds=1,
    )
    cfg = sweep._load_yaml(str(sweep_path))
    tasks = sweep.enumerate_tasks(cfg)
    assert len(tasks) == 3

    # Group 1 tasks carry both keys; group 2 task carries only the reg flag.
    reg_keys_per_task = [set(ov.keys()) for ov, _ in tasks]
    assert reg_keys_per_task[0] == {"training.use_custom_regularization", "model.custom_params.learn_L"}
    assert reg_keys_per_task[1] == {"training.use_custom_regularization", "model.custom_params.learn_L"}
    assert reg_keys_per_task[2] == {"training.use_custom_regularization"}

    # And no task in the no-reg group is a duplicate that varies only by learn_L.
    no_reg_tasks = [ov for ov, _ in tasks if ov["training.use_custom_regularization"] is False]
    assert len(no_reg_tasks) == 1


def test_multi_group_seeds_replicate_within_each_group(tmp_path):
    """n_seeds applies inside every group, not just once across the union."""
    sweep_path = _write_sweep(
        tmp_path,
        search_space=[{"a": [1]}, {"b": [2]}],
        n_seeds=4,
    )
    cfg = sweep._load_yaml(str(sweep_path))
    tasks = sweep.enumerate_tasks(cfg)
    assert len(tasks) == 2 * 4
    seeds_for_a = sorted(s for ov, s in tasks if "a" in ov)
    seeds_for_b = sorted(s for ov, s in tasks if "b" in ov)
    assert seeds_for_a == [0, 1, 2, 3]
    assert seeds_for_b == [0, 1, 2, 3]


# --- sweep tags (set before training so they survive a failed run) ---------

def test_build_sweep_tags_includes_metadata_and_stringifies():
    """Tags carry sweep identity; ids/seeds are stringified for MLflow."""
    tags = sweep.build_sweep_tags(
        {"sweep_name": "duffing"},
        task_id=7,
        seed=2,
        overrides={"optimizer.learning_rate": 5e-3, "model.model_type": "crnn"},
    )
    assert tags["sweep_name"] == "duffing"
    assert tags["sweep_task_id"] == "7"
    assert tags["sweep_seed"] == "2"
    # Overrides are carried through under their full dot-keys, floats shortened.
    assert tags["optimizer.learning_rate"] == "0.005"
    assert tags["model.model_type"] == "crnn"


def test_build_sweep_tags_defaults_sweep_name():
    """Missing sweep_name falls back to the same default used elsewhere."""
    tags = sweep.build_sweep_tags({}, task_id=0, seed=0, overrides={})
    assert tags["sweep_name"] == "sweep"


# --- error surface ---------------------------------------------------------

def test_invalid_search_space_type_rejected(tmp_path):
    """A scalar search_space is a clear user error; --count must exit non-zero
    with a message rather than silently doing the wrong thing."""
    sweep_path = _write_sweep(tmp_path, search_space=42)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "sweep.py"),
         "--sweep-config", str(sweep_path), "--count"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "search_space" in (result.stderr + result.stdout)


def test_invalid_group_element_rejected(tmp_path):
    """List form must contain dicts; mixing in a non-dict surfaces an error
    pointing at the offending index."""
    sweep_path = _write_sweep(
        tmp_path,
        search_space=[{"a": [1]}, "not-a-dict"],
    )
    with pytest.raises(ValueError, match=r"search_space\[1\]"):
        sweep._search_space_groups(sweep._load_yaml(str(sweep_path)))


# --- find_ood_sibling ----------------------------------------------------------

def test_find_ood_sibling_returns_ood_when_present(tmp_path):
    """Path with an 'id' component maps to its 'ood' sibling when that dir exists."""
    id_dir = tmp_path / "Duffing" / "id"
    ood_dir = tmp_path / "Duffing" / "ood"
    id_dir.mkdir(parents=True)
    ood_dir.mkdir(parents=True)
    result = sweep.find_ood_sibling(str(id_dir))
    assert result == ood_dir.resolve()


def test_find_ood_sibling_returns_none_when_ood_missing(tmp_path):
    """No 'ood' sibling on disk → None (no directory creation side-effects)."""
    id_dir = tmp_path / "Duffing" / "id"
    id_dir.mkdir(parents=True)
    assert sweep.find_ood_sibling(str(id_dir)) is None


def test_find_ood_sibling_returns_none_when_no_id_component(tmp_path):
    """Paths without an 'id' component are left untouched."""
    data_dir = tmp_path / "Duffing" / "train"
    data_dir.mkdir(parents=True)
    assert sweep.find_ood_sibling(str(data_dir)) is None


def test_find_ood_sibling_case_insensitive(tmp_path):
    """Component matching is case-insensitive ('ID', 'Id' all match)."""
    id_dir = tmp_path / "Duffing" / "ID"
    ood_dir = tmp_path / "Duffing" / "ood"
    id_dir.mkdir(parents=True)
    ood_dir.mkdir(parents=True)
    result = sweep.find_ood_sibling(str(id_dir))
    assert result == ood_dir.resolve()


# --- integration: full train → evaluate → post_process pipeline ---------------

def _make_traj(rng, n_steps):
    u = rng.standard_normal(n_steps).astype(np.float64) * 0.2
    q = np.zeros(n_steps, dtype=np.float64)
    q_dot = np.zeros(n_steps, dtype=np.float64)
    for k in range(n_steps - 1):
        q[k + 1] = q[k] + 0.05 * q_dot[k]
        q_dot[k + 1] = q_dot[k] + 0.05 * (-q[k] - 0.3 * q_dot[k] + u[k])
    return u, q, q_dot


def _write_csvs(folder: Path, n_files: int, n_steps: int, seed: int):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_files):
        u, q, q_dot = _make_traj(rng, n_steps)
        pd.DataFrame({"u": u, "q": q, "q_dot": q_dot}).to_csv(
            folder / f"traj_{i:03d}.csv", index=False
        )


@pytest.fixture(scope="module")
def sweep_integration_root(tmp_path_factory):
    """Hermetic data root + sweep config for the full-pipeline smoke test."""
    root = tmp_path_factory.mktemp("sweep_int")
    data_dir = root / "data" / "SmokeData"

    for split, n, seed in [("train", 4, 0), ("validation", 2, 1), ("test", 2, 2)]:
        _write_csvs(data_dir / split, n_files=n, n_steps=200, seed=seed)

    base_cfg = {
        "data": {
            "train_path": str(data_dir),
            "input_col": ["u"],
            "output_col": ["q"],
            "state_col": ["q", "q_dot"],
            "pattern": "*.csv",
            "normalize": True,
            "normalization_method": "scale_only",
            "batch_size": 2,
            "train_sequence_length": 50,
            "sequence_stride": 50,
            "shuffle": True,
            "num_workers": 0,
            "sampling_time": 0.05,
        },
        "model": {
            "model_type": "crnn",
            "nw": 4,
            "nx": 2,
            "activation": "dzn",
            "custom_params": {
                "learn_L": True,
                "structural_constraints": {
                    "D": {"fixed": True, "value": 0.0},
                    "D12": {"fixed": True, "value": 0.0},
                },
            },
            "initialization": {"method": "identity"},
        },
        "optimizer": {
            "optimizer_type": "adam",
            "learning_rate": 0.005,
            "use_scheduler": True,
            "scheduler_type": "reduce_on_plateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
        },
        "training": {
            "max_epochs": 2,
            "gradient_clip_value": 10.0,
            "loss_type": "mse",
            "use_custom_regularization": True,
            "min_regularization_weight": 1e-7,
            "regularization_weight": 1e-2,
            "decay_regularization_weight": True,
            "regularization_decay_factor": 0.5,
            "device": "cpu",
            "log_gradients": True,
            "warmup_steps": 10,
            "input_regularization_weight": 1e-2,
        },
        "mlflow": {
            "tracking_uri": f"file:{root}/mlruns",
            "experiment_name": "sweep_smoke",
            "run_name": None,
        },
        "evaluation": {"metrics": ["rmse", "nrmse"]},
        "root_dir": str(root),
        "seed": 42,
    }
    base_cfg_path = root / "base_config.yaml"
    with open(base_cfg_path, "w") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)

    sweep_cfg = {
        "sweep_name": "smoke",
        "base_config": str(base_cfg_path),
        "n_seeds": 1,
        "post_process_args": ["--rv-num-trajectories", "2", "--rv-horizon", "50"],
        "search_space": {"training.use_custom_regularization": [True]},
    }
    sweep_cfg_path = root / "sweep_debug.yaml"
    with open(sweep_cfg_path, "w") as f:
        yaml.safe_dump(sweep_cfg, f, sort_keys=False)

    return root, sweep_cfg_path


def test_sweep_full_pipeline_single_task(sweep_integration_root):
    """sweep.py --task-id 0 must exit 0 and produce model + evaluation artefacts."""
    root, sweep_cfg_path = sweep_integration_root
    subprocess.run(
        [sys.executable, str(SCRIPTS / "sweep.py"),
         "--sweep-config", str(sweep_cfg_path),
         "--task-id", "0",
         "--device", "cpu"],
        check=True,
        cwd=str(root),
    )
    models_dir = root / "models" / "crnn"
    assert models_dir.exists(), "models/crnn/ not created"
    run_dirs = list(models_dir.iterdir())
    assert run_dirs, "No run directory under models/crnn/"
    run_dir = run_dirs[0]
    assert (run_dir / "best_model.pt").exists(), "best_model.pt missing"
    eval_dir = root / "outputs" / "crnn" / run_dir.name / "evaluation"
    assert eval_dir.exists(), "evaluation/ artefact dir missing"
