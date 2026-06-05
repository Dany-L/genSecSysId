"""Smoke tests for the sweep schema (single-group dict and multi-group list).

Covers the two consumers of the schema:
  - ``sweep.py --count`` (used by ``slurm/submit_sweep.sh`` to size the
    SLURM array)
  - ``enumerate_tasks`` (used inside ``sweep.py`` to map task-id -> overrides)

Does NOT exercise train/evaluate/post_process — those are covered by
``test_scripts_smoke.py``. The new code path here is the schema
normalization, which is pure config handling and runs in milliseconds.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_PY / "scripts"

# Make ``sweep.py`` importable for the in-process tests below.
sys.path.insert(0, str(SCRIPTS))
import sweep  # noqa: E402


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
