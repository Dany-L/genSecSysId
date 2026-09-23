"""End-to-end smoke tests for scripts/.

Each test invokes a script as a subprocess against synthetic data laid out
in a tempdir, so the tests are hermetic — they don't depend on the user's
~/genSecSysId-Data directory or any existing trained run.

The contract: every script must exit 0 on a happy-path invocation.
`subprocess.run(check=True)` raises CalledProcessError on non-zero exit,
which pytest reports as a failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.solver_utils import requires_mosek

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_PY / "scripts"

# Every test here runs train.py, whose crnn init solves the MaxS certificate
# SDP via MOSEK; skip the whole module when MOSEK is unavailable (e.g. CI).
pytestmark = requires_mosek


def _make_traj(rng, n_steps):
    """Simple stable linear surrogate so 2-epoch training is well-behaved."""
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
def smoke_root(tmp_path_factory):
    """Fresh data_root with synthetic train/val/test splits."""
    root = tmp_path_factory.mktemp("smoke_root")
    data_dir = root / "data" / "SmokeData"
    _write_csvs(data_dir / "train", n_files=4, n_steps=200, seed=0)
    _write_csvs(data_dir / "validation", n_files=2, n_steps=200, seed=1)
    _write_csvs(data_dir / "test", n_files=2, n_steps=200, seed=2)
    return root


@pytest.fixture(scope="module")
def smoke_config(smoke_root):
    """Minimal YAML config — tiny model, 2 epochs, file-based MLflow."""
    data_dir = smoke_root / "data" / "SmokeData"
    cfg = {
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
            "tracking_uri": f"file:{smoke_root}/mlruns",
            "experiment_name": "smoke",
            "run_name": None,
        },
        "evaluation": {"metrics": ["rmse", "nrmse"]},
        "root_dir": str(smoke_root),
        "seed": 42,
    }
    path = smoke_root / "smoke_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


# ── multi-output (ne > 1) ─────────────────────────────────────────────────────
# The SISO fixtures above cannot catch ne > 1 regressions, and the scalar
# assumptions are easy to reintroduce: post_process.py computed
# ybar = sigma * s * sqrt((C P C^T).item()), which is a ValueError the moment
# C P C^T stops being 1x1.
def _make_traj_3state(rng, n_steps):
    """Stable 3rd-order surrogate: the 2-state oscillator plus a first-order lag.

    THREE genuinely independent states are needed. Reading three channels off
    the 2-state surrogate makes the third an exact combination of the first
    two, so C P C^T is rank 2, the certified output set is degenerate and the
    bootstrap SDP is ill-conditioned enough that MOSEK fails outright.
    """
    u, q, q_dot = _make_traj(rng, n_steps)
    lag = np.zeros(n_steps, dtype=np.float64)
    for k in range(n_steps - 1):
        lag[k + 1] = 0.9 * lag[k] + 0.1 * u[k]
    return u, q, q_dot, lag


def _write_multi_output_csvs(folder: Path, n_files: int, n_steps: int, seed: int):
    """The 3-state surrogate read out on THREE channels at different scales."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_files):
        u, q, q_dot, lag = _make_traj_3state(rng, n_steps)
        pd.DataFrame({
            "u": u,
            # Distinct scales, so a channel-0 collapse of output_std is wrong
            # rather than merely redundant.
            "y1": q,
            "y2": 2.0 * q_dot,
            "y3": 5.0 * lag,
        }).to_csv(folder / f"traj_{i:03d}.csv", index=False)


@pytest.fixture(scope="module")
def multi_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("multi_root")
    data_dir = root / "data" / "MultiData"
    _write_multi_output_csvs(data_dir / "train", n_files=4, n_steps=200, seed=0)
    _write_multi_output_csvs(data_dir / "validation", n_files=2, n_steps=200, seed=1)
    _write_multi_output_csvs(data_dir / "test", n_files=2, n_steps=200, seed=2)
    return root


@pytest.fixture(scope="module")
def multi_config(multi_root, smoke_config):
    """The smoke config with three outputs and nx >= ne."""
    cfg = yaml.safe_load(smoke_config.read_text())
    cfg["data"]["train_path"] = str(multi_root / "data" / "MultiData")
    cfg["data"]["output_col"] = ["y1", "y2", "y3"]
    cfg["data"]["state_col"] = None
    # nx >= ne, or the certified output set is flat and ybar is identically 0.
    cfg["model"]["nx"] = 3
    # Draw A by placing its discrete poles. The default {scale} spec draws
    # continuous damping and Euler-discretizes it, so with three eigenvalues
    # instead of two it lands above alpha_0 often enough that the fixture is
    # flaky; {radius} makes rho(A) = max radius hold by construction.
    cfg["model"]["custom_params"]["identity_init"] = {
        "A": {"radius": [0.5, 0.9]},
        "B2": {"std": 0.01},
        "C2": {"std": 0.1},
    }
    cfg["mlflow"]["tracking_uri"] = f"file:{multi_root}/mlruns"
    cfg["root_dir"] = str(multi_root)
    path = multi_root / "multi_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


@pytest.fixture(scope="module")
def multi_trained_run_id(multi_root, multi_config, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multi_train")
    run_id_out = tmp / "run_id.txt"
    _run_script("train.py", "--config", multi_config,
                "--run-id-out", run_id_out, cwd=multi_root)
    run_id = run_id_out.read_text().strip()
    assert run_id, "train.py did not write a run_id for the 3-output config"
    return run_id


def test_train_smoke_multi_output(multi_trained_run_id):
    """train.py survives ne > 1 (identity init used to raise on the C scaling)."""
    assert multi_trained_run_id


def test_post_process_smoke_multi_output(multi_root, multi_trained_run_id):
    """post_process.py survives ne > 1.

    The regression: it computed the certified output half-width by hand as
    ``sigma * s * sqrt((C P C^T).item())``, which raises ValueError for ne > 1
    because C P C^T is (ne, ne). It now goes through
    sysid.optimization.output_set.output_ellipsoid like the model does.
    """
    _run_script(
        "post_process.py",
        "--run-id", multi_trained_run_id,
        "--data-root", multi_root,
        "--rv-num-trajectories", "2",
        "--rv-horizon", "50",
        cwd=multi_root,
    )


def _run_script(name: str, *args, cwd: Path):
    """Invoke a script as a subprocess; raise on non-zero exit.

    Stdout/stderr are inherited so failures are visible in pytest output.
    """
    cmd = [sys.executable, str(SCRIPTS / name)] + [str(a) for a in args]
    subprocess.run(cmd, check=True, cwd=str(cwd))


@pytest.fixture(scope="module")
def trained_run_id(smoke_root, smoke_config, tmp_path_factory):
    """Run train.py once and yield the resulting MLflow run_id."""
    tmp = tmp_path_factory.mktemp("smoke_train")
    run_id_out = tmp / "run_id.txt"
    _run_script(
        "train.py",
        "--config", smoke_config,
        "--run-id-out", run_id_out,
        cwd=smoke_root,
    )
    run_id = run_id_out.read_text().strip()
    assert run_id, "train.py did not write a run_id"
    return run_id


def test_train_smoke(trained_run_id, smoke_root):
    """train.py wrote the standard run layout under smoke_root."""
    run_dir = smoke_root / "models" / "crnn" / trained_run_id
    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "normalizer.json").exists()
    assert (run_dir / "run_info.json").exists()
    config_path = smoke_root / "outputs" / "crnn" / trained_run_id / "config.yaml"
    assert config_path.exists()


def test_evaluate_smoke(smoke_root, trained_run_id):
    test_data = smoke_root / "data" / "SmokeData"
    _run_script(
        "evaluate.py",
        "--run-id", trained_run_id,
        "--data-root", smoke_root,
        "--test-data", test_data,
        cwd=smoke_root,
    )


def test_post_process_smoke(smoke_root, trained_run_id):
    # Keep the regional-verification workload tiny for smoke speed.
    _run_script(
        "post_process.py",
        "--run-id", trained_run_id,
        "--data-root", smoke_root,
        "--rv-num-trajectories", "2",
        "--rv-horizon", "50",
        cwd=smoke_root,
    )


def test_export_for_matlab_smoke(smoke_root, trained_run_id, tmp_path):
    out = tmp_path / "exported.mat"
    _run_script(
        "export_for_matlab.py",
        "--run-id", trained_run_id,
        "--data-root", smoke_root,
        "--output", out,
        cwd=smoke_root,
    )
    assert out.exists()


def test_compare_smoke(smoke_root, smoke_config, trained_run_id, tmp_path_factory, tmp_path):
    """Train a second run with a different seed so compare.py has two
    columns of data to put side by side."""
    tmp = tmp_path_factory.mktemp("smoke_train2")
    run_id_out = tmp / "run_id.txt"
    _run_script(
        "train.py",
        "--config", smoke_config,
        "--seed", "7",
        "--run-id-out", run_id_out,
        cwd=smoke_root,
    )
    second = run_id_out.read_text().strip()
    assert second and second != trained_run_id

    output_dir = tmp_path / "comparison"
    _run_script(
        "compare.py",
        "--run-ids", trained_run_id, second,
        "--data-root", smoke_root,
        "--output-dir", output_dir,
        cwd=smoke_root,
    )
    assert output_dir.exists()
