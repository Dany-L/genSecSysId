"""Tests for the Silverbox benchmark dataset (scripts/prepare_benchmark.py).

Three things are pinned here:

1. The layout the prep script writes is one ``sysid.data.direct_loader`` can
   read back — the uniform-row-count rule that ``np.stack`` imposes on every
   converging split is easy to break and fails far from the cause.
2. The three CRNN arms the Silverbox configs define all train end to end:
   ``(reg, learn_L)`` = ``(True, True)`` regional, ``(True, False)`` global,
   ``(False, False)`` none.
3. The shipped configs in ``genSecSysId-Data/configs`` really carry those flag
   combinations and load without dropping keys. Unknown YAML keys are dropped
   with only a warning, so a renamed field silently turns an arm into a no-op
   that looks like a null result.

Everything except (3) is hermetic: the prep script's download is confined to
``load_official_silverbox``, so the tests drive ``prepare_dataset`` with
synthetic records instead of fetching 6 MB from Google Drive.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from sysid.config import Config
from tests.solver_utils import requires_mosek

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_PY / "scripts"

# Where the shipped configs live. Outside the repo on purpose (they are the
# live experiment configs), so the tests that read them skip when it is absent.
SHIPPED_CONFIG_DIR = Path("~/genSecSysId-Data/configs").expanduser()

# (config stem, use_custom_regularization, learn_L) for the three arms.
ARMS = [
    ("crnn_regional_silverbox", True, True),
    ("crnn_global_silverbox", True, False),
    ("crnn_none_silverbox", False, False),
]


def _load_script_module(name: str, path: Path):
    """Import a scripts/ module by file path without mutating sys.path.

    Same rationale as tests/test_sweep_smoke.py: scripts/ modules have generic
    names, so a persistent sys.path entry would shadow real imports.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_benchmark = _load_script_module(
    "prepare_benchmark", SCRIPTS / "prepare_benchmark.py"
)


# --------------------------------------------------------------------------
# synthetic stand-in for the real records
# --------------------------------------------------------------------------
def _lure_record(n_steps: int, seed: int, amplitude: float = 0.02):
    """A stable 2nd-order Lur'e record shaped like the Silverbox.

    The Silverbox is a 2nd-order LTI system with a static nonlinearity in
    feedback; this is the same structure with a dead zone instead of a cubic,
    so the CRNN's model class actually contains the data-generating system and
    a 2-epoch run is well behaved.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n_steps) * amplitude
    x1 = np.zeros(n_steps)
    x2 = np.zeros(n_steps)
    for k in range(n_steps - 1):
        w = np.sign(x1[k]) * max(abs(x1[k]) - 0.05, 0.0)  # dead zone
        x1[k + 1] = 0.95 * x1[k] + 0.05 * x2[k]
        x2[k + 1] = -0.05 * x1[k] + 0.92 * x2[k] + 0.5 * u[k] - 0.1 * w
    return u.reshape(-1, 1), x1.reshape(-1, 1)


def _synthetic_records(n_train_val=4000, n_test=1200):
    """(train_records, test_records) in the shape `prepare` expects."""
    from sysid.data.benchmark_registry import Record

    train = [Record("multisine_train", *_lure_record(n_train_val, seed=0))]
    tests = [
        Record("multisine", *_lure_record(n_test, seed=1)),
        # The real arrow record extrapolates past the training amplitude; mirror
        # that here, including its different length, so the ragged-test rule is
        # exercised.
        Record("arrow", *_lure_record(n_test + 300, seed=2, amplitude=0.05)),
        Record("arrow_no_extrapolation", *_lure_record(n_test, seed=3, amplitude=0.015)),
    ]
    return train, tests


@pytest.fixture(scope="module")
def prepared_dir(tmp_path_factory):
    """A full prepared Silverbox layout built from synthetic records.

    Goes through the generic adapter (`scripts/prepare_benchmark.py`), which
    replaced the per-benchmark prep scripts. The arms below are what this file
    is really for: they check that a CRNN trains end to end on this layout.
    """
    from sysid.data.benchmark_registry import get

    out = tmp_path_factory.mktemp("silverbox") / "id"
    train, tests = _synthetic_records()
    prepare_benchmark.prepare(
        get("Silverbox"), out, train, tests, 1 / 610.35,
        val_fraction=0.2, test_record="multisine",
    )
    return out


# --------------------------------------------------------------------------
# CSV writing
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# the prepared layout
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# the three model arms, end to end
# --------------------------------------------------------------------------
def _arm_config(prepared_dir, tmp_path, use_reg: bool, learn_L: bool) -> Path:
    """A shrunk version of the shipped Silverbox configs for one arm.

    Same structure as crnn_*_silverbox.yaml (SISO u/y, no state_col, scale_only,
    windowed training, warmup to cover the x0 = 0 washout), sized so two epochs
    run in seconds.
    """
    cfg = {
        "data": {
            "train_path": str(prepared_dir),
            "input_col": ["u"],
            "output_col": ["y"],
            "pattern": "*.csv",
            "normalize": True,
            "normalization_method": "scale_only",
            "batch_size": 4,
            "train_sequence_length": 200,
            "sequence_stride": 200,
            "shuffle": False,
            "num_workers": 0,
            "sampling_time": 1 / 610.35,
            "use_diverging_trajectories": False,
        },
        "model": {
            "model_type": "crnn",
            "nw": 4,
            "nx": 2,
            "activation": "dzn",
            "custom_params": {
                "learn_L": learn_L,
                # Mirrors the shipped configs, and is the reason this test is
                # worth running at all — both values are load-bearing:
                #   A: the identity init's DEFAULT A = I + ts*A_ct puts
                #      |eig| ~ 0.99995 at Silverbox's sampling rate, above
                #      alpha_0 = 0.9999, so no P exists and every certified arm
                #      dies in the D21 bootstrap before epoch 0.
                #   B2: above ~3e-4 the GLOBAL arm's bootstrap comes back
                #      infeasible — with L = 0 the sector condition has to hold
                #      everywhere, and A's resonance amplifies the w -> x loop
                #      by ~100x.
                "identity_init": {
                    "A": {"value": [[0.0, 1.0], [-0.980100, 1.492807]]},
                    "B": {"value": [[0.0], [0.0005]]},
                    "B2": {"std": 3.0e-4},
                    "C2": {"std": 1.0},
                    "D21": {"std": 0.5},
                },
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
            "use_scheduler": False,
        },
        "training": {
            "max_epochs": 2,
            "gradient_clip_value": 1.0,
            "loss_type": "mse",
            "use_custom_regularization": use_reg,
            "regularization_weight": 1e-2,
            "min_regularization_weight": 1e-7,
            "decay_regularization_weight": True,
            "regularization_decay_factor": 0.1,
            "device": "cpu",
            "log_gradients": False,
            "warmup_steps": 20,
            "input_regularization_weight": 0.0,
            # Only ever fires in the regional arm — _maybe_maximize_s returns
            # early when learn_L is false or the barrier is off.
            "max_s_trigger": "on_violation" if (use_reg and learn_L) else "never",
        },
        "mlflow": {
            "tracking_uri": f"file:{tmp_path}/mlruns",
            "experiment_name": "smoketest-silverbox",
            "run_name": None,
        },
        "evaluation": {"metrics": ["rmse", "nrmse"]},
        "root_dir": str(tmp_path),
        "seed": 42,
    }
    path = tmp_path / f"silverbox_reg{int(use_reg)}_L{int(learn_L)}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def test_none_arm_trains_without_a_solver(prepared_dir, tmp_path):
    """reg=False, learn_L=False.

    Kept out of the MOSEK-gated class deliberately: with the barrier off no SDP
    is solved at init or during training, so this arm must run on a machine
    with no MOSEK license at all. If it ever starts needing one, that is a
    regression in the ``use_custom_regularization=False`` path.
    """
    cfg = _arm_config(prepared_dir, tmp_path, use_reg=False, learn_L=False)
    subprocess.run(
        [sys.executable, str(SCRIPTS / "train.py"), "--config", str(cfg)],
        check=True, cwd=str(REPO_PY),
    )


@requires_mosek
@pytest.mark.parametrize(
    "use_reg,learn_L,arm",
    [(True, True, "regional"), (True, False, "global")],
)
def test_certified_arms_train(prepared_dir, tmp_path, use_reg, learn_L, arm):
    """The two arms whose init and per-batch repair solve the certificate SDP."""
    cfg = _arm_config(prepared_dir, tmp_path, use_reg=use_reg, learn_L=learn_L)
    subprocess.run(
        [sys.executable, str(SCRIPTS / "train.py"), "--config", str(cfg)],
        check=True, cwd=str(REPO_PY),
    )


# --------------------------------------------------------------------------
# the shipped configs
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not SHIPPED_CONFIG_DIR.is_dir(),
    reason=f"{SHIPPED_CONFIG_DIR} not present (data repo not checked out)",
)
class TestShippedConfigs:
    @pytest.mark.parametrize("stem,use_reg,learn_L", ARMS)
    def test_arm_flags(self, stem, use_reg, learn_L):
        path = SHIPPED_CONFIG_DIR / f"{stem}.yaml"
        if not path.exists():
            pytest.skip(f"{path} not present")
        config = Config.from_yaml(str(path))

        assert config.training.use_custom_regularization is use_reg
        assert config.model.custom_params.get("learn_L") is learn_L
        assert config.model.model_type == "crnn"
        # nx = 2 is the Silverbox plant order (2nd-order LTI + feedback
        # nonlinearity), not a tuned hyperparameter.
        assert config.model.nx == 2
        assert config.data.input_col == ["u"]
        assert config.data.output_col == ["y"]
        assert "Silverbox" in config.data.train_path

    @pytest.mark.parametrize("stem,use_reg,learn_L", ARMS)
    def test_no_keys_are_silently_dropped(self, stem, use_reg, learn_L, caplog):
        """Unknown YAML keys are dropped with a warning, turning an arm into a
        no-op that still looks like a result. Fail instead."""
        path = SHIPPED_CONFIG_DIR / f"{stem}.yaml"
        if not path.exists():
            pytest.skip(f"{path} not present")
        with caplog.at_level("WARNING", logger="sysid.config"):
            Config.from_yaml(str(path))
        dropped = [r.getMessage() for r in caplog.records if "Ignoring unknown" in r.getMessage()]
        assert not dropped, f"{stem}.yaml has stale keys: {dropped}"

    @pytest.mark.parametrize("stem,use_reg,learn_L", ARMS)
    def test_a_init_is_inside_the_contraction_rate(self, stem, use_reg, learn_L):
        """rho(A_init) < alpha_0, or initialization cannot start.

        The stability LMI asks for A'PA - alpha^2 P < 0, which has no solution
        once rho(A) >= alpha. The identity init's default A = I + ts*A_ct lands
        at |eig| ~ 0.99995 at Silverbox's 610 Hz — above the alpha_0 = 0.9999
        default — so these configs set A explicitly. Pinned because the failure
        is a RuntimeError from the bootstrap SDP that names D21, not A.
        """
        path = SHIPPED_CONFIG_DIR / f"{stem}.yaml"
        if not path.exists():
            pytest.skip(f"{path} not present")
        custom = Config.from_yaml(str(path)).model.custom_params
        A_spec = custom["identity_init"]["A"]
        assert "value" in A_spec, "A must be set explicitly, not drawn"

        alpha_0 = float(custom.get("alpha_0", 0.9999))
        rho = float(np.abs(np.linalg.eigvals(np.array(A_spec["value"], dtype=float))).max())
        assert rho < alpha_0, f"rho(A_init)={rho} >= alpha_0={alpha_0}"

    def test_all_arms_share_one_initialization(self):
        """The three arms must differ in the certificate machinery ALONE.

        B2 is capped at 3e-4 by the global arm's feasibility, and the other two
        arms adopt that value rather than keeping a larger one — otherwise the
        comparison confounds the certificate with a different starting point.
        """
        inits = {}
        for stem, _, _ in ARMS:
            path = SHIPPED_CONFIG_DIR / f"{stem}.yaml"
            if not path.exists():
                pytest.skip(f"{path} not present")
            inits[stem] = Config.from_yaml(str(path)).model.custom_params["identity_init"]
        first = inits[ARMS[0][0]]
        for stem, spec in inits.items():
            assert spec == first, f"{stem}.yaml starts from a different theta"
        assert float(first["B2"]["std"]) <= 3.0e-4, (
            "B2 above 3e-4 makes the global arm's bootstrap SDP infeasible"
        )

    def test_max_s_trigger_matches_the_arm(self):
        """Only the regional arm can move s; the others say so explicitly."""
        triggers = {}
        for stem, _, _ in ARMS:
            path = SHIPPED_CONFIG_DIR / f"{stem}.yaml"
            if not path.exists():
                pytest.skip(f"{path} not present")
            triggers[stem] = Config.from_yaml(str(path)).training.max_s_trigger
        assert triggers["crnn_regional_silverbox"] == "on_violation"
        assert triggers["crnn_global_silverbox"] == "never"
        assert triggers["crnn_none_silverbox"] == "never"


# --------------------------------------------------------------------------
# the real records — only when they are already cached
# --------------------------------------------------------------------------
def _silverbox_is_cached() -> bool:
    """True iff nonlinear_benchmarks already holds the Silverbox .mat locally.

    Gating on the cache rather than on a marker keeps this test honest in both
    directions: it runs (offline, in ~1 s) on any machine that has prepared the
    dataset, and skips instead of pulling 6 MB from Google Drive on a fresh
    checkout or a CI runner.
    """
    if importlib.util.find_spec("nonlinear_benchmarks") is None:
        return False
    from nonlinear_benchmarks.utilities import get_tmp_benchmark_directory

    cache = Path(get_tmp_benchmark_directory()) / "Silverbox" / "SilverboxFiles"
    return (cache / "SNLS80mV.mat").exists()


@pytest.mark.skipif(
    not _silverbox_is_cached(),
    reason="Silverbox not in the nonlinear_benchmarks cache; "
           "run scripts/prepare_benchmark.py once to fetch it",
)
@pytest.mark.benchmark_data
def test_official_split_shapes_are_the_documented_ones():
    """Pins the package's split against what the configs and docstring assume.

    The only test that would notice nonlinear_benchmarks changing the split out
    from under the prepared dataset — every published Silverbox number depends
    on these exact index ranges.
    """
    from sysid.data.benchmark_registry import REGISTRY

    try:
        train, test, ts = REGISTRY["Silverbox"].fetch()
    except Exception as exc:  # not cached / no network
        pytest.skip(f"Silverbox unavailable: {exc}")

    by_name = {r.name: r for r in test}
    u_tv = train[0].u
    assert len(u_tv) == 65062, "train_val block changed"
    assert len(by_name["multisine"].u) == 21688
    assert len(by_name["arrow"].u) == 40475
    assert len(by_name["arrow_no_extrapolation"].u) == 32000
    assert ts == pytest.approx(1 / 610.35)
    # The naming assumption the package's own .name attributes get wrong:
    # test[1] is the arrow record, and it extrapolates past the training range.
    assert np.abs(by_name["arrow"].u).max() > np.abs(u_tv).max()
    assert np.abs(by_name["arrow_no_extrapolation"].u).max() <= np.abs(u_tv).max()
