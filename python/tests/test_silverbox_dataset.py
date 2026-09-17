"""Tests for the Silverbox benchmark dataset (scripts/prepare_silverbox_dataset.py).

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
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sysid.config import Config
from sysid.data.direct_loader import load_split_data
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


prep = _load_script_module(
    "prepare_silverbox_dataset", SCRIPTS / "prepare_silverbox_dataset.py"
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
    """(train_val, test_records) in the shape prepare_dataset expects."""
    train_val = _lure_record(n_train_val, seed=0)
    return train_val, {
        "multisine": _lure_record(n_test, seed=1),
        # The real arrow record extrapolates past the training amplitude; mirror
        # that here so a test can assert the metadata reports it.
        "arrow": _lure_record(n_test + 300, seed=2, amplitude=0.05),
        "arrow_no_extrapolation": _lure_record(n_test, seed=3, amplitude=0.015),
    }


@pytest.fixture(scope="module")
def prepared_dir(tmp_path_factory):
    """A full prepared Silverbox layout built from synthetic records."""
    out = tmp_path_factory.mktemp("silverbox") / "id"
    train_val, tests = _synthetic_records()
    prep.prepare_dataset(
        train_val=train_val,
        test_records=tests,
        out_dir=out,
        sampling_time=1 / 610.35,
        val_fraction=0.2,
        subsequence_length=None,
        test_set="multisine",
    )
    return out


# --------------------------------------------------------------------------
# split arithmetic
# --------------------------------------------------------------------------
class TestSplitTrainVal:
    def test_split_is_contiguous_and_lossless(self):
        u = np.arange(1000, dtype=float).reshape(-1, 1)
        y = u * 2.0
        u_tr, y_tr, u_va, y_va = prep.split_train_val(u, y, val_fraction=0.2)

        assert len(u_tr) == 800 and len(u_va) == 200
        # Contiguous, validation is the TAIL — a shuffled split would leak the
        # validation dynamics into training through overlapping windows.
        np.testing.assert_array_equal(np.concatenate([u_tr, u_va]), u)
        np.testing.assert_array_equal(u_va[0], [800.0])
        np.testing.assert_array_equal(y_tr, u_tr * 2.0)
        np.testing.assert_array_equal(y_va, u_va * 2.0)

    def test_accepts_1d_signals(self):
        u_tr, _, u_va, _ = prep.split_train_val(
            np.arange(100, dtype=float), np.arange(100, dtype=float), 0.25
        )
        assert u_tr.shape == (75, 1) and u_va.shape == (25, 1)

    @pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_degenerate_fractions(self, frac):
        with pytest.raises(ValueError):
            prep.split_train_val(np.zeros(100), np.zeros(100), frac)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="differ in length"):
            prep.split_train_val(np.zeros(100), np.zeros(90), 0.2)


# --------------------------------------------------------------------------
# CSV writing
# --------------------------------------------------------------------------
class TestWriteSplit:
    def test_single_file_keeps_every_sample(self, tmp_path):
        u, y = _lure_record(500, seed=7)
        written = prep.write_split(tmp_path / "train", u, y)

        assert len(written) == 1
        df = pd.read_csv(written[0])
        assert list(df.columns) == ["u", "y"]
        assert len(df) == 500
        np.testing.assert_allclose(df["u"].values, u[:, 0])

    def test_chunking_drops_the_short_remainder(self, tmp_path):
        """A short trailing file would break load_split_data's np.stack."""
        u, y = _lure_record(1050, seed=8)
        written = prep.write_split(tmp_path / "train", u, y, subsequence_length=100)

        assert len(written) == 10  # 1050 // 100, the last 50 samples dropped
        lengths = {len(pd.read_csv(p)) for p in written}
        assert lengths == {100}
        # Chunks are consecutive, not shuffled.
        np.testing.assert_allclose(pd.read_csv(written[1])["u"].values, u[100:200, 0])

    def test_rejects_subsequence_longer_than_the_record(self, tmp_path):
        u, y = _lure_record(50, seed=9)
        with pytest.raises(ValueError, match="exceeds the 50-sample record"):
            prep.write_split(tmp_path / "train", u, y, subsequence_length=100)


# --------------------------------------------------------------------------
# the prepared layout
# --------------------------------------------------------------------------
class TestPrepareDataset:
    def test_writes_every_folder_the_loader_and_the_study_need(self, prepared_dir):
        for name in [
            "train", "validation", "test",
            "test_multisine", "test_arrow", "test_arrow_no_extrapolation",
        ]:
            folder = prepared_dir / name
            assert folder.is_dir(), f"missing {name}/"
            assert list(folder.glob("*.csv")), f"{name}/ has no CSVs"

    def test_loader_reads_it_back(self, prepared_dir):
        """The actual contract: direct_loader.load_split_data must not raise."""
        (train_in, train_out, val_in, val_out, test_in, test_out, *_) = load_split_data(
            str(prepared_dir), input_col=["u"], output_col=["y"]
        )
        # One full-length CSV per split -> (1, T, 1).
        assert train_in.shape == (1, 3200, 1)
        assert val_in.shape == (1, 800, 1)
        assert test_in.shape == (1, 1200, 1)
        assert train_out.shape == train_in.shape
        assert val_out.shape == val_in.shape
        assert test_out.shape == test_in.shape

    def test_chunked_layout_also_loads(self, tmp_path):
        train_val, tests = _synthetic_records()
        out = tmp_path / "chunked"
        prep.prepare_dataset(
            train_val=train_val, test_records=tests, out_dir=out,
            sampling_time=1 / 610.35, subsequence_length=400,
        )
        train_in, *_ = load_split_data(str(out), input_col=["u"], output_col=["y"])
        assert train_in.shape == (8, 400, 1)  # 3200 // 400

    def test_test_folder_mirrors_the_selected_record(self, prepared_dir):
        a = pd.read_csv(next((prepared_dir / "test").glob("*.csv")))
        b = pd.read_csv(next((prepared_dir / "test_multisine").glob("*.csv")))
        pd.testing.assert_frame_equal(a, b)

    def test_test_set_flag_selects_the_arrow_record(self, tmp_path):
        train_val, tests = _synthetic_records()
        out = tmp_path / "arrow"
        prep.prepare_dataset(
            train_val=train_val, test_records=tests, out_dir=out,
            sampling_time=1 / 610.35, test_set="arrow",
        )
        _, _, _, _, test_in, *_ = load_split_data(
            str(out), input_col=["u"], output_col=["y"]
        )
        assert test_in.shape[1] == 1500  # the longer arrow record

    def test_metadata_records_provenance_and_amplitudes(self, prepared_dir):
        meta = json.loads((prepared_dir / "metadata.json").read_text())

        assert meta["dataset"] == "Silverbox"
        assert "official train/test split" in meta["source"]
        assert meta["input_col"] == ["u"] and meta["output_col"] == ["y"]
        assert meta["test_set_in_test_folder"] == "multisine"
        assert meta["benchmark_state_initialization_window_length"] == 50
        assert meta["sampling_time"] == pytest.approx(1 / 610.35)
        # The whole point of keeping the arrow record separate: it leaves the
        # amplitude range the certificate was fit over.
        assert (
            meta["records"]["test_arrow"]["u_abs_max"]
            > meta["records"]["train"]["u_abs_max"]
        )

    def test_clean_removes_stale_csvs(self, tmp_path):
        """Re-preparing with a different chunk size must not mix row counts."""
        train_val, tests = _synthetic_records()
        out = tmp_path / "restage"
        prep.prepare_dataset(
            train_val=train_val, test_records=tests, out_dir=out,
            sampling_time=1 / 610.35, subsequence_length=400,
        )
        prep.prepare_dataset(
            train_val=train_val, test_records=tests, out_dir=out,
            sampling_time=1 / 610.35, subsequence_length=800,
        )
        lengths = {
            len(pd.read_csv(p)) for p in (out / "train").glob("*.csv")
        }
        assert lengths == {800}, "stale 400-row CSVs survived the re-prepare"

    def test_rejects_an_unknown_test_set(self, tmp_path):
        train_val, tests = _synthetic_records()
        with pytest.raises(ValueError, match="test_set must be one of"):
            prep.prepare_dataset(
                train_val=train_val, test_records=tests, out_dir=tmp_path / "x",
                sampling_time=1 / 610.35, test_set="schroeder",
            )

    def test_rejects_incomplete_test_records(self, tmp_path):
        train_val, tests = _synthetic_records()
        tests.pop("arrow")
        with pytest.raises(ValueError, match="missing"):
            prep.prepare_dataset(
                train_val=train_val, test_records=tests, out_dir=tmp_path / "x",
                sampling_time=1 / 610.35,
            )


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
           "run scripts/prepare_silverbox_dataset.py once to fetch it",
)
def test_official_split_shapes_are_the_documented_ones():
    """Pins the package's split against what the configs and docstring assume.

    The only test that would notice nonlinear_benchmarks changing the split out
    from under the prepared dataset — every published Silverbox number depends
    on these exact index ranges.
    """
    (u_tv, y_tv), records, ts = prep.load_official_silverbox()

    assert len(u_tv) == 65062, "train_val block changed"
    assert len(records["multisine"][0]) == 21688
    assert len(records["arrow"][0]) == 40475
    assert len(records["arrow_no_extrapolation"][0]) == 32000
    assert ts == pytest.approx(1 / 610.35)
    # The ordering assumption the package's own .name attributes get wrong:
    # test[1] is the arrow record and it extrapolates past the training range.
    assert np.abs(records["arrow"][0]).max() > np.abs(u_tv).max()
    assert np.abs(records["arrow_no_extrapolation"][0]).max() <= np.abs(u_tv).max()
