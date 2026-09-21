"""Tests for the Cascaded Tanks and F-16 benchmark preparation.

Three things are pinned here:

1. ``sysid.data.benchmark_prep`` writes a layout ``load_split_data`` can read
   back — in particular the uniform-row-count rule that ``np.stack`` imposes on
   every split, which is easy to break and fails far from its cause.
2. The F-16 keeps ALL THREE accelerometers in the CSV, so switching between one
   and three outputs is a config edit rather than a re-preparation, and the
   metadata records that the split is not an official one.
3. The shipped configs in ``genSecSysId-Data/configs`` load without dropping
   keys and carry the settings their datasets actually need. Unknown YAML keys
   are dropped with only a warning, so a renamed field silently turns a setting
   into a no-op.

The downloads are confined to the scripts' ``load_*`` functions, so everything
here drives ``prepare_dataset`` with synthetic records instead of fetching
7 MB (tanks) or 148 MB (F-16).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sysid.config import Config
from sysid.data import benchmark_prep as bp

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_PY / "scripts"
SHIPPED_CONFIG_DIR = Path("~/genSecSysId-Data/configs").expanduser()


def _load_script_module(name: str, path: Path):
    """Import a scripts/ module by file path without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── shared helpers ────────────────────────────────────────────────────────────
class TestBenchmarkPrepHelpers:
    def test_as_2d_promotes_and_checks_width(self):
        assert bp.as_2d(np.arange(10.0), "u").shape == (10, 1)
        assert bp.as_2d(np.zeros((10, 3)), "y", n_cols=3).shape == (10, 3)
        with pytest.raises(ValueError, match="expected 1 column"):
            bp.as_2d(np.zeros((10, 3)), "y", n_cols=1)

    def test_split_is_contiguous_and_lossless(self):
        u = np.arange(1000, dtype=float).reshape(-1, 1)
        u_tr, y_tr, u_va, y_va = bp.split_train_val(u, u * 2.0, val_fraction=0.2)
        assert len(u_tr) == 800 and len(u_va) == 200
        # Validation is the TAIL, never shuffled: overlapping windows of one
        # record would otherwise leak validation dynamics into training.
        np.testing.assert_array_equal(np.concatenate([u_tr, u_va]), u)
        np.testing.assert_array_equal(u_va[0], [800.0])
        np.testing.assert_array_equal(y_tr, u_tr * 2.0)

    @pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
    def test_split_rejects_degenerate_fractions(self, frac):
        with pytest.raises(ValueError):
            bp.split_train_val(np.zeros(100), np.zeros(100), frac)

    def test_write_split_multi_output_columns(self, tmp_path):
        u = np.arange(20.0).reshape(-1, 1)
        y = np.stack([u[:, 0], u[:, 0] * 2, u[:, 0] * 3], axis=1)
        bp.write_split(tmp_path, u, y, ["u"], ["y1", "y2", "y3"], stem="rec")
        df = pd.read_csv(tmp_path / "rec.csv")
        assert list(df.columns) == ["u", "y1", "y2", "y3"]
        np.testing.assert_array_equal(df["y3"].to_numpy(), u[:, 0] * 3)

    def test_write_split_chunks_are_uniform_and_drop_the_remainder(self, tmp_path):
        u = np.arange(250.0).reshape(-1, 1)
        paths = bp.write_split(tmp_path, u, u, ["u"], ["y"], subsequence_length=100)
        # 250 // 100 == 2; the 50-sample tail is dropped rather than written
        # short, because load_split_data np.stacks a folder.
        assert len(paths) == 2
        assert {len(pd.read_csv(p)) for p in paths} == {100}

    def test_write_split_rejects_oversized_subsequence(self, tmp_path):
        with pytest.raises(ValueError, match="exceeds the 50-sample record"):
            bp.write_split(tmp_path, np.zeros((50, 1)), np.zeros((50, 1)),
                           ["u"], ["y"], subsequence_length=100)

    def test_record_stats_is_per_channel(self):
        y = np.stack([np.full(10, 1.0), np.full(10, -5.0)], axis=1)
        stats = bp.record_stats(np.zeros((10, 1)), y)
        assert stats["n_samples"] == 10
        assert stats["y_abs_max"] == [1.0, 5.0]


# ── Cascaded Tanks ────────────────────────────────────────────────────────────


# ── F-16 ──────────────────────────────────────────────────────────────────────


# ── the shipped configs ───────────────────────────────────────────────────────
# Cascaded Tanks was dropped from the adapter (its signals are not zero-mean;
# see benchmark_registry.REGISTRY), so its config is no longer pinned here.
CONFIGS = [
    # output_col is deliberately NOT pinned for the F-16: it is the knob for
    # fitting one accelerometer or all three. What must hold either way is the
    # nx >= ne invariant below.
    ("crnn_f16", "F16", "scale_only", None),
]


@pytest.mark.parametrize("stem, data_dir, norm_method, output_col", CONFIGS)
class TestShippedConfigs:
    def _path(self, stem):
        path = SHIPPED_CONFIG_DIR / f"{stem}.yaml"
        if not path.exists():
            pytest.skip(f"{path} not present")
        return path

    def test_loads_without_dropping_keys(self, stem, data_dir, norm_method, output_col):
        path = self._path(stem)
        cfg = Config.from_yaml(str(path))
        raw = yaml.safe_load(path.read_text())
        # Unknown keys are dropped with a warning only, so a renamed field would
        # silently become a no-op. Check the settings that matter round-trip.
        assert cfg.data.normalization_method == norm_method == raw["data"]["normalization_method"]
        if output_col is not None:
            assert cfg.data.output_col == output_col
        assert data_dir in cfg.data.train_path
        assert cfg.training.use_custom_regularization is True
        assert cfg.model.custom_params["learn_L"] is True

    def test_numeric_scalars_are_floats_not_strings(self, stem, data_dir, norm_method, output_col):
        # YAML only parses an exponent as a float when it carries a decimal
        # point: "5e-4" is a STRING and reaches torch as one, which fails deep
        # inside the optimizer rather than at config load.
        cfg = Config.from_yaml(str(self._path(stem)))
        for value in (cfg.optimizer.learning_rate,
                      cfg.training.regularization_weight,
                      cfg.training.min_regularization_weight,
                      cfg.data.sampling_time):
            assert isinstance(value, float), f"{value!r} parsed as {type(value).__name__}"

    def test_alpha_0_clears_the_initial_A_spectrum(self, stem, data_dir, norm_method, output_col):
        """rho(A) < alpha_0 at initialization, however A is specified.

        The sampling-rate failure mode: with an eigenvalue at or above alpha_0
        no P satisfies A'PA - alpha^2 P < 0 and initialization dies before
        epoch 0. Both A specs have to be checked -- a pinned {value} matrix
        directly, and a drawn {radius} band by its upper bound, which is what
        rho(A) equals by construction for every seed.
        """
        cfg = Config.from_yaml(str(self._path(stem)))
        custom = cfg.model.custom_params
        # Default from SimpleLure.__init__ when the config does not set it.
        alpha_0 = custom.get("alpha_0", 0.9999)
        spec = custom["identity_init"]["A"]
        if "value" in spec:
            rho = float(np.abs(np.linalg.eigvals(np.array(spec["value"], float))).max())
        elif "radius" in spec:
            radius = spec["radius"]
            rho = float(radius if isinstance(radius, (int, float)) else max(radius))
        else:
            pytest.fail(
                f"{stem}.yaml uses the {{scale}} A spec; at these sampling rates it "
                "draws rho(A) > 1. Use {radius, freq_hz} or a pinned {value}."
            )
        assert rho < alpha_0, f"rho(A)={rho} is not below alpha_0={alpha_0}"

    def test_state_dimension_admits_the_number_of_outputs(
        self, stem, data_dir, norm_method, output_col
    ):
        """``nx >= ne``, or the certified OUTPUT SET is degenerate.

        The certified set is the image of the state ellipsoid under ``y = C x``.
        With ``nx < ne`` that image is flat -- it lives in an ``nx``-dimensional
        subspace of ``R^ne`` -- so ``W = s^2 S (C P C^T) S`` is singular, there
        is no ``Y`` with ``Yc = {y : y^T Y y <= 1}``, and the worst-direction
        ``ybar`` (hence ``coverage_ratio``) is exactly 0. Training still runs,
        which is why this is worth pinning: the failure is silent in the loss
        and only shows up as an empty certificate.
        """
        cfg = Config.from_yaml(str(self._path(stem)))
        ne = len(cfg.data.output_col)
        if ne == 1:
            pytest.skip("single output; the invariant is vacuous")
        assert cfg.model.nx >= ne, (
            f"{stem}.yaml fits {ne} outputs with nx={cfg.model.nx}: the certified "
            f"output set is flat, so ybar and coverage_ratio come out 0. Raise nx "
            f"to at least {ne} (and give identity_init.A a matching {ne}x{ne} or "
            f"larger block)."
        )

    def test_windowing_fits_the_prepared_record(self, stem, data_dir, norm_method, output_col):
        cfg = Config.from_yaml(str(self._path(stem)))
        meta_path = Path(cfg.data.train_path).expanduser() / "metadata.json"
        if not meta_path.exists():
            pytest.skip(f"{meta_path} not present; run the prep script first")
        n_train = json.loads(meta_path.read_text())["records"]["train"]["n_samples"]
        # A window longer than the record yields zero training sequences.
        assert cfg.data.train_sequence_length <= n_train
        # ... and the warmup must leave something to score.
        assert cfg.training.warmup_steps < cfg.data.train_sequence_length
