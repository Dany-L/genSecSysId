"""
Tests for the configurable random-init scales in ``SimpleLure._init_identity``.

Verifies that the ``custom_params['identity_init']`` block actually controls the
empirical std/scale of the generated B2, C2, D21 tensors (and the `scale` knob
on A's random row), and that the `value` / `load_from` overrides work.
"""

import numpy as np
import pytest
import torch

from sysid.models.constrained_rnn import SimpleLure


class _MockNormalizer:
    """Minimal normalizer stand-in for ``_init_identity``."""

    def __init__(self, input_std=1.0, output_std=1.0):
        self.input_std = np.array([[input_std]])
        self.output_std = np.array([[output_std]])


def _make_model(nx=2, nw=400, nd=1, ne=1, ts=0.1, custom_params=None):
    """
    Build a SimpleLure sized large enough that empirical std of B2/C2/D21
    (with nx*nw or nz*nx samples) is a tight estimate of the true std.
    """
    return SimpleLure(
        nd=nd, ne=ne, nx=nx, nw=nw,
        activation="tanh",
        delta=0.1,
        custom_params=custom_params,
        ts=ts,
    )


# ---------------------------------------------------------------------------
# Default behavior (no identity_init block) -- must match pre-refactor defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    """No identity_init block: defaults are A.scale=1, B2.std=ts, C2.std=1, D21.std=1."""

    def test_defaults_match_documented_scales(self):
        ts = 0.1
        model = _make_model(ts=ts)
        model._init_identity(normalizer=_MockNormalizer())

        # B2 ~ N(0, ts^2)
        b2 = model.B2.detach().numpy()
        assert b2.std() == pytest.approx(ts, rel=0.15)

        # C2 ~ N(0, 1)
        c2 = model.C2.detach().numpy()
        assert c2.std() == pytest.approx(1.0, rel=0.15)

        # D21 has shape (nz, nd) = (nw, 1). Plenty of samples too.
        d21 = model.D21.detach().numpy()
        assert d21.std() == pytest.approx(1.0, rel=0.15)


# ---------------------------------------------------------------------------
# Configurable std for B2, C2, D21
# ---------------------------------------------------------------------------

class TestConfigurableStd:
    """The `std` knob actually scales the generated Gaussian random parameters."""

    @pytest.mark.parametrize("std", [0.01, 0.1, 0.5, 2.0])
    def test_b2_std_matches_config(self, std):
        model = _make_model(custom_params={
            "identity_init": {"B2": {"std": std}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert model.B2.detach().numpy().std() == pytest.approx(std, rel=0.15)

    @pytest.mark.parametrize("std", [0.01, 0.1, 0.5, 2.0])
    def test_c2_std_matches_config(self, std):
        model = _make_model(custom_params={
            "identity_init": {"C2": {"std": std}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert model.C2.detach().numpy().std() == pytest.approx(std, rel=0.15)

    @pytest.mark.parametrize("std", [0.01, 0.1, 0.5, 2.0])
    def test_d21_std_matches_config(self, std):
        model = _make_model(custom_params={
            "identity_init": {"D21": {"std": std}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert model.D21.detach().numpy().std() == pytest.approx(std, rel=0.15)

    def test_per_parameter_stds_are_independent(self):
        """Setting B2.std should not change C2 or D21's empirical std."""
        model = _make_model(custom_params={
            "identity_init": {"B2": {"std": 0.05}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        # B2 follows the override
        assert model.B2.detach().numpy().std() == pytest.approx(0.05, rel=0.15)
        # C2 and D21 use their defaults (std=1)
        assert model.C2.detach().numpy().std() == pytest.approx(1.0, rel=0.15)
        assert model.D21.detach().numpy().std() == pytest.approx(1.0, rel=0.15)


# ---------------------------------------------------------------------------
# A's random scale (uniform on last row of A_ct)
# ---------------------------------------------------------------------------

class TestAScale:
    """A's last row of A_ct is -A_scale * U(0,1).

    A_dt = I + A_ct*ts (forward Euler), so A_ct = (A - I)/ts inverts it exactly.
    """

    @pytest.mark.parametrize("scale", [0.5, 1.0, 4.0])
    def test_a_random_row_mean_matches_scale(self, scale):
        # Estimate the mean of the random component by sampling many models.
        # E[-scale * U(0,1)] = -scale/2, so E[A_dt[1,j]] = ts * (-scale/2) for j != diagonal.
        # We can read the last row of A_ct back by: (A - I)/ts == A_ct, then check row 1.
        ts = 0.1
        nx = 2
        n_trials = 200
        last_row_samples = []
        for trial in range(n_trials):
            torch.manual_seed(trial)
            np.random.seed(trial)
            model = _make_model(nx=nx, ts=ts, custom_params={
                "identity_init": {"A": {"scale": scale}},
            })
            model._init_identity(normalizer=_MockNormalizer())
            A_ct = (model.A.detach().numpy() - np.eye(nx)) / ts
            last_row_samples.append(A_ct[1, :])  # the random row

        samples = np.array(last_row_samples)  # (trials, nx)
        # Each sample is from Uniform(-scale, 0)
        # Mean = -scale/2, range width = scale
        mean_est = samples.mean()
        range_est = samples.max() - samples.min()
        assert mean_est == pytest.approx(-scale / 2.0, abs=0.05 * scale)
        # Observed range should approach `scale` as trials grow
        assert range_est == pytest.approx(scale, rel=0.1)


# ---------------------------------------------------------------------------
# `value` and `load_from` overrides
# ---------------------------------------------------------------------------

class TestValueOverride:
    """Inline `value` overrides skip random init and use the given tensor exactly."""

    def test_value_override_b2(self):
        nx, nw = 2, 5
        fixed = [[0.1, 0.2, 0.3, 0.4, 0.5],
                 [1.0, 2.0, 3.0, 4.0, 5.0]]
        model = _make_model(nx=nx, nw=nw, custom_params={
            "identity_init": {"B2": {"value": fixed}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert np.allclose(model.B2.detach().numpy(), np.array(fixed))

    def test_value_override_a(self):
        nx = 2
        fixed = [[0.95, 0.05], [-0.05, 0.95]]
        model = _make_model(nx=nx, custom_params={
            "identity_init": {"A": {"value": fixed}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert np.allclose(model.A.detach().numpy(), np.array(fixed))

    def test_value_shape_mismatch_raises(self):
        # B2 should be (nx, nw) = (2, 4); supply wrong shape
        bad = [[1.0, 2.0]]
        model = _make_model(nx=2, nw=4, custom_params={
            "identity_init": {"B2": {"value": bad}},
        })
        with pytest.raises(ValueError, match="shape"):
            model._init_identity(normalizer=_MockNormalizer())


class TestLoadFromOverride:
    """`load_from` reads a .npy file and uses it as the init tensor."""

    def test_load_from_npy(self, tmp_path):
        nx, nw = 2, 5
        expected = np.random.randn(nx, nw).astype(np.float64)
        path = tmp_path / "B2_test.npy"
        np.save(path, expected)

        model = _make_model(nx=nx, nw=nw, custom_params={
            "identity_init": {"B2": {"load_from": str(path)}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert np.allclose(model.B2.detach().numpy(), expected)

    def test_load_from_expands_tilde(self, tmp_path, monkeypatch):
        # Point HOME at tmp_path so "~/..." resolves under our control
        monkeypatch.setenv("HOME", str(tmp_path))
        nx, nw = 2, 5
        expected = np.full((nx, nw), 0.123, dtype=np.float64)
        sub = tmp_path / "init_values"
        sub.mkdir()
        np.save(sub / "B2.npy", expected)

        model = _make_model(nx=nx, nw=nw, custom_params={
            "identity_init": {"B2": {"load_from": "~/init_values/B2.npy"}},
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert np.allclose(model.B2.detach().numpy(), expected)

    def test_missing_file_raises(self):
        model = _make_model(custom_params={
            "identity_init": {"B2": {"load_from": "/nonexistent/path/file.npy"}},
        })
        with pytest.raises(FileNotFoundError):
            model._init_identity(normalizer=_MockNormalizer())

    def test_loaded_shape_mismatch_raises(self, tmp_path):
        # Save a wrong-shape file
        path = tmp_path / "wrong.npy"
        np.save(path, np.zeros((5, 5)))  # B2 expected shape (2, 4)

        model = _make_model(nx=2, nw=4, custom_params={
            "identity_init": {"B2": {"load_from": str(path)}},
        })
        with pytest.raises(ValueError, match="shape"):
            model._init_identity(normalizer=_MockNormalizer())


# ---------------------------------------------------------------------------
# Structural constraints still respected when identity_init is used
# ---------------------------------------------------------------------------

class TestInteractionWithStructuralConstraints:
    """A fully-fixed parameter must not be overwritten even with identity_init present."""

    def test_fixed_parameter_ignores_identity_init(self):
        # Fix B2 via structural_constraints; identity_init.B2 should be ignored
        # because the parameter is fully fixed (skip flag).
        fixed_value = 0.42
        model = _make_model(custom_params={
            "structural_constraints": {
                "B2": {"fixed": True, "value": fixed_value},
            },
            "identity_init": {
                "B2": {"std": 5.0},  # would normally produce large std
            },
        })
        model._init_identity(normalizer=_MockNormalizer())
        assert np.allclose(model.B2.detach().numpy(), fixed_value)
