"""``use_custom_regularization: false`` -> the initial theta need not be feasible.

Without the LMI barrier in the loss the trainer never keeps theta feasible (its
repair/rollback is gated on ``regularization_weight > 0``), so there is nothing
for a feasible start to preserve. ``initialize_parameters`` therefore stops after
the identity draw: no MaxS/D21 repair solve, no solver dependency, and an
infeasible draw is not an error.

Companion to ``test_d21_bootstrap.py``, which pins the opposite case (barrier on
-> the solve runs). These tests need no MOSEK precisely because nothing solves.
"""

import numpy as np
import torch

from sysid.config import InitializationConfig
from sysid.models.constrained_rnn import SimpleLure


class _Normalizer:
    """Normalizer stand-in with the Duffing training-set scales."""

    def __init__(self, input_std=0.19058719, output_std=0.14378139):
        self.input_std = np.array([[[input_std]]])
        self.output_std = np.array([[[output_std]]])

    def transform_inputs(self, x):
        return np.asarray(x) / self.input_std

    def transform_outputs(self, x):
        return np.asarray(x) / self.output_std


def _training_data(n_seq=3, n_steps=200, seed=0):
    rng = np.random.default_rng(seed)
    u = 0.19 * rng.standard_normal((n_seq, n_steps, 1))
    u[0, 0, 0] = 1.85  # the peak that makes |d_n| ~ 9.7 after scale_only
    q = 0.14 * rng.standard_normal((n_seq, n_steps, 1))
    x = np.concatenate([q, 0.5 * rng.standard_normal((n_seq, n_steps, 1))], axis=-1)
    return u, x, q


def _make_model(seed=0):
    torch.manual_seed(seed)
    return SimpleLure(
        nd=1, ne=1, nx=2, nw=20, activation="dzn", ts=0.05,
        custom_params={
            "learn_L": True,
            "identity_init": {
                "A": {"scale": 1.0},
                "B2": {"std": 0.1},
                "C2": {"std": 1.0},
                "D21": {"std": 1.0},
            },
            "structural_constraints": {
                "D": {"fixed": True, "value": 0.0},
                "D12": {"fixed": True, "value": 0.0},
            },
        },
    )


def _init(model, seed=0, **kw):
    torch.manual_seed(seed)
    u, x, q = _training_data()
    return model.initialize_parameters(
        u, x, q,
        init_config=InitializationConfig(method="identity"),
        normalizer=_Normalizer(),
        **kw,
    )


class TestNoSolveWithoutRegularization:
    def test_solve_is_not_called(self, monkeypatch):
        m = _make_model(seed=1)
        monkeypatch.setattr(
            SimpleLure, "analysis_problem_init",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("analysis_problem_init must not run")
            ),
        )
        _init(m, seed=1, use_custom_regularization=False)

    def test_identity_draw_is_kept_verbatim(self):
        """theta must be exactly what ``_init_identity`` produced."""
        expected = _make_model(seed=2)
        torch.manual_seed(2)
        expected._init_identity(_Normalizer())

        m = _make_model(seed=2)
        _init(m, seed=2, use_custom_regularization=False)

        for name in ("A", "B", "B2", "C", "C2", "D", "D12", "D21"):
            assert torch.equal(
                getattr(m, name).detach(), getattr(expected, name).detach()
            ), f"{name} was modified"

    def test_certificate_params_stay_at_constructor_values(self):
        """No solve ran, so (P, L, s) are still I / 0 / 1."""
        m = _make_model(seed=3)
        _init(m, seed=3, use_custom_regularization=False)

        assert torch.equal(m.P.detach(), torch.eye(m.nx, dtype=m.P.dtype))
        assert float(torch.linalg.norm(m.L.detach())) == 0.0
        assert float(m.s) == 1.0

    def test_infeasible_draw_is_not_an_error(self):
        """The barrier-on path raises on an infeasible draw; this one must not."""
        m = _make_model(seed=4)
        report = _init(m, seed=4, use_custom_regularization=False)

        # The point of the test: the D21 draw does land infeasible here.
        assert not m.check_constraints()
        assert report is not None
        assert report.constraints_satisfied is False

    def test_buffers_and_diagnostics_still_filled(self):
        """Only the repair solve is skipped — y_max / dead-zone report remain."""
        m = _make_model(seed=5)
        report = _init(m, seed=5, use_custom_regularization=False)

        _, _, q = _training_data()
        assert report.y_max == float(np.abs(q).max())
        assert float(m.y_max) == report.y_max
        assert report.firing_rate is not None

    def test_default_still_solves(self, monkeypatch):
        """Omitting the flag must keep the barrier-on behaviour (solve attempted)."""
        called = []

        def _fake(self, learn_B=False, learn_D21=True):
            called.append((learn_B, learn_D21))
            return True

        monkeypatch.setattr(SimpleLure, "analysis_problem_init", _fake)
        m = _make_model(seed=6)
        _init(m, seed=6)  # no use_custom_regularization argument

        assert called == [(False, True)]
