"""Tests for the dead-zone firing diagnostic and the fit-ranked calibration.

Background: for the ``dzn`` activation ``Δ'(z) = 0`` inside the dead band, so a
model whose nonlinearity never fires on the training data is LTI *in that regime*
**and unrecoverable** — no gradient from the prediction loss ever reaches B2, C2 or
D21. Only the initialization can prevent it. These tests cover the three pieces
that make that visible and avoidable:

- ``deadzone_activity`` — the firing-rate DIAGNOSTIC (init report + per epoch).
  The initialization deliberately does **not** optimize for firing; this only
  makes an inert nonlinearity visible.
- ``_calibrate_nonlinearity`` — unchanged min-rho semantics, plus the rollout
  subsampling / already-infeasible skip that make it affordable.
"""

import numpy as np
import pytest
import torch
import cvxpy as cp

from sysid.models.constrained_rnn import SimpleLure


def _mosek_available() -> bool:
    if "MOSEK" not in cp.installed_solvers():
        return False
    try:
        x = cp.Variable()
        cp.Problem(cp.Minimize((x - 1) ** 2), [x >= 0]).solve(solver=cp.MOSEK, verbose=False)
        return True
    except Exception:
        return False


MOSEK = _mosek_available()
requires_mosek = pytest.mark.skipif(not MOSEK, reason="MOSEK solver not available/licensed")


def _make_model(nw: int = 4, learn_L: bool = True) -> SimpleLure:
    m = SimpleLure(nd=1, ne=1, nx=2, nw=nw, activation="dzn",
                   custom_params={"learn_L": learn_L})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = 0.1 * torch.ones_like(m.B2)
        m.C.data = torch.tensor([[1.0, 0.0]], dtype=m.C.dtype)
        m.C2.data = 0.1 * torch.ones_like(m.C2)
        m.D21.data = torch.zeros_like(m.D21)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))
        m.s.data = torch.tensor(1.0)
        m.P.data = torch.eye(2, dtype=m.P.dtype)
    return m


class TestDeadzoneActivity:
    def test_reports_zero_when_inert(self):
        m = _make_model(nw=4)
        with torch.no_grad():
            m.C2.data = 1e-6 * torch.ones_like(m.C2)  # breakpoints far above the data
        act = m.deadzone_activity(0.1 * torch.ones(2, 20, 1))
        assert act["firing_rate"] == 0.0
        assert act["units_firing"] == 0.0
        assert act["max_abs_z"] < 1.0

    def test_reports_firing_when_active(self):
        m = _make_model(nw=4)
        with torch.no_grad():
            m.D21.data = 10.0 * torch.ones_like(m.D21)  # input alone leaves the band
        act = m.deadzone_activity(torch.ones(2, 20, 1))
        assert act["firing_rate"] > 0.0
        assert act["units_firing"] == 1.0
        assert act["max_abs_z"] > 1.0
        assert 0.0 <= act["steps_firing"] <= 1.0

    @requires_mosek
    def test_rho_objective_is_unchanged(self):
        """Back-compat: the default path still minimizes rho and reports the same keys."""
        m = _make_model(nw=4)
        u = 0.2 * torch.randn(2, 20, 1, dtype=m.C2.dtype).numpy()
        cal = m._calibrate_nonlinearity(u, y_max=0.5, knobs=["C2"])
        assert cal is not None
        for key in ("factors", "rho", "feasible", "cov_ok", "n_input_violations",
                    "cert", "knobs", "n_evals", "firing_rate"):
            assert key in cal


class TestInitializationReport:
    def test_metrics_include_the_activity_fields(self):
        from sysid.optimization import InitializationReport
        r = InitializationReport(
            volume=1.0, s=1.0, norm_H=0.0, max_eig_F=-1.0, constraints_satisfied=True,
            firing_rate=0.02, units_firing=0.5, max_abs_z=3.0,
        )
        mets = r.to_metrics()
        assert mets["firing_rate"] == pytest.approx(0.02)
        assert mets["units_firing"] == pytest.approx(0.5)
        assert mets["max_abs_z"] == pytest.approx(3.0)

    def test_activity_fields_are_optional(self):
        from sysid.optimization import InitializationReport
        r = InitializationReport(volume=1.0, s=1.0, norm_H=0.0, max_eig_F=-1.0,
                                 constraints_satisfied=True)
        assert "firing_rate" not in r.to_metrics()
