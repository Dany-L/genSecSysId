"""Tests for the shared max-feasible-s SDP and its training-time wrappers.

- ``maximize_s_on_violation`` only re-solves when the input constraint is
  currently violated (c > 0); the gate itself is solver-free.
- ``solve_max_s`` and ``post_process`` share the same ``_max_s_sdp`` core, so a
  successful solve writes a constraint-satisfying certificate and the two entry
  points agree. These need MOSEK and are skipped otherwise.
"""

import numpy as np
import pytest
import torch
import cvxpy as cp
from torch.utils.data import DataLoader, TensorDataset

from sysid.models.constrained_rnn import SimpleLure
from sysid.training.trainer import Trainer
from sysid.training import get_loss_function, get_optimizer


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


def _make_model(s_value: float = 0.05) -> SimpleLure:
    """A small, stable Lure system for which the max-s SDP is feasible."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params={"learn_L": True})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = torch.zeros_like(m.B2)
        m.C2.data = torch.tensor([[0.1, 0.1]], dtype=m.C2.dtype)
        m.D21.data = torch.tensor([[0.1]], dtype=m.D21.dtype)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))  # alpha = 0.9
        m.s.data = torch.tensor(float(s_value))
    return m


class TestMaximizeSGate:
    """The c>0 gate must be solver-free and inert when the constraint holds."""

    def test_no_solve_when_constraint_satisfied(self):
        m = _make_model(s_value=5.0)  # s^2 = 25, generous
        with torch.no_grad():
            m.P.data = torch.eye(2)
        u = torch.tensor([[[0.1], [0.3], [0.2]]], dtype=torch.float32)  # ||u||^2 <= 0.09
        x = torch.zeros(1, 3, 2)

        before = (m.P.detach().clone(), m.s.detach().clone(), m.L.detach().clone())
        res = m.maximize_s_on_violation(u, x)

        assert res is None  # satisfied -> no SDP solved
        assert torch.equal(m.P, before[0])
        assert torch.equal(m.s, before[1])
        assert torch.equal(m.L, before[2])


@requires_mosek
class TestMaxSSdp:
    """The SDP path (requires MOSEK)."""

    def test_solve_max_s_updates_model_and_is_feasible(self):
        m = _make_model()
        s_new = m.solve_max_s()
        assert s_new is not None and s_new > 0
        assert float(m.s) == pytest.approx(s_new)
        # The written-back certificate must satisfy the model constraints.
        assert m.check_constraints()

    def test_post_process_matches_shared_core(self):
        """post_process routes through the same _max_s_sdp, so its s_opt equals
        what solve_max_s computes from the same starting parameters."""
        s_direct = _make_model().solve_max_s()
        out = _make_model().post_process()
        assert out["success"]
        assert out["s_opt"] == pytest.approx(s_direct, rel=1e-4)

    def test_on_violation_enlarges_s(self):
        m = _make_model(s_value=0.05)  # tiny s -> data violates
        u = torch.tensor([[[0.0], [0.3], [0.0]]], dtype=torch.float32)  # ||u||^2 = 0.09
        x = torch.zeros(1, 3, 2)
        s_before = float(m.s)
        res = m.maximize_s_on_violation(u, x)
        assert res is not None
        assert res > s_before


def _make_loader(u_amp: float, N: int = 5, B: int = 4) -> DataLoader:
    """Constant-amplitude input batches (default float64 to match the model)."""
    d = u_amp * torch.ones(B, N, 1)
    e = torch.zeros(B, N, 1)
    return DataLoader(TensorDataset(d, e), batch_size=2)


def _make_trainer(tmp_path, model, loader, **kwargs) -> Trainer:
    optimizer = get_optimizer(model.parameters(), learning_rate=1e-3)
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=get_loss_function("mse"),
        optimizer=optimizer,
        device="cpu",
        output_dir=str(tmp_path / "o"),
        model_dir=str(tmp_path / "m"),
        log_dir=str(tmp_path / "l"),
        mlflow_tracking=False,
        **kwargs,
    )


class TestTrainerIntegration:
    """Trainer wiring of maximize_s_on_violation (default off)."""

    def test_flag_defaults_to_false(self, tmp_path):
        t = _make_trainer(tmp_path, _make_model(), _make_loader(0.1))
        assert t.solve_max_s_on_violation is False

    def test_noop_when_satisfied(self, tmp_path):
        """No SDP and no mutation when the data does not breach s (solver-free)."""
        m = _make_model(s_value=5.0)  # s^2 = 25, generous
        t = _make_trainer(
            tmp_path, m, _make_loader(0.1), solve_max_s_on_violation=True
        )
        before = (m.P.detach().clone(), m.s.detach().clone(), m.L.detach().clone())
        res = t._maybe_maximize_s(epoch=0)
        assert res is None
        assert torch.equal(m.P, before[0])
        assert torch.equal(m.s, before[1])
        assert torch.equal(m.L, before[2])

    @requires_mosek
    def test_solves_when_violated(self, tmp_path):
        m = _make_model(s_value=0.05)  # tiny s -> input amp 0.3 violates
        t = _make_trainer(
            tmp_path, m, _make_loader(0.3), solve_max_s_on_violation=True
        )
        s_before = float(m.s)
        res = t._maybe_maximize_s(epoch=0)
        assert res is not None
        assert res > s_before
        assert float(m.s) == pytest.approx(res)


class TestMaybeMaximizeSAcrossBatches:
    """_maybe_maximize_s scans the whole training set, then solves at most once.

    The batching is decoupled from the SDP: we first find the peak constraint
    margin over every batch, then solve a single (data-independent) SDP only if
    something is violated. These tests count solve_max_s calls, so they don't
    need MOSEK (the SDP itself is exercised by the MOSEK tests above).
    """

    @staticmethod
    def _count_solve_calls(model):
        """Shadow model.solve_max_s with a counter that pretends to enlarge s."""
        calls = {"n": 0}
        s_new = float(model.s) + 1.0

        def fake_solve_max_s():
            calls["n"] += 1
            return s_new

        model.solve_max_s = fake_solve_max_s
        return calls

    def test_solves_once_when_multiple_batches_violate(self, tmp_path):
        """Every batch violates -> still exactly ONE SDP for the whole epoch."""
        m = _make_model(s_value=0.05)  # tiny s -> input amp 0.3 violates everywhere
        loader = _make_loader(0.3, N=5, B=6)  # 3 batches (batch_size=2)
        t = _make_trainer(tmp_path, m, loader, solve_max_s_on_violation=True)
        calls = self._count_solve_calls(m)

        res = t._maybe_maximize_s(epoch=0)

        assert res is not None
        assert calls["n"] == 1  # one solve per epoch, not one per batch

    def test_no_solve_when_all_batches_satisfied(self, tmp_path):
        """Nothing violates -> the gate stays solver-free across all batches."""
        m = _make_model(s_value=5.0)  # s^2 = 25, generous
        loader = _make_loader(0.1, N=5, B=6)
        t = _make_trainer(tmp_path, m, loader, solve_max_s_on_violation=True)
        calls = self._count_solve_calls(m)

        res = t._maybe_maximize_s(epoch=0)

        assert res is None
        assert calls["n"] == 0

    def test_solves_when_only_a_later_batch_violates(self, tmp_path):
        """A violation in a non-first batch must still be detected (the previous
        per-batch loop that returned early made this order-dependent)."""
        m = _make_model(s_value=0.05)  # tiny s
        # batch_size=1 keeps order: batch 0 = zero input (V=0, c=-s^2 < 0,
        # satisfied), batch 1 = amp 0.3 (violates).
        d = torch.cat([torch.zeros(1, 5, 1), 0.3 * torch.ones(1, 5, 1)], dim=0)
        e = torch.zeros(2, 5, 1)
        loader = DataLoader(TensorDataset(d, e), batch_size=1)
        t = _make_trainer(tmp_path, m, loader, solve_max_s_on_violation=True)
        calls = self._count_solve_calls(m)

        res = t._maybe_maximize_s(epoch=0)

        assert res is not None
        assert calls["n"] == 1


def _make_moderate_model() -> SimpleLure:
    """A feasible system whose max feasible s is moderate (not the EPS-capped
    near-global value of the degenerate _make_model)."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=2, activation="dzn", custom_params={"learn_L": True})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.9, 0.1], [0.0, 0.9]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = 0.3 * torch.ones_like(m.B2)
        m.C2.data = 0.5 * torch.ones_like(m.C2)
        m.D21.data = 0.1 * torch.ones_like(m.D21)
        m.tau.data = torch.tensor(float(np.log(0.99 / 0.01)))  # alpha = 0.99
    return m


@requires_mosek
class TestAnalysisProblemProjection:
    """analysis_problem is the within-epoch fallback: it PROJECTS s onto the
    feasible set (s = min(s_current, s_max)) via the max-s SDP with
    s_upper = s_current, and enforces M >= 0. It keeps a feasible s, clamps an
    overshooting s down to s_max, and rolls back (returns False) only when the
    dynamics are uncertifiable at any s."""

    def test_repair_clamps_overshooting_s_down_to_smax(self):
        """If s has overshot the max feasible value, the projection clamps it back
        down to ~s_max instead of failing (the key win over a fixed-s repair,
        which would be infeasible here and stall training)."""
        m = _make_moderate_model()
        s_max = m.solve_max_s()
        assert s_max is not None

        # Overshoot: push s far above the max feasible value -> certificate breaks.
        with torch.no_grad():
            m.s.data = torch.tensor(s_max * 5.0)
        assert not m.check_constraints()

        assert m.analysis_problem() is True
        assert m.check_constraints()
        assert float(m.s) == pytest.approx(s_max, rel=1e-3)

    def test_repair_keeps_feasible_s_without_inflating(self):
        """When the current s is feasible (< s_max), the projection keeps it and
        does NOT inflate it up to s_max (the key difference from a max-s repair)."""
        m = _make_moderate_model()
        s_max = m.solve_max_s()
        assert s_max is not None
        s_small = 0.5 * s_max
        with torch.no_grad():
            m.s.data = torch.tensor(s_small)

        assert m.analysis_problem() is True  # projects with s_upper=s_small
        assert m.check_constraints()
        # s stays ~s_small, NOT inflated to s_max.
        assert float(m.s) == pytest.approx(s_small, rel=1e-3)
        assert float(m.s) < 0.9 * s_max

    def test_repair_rolls_back_only_on_genuine_infeasibility(self):
        """If the dynamics are uncertifiable at any s (unstable A), it returns False."""
        m = _make_moderate_model()
        with torch.no_grad():
            m.A.data = torch.tensor([[1.5, 0.0], [0.0, 1.5]], dtype=m.A.dtype)  # unstable
        assert m.analysis_problem() is False

    def test_projection_never_inflates_above_s_upper(self):
        """The s_upper cap (S_hat >= 1/s_upper^2) is one-directional: the projected
        s is always <= s_upper, even when the dynamics could certify a larger s."""
        m = _make_moderate_model()
        s_max = m.solve_max_s()
        assert s_max is not None

        for frac in (0.3, 0.7):
            s_upper = frac * s_max
            sol = m._max_s_sdp(s_upper=s_upper)
            assert sol is not None
            assert sol["s"] <= s_upper * (1 + 1e-4)  # capped at s_upper
            assert sol["s"] == pytest.approx(s_upper, rel=1e-3)  # and binds it

    def test_multiplier_is_nonnegative(self):
        m = _make_moderate_model()
        assert m.solve_max_s() is not None
        assert float(m.la.min()) >= -1e-9  # M = diag(la) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
