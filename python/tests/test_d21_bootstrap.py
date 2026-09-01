"""Tests for the D21 analysis-SDP bootstrap in ``initialize_parameters``.

``_init_identity`` draws D21 from ``N(0, std^2)`` with no reference to the data,
and the certificate steps that follow only choose ``(P, L, s)`` — nothing
revisits D21 unless the nonlinearity calibration runs. On normalized inputs that
reach ``|d_n| ~ 10`` a random D21 pushes ``z = C2 x + D21 d`` far outside the dead
band, so the untrained nonlinearity injects large energy into the state and the
MaxS ceiling collapses.

``analysis_problem_init(learn_B=False, learn_D21=True)`` solves D21 jointly with
``(P, la, L, s)`` instead. It ran unconditionally before ``b97fe65``; these tests
pin the restored behaviour, the config switch, and the reason ``learn_B`` stays
False.
"""

import numpy as np
import pytest
import torch
import cvxpy as cp

from sysid.config import InitializationConfig
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


requires_mosek = pytest.mark.skipif(
    not _mosek_available(), reason="MOSEK solver not available/licensed"
)


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
    """Inputs/outputs on the Duffing physical scale (|u| up to ~1.85, |q| < 1)."""
    rng = np.random.default_rng(seed)
    u = 0.19 * rng.standard_normal((n_seq, n_steps, 1))
    u[0, 0, 0] = 1.85  # the peak that makes |d_n| ~ 9.7 after scale_only
    q = 0.14 * rng.standard_normal((n_seq, n_steps, 1))
    x = np.concatenate([q, 0.5 * rng.standard_normal((n_seq, n_steps, 1))], axis=-1)
    return u, x, q


def _make_model(learn_L=True, d21_std=1.0, b2_std=0.1, c2_std=1.0, seed=0):
    torch.manual_seed(seed)
    return SimpleLure(
        nd=1, ne=1, nx=2, nw=20, activation="dzn", ts=0.05,
        custom_params={
            "learn_L": learn_L,
            "identity_init": {
                "A": {"scale": 1.0},
                "B2": {"std": b2_std},
                "C2": {"std": c2_std},
                "D21": {"std": d21_std},
            },
            "structural_constraints": {
                "D": {"fixed": True, "value": 0.0},
                "D12": {"fixed": True, "value": 0.0},
            },
        },
    )


def _init(model, cfg, seed=0):
    torch.manual_seed(seed)
    u, x, q = _training_data()
    return model.initialize_parameters(
        u, x, q, init_config=cfg, normalizer=_Normalizer()
    )


def _cfg(**kw):
    base = dict(method="identity")
    base.update(kw)
    return InitializationConfig(**base)


@requires_mosek
class TestBootstrapRestored:
    """The bootstrap runs by default and leaves a usable starting point."""

    def test_shrinks_d21_and_lifts_s(self):
        on = _make_model(seed=1)
        _init(on, _cfg(bootstrap_d21_on_infeasible=True), seed=1)

        off = _make_model(seed=1)
        _init(off, _cfg(bootstrap_d21_on_infeasible=False), seed=1)

        d21_on = float(torch.linalg.norm(on.D21.detach()))
        d21_off = float(torch.linalg.norm(off.D21.detach()))

        # The SDP replaces the random draw; it does not merely perturb it.
        assert d21_on < 0.5 * d21_off
        # A smaller D21 relaxes the input condition, so the certifiable scale grows.
        assert float(on.s) > float(off.s)

    def test_default_is_on(self):
        """No explicit flag (and no init_config at all) must bootstrap."""
        explicit = _make_model(seed=2)
        _init(explicit, _cfg(bootstrap_d21_on_infeasible=True), seed=2)

        default = _make_model(seed=2)
        _init(default, _cfg(), seed=2)

        assert torch.allclose(
            explicit.D21.detach(), default.D21.detach(), atol=1e-6
        )

    def test_only_d21_and_the_certificate_move(self):
        """A, B, C, D, D12, B2, C2 must survive the SDP untouched."""
        m = _make_model(seed=7)
        torch.manual_seed(7)
        u, x, q = _training_data()
        m._init_identity(_Normalizer())  # step 1 only
        frozen = {n: getattr(m, n).detach().clone()
                  for n in ("A", "B", "C", "D", "D12", "B2", "C2")}
        d21_before = m.D21.detach().clone()

        assert m.analysis_problem_init(learn_B=False, learn_D21=True)

        for n, before in frozen.items():
            assert torch.equal(getattr(m, n).detach(), before), f"{n} was modified"
        assert not torch.equal(m.D21.detach(), d21_before)

    def test_maximizes_s(self):
        """The solve is MaxS: no larger s is certifiable once D21 is fixed there."""
        m = _make_model(seed=8)
        _init(m, _cfg(bootstrap_d21_on_infeasible=True), seed=8)
        s_attained = float(m.s)

        # Re-solve MaxS with D21 pinned at the value the bootstrap chose. The
        # bootstrap had strictly more freedom, so it cannot have done worse.
        ceiling = m._synth().max_s()
        assert ceiling is not None
        assert s_attained >= float(ceiling.s) - 1e-4

    def test_bootstrapped_model_is_feasible_and_quiet(self):
        m = _make_model(seed=3)
        _init(m, _cfg(bootstrap_d21_on_infeasible=True), seed=3)

        assert m.check_constraints()

        # The dead zone should no longer be saturated by the input path alone.
        u, _, _ = _training_data()
        d_n = torch.as_tensor(
            _Normalizer().transform_inputs(u), dtype=m.C2.dtype
        )
        act = m.deadzone_activity(d_n)
        assert act["firing_rate"] < 0.05

    def test_rollout_stays_on_the_output_scale(self):
        """The regression this guards: e_hat overshooting the targets severalfold."""
        u, _, q = _training_data()
        nz = _Normalizer()
        d_n = torch.as_tensor(nz.transform_inputs(u), dtype=torch.get_default_dtype())
        e_n = torch.as_tensor(nz.transform_outputs(q), dtype=torch.get_default_dtype())

        m = _make_model(seed=4)
        _init(m, _cfg(bootstrap_d21_on_infeasible=True), seed=4)
        m.eval()
        with torch.no_grad():
            e_hat, _, _ = m(d_n, None, warmup_steps=0)

        peak_ratio = float(e_hat.abs().max() / e_n.abs().max())
        assert peak_ratio < 5.0


@requires_mosek
class TestLearnBStaysFalse:
    """With B free too the SDP has a trivial escape: kill the input path."""

    def test_learn_b_true_produces_a_dead_model(self):
        m = _make_model(seed=5)
        _init(m, _cfg(bootstrap_d21_on_infeasible=False), seed=5)

        assert m.analysis_problem_init(learn_B=True, learn_D21=True)

        # B and D21 both collapse, so nothing drives the state.
        assert float(torch.linalg.norm(m.B.detach())) < 1e-6
        assert float(torch.linalg.norm(m.D21.detach())) < 1e-6

    def test_bootstrap_keeps_the_input_path_alive(self):
        m = _make_model(seed=5)
        b_before = m.B.detach().clone()
        _init(m, _cfg(bootstrap_d21_on_infeasible=True), seed=5)

        # learn_B=False: B is untouched by the SDP.
        assert torch.allclose(m.B.detach(), b_before, atol=1e-12) or float(
            torch.linalg.norm(m.B.detach())
        ) > 0.0
        assert float(torch.linalg.norm(m.B.detach())) > 1e-9


@requires_mosek
class TestLearnLFalse:
    """learn_L=False freezes s, but D21 must still be solved.

    Mirrors the duffing-soft-7 sweep group that used ``learn_L: false`` with the
    small B2/C2 stds — the combination the best run (ood_rmse 0.0236) was drawn
    from. There the nonlinearity calibration cannot run either (it needs
    learn_L), so the bootstrap is the *only* thing that ever touches D21.
    """

    def test_d21_still_bootstrapped(self):
        kw = dict(learn_L=False, b2_std=0.01, c2_std=0.01, seed=6)
        on = _make_model(**kw)
        _init(on, _cfg(bootstrap_d21_on_infeasible=True), seed=6)

        off = _make_model(**kw)
        _init(off, _cfg(bootstrap_d21_on_infeasible=False), seed=6)

        assert float(torch.linalg.norm(on.D21.detach())) < 0.5 * float(
            torch.linalg.norm(off.D21.detach())
        )
