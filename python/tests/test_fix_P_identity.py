"""Tests for ``custom_params.fix_P_identity``: the Lyapunov matrix pinned to P = I.

Fixing P = I is without loss of generality — the state change x = T x̃ with
P = T Tᵀ maps any certificate to P̃ = I — so the flag must (a) keep P out of the
gradient, (b) leave the input-output map and certificate quantities invariant
when it re-coordinatizes, and (c) bring every SDP write-back back to P = I.
"""

import numpy as np
import pytest
import torch

from sysid.models.constrained_rnn import SimpleLure

from .solver_utils import requires_mosek


def _make_model(fix_P: bool, nx: int = 2, nw: int = 2) -> SimpleLure:
    m = SimpleLure(
        nd=1, ne=1, nx=nx, nw=nw, activation="dzn",
        custom_params={"learn_L": True, "fix_P_identity": fix_P},
    )
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        m.A.data = 0.5 * torch.eye(nx, dtype=m.A.dtype)
        m.B.data = 0.1 * torch.ones_like(m.B)
        m.B2.data = 0.05 * torch.randn(m.B2.shape, generator=g, dtype=m.B2.dtype)
        m.C.data = torch.randn(m.C.shape, generator=g, dtype=m.C.dtype)
        m.C2.data = 0.5 * torch.randn(m.C2.shape, generator=g, dtype=m.C2.dtype)
        m.D21.data = 0.1 * torch.ones_like(m.D21)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))  # alpha = 0.9
    return m


def _set_nonidentity_certificate(m: SimpleLure) -> None:
    """Put a generic SPD P and a nonzero L into the model (no SDP needed)."""
    with torch.no_grad():
        Q = torch.tensor([[2.0, 0.3], [0.3, 0.5]], dtype=m.P.dtype)
        m.P.data = Q
        m.L.data = 0.1 * torch.arange(m.L.numel(), dtype=m.L.dtype).reshape(m.L.shape)
        m.s.data = torch.tensor(0.7, dtype=m.s.dtype)


def test_default_keeps_P_learnable():
    m = _make_model(fix_P=False)
    assert m.P.requires_grad
    assert not m.fix_P_identity


def test_fix_P_identity_freezes_P():
    m = _make_model(fix_P=True)
    assert not m.P.requires_grad
    assert torch.equal(m.P, torch.eye(2, dtype=m.P.dtype))
    assert all(p is not m.P for p in m.parameters() if p.requires_grad)


def test_structural_constraint_on_state_map_rejected():
    with pytest.raises(ValueError, match="fix_P_identity"):
        SimpleLure(
            nd=1, ne=1, nx=2, nw=2, activation="dzn",
            custom_params={
                "fix_P_identity": True,
                "structural_constraints": {"A": {"fixed": True, "value": 0.5}},
            },
        )
    # Constraints on maps that do not touch the state are fine.
    SimpleLure(
        nd=1, ne=1, nx=2, nw=2, activation="dzn",
        custom_params={
            "fix_P_identity": True,
            "structural_constraints": {"D21": {"fixed": True, "value": 0.0}},
        },
    )


def test_transform_preserves_io_map_and_certificate():
    """x = T x̃: same outputs, P̃ = I, invariant LMI feasibility and output set."""
    m = _make_model(fix_P=True)
    _set_nonidentity_certificate(m)
    d = 0.5 * torch.randn(3, 60, 1, generator=torch.Generator().manual_seed(1), dtype=m.A.dtype)

    y_before, _, _ = m(d)
    lmi_before = [float(torch.linalg.eigvalsh(f()).min()) for f in m.get_lmis()]
    out_before = m.certified_output_set().y_bar
    s_before = float(m.s)

    m._transform_to_identity_P()

    y_after, _, _ = m(d)
    lmi_after = [float(torch.linalg.eigvalsh(f()).min()) for f in m.get_lmis()]
    torch.testing.assert_close(m.P.data, torch.eye(2, dtype=m.P.dtype))
    torch.testing.assert_close(y_after, y_before, rtol=1e-10, atol=1e-10)
    assert float(m.s) == s_before
    assert m.certified_output_set().y_bar == pytest.approx(out_before, rel=1e-10)
    # Congruence preserves the inertia, so each LMI keeps the sign of its margin.
    assert [v > 0 for v in lmi_after] == [v > 0 for v in lmi_before]


def test_transform_rejects_non_pd_P():
    m = _make_model(fix_P=True)
    with torch.no_grad():
        m.P.data = torch.diag(torch.tensor([1.0, -0.1], dtype=m.P.dtype))
    with pytest.raises(RuntimeError, match="positive"):
        m._transform_to_identity_P()


@requires_mosek
@pytest.mark.parametrize("fix_P", [False, True])
def test_max_s_writeback(fix_P):
    """A MaxS write-back keeps P = I under the flag, with s and the IO map unchanged."""
    m = _make_model(fix_P=fix_P)
    d = 0.2 * torch.randn(2, 40, 1, generator=torch.Generator().manual_seed(2), dtype=m.A.dtype)
    y_before, _, _ = m(d)

    sol = m._synth().max_s()
    assert sol is not None
    m._apply_certificate_solution(sol)

    assert m.check_constraints()
    assert float(m.s) == pytest.approx(float(sol.s))
    y_after, _, _ = m(d)
    torch.testing.assert_close(y_after, y_before, rtol=1e-8, atol=1e-10)
    if fix_P:
        torch.testing.assert_close(m.P.data, torch.eye(2, dtype=m.P.dtype))
    else:
        np.testing.assert_allclose(m.P.detach().numpy(), sol.P)


@requires_mosek
def test_feasibility_repair_returns_to_identity_P():
    """The within-epoch repair (SDP over a free P) ends at P = I and feasible."""
    m = _make_model(fix_P=True)
    m._apply_certificate_solution(m._synth().max_s())
    with torch.no_grad():  # a "gradient step" that breaks the LMIs at P = I
        m.A.data = m.A.data + torch.tensor([[0.0, 0.45], [0.0, 0.0]], dtype=m.A.dtype)
    if m.check_constraints():
        pytest.skip("perturbation did not break the LMIs at P = I")
    assert m.feasibility_problem()
    assert m.check_constraints()
    torch.testing.assert_close(m.P.data, torch.eye(2, dtype=m.P.dtype))


# --------------------------------------------------------------------- strict
# ``fix_P_identity: strict`` holds P = I inside the SDPs too, so θ is never
# re-coordinatized and every solution must come back with P exactly I.


def _make_strict_model() -> SimpleLure:
    m = _make_model(fix_P=True)
    m.fix_P_mode = "strict"
    return m


def test_mode_parsing():
    for raw, mode in [(False, None), (True, "transform"),
                      ("transform", "transform"), ("strict", "strict")]:
        m = SimpleLure(nd=1, ne=1, nx=2, nw=2, activation="dzn",
                       custom_params={"fix_P_identity": raw})
        assert m.fix_P_mode == mode
        assert m.P.requires_grad == (mode is None)
    with pytest.raises(ValueError, match="fix_P_identity"):
        SimpleLure(nd=1, ne=1, nx=2, nw=2, activation="dzn",
                   custom_params={"fix_P_identity": "sometimes"})


def test_strict_allows_structural_constraints_on_state_maps():
    """Strict never transforms θ, so pinned A/B/C are no conflict."""
    m = SimpleLure(
        nd=1, ne=1, nx=2, nw=2, activation="dzn",
        custom_params={
            "fix_P_identity": "strict",
            "structural_constraints": {"A": {"fixed": True, "value": 0.5}},
        },
    )
    assert m.fix_P_mode == "strict"


def test_strict_synthesizer_carries_P_fixed():
    assert _make_model(fix_P=True)._synth().P_fixed is None
    np.testing.assert_array_equal(_make_strict_model()._synth().P_fixed, np.eye(2))


@requires_mosek
def test_strict_max_s_keeps_identity_and_theta():
    m = _make_strict_model()
    theta = {n: getattr(m, n).detach().clone() for n in ("A", "B", "B2", "C", "C2")}
    sol = m._synth().max_s()
    assert sol is not None
    np.testing.assert_array_equal(sol.P, np.eye(2))
    m._apply_certificate_solution(sol)
    assert m.check_constraints()
    for n, v in theta.items():
        assert torch.equal(getattr(m, n), v), f"{n} moved under strict"


@requires_mosek
def test_strict_max_s_is_a_restriction():
    """Same θ, regional regime (strong dead zone): the P = I ceiling sits strictly
    below the free-P one (≈0.74 vs ≈1.48 here). The default test model is not
    usable for this — both solves hit the ε-imposed s ≈ 1000 cap there."""
    def regional(m):
        with torch.no_grad():
            m.A.data = torch.tensor([[0.5, 0.3], [0.0, 0.4]], dtype=m.A.dtype)
            m.B2.data = torch.ones_like(m.B2)
            m.C2.data = torch.ones_like(m.C2)
        return m

    s_strict = regional(_make_strict_model())._synth().max_s().s
    s_free = regional(_make_model(fix_P=False))._synth().max_s().s
    assert s_strict < 0.9 * s_free


@requires_mosek
def test_strict_feasibility_infeasible_when_only_P_neq_I_certifies():
    """A θ certifiable only by a non-identity P is a rollback under strict."""
    m = _make_strict_model()
    with torch.no_grad():
        # Non-normal A: ||A||_2 > 1 (no contraction in the Euclidean norm, so
        # P = I fails) but rho(A) = 0.5 (a weighted P certifies it).
        m.A.data = torch.tensor([[0.5, 3.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B2.data.zero_()
    assert not m.feasibility_problem()
    free = _make_model(fix_P=False)
    with torch.no_grad():
        free.A.data = m.A.data.clone()
        free.B2.data.zero_()
    assert free._synth().feasibility(None) is not None
