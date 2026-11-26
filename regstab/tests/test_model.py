"""DRN rollout and the regional certificate in closed loop."""

import pytest
import torch

from regstab.model import DRN, LSTM, dzn, simulate


def _step(m, x, u):
    w = dzn(x @ m["C2"].T + u @ m["D21"].T)
    return x @ m["A"].T + u @ m["B"].T + w @ m["B2"].T


def test_dzn():
    v = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.7, 1.0, 2.5])
    assert torch.equal(dzn(v), torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5]))


def test_simulate_matches_naive_loop():
    torch.manual_seed(0)
    model = DRN(nu=2, ny=3, nx=4, nz=6)
    u = torch.randn(5, 30, 2, dtype=torch.float64) * 3
    y, x = model.simulate(u)
    m = model.matrices()
    xk = torch.zeros(5, 4, dtype=torch.float64)
    for k in range(30):
        assert torch.allclose(x[:, k], xk)
        assert torch.allclose(y[:, k], xk @ m["C"].T)
        xk = _step(m, xk, u[:, k])


@pytest.mark.parametrize("certificate", ["regional", "global"])
@pytest.mark.parametrize("scale", [0.02, 0.3, 2.0])
def test_unit_ball_is_forward_invariant_for_admissible_inputs(certificate, scale):
    """Theorem 1: x_0 ∈ B₁ and ‖u_k‖² ≤ s²(1 - α²‖x_k‖²) keep x_k ∈ B₁ and
    ‖x_{k+1}‖² ≤ α²‖x_k‖² + ‖u_k‖²/s²."""
    g = torch.Generator().manual_seed(1)
    nx, nu, nz = 3, 2, 8
    model = DRN(nu, 1, nx, nz, certificate, alpha=0.95, s=2.0)
    model.sample_free_parameters(scale, generator=g)
    with torch.no_grad():
        m = model.matrices()
        alpha, s = float(model.alpha), float(model.s)
        x = torch.randn(2000, nx, generator=g, dtype=torch.float64)
        x = x / x.norm(dim=1, keepdim=True) * torch.rand(2000, 1, generator=g, dtype=torch.float64)
        for _ in range(50):
            radius = s * torch.sqrt((1 - alpha**2 * (x**2).sum(1, keepdim=True)).clamp_min(0))
            d = torch.randn(2000, nu, generator=g, dtype=torch.float64)
            u = d / d.norm(dim=1, keepdim=True) * radius  # on the boundary of U(x)
            x_next = _step(m, x, u)
            lhs = (x_next**2).sum(1)
            rhs = alpha**2 * (x**2).sum(1) + (u**2).sum(1) / s**2
            assert (lhs <= rhs + 1e-10).all()
            assert ((x_next**2).sum(1) <= 1 + 1e-10).all()
            x = x_next


def test_uncertified_model_has_free_matrices_and_fixed_scalars():
    model = DRN(1, 1, 2, 4, certificate="none")
    names = {n for n, _ in model.named_parameters()}
    assert {"theta.A", "theta.B", "theta.B2", "theta.C2", "theta.D21", "C"} == names
    assert not model.certified
    y = model(torch.randn(3, 10, 1, dtype=torch.float64))
    assert y.shape == (3, 10, 1)


def test_scalars_trainable_flags():
    model = DRN(1, 1, 2, 4, alpha=0.9, s=3.0, train_alpha=False, train_s=True)
    assert not model.alpha_raw.requires_grad and model.log_s.requires_grad
    assert abs(float(model.alpha) - 0.9) < 1e-12 and abs(float(model.s) - 3.0) < 1e-12


def test_invalid_arguments():
    with pytest.raises(ValueError):
        DRN(1, 1, 2, 4, certificate="local")
    with pytest.raises(ValueError):
        DRN(1, 1, 2, 4, alpha=1.0)


def test_lstm_shapes():
    model = LSTM(2, 3, hidden_size=8)
    assert model(torch.randn(4, 12, 2)).shape == (4, 12, 3)
    assert not model.certified
