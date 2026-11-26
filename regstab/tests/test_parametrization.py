"""Γ produces matrices that satisfy Theorem 1 for every free parameter value."""

import numpy as np
import pytest
import torch

from regstab.parametrization import (cayley, certificate_margins, contraction,
                                     contraction_shapes, free_parameter_shapes, gamma)

SHAPES = [(2, 1, 20), (2, 1, 4), (3, 2, 2), (1, 1, 1), (8, 2, 64), (4, 3, 5)]  # incl. q < n+m


def _random_theta(nx, nu, nz, regional, scale, g):
    shapes = free_parameter_shapes(nx, nu, nz, regional)
    return {k: torch.randn(*v, generator=g, dtype=torch.float64) * scale
            for k, v in shapes.items()}


def _margins(m, alpha, s):
    return certificate_margins(*(m[k].detach().numpy() for k in
                                 ("A", "B", "B2", "C2", "D21", "Lam", "H")), alpha=alpha, s=s)


@pytest.mark.parametrize("rows,cols", [(5, 3), (3, 5), (4, 4), (1, 6)])
def test_contraction_has_norm_below_one(rows, cols):
    g = torch.Generator().manual_seed(0)
    for scale in (0.01, 1.0, 10.0):
        p = {k: torch.randn(*v, generator=g, dtype=torch.float64) * scale
             for k, v in contraction_shapes(rows, cols).items()}
        N = contraction(p["X"], p["Y"], p["Z"], rows, cols, eps=1e-6)
        assert N.shape == (rows, cols)
        assert torch.linalg.matrix_norm(N, ord=2) < 1.0


def test_cayley_without_dissipation_is_semi_orthogonal():
    g = torch.Generator().manual_seed(1)
    Y = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Z = torch.randn(2, 3, generator=g, dtype=torch.float64)
    N = cayley(torch.zeros(3, 3, dtype=torch.float64), Y, Z, eps=0.0)
    assert torch.allclose(N.T @ N, torch.eye(3, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("nx,nu,nz", SHAPES)
@pytest.mark.parametrize("regional", [True, False])
def test_conditions_hold_by_construction(nx, nu, nz, regional):
    g = torch.Generator().manual_seed(nx * 100 + nu * 10 + nz)
    for scale in (0.01, 0.1, 1.0, 3.0):
        for alpha, s in ((0.5, 0.1), (0.99, 1.0), (0.9999, 100.0)):
            p = _random_theta(nx, nu, nz, regional, scale, g)
            m = gamma(p, nx, nu, nz, torch.tensor(alpha, dtype=torch.float64),
                      torch.tensor(s, dtype=torch.float64), regional)
            mg = _margins(m, alpha, s)
            # F ≺ 0 up to round-off relative to the size of F
            size = 1.0 + float(m["Lam"].max()) + 1.0 / s**2
            assert mg["max_eig_F"] < 1e-12 * size, (scale, alpha, s, mg)
            assert mg["max_h_norm"] <= 1.0
            if not regional:
                assert mg["max_h_norm"] == 0.0


def test_shapes_and_contraction_rate():
    nx, nu, nz = 3, 2, 7
    g = torch.Generator().manual_seed(3)
    p = _random_theta(nx, nu, nz, True, 0.5, g)
    alpha = 0.9
    m = gamma(p, nx, nu, nz, torch.tensor(alpha, dtype=torch.float64),
              torch.tensor(2.0, dtype=torch.float64), True)
    assert m["A"].shape == (nx, nx) and m["B"].shape == (nx, nu) and m["B2"].shape == (nx, nz)
    assert m["C2"].shape == (nz, nx) and m["D21"].shape == (nz, nu) and m["H"].shape == (nz, nx)
    # ‖A‖ ≤ α because A = α V[:, :n] with ‖V‖ < 1
    assert torch.linalg.matrix_norm(m["A"], ord=2) < alpha


def test_gradients_flow_to_all_free_parameters_and_scalars():
    nx, nu, nz = 2, 1, 6
    g = torch.Generator().manual_seed(4)
    p = {k: v.requires_grad_() for k, v in _random_theta(nx, nu, nz, True, 0.3, g).items()}
    alpha = torch.tensor(0.95, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    m = gamma(p, nx, nu, nz, alpha, s, True)
    loss = sum((m[k] ** 2).sum() for k in ("A", "B", "B2", "C2", "D21"))
    loss.backward()
    for k, v in p.items():
        assert v.grad is not None and torch.isfinite(v.grad).all(), k
    assert torch.isfinite(alpha.grad) and alpha.grad != 0
    assert torch.isfinite(s.grad) and s.grad != 0


def test_input_channel_scales_with_one_over_s():
    """B and D21 are proportional to 1/s for fixed θ̂ (the input enters as u/s)."""
    nx, nu, nz = 2, 1, 5
    g = torch.Generator().manual_seed(5)
    p = _random_theta(nx, nu, nz, True, 0.3, g)
    a = torch.tensor(0.9, dtype=torch.float64)
    m1 = gamma(p, nx, nu, nz, a, torch.tensor(1.0, dtype=torch.float64), True)
    m4 = gamma(p, nx, nu, nz, a, torch.tensor(4.0, dtype=torch.float64), True)
    assert torch.allclose(m4["B"], m1["B"] / 4) and torch.allclose(m4["D21"], m1["D21"] / 4)
    for k in ("A", "B2", "C2"):
        assert torch.allclose(m4[k], m1[k])


def test_certificate_margins_detects_violation():
    nx, nu, nz = 2, 1, 3
    g = torch.Generator().manual_seed(6)
    p = _random_theta(nx, nu, nz, True, 0.3, g)
    m = gamma(p, nx, nu, nz, torch.tensor(0.9, dtype=torch.float64),
              torch.tensor(1.0, dtype=torch.float64), True)
    mats = {k: v.detach().numpy().copy() for k, v in m.items()}
    mats["A"] = mats["A"] + 2.0 * np.eye(nx)  # unstable
    mg = certificate_margins(*(mats[k] for k in ("A", "B", "B2", "C2", "D21", "Lam", "H")),
                             alpha=0.9, s=1.0)
    assert mg["max_eig_F"] > 0
