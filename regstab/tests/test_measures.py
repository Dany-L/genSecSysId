"""Certificate size measures and the regularization penalties."""

import math

import pytest
import torch

from regstab.config import Config, RegTerm
from regstab.measures import all_sizes, input_size, output_size, penalty
from regstab.model import DRN
from regstab.trainer import regularization

f64 = dict(dtype=torch.float64)


def test_input_sizes_closed_form():
    s, a = torch.tensor(4.0, **f64), torch.tensor(0.6, **f64)
    assert float(input_size("s", s, a, 2)) == 4.0
    assert math.isclose(float(input_size("sigma", s, a, 2)), 4.0 * 0.8)
    # n = 1: mean over [0, 1] of √(1 - α²ρ²) = (α√(1-α²) + asin α) / (2α)
    expected = 4.0 * (0.6 * 0.8 + math.asin(0.6)) / (2 * 0.6)
    assert math.isclose(float(input_size("mean_radius", s, a, 1)), expected, rel_tol=1e-10)


@pytest.mark.parametrize("nx", [1, 2, 5])
def test_mean_radius_matches_monte_carlo(nx):
    g = torch.Generator().manual_seed(nx)
    s, a = torch.tensor(2.0, **f64), torch.tensor(0.95, **f64)
    d = torch.randn(400_000, nx, generator=g, **f64)
    r = torch.rand(400_000, 1, generator=g, **f64) ** (1.0 / nx)  # uniform in the ball
    x = d / d.norm(dim=1, keepdim=True) * r
    mc = float((s * torch.sqrt(1 - a**2 * (x**2).sum(1))).mean())
    assert math.isclose(float(input_size("mean_radius", s, a, nx)), mc, rel_tol=3e-3)


def test_sizes_are_ordered():
    s, a = torch.tensor(1.0, **f64), torch.tensor(0.9, **f64)
    sig, mean, s0 = (float(input_size(m, s, a, 3)) for m in ("sigma", "mean_radius", "s"))
    assert sig < mean < s0


def test_output_sizes_for_diagonal_C():
    C = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], **f64)
    assert math.isclose(float(output_size("trace", C)), 5.0)
    assert math.isclose(float(output_size("min_eig", C)), 3.0)
    assert math.isclose(float(output_size("det", C)), math.sqrt(12.0))


def test_penalties_decrease_with_size():
    small, large = torch.tensor(1.0, **f64), torch.tensor(2.0, **f64)
    for form in ("inverse_square", "neg_log"):
        assert penalty(large, form) < penalty(small, form)
    assert float(penalty(torch.tensor(2.0, **f64), "inverse_square")) == 0.25  # ŝ = 1/s²
    with pytest.raises(ValueError):
        penalty(small, "square")


@pytest.mark.parametrize("measure", ["s", "sigma", "mean_radius"])
def test_input_regularization_pushes_s_up(measure):
    model = DRN(1, 1, 2, 4, alpha=0.9, s=1.0)
    cfg = Config()
    cfg.regularization.input = RegTerm(measure, 1.0)
    cfg.regularization.output = RegTerm("trace", 0.0)
    regularization(model, cfg).backward()
    assert model.log_s.grad < 0  # gradient descent increases s
    if measure != "s":
        assert model.alpha_raw.grad > 0  # ... and decreases α


def test_output_regularization_pushes_C_up():
    model = DRN(1, 1, 2, 4)
    cfg = Config()
    cfg.regularization.output = RegTerm("trace", 1.0)
    reg = regularization(model, cfg)
    reg.backward()
    # the descent direction -grad points along C, i.e. enlarges ‖C‖
    assert float((-model.C.grad * model.C).sum()) > 0


def test_regularization_is_zero_for_uncertified_models():
    cfg = Config()
    cfg.regularization.input = RegTerm("s", 1.0)
    assert float(regularization(DRN(1, 1, 2, 4, certificate="none"), cfg)) == 0.0


def test_all_sizes_keys():
    keys = all_sizes(DRN(1, 1, 2, 4)).keys()
    assert {"size_in_s", "size_in_sigma", "size_in_mean_radius",
            "size_out_trace", "size_out_min_eig", "size_out_det"} == set(keys)
