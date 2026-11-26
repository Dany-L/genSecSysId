"""Size measures of the regional certificate (U, Y) and the regularization penalties.

Problem: small prediction error, large admissible input map U, large output set Y. The
objectives conflict, so we scalarize them (Pareto optimality)::

    min_θ̂  L(Γ(θ̂)) + ν₁ pen(size_U) + ν₂ pen(size_Y)

Every size is a length in the units of the (normalized) signal, so it can be compared
directly with the data, e.g. ``s / max‖u‖``.

Input map U(x) = {u : ‖u‖² ≤ s²(1 - α²‖x‖²)} on the unit ball B₁ ⊂ R^n:
    * ``s``: admissible radius at x = 0.
    * ``sigma``: σ_U = s√(1-α²), admissible radius everywhere in B₁.
    * ``mean_radius``: admissible radius averaged over B₁ (uniform),
      s ∫₀¹ n ρ^{n-1} √(1 - α²ρ²) dρ.

Output set Y = {Cx : ‖x‖ ≤ 1} = E((CCᵀ)⁻¹):
    * ``trace``: √tr(CCᵀ) = ‖C‖_F.
    * ``min_eig``: √λ_min(CCᵀ), the radius of the largest ball inside Y.
    * ``det``: det(CCᵀ)^{1/(2p)}, the geometric mean of the semi-axes.

Penalty forms: ``inverse_square`` (1/size²; for size = s this is ŝ = 1/s² from the
slides) and ``neg_log`` (-log size).
"""

from __future__ import annotations

import numpy as np
import torch

Tensor = torch.Tensor

INPUT_MEASURES = ("s", "sigma", "mean_radius")
OUTPUT_MEASURES = ("trace", "min_eig", "det")
PENALTY_FORMS = ("inverse_square", "neg_log")

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)
_RHO = 0.5 * (_NODES + 1.0)  # Gauss-Legendre on [0, 1]
_W = 0.5 * _WEIGHTS


def input_size(measure: str, s: Tensor, alpha: Tensor, nx: int) -> Tensor:
    if measure == "s":
        return s
    if measure == "sigma":
        return s * torch.sqrt(1.0 - alpha**2)
    if measure == "mean_radius":
        rho = torch.as_tensor(_RHO, dtype=s.dtype, device=s.device)
        w = torch.as_tensor(_W, dtype=s.dtype, device=s.device)
        density = nx * rho ** (nx - 1)
        return s * (w * density * torch.sqrt(1.0 - alpha**2 * rho**2)).sum()
    raise ValueError(f"input measure must be one of {INPUT_MEASURES}, got {measure!r}")


def output_size(measure: str, C: Tensor) -> Tensor:
    CCt = C @ C.T
    if measure == "trace":
        return torch.sqrt(torch.trace(CCt))
    if measure == "min_eig":
        return torch.sqrt(torch.linalg.eigvalsh(CCt)[0].clamp_min(0.0))
    if measure == "det":
        return torch.det(CCt).clamp_min(0.0) ** (1.0 / (2 * C.shape[0]))
    raise ValueError(f"output measure must be one of {OUTPUT_MEASURES}, got {measure!r}")


def penalty(size: Tensor, form: str) -> Tensor:
    if form == "inverse_square":
        return size ** -2
    if form == "neg_log":
        return -torch.log(size)
    raise ValueError(f"penalty form must be one of {PENALTY_FORMS}, got {form!r}")


def all_sizes(model) -> dict:
    """Every input and output size of a certified DRN, as floats (for logging)."""
    with torch.no_grad():
        out = {f"size_in_{m}": float(input_size(m, model.s, model.alpha, model.nx))
               for m in INPUT_MEASURES}
        out.update({f"size_out_{m}": float(output_size(m, model.C)) for m in OUTPUT_MEASURES})
    return out
