"""Tests for the three divergence checks (sysid.evaluation.divergence_checks).

No real model/MLflow: a minimal accumulator stub inherits the *real*
``LureRegularizationMixin`` so the safe-set margin ``c`` is genuine
(``‖u‖² − s² + α² xᵀ P⁻¹ x``), while a trivial identity normalizer keeps the
physical/normalized spaces equal. That makes each check's divergence and
output-bound outcomes exactly predictable.
"""

import numpy as np
import torch

from sysid.evaluation.divergence_checks import (
    check_initial_state,
    check_input_scaling,
    check_output_bounds,
    diverged_at_end,
    left_safe_set,
)
from sysid.models._lure_regularization import LureRegularizationMixin


class _AccumulatorLure(LureRegularizationMixin):
    """State accumulates the scalar input: ``x_{k+1}[0] = x_k[0] + d_k``.

    Second state component stays 0; the output equals the post-step state. With
    a zero-input tail the accumulated state (hence the final margin ``c[-1]``)
    settles at ``sum(inputs)``, so scaling the input scales divergence. P = I so
    ``xᵀ P⁻¹ x = x[0]²``; ``tau = 0`` ⇒ ``α = σ(0) = 0.5``.
    """

    def __init__(self, s=3.0, tau=0.0):
        self.s = torch.tensor(float(s), dtype=torch.float64)
        self.tau = torch.tensor(float(tau), dtype=torch.float64)
        self.P = torch.eye(2, dtype=torch.float64)

    def forward_unfiltered(self, d, x0=None):
        B, N, _ = d.shape
        xs = torch.zeros(B, N + 1, 2, dtype=d.dtype)
        if x0 is not None:
            xs[:, 0, :] = x0[:, :2].to(d.dtype)
        acc = xs[:, 0, 0].clone()                # component 0 accumulates input
        const1 = xs[:, 0, 1].clone()             # component 1 held constant
        for k in range(N):
            acc = acc + d[:, k, 0]
            xs[:, k + 1, 0] = acc
            xs[:, k + 1, 1] = const1
        y = xs[:, 1:, 0:1]                       # (B, N, 1)
        w = torch.zeros(B, N, 1, dtype=d.dtype)
        return y, (xs, w), d


class _IdentityNormalizer:
    def transform_inputs(self, u):
        return np.asarray(u)

    def inverse_transform_inputs(self, u):
        return np.asarray(u)

    def inverse_transform_outputs(self, y):
        return np.asarray(y)


# ── pure margin classification ────────────────────────────────────────────────
def test_diverged_at_end_and_left_safe_set():
    c = np.array([
        [-1.0, -0.5, 0.2],   # ends outside  -> diverged + left
        [-1.0, 0.3, -0.1],   # transient only -> left, not diverged
        [-2.0, -1.0, -0.5],  # always inside  -> neither
    ])
    assert list(diverged_at_end(c)) == [True, False, False]
    assert list(left_safe_set(c)) == [True, True, False]


# ── case 1: output bounds on true diverging inputs ────────────────────────────
def test_check_output_bounds_flags_bounds_and_divergence():
    model = _AccumulatorLure(s=3.0)          # alpha=0.5
    norm = _IdentityNormalizer()
    inputs_div = np.ones((1, 3, 1))          # one trajectory, cumulative peak = 3
    out = check_output_bounds(
        model, norm, inputs_div, y_bar=5.0, y_max=2.5, pad_zeros=2,
    )
    row = out["summary"].iloc[0]
    assert row["peak_abs_y"] == 3.0          # max cumulative state
    assert bool(row["within_y_bar"]) is True     # 3 < 5
    assert bool(row["within_y_max"]) is False    # 3 !< 2.5
    # settled c[-1] = 2.25 f^2 - 9 with f=1 -> negative -> not diverged
    assert bool(row["diverged"]) is False
    assert len(out["y_hat"]) == 1 and len(out["c"]) == 1


# ── case 2: scaling until instability ─────────────────────────────────────────
def test_check_input_scaling_finds_onset_factor():
    model = _AccumulatorLure(s=3.0)          # alpha=0.5, P=I
    norm = _IdentityNormalizer()
    inputs_div = np.ones((1, 3, 1))
    # settled margin c[-1] = alpha^2 (3f)^2 - s^2 = 2.25 f^2 - 9  > 0  <=>  f > 2
    out = check_input_scaling(
        model, norm, inputs_div, factors=[1.0, 2.0, 3.0], pad_zeros=2,
    )
    n_div = list(out["summary"]["n_diverged"])
    assert n_div == [0, 0, 1]                # only f=3 diverges
    assert out["onset_factor"] == 3.0
    # max margin is monotone in the factor
    peaks = list(out["summary"]["max_peak_c"])
    assert peaks[0] < peaks[1] < peaks[2]


# ── case 3: initial states outside the safe ellipsoid ─────────────────────────
def test_check_initial_state_diverges_only_outside():
    model = _AccumulatorLure(s=3.0)          # alpha=0.5
    # radius = scale * s/alpha on {xᵀ X x = r²} => alpha^2 xᵀXx = scale^2 s^2,
    # so with zero input c = (scale^2 - 1) s^2: inside for scale<1, outside for >1.
    out = check_initial_state(
        model, scales=[0.5, 1.5], n_traj=8, horizon=5, seed=0,
        input_amp_frac=0.0,                  # no excitation -> state stays at x0
    )
    frac = dict(zip(out["summary"]["scale"], out["summary"]["diverged_frac"]))
    assert frac[0.5] == 0.0                  # inside the safe set
    assert frac[1.5] == 1.0                  # every sample outside


def test_check_initial_state_summary_columns():
    model = _AccumulatorLure()
    out = check_initial_state(model, scales=[2.0], n_traj=4, horizon=3,
                              input_amp_frac=0.0)
    assert set(out["summary"].columns) == {
        "scale", "n_diverged", "diverged_frac", "n_traj"
    }
    assert out["xs"][2.0].shape[0] == 4      # n_traj trajectories stored
