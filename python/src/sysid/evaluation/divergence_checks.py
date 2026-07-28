"""Three divergence checks for a post-processed Lure model.

Answers the notebook question *"does this model diverge?"* three complementary
ways. All three share the same safe-set membership margin the training
regularizer uses (``get_regularization_input(..., return_c=True)``): the state
has left the safe set ``α² xᵀ P⁻¹ x + ‖u‖² ≤ s²`` exactly where the margin
``c > 0``. A trajectory is called *diverged* when its **final** state is still
outside (``c[-1] > 0``) — a transient excursion that returns is not divergence —
and *left the safe set* when ``c`` is ever positive.

  1. :func:`check_output_bounds` — drive the model with the true system's
     diverging inputs and check the model output stays within the certified
     bound ``y_bar`` and the training envelope ``y_max``.
  2. :func:`check_input_scaling` — scale those diverging inputs by increasing
     factors and find the smallest factor at which the model itself diverges.
  3. :func:`check_initial_state` — start from initial states placed *outside* the
     safe ellipsoid and check whether the model settles back or runs away.

Cases 2 and 3 mirror the input- / initial-state-violation regimes of
:func:`~sysid.evaluation.regional_verification.regional_verification` but operate
on the given model + arrays directly (no MLflow, no artifacts, no file IO), so
they can be called interactively from a notebook. The heavy classification /
sampling primitives are shared with that module.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .regional_verification import _make_lp_noise, _sample_on_ellipsoid


# ── safe-set margin classification (pure, array-only) ─────────────────────────
def diverged_at_end(c: np.ndarray) -> np.ndarray:
    """Per-trajectory ``c[:, -1] > 0`` — final state left the safe set."""
    return np.asarray(c)[:, -1] > 0


def left_safe_set(c: np.ndarray) -> np.ndarray:
    """Per-trajectory ``any(c > 0)`` — ever left the safe set (incl. transient)."""
    return np.any(np.asarray(c) > 0, axis=1)


# ── model plumbing ────────────────────────────────────────────────────────────
def _forward(model, u_t: torch.Tensor, x0: Optional[torch.Tensor] = None,
             warmup_steps: int = 0):
    """Roll the model out with the safety filter bypassed when present.

    Mirrors ``regional_verification.simulate_model`` / the notebook's own
    ``forward_unfiltered if hasattr(...) else model(...)`` pattern, so the margin
    ``c`` reflects raw (unprotected) behaviour rather than the filtered input.
    """
    if hasattr(model, "forward_unfiltered"):
        return model.forward_unfiltered(u_t, x0)
    return model(u_t, x0, warmup_steps=warmup_steps)


def _rollout(model, u_n, x0=None, warmup_steps: int = 0):
    """Run ``u_n`` (normalized, ``(B,T,nd)`` or ``(T,nd)``) through the model.

    Returns ``(y_hat_n, xs, u_used, c)`` as numpy arrays, where ``c`` is the
    safe-set margin from ``get_regularization_input``.
    """
    dtype = model.P.dtype
    u_t = torch.as_tensor(np.asarray(u_n), dtype=dtype)
    if u_t.dim() == 2:
        u_t = u_t.unsqueeze(0)
    x0_t = None if x0 is None else torch.as_tensor(np.asarray(x0), dtype=dtype)
    with torch.no_grad():
        y_hat_n, (xs, _), u_used = _forward(model, u_t, x0_t, warmup_steps)
        _, c = model.get_regularization_input(
            u_used, xs, return_c=True, warmup_steps=warmup_steps
        )
    return (
        y_hat_n.cpu().numpy(),
        xs.cpu().numpy(),
        u_used.cpu().numpy(),
        c.cpu().numpy(),
    )


def _model_safe_set(model):
    """Extract ``(s, alpha, P, X)`` (``X = P⁻¹``, ``alpha = σ(tau)``)."""
    s = float(model.s.detach().cpu().numpy())
    alpha = float(torch.sigmoid(model.tau.detach()).cpu().numpy())
    P = model.P.detach().cpu().numpy()
    X = np.linalg.inv(P)
    return s, alpha, P, X


def _pad_input(u_i: np.ndarray, nd: int, pad_zeros: int) -> np.ndarray:
    """``(T, nd)`` physical input -> ``(1, T + pad_zeros, nd)`` with a zero tail.

    NaNs (ragged/short trajectories) are zero-filled; the trailing zero-input
    steps let the state settle so the ``c[-1] > 0`` divergence test is meaningful.
    """
    u = np.nan_to_num(np.asarray(u_i), nan=0.0).reshape(1, -1, nd)
    if pad_zeros:
        u = np.concatenate([u, np.zeros((1, pad_zeros, nd))], axis=1)
    return u


# ── case 1: output bounds on the true diverging inputs ────────────────────────
def check_output_bounds(
    model,
    normalizer,
    inputs_div: np.ndarray,
    y_bar: float,
    y_max: float,
    pad_zeros: int = 1000,
    warmup_steps: int = 0,
) -> Dict[str, object]:
    """Case 1: model outputs driven by the true system's diverging inputs.

    For each trajectory in ``inputs_div`` (padded with ``pad_zeros`` zero-input
    steps) roll the model out unfiltered and record the peak absolute *physical*
    output, whether it stays within the certified bound ``y_bar`` and the
    training envelope ``y_max``, and whether the final state left the safe set.

    Returns ``{"summary": DataFrame, "y_hat": [...], "xs": [...], "c": [...]}``;
    the arrays are kept for plotting. ``summary`` has one row per trajectory plus
    boolean ``within_y_bar`` / ``within_y_max`` / ``diverged`` columns.
    """
    records: List[dict] = []
    y_hats, xs_all, c_all = [], [], []
    for i, u_i in enumerate(inputs_div):
        nd = np.asarray(u_i).shape[-1]
        u_phys = _pad_input(u_i, nd, pad_zeros)
        u_n = normalizer.transform_inputs(u_phys)
        y_hat_n, xs, _, c = _rollout(model, u_n, warmup_steps=warmup_steps)
        y_hat = normalizer.inverse_transform_outputs(y_hat_n)
        peak = float(np.max(np.abs(y_hat[0, warmup_steps:, 0])))
        records.append({
            "traj": i,
            "peak_abs_y": peak,
            "within_y_bar": peak < y_bar,
            "within_y_max": peak < y_max,
            "diverged": bool(diverged_at_end(c)[0]),
        })
        y_hats.append(y_hat[0])
        xs_all.append(xs[0])
        c_all.append(c[0])
    return {
        "summary": pd.DataFrame.from_records(records),
        "y_hat": y_hats,
        "xs": xs_all,
        "c": c_all,
    }


# ── case 2: scale the true diverging inputs until instability ──────────────────
def check_input_scaling(
    model,
    normalizer,
    inputs_div: np.ndarray,
    factors: Sequence[float],
    pad_zeros: int = 1000,
    warmup_steps: int = 0,
) -> Dict[str, object]:
    """Case 2: scale the true diverging inputs and see if the model can be driven
    unstable.

    For every factor in ``factors`` scale each diverging input by that factor,
    roll the model out unfiltered, and report the fraction of trajectories whose
    final state leaves the safe set (``c[-1] > 0``) together with the largest
    margin reached. ``factor = 1`` is the unscaled baseline (matches case 1's
    divergence column); ``factor > 1`` amplifies the excitation.

    Returns ``{"summary": DataFrame, "onset_factor": float|None, "xs": {...},
    "c": {...}}``. ``onset_factor`` is the smallest factor with any divergence
    (``None`` if the model never diverges). ``xs`` / ``c`` are keyed by factor
    for plotting.
    """
    records: List[dict] = []
    xs_by_factor: Dict[float, np.ndarray] = {}
    c_by_factor: Dict[float, np.ndarray] = {}
    onset = None
    for f in factors:
        diverged, peak_c, xs_list, c_list = [], [], [], []
        for u_i in inputs_div:
            nd = np.asarray(u_i).shape[-1]
            u_phys = float(f) * _pad_input(u_i, nd, pad_zeros)
            u_n = normalizer.transform_inputs(u_phys)
            _, xs, _, c = _rollout(model, u_n, warmup_steps=warmup_steps)
            diverged.append(bool(diverged_at_end(c)[0]))
            peak_c.append(float(np.max(c[0])))
            xs_list.append(xs[0])
            c_list.append(c[0])
        n_div = int(np.sum(diverged))
        records.append({
            "factor": float(f),
            "n_diverged": n_div,
            "diverged_frac": float(np.mean(diverged)),
            "max_peak_c": float(np.max(peak_c)),
        })
        xs_by_factor[float(f)] = np.array(xs_list)
        c_by_factor[float(f)] = np.array(c_list)
        if n_div > 0 and onset is None:
            onset = float(f)
    return {
        "summary": pd.DataFrame.from_records(records),
        "onset_factor": onset,
        "xs": xs_by_factor,
        "c": c_by_factor,
    }


# ── case 3: initial states outside the safe ellipsoid ─────────────────────────
def check_initial_state(
    model,
    scales: Sequence[float],
    n_traj: int = 20,
    horizon: int = 200,
    seed: int = 0,
    input_amp_frac: float = 0.01,
    sampling_time: float = 0.05,
) -> Dict[str, object]:
    """Case 3: initial states placed outside the safe ellipsoid.

    For each scale sample ``n_traj`` states on the ellipsoid
    ``{x : xᵀ X x = (scale · s/α)²}`` (``scale > 1`` ⇒ outside the safe set),
    drive the model with modest LP-filtered noise (peak ``input_amp_frac · s`` in
    normalized units, well inside the input bound) and report the fraction whose
    final state is still outside / diverges.

    Returns ``{"summary": DataFrame, "xs": {...}, "c": {...}}`` keyed by scale.
    Requires a 2-state-input-free model with scalar input (``nd == 1``); the LP
    noise excitation is scalar (as in ``regional_verification``).
    """
    s, alpha, _, X = _model_safe_set(model)
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    xs_by_scale: Dict[float, np.ndarray] = {}
    c_by_scale: Dict[float, np.ndarray] = {}
    for scale in scales:
        radius = float(scale) * s / max(alpha, 1e-12)
        x0 = _sample_on_ellipsoid(rng, X, radius=radius, n=n_traj)
        u_n = np.stack([
            _make_lp_noise(rng, horizon, amp_max=input_amp_frac * s, Ts=sampling_time)
            for _ in range(n_traj)
        ])
        _, xs, _, c = _rollout(model, u_n[..., None], x0=x0)
        diverged = diverged_at_end(c)
        records.append({
            "scale": float(scale),
            "n_diverged": int(diverged.sum()),
            "diverged_frac": float(diverged.mean()),
            "n_traj": n_traj,
        })
        xs_by_scale[float(scale)] = xs
        c_by_scale[float(scale)] = c
    return {
        "summary": pd.DataFrame.from_records(records),
        "xs": xs_by_scale,
        "c": c_by_scale,
    }
