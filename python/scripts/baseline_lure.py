"""Reference baseline for a *fixed* Lur'e model given as an ``.npz``.

The Duffing benchmark ships a hand-constructed Lur'e model
(``lure_model_params.npz``, built in ``notebooks/Duffing/duffing_benchmark.ipynb``)
that is the data generator's own discretisation plus a dead-zone fit of the cubic
spring. It is the best in-class θ we have: training is not expected to beat it.
This script turns it into a *baseline record* — one file holding both halves of
the claim:

* **the fit** — open-loop NRMSE/RMSE/... on train, validation, test and on the
  three diverging splits, through the same loaders, normalizer, warmup and
  metric the training/evaluation pipeline uses, so the numbers are directly
  comparable to a trained run;
* **the certificate** — the MaxS solution for that fixed θ (the largest regional
  invariant set these parameters admit) with ``s``, ``P``, ``X = P⁻¹``, ``L``,
  ``H = L P⁻¹``, ``Λ``, the certified output half-width ``ȳ = σ·s·√(C P Cᵀ)``,
  the coverage ratio ``ρ = (ȳ/y_max)ⁿˣ``, and the input-condition violation
  counts per split.

Everything is reused from the package: :func:`load_split_data` +
:func:`create_dataloaders` for the data, :class:`SimpleLure` for the dynamics,
:meth:`SimpleLure.post_process` for the certificate SDPs, and
:class:`Evaluator` for the metrics.

Usage::

    python scripts/baseline_lure.py --npz <lure_model_params.npz> --data <dataset root>
    python scripts/baseline_lure.py --npz ... --data ... --output baseline.yaml

The ``.npz`` holds θ in **physical** units (u, q, q̇ as generated). The model runs
in the pipeline's normalized units, so the matrices are rescaled with the
normalizer fitted on the training split — ``B ← B·σ_u``, ``C ← C/σ_y``,
``D21 ← D21·σ_u``, ``D ← D·σ_u/σ_y``, ``D12 ← D12/σ_y`` — leaving the state in
physical units, exactly as the pipeline treats ``state_col``.

No training config is read: everything the record depends on is either derived
from the data or pinned in the constants below, because a baseline is only
comparable across runs if every run reads the data and solves the SDP the same
way.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from scipy.signal import welch

from sysid.data import create_dataloaders
from sysid.data.direct_loader import load_split_data
from sysid.evaluation import Evaluator, compute_metrics
from sysid.models import SimpleLure


# The physical-unit system matrices expected in the .npz.
THETA_KEYS = ("A", "B", "B2", "C", "C2", "D", "D12", "D21")

# --- the dataset contract -------------------------------------------------
# CSV columns and normalization, matching the benchmark's training pipeline.
INPUT_COL = ["u"]
OUTPUT_COL = ["q"]
PATTERN = "*.csv"
BATCH_SIZE = 32
NORMALIZATION = "scale_only"     # the only pure rescaling; see to_normalized_units

# --- fixed analysis settings ----------------------------------------------
# Pinned, not exposed: varying any of these silently changes what "the baseline"
# means, and the point of the record is that two of them can be compared.
ALPHA = 0.9999                   # contraction rate in the LMIs (the model's init value)
COVERAGE_N_GRID = 20             # points in the tightest-coverage s-sweep
COVERAGE_S_MAX = 100.0           # upper end of that sweep
DIVERGENCE_FACTOR = 10.0         # predicted peak / target peak that counts as a blow-up
TS_FALLBACK = 0.05               # sampling time when the .npz carries no TS
DTYPE = torch.float64            # reference precision (float32 agrees to 5 decimals)
NOISE_BAND_FRAC = 0.5            # noise floor is read off the PSD above this x Nyquist

logger = logging.getLogger("baseline_lure")


# --------------------------------------------------------------------------- io
def jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and arrays to plain Python."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return jsonable(obj.detach().cpu().numpy())
    # bool before int: Python's bool subclasses int, so the int branch would
    # turn True into 1 and quietly demote every flag in the record.
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def write_record(record: Dict[str, Any], path: Path) -> None:
    """Write the record as YAML when the suffix says so, JSON otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = jsonable(record)
    with open(path, "w") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        else:
            json.dump(payload, f, indent=2)


# ------------------------------------------------------------------- rescaling
def to_normalized_units(npz: Dict[str, np.ndarray], normalizer) -> Dict[str, np.ndarray]:
    """Rescale θ from physical to the pipeline's normalized input/output units.

    The state is left in physical units (the pipeline never normalizes
    ``state_col``), so only the input and output channels move::

        A ← A          B ← B·S_u      B2 ← B2
        C ← S_y⁻¹·C    D ← S_y⁻¹·D·S_u    D12 ← S_y⁻¹·D12
        C2 ← C2        D21 ← D21·S_u

    ``normalizer=None`` (unnormalized pipeline) returns θ unchanged.
    """
    theta = {k: np.asarray(npz[k], dtype=float) for k in THETA_KEYS}
    if normalizer is None:
        return theta

    if normalizer.method != "scale_only":
        raise ValueError(
            f"normalization_method='{normalizer.method}' is not supported: only "
            "'scale_only' is a pure diagonal rescaling. 'standard'/'minmax' add an "
            "offset the Lur'e structure cannot absorb, so the .npz cannot be mapped "
            "into those units without changing the model class."
        )

    S_u = np.diag(np.asarray(normalizer.input_std, dtype=float).reshape(-1))
    S_y_inv = np.diag(1.0 / np.asarray(normalizer.output_std, dtype=float).reshape(-1))

    theta["B"] = theta["B"] @ S_u
    theta["D21"] = theta["D21"] @ S_u
    theta["C"] = S_y_inv @ theta["C"]
    theta["D12"] = S_y_inv @ theta["D12"]
    theta["D"] = S_y_inv @ theta["D"] @ S_u
    return theta


def build_model(theta_n: Dict[str, np.ndarray], *, activation: str, ts: float,
                alpha: float, device: str) -> SimpleLure:
    """A :class:`SimpleLure` carrying exactly the given (normalized) matrices.

    ``learn_L=True`` regardless of what the training config says: the baseline's
    point is the *regional* certificate, and ``learn_L=False`` pins ``L = 0``,
    which is the global sector condition and a different (much weaker) claim.
    """
    nx, nz = theta_n["A"].shape[0], theta_n["C2"].shape[0]
    model = SimpleLure(
        nd=theta_n["B"].shape[1],
        ne=theta_n["C"].shape[0],
        nx=nx,
        nw=nz,
        activation=activation,
        custom_params={"learn_L": True},
        ts=ts,
    )
    dtype = model.P.dtype
    with torch.no_grad():
        for key, value in theta_n.items():
            getattr(model, key).data = torch.tensor(value, dtype=dtype, device=device)
        # alpha enters the LMIs through tau via the sigmoid; store its logit.
        model.tau.data = torch.tensor(float(np.log(alpha / (1.0 - alpha))),
                                      dtype=dtype, device=device)
    return model.to(device)


# --------------------------------------------------------------------- metrics
def input_violations(c: Optional[np.ndarray]) -> Dict[str, Any]:
    """Trajectories whose constraint margin ``c_k`` ever goes positive.

    ``c_k = ‖u_k‖² − s² + α²·xₖᵀP⁻¹xₖ`` comes straight out of the evaluator, so
    this reuses the rollout that produced the metrics."""
    if c is None or np.size(c) == 0:
        return {"n_trajectories": 0, "n_violating": None, "max_margin": None}
    viol = np.any(np.nan_to_num(c, nan=-np.inf) > 0, axis=1)
    return {
        "n_trajectories": int(c.shape[0]),
        "n_violating": int(viol.sum()),
        "max_margin": float(np.nanmax(c)),
    }


def measured_noise_floor(e: np.ndarray, fs: float,
                         output_scale: float) -> Optional[Dict[str, Any]]:
    """The NRMSE an *exact* model would still score, read off the targets alone.

    The measurement noise is white, so it is flat across the spectrum, while the
    plant output is band-limited far below Nyquist (a smooth second-order system
    driven by low-pass-filtered excitation). Everything in the top of the band is
    therefore noise: ``σ² = mean PSD there × f_nyq``. Pooling ``σ²`` over
    trajectories and dividing by the NRMSE denominator gives the floor the fit
    can approach but not cross.

    ``flatness_ratio`` is the sanity check — mean PSD in the top quarter of the
    band over the rest of the measurement band. It sits at 1 when the assumption
    holds; far from 1 means signal is leaking into the band and the floor is
    overstated. Returns ``None`` when the trajectories are too short to estimate.
    """
    n_traj, n_steps, n_out = e.shape
    f_nyq = fs / 2.0
    nperseg = int(min(512, max(32, n_steps // 4)))
    if n_steps < nperseg:
        return None

    sigma_sq, flatness = [], []
    for b in range(n_traj):
        for j in range(n_out):
            x = e[b, :, j]
            x = x[np.isfinite(x)]
            if x.size < nperseg:
                continue
            f, pxx = welch(x, fs=fs, nperseg=nperseg)
            band = f >= NOISE_BAND_FRAC * f_nyq
            if not band.any():
                continue
            sigma_sq.append(float(pxx[band].mean() * f_nyq))
            top, mid = f >= 0.75 * f_nyq, band & (f < 0.75 * f_nyq)
            if top.any() and mid.any() and pxx[mid].mean() > 0:
                flatness.append(float(pxx[top].mean() / pxx[mid].mean()))
    if not sigma_sq:
        return None

    sigma = float(np.sqrt(np.mean(sigma_sq)))
    return {
        "sigma": sigma,
        "nrmse": sigma / output_scale,
        "method": "white-noise PSD in the upper band of the target spectrum",
        "band_hz": [NOISE_BAND_FRAC * f_nyq, f_nyq],
        "flatness_ratio": float(np.mean(flatness)) if flatness else None,
        "n_estimates": len(sigma_sq),
    }


def declared_noise_floor(e: np.ndarray, snr_db: float,
                         output_scale: float) -> Dict[str, Any]:
    """The same floor from the dataset's declared SNR — an independent cross-check.

    The generator draws each trajectory's noise as ``σ_i = std(y_i)/10^(SNR/20)``
    over the **full** trajectory, so this uses the full target, not the metric
    window."""
    ratio = 10.0 ** (snr_db / 20.0)
    sigma_i = np.nanstd(e, axis=1) / ratio        # (n_traj, n_out)
    sigma = float(np.sqrt(np.nanmean(sigma_i ** 2)))
    return {"snr_db": float(snr_db), "sigma": sigma, "nrmse": sigma / output_scale}


def diverged_mask(e_hat: np.ndarray, e: np.ndarray, factor: float) -> np.ndarray:
    """Per-trajectory blow-up flag: non-finite, or a peak ``factor``× the target's.

    A single diverging trajectory drives the pooled NRMSE to ``inf`` (or to
    ``1e80``), which says "it broke" but destroys the number's comparability. The
    flag lets the record report both: the pooled metric the pipeline reports, and
    the metric over the trajectories the model actually tracked."""
    peak_target = np.nanmax(np.abs(e)) if np.isfinite(e).any() else np.inf
    with np.errstate(invalid="ignore"):
        peak_pred = np.nanmax(np.abs(np.where(np.isfinite(e_hat), e_hat, np.inf)),
                              axis=tuple(range(1, e_hat.ndim)))
    return ~np.isfinite(e_hat).all(axis=tuple(range(1, e_hat.ndim))) | (
        peak_pred > factor * peak_target
    )


def evaluate_conv(evaluator: Evaluator, loader, normalizer, *, warmup: int,
                  divergence_factor: float, fs: float,
                  snr_db: Optional[float]) -> Dict[str, Any]:
    """NRMSE & co. on a converging split, plus its noise floor and input-condition
    tally.

    Reports the pooled metrics (pipeline convention) and, when any trajectory
    blows up, the same metrics restricted to those that did not. The noise floor
    turns the NRMSE into a *ratio*: 1.0 means the residual is pure measurement
    noise and the model cannot be improved on this data."""
    res = evaluator.evaluate(loader, normalizer=normalizer,
                             print_results=False, save_files=False)
    e_hat, e = res["e_hat"], res["e"]
    div = diverged_mask(e_hat, e, divergence_factor)
    scale = normalizer.get_output_scale() if normalizer is not None else None

    out = {
        "n_trajectories": int(e.shape[0]),
        "n_steps": int(e.shape[1]),
        "n_diverged": int(div.sum()),
        "diverged_trajectories": np.flatnonzero(div),
        "metrics": {k: v for k, v in res["metrics"].items() if k != "per_step"},
        "input_condition": input_violations(res.get("c")),
    }
    if div.any() and (~div).any():
        keep = ~div
        out["metrics_non_diverged"] = compute_metrics(
            e_hat[keep, warmup:, :].reshape(-1, e_hat.shape[-1]),
            e[keep, warmup:, :].reshape(-1, e.shape[-1]),
            output_scale=scale,
        )

    # The floor is estimated on the metric window, so it is the floor of exactly
    # the number it is compared against.
    floor = measured_noise_floor(e[:, warmup:, :], fs, scale) if scale else None
    if floor is not None:
        if snr_db is not None:
            floor["declared"] = declared_noise_floor(e, snr_db, scale)
        floor["nrmse_ratio"] = out["metrics"]["nrmse"] / floor["nrmse"]
        if "metrics_non_diverged" in out:
            floor["nrmse_ratio_non_diverged"] = (
                out["metrics_non_diverged"]["nrmse"] / floor["nrmse"]
            )
        out["noise_floor"] = floor
    return out


def evaluate_div(evaluator: Evaluator, loader, normalizer) -> Dict[str, Any]:
    """Pooled metrics on a diverging split (full trajectories, no warmup skip).

    These targets diverge by construction and the model diverges at its own rate,
    so the pooled NRMSE is routinely astronomical — it measures *how differently*
    two blow-ups blow up. ``finite_fraction`` is the honest companion number."""
    res = evaluator.evaluate_diverging(loader, normalizer=normalizer,
                                       print_results=False, save_files=False)
    e_hat = res["e_hat_div"]
    return {
        "n_trajectories": int(res["num_trajectories_div"]),
        "metrics": {k: v for k, v in res["metrics_div"].items() if k != "per_step"},
        "metrics_per_trajectory": res["metrics_div_per_traj"],
        "finite_fraction": float(np.isfinite(e_hat).mean()) if e_hat is not None else None,
    }


# ----------------------------------------------------------------- certificate
def certificate_record(model: SimpleLure, summary: Dict[str, Any],
                       sigma_u: float) -> Dict[str, Any]:
    """The applied (MaxS) certificate, as matrices plus the derived scalars.

    **Units.** The state stays physical (see :func:`to_normalized_units`) but the
    input port is normalized, so ``s`` and ``u_max`` are in ``u/σ_u`` — that is
    the space the input condition ``‖u‖² ≤ s² − α²xᵀP⁻¹x`` is written in. Three
    consequences, because all three invite misreading:

    * ``u_max = 99.7`` is *not* a physical input of 99.7; it is
      ``(1.10/0.110)² = 99.7``. ``u_max_physical`` carries the raw number.
    * ``s`` is likewise normalized and is **not** comparable to an ``s`` obtained
      by solving the same LMIs on the raw θ; ``s_physical = σ_u·s`` is.
    * Rescaling the input port is an exact re-parameterization of the certificate:
      ``(P, L, M, s) → (P/σ_u², L/σ_u², M/σ_u², σ_u·s)`` maps a feasible solution
      to a feasible one and leaves ``𝒳 = {x : xᵀP⁻¹x ≤ s²}`` unchanged. It is
      *not* neutral in practice, because the LMIs' ``ε`` floors are absolute
      while ``P`` scales by ``σ_u²`` — with ``σ_u < 1`` the normalized problem is
      the more constrained one, so MaxS lands slightly short of the physical
      solve (≈6 % on the Duffing reference). The certificate is valid either
      way; the normalized one is the conservative one, and it is the one training
      works in.

    ``set_shape = P⁻¹/s²`` records the certified set itself,
    ``𝒳 = {x : xᵀ·set_shape·x ≤ 1}`` in physical state units. Note MaxS pins only
    ``s`` — ``P`` is not unique at the optimum — so treat ``set_shape`` as *a*
    certified set for this θ, not a canonical fingerprint of it.
    """
    P = model.P.detach().cpu().numpy()
    L = model.L.detach().cpu().numpy()
    X = np.linalg.inv(P)
    H = L @ X
    s = float(model.s)
    alpha = float(torch.sigmoid(model.tau))
    u_max = float(model.u_max) if not bool(torch.isnan(model.u_max)) else None
    set_shape = X / s ** 2

    # post_process only fills rho when the coverage sweep succeeded, but rho is a
    # property of the MaxS solution alone — and rho < 1 (under-coverage) is
    # exactly the case where that sweep fails, i.e. the case worth reporting.
    y_bar, y_max = summary["max_s"]["y_bar"], summary["y_max"]
    rho = summary["max_s"]["rho"]
    if rho is None and y_bar is not None and y_max:
        rho = float((y_bar / y_max) ** model.nx)

    return {
        "objective": "max_s",
        "units": {
            "state": "physical (the .npz / CSV state columns)",
            "input": f"normalized, u/sigma_u with sigma_u = {sigma_u:.6g}",
            "output": "physical (y_bar, y_max); C is normalized, sigma_y multiplies back",
            "note": ("s and u_max are in normalized-input units — multiply by sigma_u "
                     "(resp. sigma_u^2) to compare against a physical-unit solve"),
        },
        "alpha": alpha,
        "s": s,
        "s_physical": s * sigma_u,
        "y_bar": y_bar,
        "y_max": y_max,
        "rho": rho,
        "coverage_ok": summary["max_s"]["coverage_ok"],
        "volume": summary["max_s"]["volume"],
        "norm_H": summary["max_s"]["norm_H"],
        "max_eig_F": summary["max_s"]["max_eig_F"],
        "constraints_satisfied": summary["constraints_satisfied"],
        "u_max": u_max,
        "u_max_physical": None if u_max is None else u_max * sigma_u ** 2,
        "s_floor": None if u_max is None else float(np.sqrt(u_max)),
        "s_meets_input_floor": None if u_max is None else bool(s ** 2 >= u_max),
        "P": P,
        "X": X,
        "L": L,
        "H": H,
        "Lambda": np.diag(model.la.detach().cpu().numpy()),
        "eig_P": np.linalg.eigvalsh(P),
        "cond_P": float(np.linalg.cond(P)),
        # X = {x : x' set_shape x <= 1}, physical state units. MaxS pins s, not P,
        # so this is *a* certified set for this theta, not a canonical fingerprint.
        "set_shape": set_shape,
        "semi_axes": 1.0 / np.sqrt(np.linalg.eigvalsh(set_shape)),
    }


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certificate + open-loop fit of a fixed Lur'e model (.npz) — the "
                    "reference baseline a trained model is measured against.",
    )
    parser.add_argument("--npz", type=str, required=True,
                        help="Lur'e model parameters in PHYSICAL units (A, B, B2, C, C2, D, D12, D21).")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset root holding train/validation/test/ and the *_div/ siblings.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .json/.yaml (default: <data>/baseline_<npz stem>.json).")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Steps skipped before the converging-split metrics start (default: 500).")
    parser.add_argument("--activation", type=str, default="dzn",
                        help="Nonlinearity of the Lur'e block (default: dzn).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)

    # The dtype must be set before any tensor is built: the datasets and
    # SimpleLure both materialize through torch.get_default_dtype().
    torch.set_default_dtype(DTYPE)

    data_root = Path(os.path.expanduser(args.data))
    npz_path = Path(os.path.expanduser(args.npz))
    warmup, activation = args.warmup_steps, args.activation

    # Optional: the generator's declared SNR, used only to cross-check the noise
    # floor that is measured from the data itself.
    snr_db = None
    params_path = data_root / "params.json"
    if params_path.exists():
        snr_db = json.loads(params_path.read_text()).get("generation", {}).get("SNR_dB")

    logger.info("=" * 78)
    logger.info("Lur'e reference baseline")
    logger.info("=" * 78)
    logger.info(f"  npz    : {npz_path}")
    logger.info(f"  data   : {data_root}")

    # ---- data ------------------------------------------------------------
    # No state_col: the evaluator rolls every split out from x0 = 0, so recorded
    # initial conditions would be loaded and then ignored.
    (
        train_inputs, train_outputs,
        val_inputs, val_outputs,
        test_inputs, test_outputs,
        train_states, val_states, test_states,
        train_div_inputs, train_div_outputs, train_div_states,
        val_div_inputs, val_div_outputs, val_div_states,
        test_div_inputs, test_div_outputs, test_div_states,
    ) = load_split_data(
        data_dir=str(data_root),
        input_col=INPUT_COL,
        output_col=OUTPUT_COL,
        state_col=None,
        pattern=PATTERN,
        load_train=True, load_val=True, load_test=True, load_div=True,
    )

    # sequence_length=None everywhere: the baseline is an open-loop simulation
    # metric, so every split is run full-length (the training chunking exists for
    # the gradient, not for reporting). shuffle=False keeps it reproducible.
    (
        train_loader, val_loader, test_loader,
        train_div_loader, val_div_loader, test_div_loader,
        normalizer,
    ) = create_dataloaders(
        train_inputs=train_inputs, train_outputs=train_outputs, train_states=train_states,
        val_inputs=val_inputs, val_outputs=val_outputs, val_states=val_states,
        test_inputs=test_inputs, test_outputs=test_outputs, test_states=test_states,
        train_div_inputs=train_div_inputs, train_div_outputs=train_div_outputs,
        train_div_states=train_div_states,
        val_div_inputs=val_div_inputs, val_div_outputs=val_div_outputs,
        val_div_states=val_div_states,
        test_div_inputs=test_div_inputs, test_div_outputs=test_div_outputs,
        test_div_states=test_div_states,
        batch_size=BATCH_SIZE,
        sequence_length=None, sequence_stride=None,
        normalize=True,
        normalization_method=NORMALIZATION,
        shuffle=False,
        num_workers=0,
        diverging_batch_size=1,
    )

    # ---- model -----------------------------------------------------------
    with np.load(npz_path) as npz:
        missing = [k for k in THETA_KEYS if k not in npz]
        if missing:
            raise KeyError(f"{npz_path} is missing {missing}; expected {list(THETA_KEYS)}")
        theta_phys = {k: np.asarray(npz[k], dtype=float) for k in THETA_KEYS}
        ts = float(npz["TS"]) if "TS" in npz else TS_FALLBACK

    theta_norm = to_normalized_units(theta_phys, normalizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(theta_norm, activation=activation, ts=ts, alpha=ALPHA, device=device)
    model.eval()

    # ---- levels the certificate is measured against ----------------------
    # Both are taken exactly as the trainer takes them: y_max is the physical
    # peak of the training outputs, u_max the peak squared NORMALIZED input.
    sigma_y = float(np.asarray(normalizer.output_std).reshape(-1)[0]) if normalizer else 1.0
    sigma_u = float(np.asarray(normalizer.input_std).reshape(-1)[0]) if normalizer else 1.0
    y_max = float(np.nanmax(np.abs(train_outputs)))
    model.set_output_coverage_level(y_max, sigma_y)

    u_n = np.asarray(normalizer.transform_inputs(train_inputs) if normalizer else train_inputs, dtype=float)
    u_max_sq = float(np.max(np.sum(u_n.reshape(-1, u_n.shape[-1]) ** 2, axis=-1)))
    model.set_input_bound(u_max_sq)
    logger.info(f"  y_max  = {y_max:.6g} (physical)   sigma_y = {sigma_y:.6g}")
    logger.info(f"  u_max  = {u_max_sq:.6g} -> s >= {np.sqrt(u_max_sq):.6g}  "
                f"[normalized u/sigma_u, sigma_u = {sigma_u:.6g}; "
                f"physical max||u|| = {np.sqrt(u_max_sq) * sigma_u:.6g}]")

    # ---- certificate: MaxS (applied) + tightest-coverage sweep (reported) --
    summary = model.post_process(
        y_max=y_max, n_grid=COVERAGE_N_GRID,
        s_min=max(float(np.sqrt(u_max_sq)), 1e-6), s_max=COVERAGE_S_MAX,
    )
    if not summary.get("success", False):
        raise RuntimeError(f"certificate synthesis failed: {summary.get('status')}")

    # ---- fit --------------------------------------------------------------
    # The evaluator insists on an output dir even with save_files=False; the
    # baseline's only artefact is the record, so give it a throwaway one.
    splits: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as scratch:
        evaluator = Evaluator(model=model, device=device, output_dir=scratch, warmup_steps=warmup)
        for name, loader in (("train", train_loader), ("validation", val_loader),
                             ("test", test_loader)):
            if loader is not None:
                r = splits[name] = evaluate_conv(
                    evaluator, loader, normalizer,
                    warmup=warmup, divergence_factor=DIVERGENCE_FACTOR,
                    fs=1.0 / ts, snr_db=snr_db,
                )
                nf = r.get("noise_floor")
                ratio = "" if nf is None else (
                    f" = {nf.get('nrmse_ratio_non_diverged', nf['nrmse_ratio']):.2f}x "
                    f"the noise floor {nf['nrmse']:.6f}"
                )
                extra = ""
                if r["n_diverged"]:
                    extra = (f", {r['n_diverged']} diverged -> nrmse "
                             f"{r.get('metrics_non_diverged', {}).get('nrmse', float('nan')):.6f} "
                             f"on the rest")
                logger.info(f"  {name:14s} nrmse = {r['metrics']['nrmse']:.6f}{ratio}  "
                            f"({r['n_trajectories']} traj, "
                            f"{r['input_condition']['n_violating']} input-cond violations{extra})")
        for name, loader in (("train_div", train_div_loader), ("validation_div", val_div_loader),
                             ("test_div", test_div_loader)):
            if loader is not None:
                r = splits[name] = evaluate_div(evaluator, loader, normalizer)
                logger.info(f"  {name:14s} nrmse = {r['metrics']['nrmse']:.6g}  "
                            f"({r['n_trajectories']} traj, "
                            f"finite fraction {r['finite_fraction']:.4f})")

    # ---- diagnostics ------------------------------------------------------
    train_in_t = torch.as_tensor(
        normalizer.transform_inputs(train_inputs) if normalizer else train_inputs,
        dtype=model.P.dtype, device=device,
    )
    activity = model.deadzone_activity(train_in_t, warmup_steps=warmup)
    logger.info(f"  dead zone: firing_rate = {activity['firing_rate']:.4f}, "
                f"units_firing = {activity['units_firing']:.4f}, max|z| = {activity['max_abs_z']:.4f}")

    # ---- record -----------------------------------------------------------
    record = {
        "baseline": "fixed Lur'e model from .npz — reference fit and certificate",
        "source": {
            "npz": str(npz_path),
            "data": str(data_root),
        },
        "protocol": {
            "dtype": str(DTYPE).replace("torch.", ""),
            "warmup_steps": int(warmup),
            "x0": "zeros (the evaluator's convention on every split)",
            "sequence_length": "full trajectories on all splits",
            "nrmse_denominator": "training-output scale from the normalizer",
            "output_scale": float(normalizer.get_output_scale()) if normalizer else None,
            "divergence_factor": DIVERGENCE_FACTOR,
        },
        "model": {
            "nx": int(model.nx), "nz": int(model.nz), "nd": int(model.nd), "ne": int(model.ne),
            "activation": activation, "ts": ts, "learn_L": True,
        },
        "normalization": {
            "method": NORMALIZATION if normalizer else None,
            "input_std": np.asarray(normalizer.input_std).reshape(-1) if normalizer else None,
            "output_std": np.asarray(normalizer.output_std).reshape(-1) if normalizer else None,
        },
        "levels": {"y_max": y_max, "u_max": u_max_sq, "s_floor": float(np.sqrt(u_max_sq))},
        "certificate": certificate_record(model, summary, sigma_u),
        "coverage_sweep": summary["coverage"],
        "deadzone_activity": activity,
        "metrics": splits,
        "theta_normalized": theta_norm,
        "theta_physical": theta_phys,
    }

    out_path = Path(os.path.expanduser(args.output)) if args.output else \
        data_root / f"baseline_{npz_path.stem}.json"
    write_record(record, out_path)

    # rho / y_bar are None when the coverage sweep found y_max unreachable or
    # ne > 1 — the summary line must survive that, it is the interesting case.
    def fmt(v, spec=".4f"):
        return "n/a" if v is None else format(v, spec)

    cert = record["certificate"]
    logger.info("-" * 78)
    logger.info(f"  s = {fmt(cert['s'])}   y_bar = {fmt(cert['y_bar'])}   "
                f"y_max = {fmt(y_max)}   rho = {fmt(cert['rho'])}   "
                f"||H|| = {fmt(cert['norm_H'])}")
    logger.info(f"Baseline written to {out_path}")


if __name__ == "__main__":
    main()
