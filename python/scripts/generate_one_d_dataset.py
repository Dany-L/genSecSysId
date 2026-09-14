"""Generate the 1-D Lur'e benchmark dataset.

The plant is the scalar Lur'e system of ``notebooks/one_D_example.ipynb``::

    x_{k+1} = A x_k + B u_k + B2 dzn(C2 x_k + D21 u_k)
    y_k     = C x_k + D u_k + D12 dzn(...)

with ``A=0.9, B=1.0, B2=1.1, C=C2=1.0, D=D12=D21=0`` and the scalar dead zone
``dzn(z) = max(|z|-1, 0) sign(z)``.

Why this system is the right testbed for the sigma(U) constraint. It admits a
*regional* certificate but no global one, and the regional one is small and
exactly computable: at ``alpha = 0.99`` MaxS gives ``s = 0.4166``, ``P = 5.76``,
so the invariant set is ``X = {|x| <= s sqrt(P)} = {|x| <= 1.0}`` and the
worst-case admissible input amplitude is

    sigma*(U) = s sqrt(1 - alpha^2) = 0.0588 .

That number is a hard ceiling on any ``sigma(U) >= c`` we can ask a model to
certify, and it is what sets ``AMP_MAX_CONV`` below: at ``0.04`` the constraint
``c = max_k|u_k| = 0.04`` sits at ``0.68 sigma*``, binding but reachable. The
ratio is scale invariant under the ``scale_only`` normalisation used at load
time (both ``s`` and ``max|u|`` scale by ``1/u_std``), so it is a property of
the data, not of the units.

Note that ``X`` is *exactly* the linear region: the dead zone breaks at
``|x| = 1`` and for ``|x| > 1`` the map becomes ``x+ = 2.0 x - 1.1``, which
diverges. So the converging split never fires the nonlinearity (measured
``max|x| = 0.295`` at ``AMP_MAX_CONV``, against a breakpoint of 1) and the **diverging split is what
carries it**. That is a property of the plant, not a defect of the sampling,
and it is why the ``_div`` folders are not optional here.

Layout mirrors ``notebooks/Duffing/duffing_benchmark.ipynb`` because the loader
requires it: the split folder names are hard-coded in
``sysid.data.direct_loader``, every CSV inside a converging folder must have the
same row count (they are ``np.stack``-ed), and the ``_div`` folders are ragged
by design.

Usage::

    python scripts/generate_one_d_dataset.py --out-dir ~/genSecSysId-Data/data/OneD/id
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

# --- plant ----------------------------------------------------------------
A, B, B2 = 0.9, 1.0, 1.1
C, C2, D21 = 1.0, 1.0, 0.0
D, D12 = 0.0, 0.0

# --- input signal (identical to notebooks/one_D_example.ipynb) ------------
TS = 1.0
LP_CUTOFF = 0.1  # Hz
LP_ORDER = 4
ENV_DECAY = 0.9

# --- sampling -------------------------------------------------------------
# Converging: 0.04 = 0.68 * sigma*(alpha=0.99) = 0.68 * 0.0588. See module docstring.
AMP_MAX_CONV = 0.04
# Diverging: from x0 = 0 the state must be driven past the |x| = 1 breakpoint.
# Measured acceptance at this amplitude is ~28/40.
AMP_MAX_DIV = 0.6
X0_CONV_RANGE = 0.3  # x0 ~ U(-0.3, 0.3), inside X = {|x| <= 1}
T_TRAJ = 500
N_CONV = 100
N_DIV = 50
# Matches the Duffing generator. At 50.0 the diverging trajectories run an
# extra decade of exponential growth, and the model's own blow-up on them
# reaches ~1e30, which swamps every gradient in the diverging pass.
DIVERGE_THRESH = 5.0
MIN_DIV_LEN = 20  # a diverging trajectory shorter than this carries no dynamics
RNG_SEED = 42
SPLIT_SEED = 42


def dzn(z: float) -> float:
    """Scalar dead zone ``max(|z| - 1, 0) sign(z)``."""
    return max(abs(z) - 1.0, 0.0) * np.sign(z)


def make_u(rng, T, amp_max, f_cut=LP_CUTOFF, order=LP_ORDER, env_decay=ENV_DECAY):
    """Low-pass filtered white noise with an exponential decay envelope.

    Verbatim from ``notebooks/one_D_example.ipynb`` (itself the Duffing
    benchmark's ``make_u`` at ``TS = 1.0``): white noise at ``1/TS`` through a
    Butterworth LP filter, multiplied by ``exp(-env_decay t)``, then peak
    normalised to a *random* fraction of ``amp_max`` so the set spans
    amplitudes rather than repeating one.
    """
    b, a = butter(order, f_cut / (0.5 / TS), btype="low")
    pad = 4 * order
    noise = rng.standard_normal(T + pad)
    u_filt = filtfilt(b, a, noise)[pad:]
    t_norm = np.linspace(0.0, 1.0, T)
    u_filt = u_filt * np.exp(-env_decay * t_norm)
    peak = np.max(np.abs(u_filt))
    if peak > 0:
        u_filt = u_filt / peak * rng.uniform(0.0, amp_max)
    return u_filt


def simulate(x0: float, u_seq: np.ndarray, cap: float = DIVERGE_THRESH):
    """Roll the Lur'e recurrence, stopping early if the state escapes.

    Returns ``(x, y, diverged)`` where ``x`` has one more entry than ``y`` (the
    trailing state has no input to pair with) — the caller drops it.
    """
    x = float(x0)
    X: List[float] = [x]
    Y: List[float] = []
    diverged = False
    for u in u_seq:
        w = dzn(C2 * x + D21 * u)
        Y.append(C * x + D * u + D12 * w)
        x = A * x + B * u + B2 * w
        X.append(x)
        if abs(x) > cap:
            diverged = True
            break
    return np.array(X), np.array(Y), diverged


def write_traj_csv(path: Path, u_seq: np.ndarray, x_seq: np.ndarray) -> int:
    """Write ``u,x`` for the real simulated steps only. Returns the row count."""
    n_real = min(len(u_seq), len(x_seq) - 1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "x"])
        for k in range(n_real):
            w.writerow([round(float(u_seq[k]), 8), round(float(x_seq[k]), 8)])
    return n_real


def generate_group(
    rng,
    n_target: int,
    want_div: bool,
    x0_fn: Callable[[], float],
    amp_max: float,
    t_traj: int,
    cap: float = DIVERGE_THRESH,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rejection-sample ``n_target`` trajectories with the requested divergence label."""
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    attempts = 0
    max_attempts = 200 * n_target
    while len(out) < n_target:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Only {len(out)}/{n_target} trajectories with diverged={want_div} "
                f"after {attempts} attempts (amp_max={amp_max}). Adjust the amplitude."
            )
        u_seq = make_u(rng, t_traj, amp_max)
        x_seq, _, diverged = simulate(x0_fn(), u_seq, cap=cap)
        if diverged != want_div:
            continue
        if want_div and len(x_seq) - 1 < MIN_DIV_LEN:
            continue
        out.append((u_seq, x_seq))
    return out


def split_60_10_30(names: List[str], seed: int):
    """60/10/30 train/validation/test on a shuffled copy, as the Duffing set does."""
    rng = np.random.default_rng(seed)
    shuffled = [names[i] for i in rng.permutation(len(names)).tolist()]
    n = len(shuffled)
    n_train = round(n * 0.60)
    n_val = round(n * 0.10)
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--out-dir",
        default="~/genSecSysId-Data/data/OneD/id",
        help="dataset root; split folders are created inside it",
    )
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--n-conv", type=int, default=N_CONV)
    ap.add_argument("--n-div", type=int, default=N_DIV)
    ap.add_argument("--t-traj", type=int, default=T_TRAJ)
    ap.add_argument("--amp-max-conv", type=float, default=AMP_MAX_CONV)
    ap.add_argument("--amp-max-div", type=float, default=AMP_MAX_DIV)
    ap.add_argument("--diverge-thresh", type=float, default=DIVERGE_THRESH)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    raw_dir = out_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)

    groups = [
        ("rand_conv", args.n_conv, False, lambda: rng.uniform(-X0_CONV_RANGE, X0_CONV_RANGE),
         args.amp_max_conv),
        ("zero_div", args.n_div, True, lambda: 0.0, args.amp_max_div),
    ]

    conv_names: List[str] = []
    div_names: List[str] = []
    peak_u_conv = 0.0
    peak_x_conv = 0.0

    for gname, n_target, want_div, x0_fn, amp_max in groups:
        trajs = generate_group(rng, n_target, want_div, x0_fn, amp_max,
                               args.t_traj, args.diverge_thresh)
        for idx, (u_seq, x_seq) in enumerate(trajs):
            fname = f"{gname}_{idx:03d}.csv"
            n_rows = write_traj_csv(raw_dir / fname, u_seq, x_seq)
            if want_div:
                div_names.append(fname)
            else:
                conv_names.append(fname)
                peak_u_conv = max(peak_u_conv, float(np.abs(u_seq[:n_rows]).max()))
                peak_x_conv = max(peak_x_conv, float(np.abs(x_seq[:n_rows]).max()))
        print(f"  {gname}: {len(trajs)} trajectories (diverged={want_div})")

    conv_split = split_60_10_30(conv_names, SPLIT_SEED)
    div_split = split_60_10_30(div_names, SPLIT_SEED)

    split_files = {
        "train": conv_split["train"],
        "validation": conv_split["validation"],
        "test": conv_split["test"],
        "train_div": div_split["train"],
        "validation_div": div_split["validation"],
        "test_div": div_split["test"],
    }
    for split, files in split_files.items():
        split_dir = out_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        for fname in files:
            shutil.copy(raw_dir / fname, split_dir / fname)
        print(f"  {split}/: {len(files)} files")

    # sigma*(alpha) of the true plant, for the record — this is what c is measured against.
    sigma_star_099 = 0.0588
    params = {
        "dataset": "OneD/id",
        "description": "1-D Lur'e system with a scalar dead zone; regionally but not "
                       "globally stable. Testbed for the sigma(U) >= c constraint.",
        "system": {"A": A, "B": B, "B2": B2, "C": C, "C2": C2, "D": D, "D12": D12,
                   "D21": D21, "nonlinearity": "dzn(z) = max(|z|-1,0)*sign(z)", "Ts": TS},
        "certificate_true_system": {
            "alpha": 0.99, "s": 0.4166, "P": 5.7619,
            "invariant_set": "|x| <= 1.0", "sigma_u": sigma_star_099,
            "note": "sigma_u peaks at alpha ~ 0.95 (0.100); alpha <= 0.90 is infeasible "
                    "since the model must contract faster than rho(A) = 0.9.",
        },
        "generation": {
            "rng_seed": args.seed, "T_traj": args.t_traj,
            "amp_max_conv": args.amp_max_conv, "amp_max_div": args.amp_max_div,
            "x0_conv_range": [-X0_CONV_RANGE, X0_CONV_RANGE], "x0_div": 0.0,
            "LP_cutoff_Hz": LP_CUTOFF, "LP_order": LP_ORDER, "env_decay": ENV_DECAY,
            "N_conv": args.n_conv, "N_div": args.n_div,
            "diverge_thresh": args.diverge_thresh, "min_div_len": MIN_DIV_LEN,
        },
        "measured": {
            "max_abs_u_converging": peak_u_conv,
            "max_abs_x_converging": peak_x_conv,
            "c_over_sigma_star": peak_u_conv / sigma_star_099,
            "deadzone_fires_on_converging": bool(peak_x_conv > 1.0),
        },
        "split": {"conv_ratio": "60/10/30", "div_ratio": "60/10/30",
                  "folders": list(split_files.keys())},
    }
    with open(out_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nWrote {out_dir}")
    print(f"  max|u| (converging) = {peak_u_conv:.5f}   -> c = {peak_u_conv:.5f}")
    print(f"  c / sigma*(0.99)    = {peak_u_conv / sigma_star_099:.3f}")
    print(f"  max|x| (converging) = {peak_x_conv:.4f}   "
          f"(dead zone fires: {peak_x_conv > 1.0})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
