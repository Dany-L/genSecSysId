"""Generate the Duffing oscillator benchmark dataset (in-distribution).

The plant is the softening Duffing oscillator of
``notebooks/duffing/duffing_benchmark.ipynb``::

    q'' = -delta_d q' - q + q^3 + u(t),   delta_d = 0.3,   x = (q, q')

sampled at ``TS = 0.05`` s with a zero-order hold on ``u``: every sample is one
``scipy.integrate.solve_ivp`` RK45 step (``rtol=1e-5, atol=1e-7``) over
``[0, TS]``. The origin is a stable spiral and ``(+-1, 0)`` are saddles, so the
system is *regionally* but not globally stable: inputs large enough to push the
state across the separatrix make it diverge.

Two trajectory groups are rejection-sampled with the RK45 ground truth deciding
the label:

* ``rand_conv`` — ``x0 ~ U(-0.6, 0.6)^2``, stays below ``DIVERGE_THRESH`` for all
  ``T_TRAJ`` samples;
* ``zero_div`` — ``x0 = 0``, some ``|x_i|`` exceeds ``DIVERGE_THRESH``. The
  trajectory is truncated at the last sample *before* the threshold crossing,
  so the ``_div`` files are ragged.

Additive white Gaussian noise at ``SNR_DB`` is added to both ``q`` and ``q_dot``
of every stored sample (``k = 0`` included, so ``zero_div`` files do not start
at exactly 0); the input ``u`` is noise free.

Provenance. This reproduces, byte for byte, the published dataset, which was
written by the notebook at commit 045a667 (2026-06-17). The notebook has since
been changed to write the Lur'e surrogate state ``X_lure`` without noise; this
script keeps the original behaviour. Reproduction relies on drawing from a single
``default_rng(RNG_SEED)`` in exactly this order: per attempt ``x0`` (rand_conv
only), then the input noise and its amplitude, then — for accepted trajectories
only — the noise on ``q`` and on ``q_dot``.

Usage::

    python scripts/generate_duffing_dataset.py --out-dir ~/genSecSysId-Data/data/Duffing/id
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt

# --- plant ----------------------------------------------------------------
DELTA_D = 0.3
TS = 0.05  # s
RK_RTOL, RK_ATOL = 1e-5, 1e-7

# --- input signal ---------------------------------------------------------
U_AMP_MAX = 3.5
LP_CUTOFF = 2.0  # Hz, ~12x the linear natural frequency 1/(2 pi) = 0.16 Hz
LP_ORDER = 4
ENV_DECAY = 0.9

# --- sampling -------------------------------------------------------------
T_TRAJ = 4000
N_RAND_CONV = 100
N_ZERO_DIV = 50
X0_RAND_RANGE = 0.6
DIVERGE_THRESH = 5.0
SNR_DB: Optional[float] = 30.0
RNG_SEED = 42
SPLIT_SEED = 42


def duffing_ct(t, x, u=0.0, delta_d=DELTA_D):
    """Continuous-time right-hand side ``q'' = -delta_d q' - q + q^3 + u``."""
    q, dq = x
    return [dq, -delta_d * dq - q + q**3 + u]


def duffing_dt(x, u=0.0, ts=TS, delta_d=DELTA_D):
    """One sampling period of the ODE with ``u`` held constant (ZOH)."""
    sol = solve_ivp(lambda t, xv: duffing_ct(t, xv, u=u, delta_d=delta_d), [0, ts], x,
                    method="RK45", rtol=RK_RTOL, atol=RK_ATOL)
    return sol.y[:, -1]


def simulate(x0, u_seq: np.ndarray, cap: float = DIVERGE_THRESH):
    """Roll the sampled ODE, stopping right after the first state with ``|x_i| > cap``.

    Returns ``(X, diverged)``; ``X`` has one more row than there are usable
    samples (the trailing state has no input to pair with).
    """
    X = [np.array(x0, dtype=float)]
    diverged = False
    for u in u_seq:
        xnew = duffing_dt(X[-1], u=u)
        X.append(xnew)
        if np.any(np.abs(xnew) > cap):
            diverged = True
            break
    return np.array(X), diverged


def make_u(rng, T, amp_max, f_cut=LP_CUTOFF, order=LP_ORDER, env_decay=ENV_DECAY):
    """Low-pass filtered white noise with an exponential decay envelope.

    White noise at ``1/TS`` through a zero-phase Butterworth LP filter,
    multiplied by ``exp(-env_decay t)`` with ``t`` spanning ``[0, 1]``, then
    peak normalised to a *random* fraction ``U(0, amp_max)``.
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


def add_noise(rng, X: np.ndarray, n: int, snr_db: Optional[float]) -> np.ndarray:
    """AWGN on the first ``n`` states, per channel ``sigma = std(x_i) / 10^(snr/20)``."""
    if snr_db is None:
        return X
    ratio = 10 ** (snr_db / 20)
    X_m = X.copy()
    for i in range(X.shape[1]):
        X_m[:n, i] += rng.normal(0.0, np.std(X[:n, i]) / ratio, n)
    return X_m


def write_traj_csv(path: Path, u_seq: np.ndarray, X: np.ndarray, n: int) -> None:
    """Write ``u,q,q_dot`` for the first ``n`` samples, rounded to 8 decimals."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "q", "q_dot"])
        for k in range(n):
            w.writerow([round(float(u_seq[k]), 8),
                        round(float(X[k, 0]), 8),
                        round(float(X[k, 1]), 8)])


def generate_group(
    rng,
    n_target: int,
    want_div: bool,
    x0_fn: Callable[[], np.ndarray],
    t_traj: int,
    amp_max: float,
    cap: float,
    snr_db: Optional[float],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], int]:
    """Rejection-sample ``n_target`` noisy trajectories with the requested label.

    Returns ``([(u, X_noisy, n_samples), ...], attempts)``.
    """
    out = []
    attempts = 0
    while len(out) < n_target:
        x0 = x0_fn()
        u_seq = make_u(rng, t_traj, amp_max)
        X, diverged = simulate(x0, u_seq, cap=cap)
        attempts += 1
        if diverged != want_div:
            continue
        n = len(X) - 1
        out.append((u_seq, add_noise(rng, X, n, snr_db), n))
    return out, attempts


def split_60_10_30(names: List[str], seed: int):
    """60/10/30 train/validation/test on a shuffled copy."""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default="~/genSecSysId-Data/data/Duffing/id",
                    help="dataset root; split folders are created inside it")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--n-conv", type=int, default=N_RAND_CONV)
    ap.add_argument("--n-div", type=int, default=N_ZERO_DIV)
    ap.add_argument("--t-traj", type=int, default=T_TRAJ)
    ap.add_argument("--amp-max", type=float, default=U_AMP_MAX)
    ap.add_argument("--diverge-thresh", type=float, default=DIVERGE_THRESH)
    ap.add_argument("--snr-db", type=float, default=SNR_DB,
                    help="measurement SNR in dB; pass a negative value to disable noise")
    args = ap.parse_args(argv)
    snr_db = None if args.snr_db is None or args.snr_db < 0 else args.snr_db

    out_dir = Path(args.out_dir).expanduser()
    raw_dir = out_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)

    # The order matters for reproducibility: zero_div is drawn first.
    groups = [
        ("zero_div", args.n_div, True, lambda: np.array([0.0, 0.0])),
        ("rand_conv", args.n_conv, False,
         lambda: rng.uniform(-X0_RAND_RANGE, X0_RAND_RANGE, 2)),
    ]

    conv_names: List[str] = []
    div_names: List[str] = []
    for gname, n_target, want_div, x0_fn in groups:
        trajs, attempts = generate_group(rng, n_target, want_div, x0_fn, args.t_traj,
                                         args.amp_max, args.diverge_thresh, snr_db)
        for idx, (u_seq, X_m, n) in enumerate(trajs):
            fname = f"{gname}_{idx:03d}.csv"
            write_traj_csv(raw_dir / fname, u_seq, X_m, n)
            (div_names if want_div else conv_names).append(fname)
        print(f"  {gname}: {len(trajs)} trajectories ({attempts} attempts)")

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

    # Same keys as the notebook's params.json, so existing readers keep working.
    params = {
        "dataset": "id",
        "description": "In-distribution Duffing oscillator dataset",
        "system": {"delta_d": DELTA_D, "Ts": TS},
        "generation": {
            "rng_seed": args.seed,
            "T_traj": args.t_traj,
            "U_amp_max": args.amp_max,
            "LP_cutoff_Hz": LP_CUTOFF,
            "LP_order": LP_ORDER,
            "env_decay": ENV_DECAY,
            "SNR_dB": snr_db,
            "N_zero_conv": 0,
            "N_zero_div": args.n_div,
            "N_rand_conv": args.n_conv,
            "N_rand_div": 0,
            "x0_rand_range": [-X0_RAND_RANGE, X0_RAND_RANGE],
            "diverge_thresh": args.diverge_thresh,
        },
        "split": {
            "conv_ratio": "60/10/30",
            "div_ratio": "60/10/30",
            "folders": list(split_files.keys()),
        },
    }
    with open(out_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
