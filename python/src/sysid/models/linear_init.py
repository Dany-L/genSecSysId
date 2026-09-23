r"""Structured initialization: a balanced linear model as the starting theta.

Why this exists
---------------
The Lur'e model is ``x+ = A x + B u + B2 w``, ``z = C2 x + D21 u``, ``w = Δ(z)``,
``y = C x + D u + D12 w``. Setting ``B2 = C2 = 0`` opens the loop: the
nonlinearity is disconnected and the stability LMI collapses to the linear
Lyapunov condition. So

    **A Schur with rho(A) < alpha  =>  a feasible P always exists.**

The LMI is continuous in ``(B2, C2)``, so a neighbourhood of zero is feasible
too. Feasibility is therefore guaranteed *by construction*, and the only real
question is how large the loop gain through the dead zone may be.

That question is about the REALIZATION, not the system. A state-space model is
defined only up to a similarity transform ``T``, and ``B2``/``C2`` are written in
whatever coordinates ``A`` happens to be in. A badly scaled realization makes a
single ``B2`` std mean wildly different loop gains in different state
directions — on the F-16 at ``nx = 9`` an Euler/companion realization put one
mode's state RMS at 242 against 0.6 for the others, a 400:1 spread that ``P``
has to absorb, which is what turned picking ``B2`` into a five-decade search.

:func:`fit_linear_model` removes that freedom by returning a **balanced**
realization (Moore 1981): the controllability and observability Gramians are
equal and diagonal, so every state carries comparable input-output energy,
``P`` is well conditioned, and ``||B2||``/``||C2||`` have one consistent meaning
across directions. ``B2 = C2 ~ 0.1`` then becomes an ordinary value rather than
a lucky find.

Units
-----
Fit on **normalized** data (the loader's ``u/input_std``, ``y/output_std``) and
the resulting ``A, B, C, D`` are already in the model's own units, so they can
be written straight to ``.npy`` and picked up by
``custom_params.identity_init.{A,B,C}.load_from``. No physical->model rescaling
is needed; that path (``_WARM_START_SCALING``) exists for theta identified on
raw data and is deliberately not used here.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

# Subspace block size relative to the target state dimension. It must exceed nx
# for the SVD to have directions to choose from; beyond that, larger averages
# more and costs memory (the block Hankel is (block*ne, N - 2*block)).
DEFAULT_BLOCK_MULTIPLIER = 3
DEFAULT_BLOCK_MIN = 12
# Identify this many times nx before truncating, so the Hankel spectrum has
# something to discard and `energy_retained` reports a real number.
OVER_IDENTIFY_MULTIPLIER = 2
# Cap on the number of B/D fit chunks, to bound the regression's memory.
MAX_FIT_CHUNKS = 16
# Block Hankel rows demanded per identified state. Measured on the F-16's
# single accelerometer at nx=9 (rows/state -> linear NRMSE):
#     ~2  -> 5.9      ~4  -> 12.1     ~9  -> 0.79
#    ~18  -> 0.66    ~36  -> 0.20
# so the extra singular directions only stop being noise well above 4. 16 is the
# knee; beyond ~32 the block costs memory for nothing.
SUBSPACE_ROWS_PER_STATE = 16


@dataclass(frozen=True)
class LinearInit:
    """A balanced linear model plus the diagnostics that justify it."""

    A: np.ndarray  # (nx, nx)
    B: np.ndarray  # (nx, nd)
    C: np.ndarray  # (ne, nx)
    D: np.ndarray  # (ne, nd)
    #: Hankel singular values of the full realization, largest first. The ones
    #: past ``nx`` are what the truncation discards -- the honest measure of how
    #: much a given nx throws away.
    hankel: np.ndarray
    #: RMS of each balanced state on the fitting record. The spread across
    #: states is the conditioning number this whole module exists to control.
    state_rms: np.ndarray
    #: One-step-ahead NRMSE per output of the fitted linear model. A sanity
    #: figure: the Lur'e model should beat it, never start worse.
    nrmse: np.ndarray

    @property
    def nx(self) -> int:
        return int(self.A.shape[0])

    @property
    def rho(self) -> float:
        """Spectral radius. ``alpha_0`` must exceed this or no P exists."""
        return float(np.abs(np.linalg.eigvals(self.A)).max())

    @property
    def state_rms_spread(self) -> float:
        """max/min state RMS. ~1 is balanced; 400 is what this replaces."""
        lo = float(np.min(self.state_rms))
        return float("inf") if lo <= 0 else float(np.max(self.state_rms) / lo)

    @property
    def energy_retained(self) -> float:
        """Fraction of total Hankel energy kept by the truncation."""
        total = float(self.hankel.sum())
        return 1.0 if total <= 0 else float(self.hankel[: self.nx].sum() / total)

    def save(self, folder) -> Dict[str, Path]:
        """Write ``A.npy``/``B.npy``/``C.npy``/``D.npy`` for ``load_from``."""
        folder = Path(folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        written = {}
        for name in ("A", "B", "C", "D"):
            path = folder / f"{name}.npy"
            np.save(path, getattr(self, name))
            written[name] = path
        return written

    def summary(self) -> Dict[str, object]:
        """JSON-friendly diagnostics for metadata.json."""
        return {
            "nx": self.nx,
            "rho_A": self.rho,
            "state_rms": [float(v) for v in self.state_rms],
            "state_rms_spread": self.state_rms_spread,
            "hankel_singular_values": [float(v) for v in self.hankel],
            "energy_retained": self.energy_retained,
            "linear_nrmse_per_output": [float(v) for v in self.nrmse],
        }


# ── step 1: subspace identification ───────────────────────────────────────────
# An ARX fit followed by tf2ss was tried first and abandoned: the companion form
# of an order-12+ polynomial is numerically catastrophic. scipy warns
# "Badly conditioned filter coefficients", and the resulting A/B/C were garbage
# (a linear NRMSE of 4.6e6 on Cascaded Tanks). Subspace identification builds the
# state-space model directly from an SVD of block Hankel data and never forms a
# high-order polynomial, so it stays conditioned and handles MIMO natively.
def _block_hankel(x: np.ndarray, i: int, j: int) -> np.ndarray:
    """``(i*cols, j)`` block Hankel matrix of a ``(N, cols)`` signal."""
    cols = x.shape[1]
    H = np.zeros((i * cols, j))
    for r in range(i):
        H[r * cols : (r + 1) * cols] = x[r : r + j].T
    return H


def _identify_A_C(u: np.ndarray, y: np.ndarray, nx: int, block: int):
    """``(A, C)`` by MOESP-style subspace identification.

    Future outputs are projected onto the orthogonal complement of the future
    inputs (removing the direct input influence), then the column space of the
    result is the extended observability matrix ``Gamma``. ``C`` is its first
    block row and ``A`` follows from its shift-invariance.
    """
    n = len(u)
    j = n - 2 * block + 1
    if j < 2 * block * (u.shape[1] + y.shape[1]):
        raise ValueError(
            f"record too short for subspace ID with block={block}: {n} samples"
        )
    Uf = _block_hankel(u[block:], block, j)
    Yf = _block_hankel(y[block:], block, j)

    # Project out the future inputs: Yf @ (I - Uf^+ Uf).
    perp = Yf - (Yf @ np.linalg.pinv(Uf)) @ Uf
    U1, s1, _ = np.linalg.svd(perp, full_matrices=False)
    Gamma = U1[:, :nx] * np.sqrt(s1[:nx])

    ne = y.shape[1]
    C = Gamma[:ne]
    A, *_ = np.linalg.lstsq(Gamma[:-ne], Gamma[ne:], rcond=None)
    return _enforce_schur(A), C


def _enforce_schur(A: np.ndarray, max_rho: float = 0.9999) -> np.ndarray:
    """Pull ``A``'s spectral radius below ``max_rho`` if the fit overshot it.

    The shift-invariance least squares is not constrained to be stable, and a
    marginally unstable ``A`` is fatal twice over: the ``B``/``D`` basis
    simulations below diverge over a 60k-sample record (numpy then reports the
    unhelpful "SVD did not converge"), and no ``P`` can satisfy
    ``A'PA - alpha^2 P < 0`` for ``alpha < 1`` afterwards.

    Only the OFFENDING eigenvalues are pulled in, radially, leaving every stable
    mode exactly where the fit put it. Scaling the whole matrix instead
    (``A * max_rho/rho``) damps the good modes too, and on the F-16 that turned a
    single bad eigenvalue into a linear NRMSE of 27.
    """
    vals, vecs = np.linalg.eig(A)
    rho = float(np.abs(vals).max())
    if not np.isfinite(rho) or rho <= max_rho:
        return A
    offending = np.abs(vals) > max_rho
    scaled = vals.copy()
    scaled[offending] *= max_rho / np.abs(vals[offending])
    repaired = vecs @ np.diag(scaled) @ np.linalg.inv(vecs)
    logger.info(
        "subspace A had rho=%.6f; pulled %d of %d eigenvalue(s) in to %.6f",
        rho, int(offending.sum()), len(vals), max_rho,
    )
    return np.real_if_close(repaired, tol=1e6).real


def _fit_B_D(A, C, u: np.ndarray, y: np.ndarray):
    """``(B, D, x0)`` by least squares with ``A``/``C`` fixed.

    With ``A`` and ``C`` known the output is LINEAR in ``(x0, B, D)``, so each
    basis element is simulated once and the coefficients are regressed. That is
    both obviously correct and better conditioned than an analytic Toeplitz
    construction.
    """
    n, nd = u.shape
    ne, nx = C.shape

    # CHUNKED, each chunk with its own free initial state. A single fit over the
    # whole record is dominated by the slowest mode: with rho ~ 0.9999 the
    # regression spans a 60k-sample horizon and comes back ill-conditioned (on
    # the F-16 that produced a linear NRMSE of 27). Chunking bounds the error
    # accumulation without assuming anything, and is just the standard
    # multiple-experiment formulation.
    chunk = max(1024, 50 * nx)
    chunk = max(chunk, int(np.ceil(n / MAX_FIT_CHUNKS)))   # bound the memory
    bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    bounds = [(a, b) for a, b in bounds if b - a > nx + 1]
    n_chunks = len(bounds)
    n_rows = sum(b - a for a, b in bounds)

    n_cols = n_chunks * nx + nx * nd + ne * nd
    Phi = np.zeros((n_rows, ne, n_cols))
    target = np.zeros((n_rows, ne))
    rows_idx = np.arange(nx)
    at = 0

    for ci, (a, b) in enumerate(bounds):
        m = b - a
        uc = u[a:b]
        # All basis responses propagate TOGETHER: the recursion is linear, so one
        # (n_basis, nx) @ (nx, nx) matmul per step replaces one simulation per
        # basis element.
        X = np.zeros((nx + nx * nd, nx))
        X[:nx] = np.eye(nx)                  # this chunk's x0 bases start at e_p
        for k in range(m):
            out = X @ C.T                    # (n_basis, ne)
            Phi[at + k, :, ci * nx : (ci + 1) * nx] = out[:nx].T
            Phi[at + k, :, n_chunks * nx : n_chunks * nx + nx * nd] = out[nx:].T
            if k == m - 1:
                break
            X = X @ A.T
            # B basis (p, q) receives e_p * u[k, q] -- a diagonal scatter.
            block = X[nx:].reshape(nx, nd, nx)
            block[rows_idx, :, rows_idx] += uc[k]
        # D is not dynamic: it is u itself.
        for r in range(ne):
            for q in range(nd):
                Phi[at : at + m, r, n_chunks * nx + nx * nd + r * nd + q] = uc[:, q]
        target[at : at + m] = y[a:b]
        at += m

    theta, *_ = np.linalg.lstsq(Phi.reshape(-1, n_cols), target.reshape(-1), rcond=None)
    B = theta[n_chunks * nx : n_chunks * nx + nx * nd].reshape(nx, nd)
    D = theta[n_chunks * nx + nx * nd :].reshape(ne, nd)
    return B, D, theta[:nx]


# ── step 2: balanced truncation ───────────────────────────────────────────────
def _gramian_sqrt(W: np.ndarray) -> np.ndarray:
    """A factor ``L`` with ``L L' = W``, falling back off Cholesky.

    The Gramians are PSD by construction but can be numerically semi-definite
    for a barely controllable/observable realization, where Cholesky fails.
    """
    try:
        return np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(W)
        return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))


def balanced_truncation(A, B, C, D, nx: int):
    """Balance (Moore 1981) and keep the ``nx`` largest Hankel singular values.

    In the balanced coordinates ``Wc = Wo = diag(sigma)``: controllability and
    observability are equalized, which is what puts every state on a comparable
    scale and makes one ``B2``/``C2`` magnitude meaningful in all directions.

    Returns ``(A, B, C, D, hankel)`` with the first four truncated to ``nx``.
    """
    A, B, C, D = map(np.asarray, (A, B, C, D))
    Wc = linalg.solve_discrete_lyapunov(A, B @ B.T)
    Wo = linalg.solve_discrete_lyapunov(A.T, C.T @ C)

    Lc = _gramian_sqrt(Wc)
    Lo = _gramian_sqrt(Wo)
    U, sigma, Vt = np.linalg.svd(Lo.T @ Lc)

    keep = min(nx, int(np.sum(sigma > sigma.max() * 1e-12)))
    if keep < nx:
        logger.warning(
            "balanced truncation: only %d of %d requested states are "
            "controllable+observable; padding the rest with zeros", keep, nx
        )
    s_inv_sqrt = 1.0 / np.sqrt(sigma[:keep])
    T = Lc @ Vt[:keep].T * s_inv_sqrt          # (n, keep)
    Tinv = (s_inv_sqrt[:, None]) * U[:, :keep].T @ Lo.T   # (keep, n)

    Ab = Tinv @ A @ T
    Bb = Tinv @ B
    Cb = C @ T

    if keep < nx:  # pad so the caller always gets the nx it asked for
        Ab = np.pad(Ab, ((0, nx - keep), (0, nx - keep)))
        Bb = np.pad(Bb, ((0, nx - keep), (0, 0)))
        Cb = np.pad(Cb, ((0, 0), (0, nx - keep)))
    return Ab, Bb, Cb, np.asarray(D), sigma


# ── the entry point ───────────────────────────────────────────────────────────
def fit_linear_model(
    u: np.ndarray,
    y: np.ndarray,
    nx: int,
    arx_order: Optional[int] = None,
    max_rho: Optional[float] = None,
    burn_in: int = 0,
) -> LinearInit:
    r"""Identify a balanced linear model of order ``nx`` from NORMALIZED data.

    Args:
        u: ``(N, nd)`` or ``(n_rec, N, nd)`` normalized input.
        y: ``(N, ne)`` or ``(n_rec, N, ne)`` normalized output. Multiple records
            are concatenated for the fit.
        nx: target state dimension.
        arx_order: subspace block size. Defaults to
            ``max(3*nx, 12)`` -- generous, because balancing throws the excess
            away in a principled order.
        max_rho: if given and ``rho(A)`` exceeds it, contract ``A`` by
            ``max_rho / rho(A)``. Prefer raising ``alpha_0`` instead; contracting
            distorts the identified dynamics and is a last resort for a fit that
            came back marginally unstable.
        burn_in: samples dropped from the front of each record before fitting.

    Returns:
        :class:`LinearInit`.
    """
    u3 = np.atleast_3d(np.asarray(u, dtype=float))
    y3 = np.atleast_3d(np.asarray(y, dtype=float))
    if u3.ndim == 2:
        u3, y3 = u3[None], y3[None]
    if u3.shape[0] != y3.shape[0] or u3.shape[1] != y3.shape[1]:
        raise ValueError(f"u {u3.shape} and y {y3.shape} disagree on records/length")

    uc = np.concatenate([r[burn_in:] for r in u3], axis=0)
    yc = np.concatenate([r[burn_in:] for r in y3], axis=0)
    ne = yc.shape[-1]

    # Identify at a HIGHER order than requested, then let balanced truncation
    # decide which states to keep. Identifying directly at nx would make the
    # truncation a no-op and the Hankel spectrum uninformative (every
    # energy_retained would read 100%), which hides how much a given nx costs.
    n_identify = OVER_IDENTIFY_MULTIPLIER * nx

    # The block Hankel has block*ne rows and we ask it for n_identify singular
    # directions, so the block must scale with BOTH. Sizing it off nx alone is
    # what broke the single-output F-16: block=27 with ne=1 gave 27 rows for 18
    # states, a ratio of 1.5, and the extra directions came back as junk modes
    # (linear NRMSE 26). At ne=3 the same block gave 81 rows and worked fine --
    # which is exactly the asymmetry this corrects.
    desired_block = arx_order or max(
        DEFAULT_BLOCK_MIN,
        DEFAULT_BLOCK_MULTIPLIER * nx,
        int(np.ceil(SUBSPACE_ROWS_PER_STATE * n_identify / ne)),
    )
    # The projection needs j = N - 2*block + 1 columns against 2*block*(nd+ne)
    # rows, i.e. roughly N >= 6*block. Short records therefore cap the block
    # HARD, and the over-identification has to give way rather than the fit
    # failing outright -- CED is 400 samples, where the block the F-16 wants
    # (288) does not remotely fit.
    max_block = max(nx + 2, len(uc) // 6)
    block = max(nx + 2, min(desired_block, max_block))
    affordable = max(nx, (block * ne) // SUBSPACE_ROWS_PER_STATE)
    n_identify = max(nx, min(n_identify, affordable, block * ne - 1))
    if n_identify < OVER_IDENTIFY_MULTIPLIER * nx:
        logger.info(
            "record of %d samples caps the subspace block at %d, so identifying "
            "%d states instead of %d before truncation",
            len(uc), block, n_identify, OVER_IDENTIFY_MULTIPLIER * nx,
        )
    A, C = _identify_A_C(uc, yc, nx=n_identify, block=block)
    B, D, _ = _fit_B_D(A, C, uc, yc)

    A, B, C, D, hankel = balanced_truncation(A, B, C, D, nx)
    A = _enforce_schur(A)  # truncation can nudge the spectrum

    if max_rho is not None:
        rho = float(np.abs(np.linalg.eigvals(A)).max())
        if rho > max_rho:
            logger.warning(
                "identified rho(A)=%.6f exceeds max_rho=%.6f; contracting A. "
                "Raising alpha_0 above rho(A) is usually the better fix.", rho, max_rho
            )
            A = A * (max_rho / rho)

    # Diagnostics on the fitting record: state scale spread and linear fit quality.
    x = np.zeros((len(uc), A.shape[0]))
    for k in range(len(uc) - 1):
        x[k + 1] = A @ x[k] + B @ uc[k]
    y_hat = x @ C.T + uc @ D.T
    resid = yc - y_hat
    denom = np.where(yc.std(axis=0) > 0, yc.std(axis=0), 1.0)
    nrmse = np.sqrt((resid ** 2).mean(axis=0)) / denom

    return LinearInit(
        A=A, B=B, C=C, D=D,
        hankel=hankel,
        state_rms=x.std(axis=0),
        nrmse=nrmse,
    )


# ── step 4: the feasibility question, answered by bisection ───────────────────
def largest_feasible_loop_gain(
    is_feasible: Callable[[float], bool],
    hi: float = 1.0,
    tolerance: float = 0.05,
    max_iter: int = 12,
) -> Tuple[float, bool]:
    r"""Largest ``g in [0, hi]`` for which ``is_feasible(g)`` holds.

    ``g`` scales ``B2`` and ``C2`` together, so it is the loop gain through the
    dead zone. **Bisection always terminates** because ``g = 0`` opens the loop
    and is feasible whenever ``A`` is Schur under ``alpha`` -- that is the whole
    point of starting from a linear model. This replaces sweeping ``B2`` over
    decades and hoping.

    Returns ``(gain, accepted_hi)``; ``accepted_hi`` is True when the requested
    ``hi`` was itself feasible, i.e. no reduction was needed.
    """
    if is_feasible(hi):
        return hi, True
    if not is_feasible(0.0):
        raise RuntimeError(
            "the open-loop (B2 = C2 = 0) LMI is infeasible, which should be "
            "impossible for a Schur A under alpha. Check rho(A) < alpha_0 and "
            "that the solver is working."
        )
    lo, best = 0.0, 0.0
    for _ in range(max_iter):
        if hi - lo <= tolerance * max(hi, 1e-12):
            break
        mid = 0.5 * (lo + hi)
        if is_feasible(mid):
            lo, best = mid, mid
        else:
            hi = mid
    return best, False
