"""Reference (ground-truth) dynamics for synthetic benchmark systems.

These are used by post-processing scripts (e.g. ``scripts/post_process.py``) to
compare an identified regionally-stable model against the true unknown system
under conditions that violate the model's input/state regional-stability
constraint. Each registered system exposes a uniform ``simulate(x0, u_seq, **)``
interface, so callers can switch systems by name.

Each spec also exposes ``converges_to_origin(x0)``: the autonomous (``u = 0``)
verdict used to colour trajectories started from a random ``x0`` by what the
TRUE system does there, independent of what the identified model does.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Duffing oscillator
# ---------------------------------------------------------------------------
# q'' = -delta_d * q' - q + q^3 + u
# Fixed points (u=0): (0,0) stable, (+/-1, 0) saddle.
# Saddle-node bifurcation in u: |u| > 2/(3*sqrt(3)) ~ 0.385 -> always diverges.

DUFFING_DELTA_D = 0.3
DUFFING_TS = 0.05
DUFFING_U_C = 2.0 / (3.0 * np.sqrt(3.0))
DUFFING_V_SADDLE = -0.25  # V(1, 0) = -1/4


def duffing_ct(t, x, u=0.0, delta_d=DUFFING_DELTA_D):
    q, dq = x
    ddq = -delta_d * dq - q + q ** 3 + u
    return [dq, ddq]


def duffing_dt(x, u=0.0, Ts=DUFFING_TS, delta_d=DUFFING_DELTA_D):
    """One RK45 step of the Duffing system (ZOH input)."""
    sol = solve_ivp(
        lambda t, xv: duffing_ct(t, xv, u=u, delta_d=delta_d),
        [0.0, Ts],
        x,
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
        dense_output=False,
    )
    return sol.y[:, -1]


def simulate_duffing(
    x0,
    u_seq,
    Ts: float = DUFFING_TS,
    delta_d: float = DUFFING_DELTA_D,
    diverge_thresh: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Simulate the discrete-time Duffing system.

    Returns ``(X, y, diverged)`` where ``X`` is the state trajectory of shape
    ``(T+1, 2)`` (or shorter if divergence triggered early), ``y = X[:-1, 0]``
    is the position output, and ``diverged`` is ``True`` if any state component
    exceeded ``diverge_thresh`` during the run.
    """
    u_seq = np.asarray(u_seq, dtype=float).reshape(-1)
    X = [np.asarray(x0, dtype=float)]
    diverged = False
    for k in range(len(u_seq)):
        x_next = duffing_dt(X[-1], u=float(u_seq[k]), Ts=Ts, delta_d=delta_d)
        X.append(x_next)
        if np.any(np.abs(x_next) > diverge_thresh) or not np.all(np.isfinite(x_next)):
            diverged = True
            break
    X = np.asarray(X)
    y = X[:-1, 0]
    return X, y, diverged


DUFFING_T_AUTONOMOUS = 200.0  # [s] horizon for the autonomous verdict
DUFFING_BLOW_UP = 10.0  # |state| beyond this counts as diverged


def duffing_converges_to_origin(
    x0,
    T: float = DUFFING_T_AUTONOMOUS,
    blow_up: float = DUFFING_BLOW_UP,
    tol: float = 1e-3,
    delta_d: float = DUFFING_DELTA_D,
) -> bool:
    """True autonomous Duffing (``u = 0``) from ``x0``: does it settle at the origin?

    ``x0`` inside the separatrix converges, outside it runs away to one of the
    ``q^3`` branches. Integrated in one shot (not step by step) because only the
    end state matters.
    """

    def escape(t, x):
        return np.max(np.abs(x)) - blow_up

    escape.terminal, escape.direction = True, 1

    sol = solve_ivp(
        lambda t, xv: duffing_ct(t, xv, u=0.0, delta_d=delta_d),
        [0.0, T],
        np.asarray(x0, dtype=float).reshape(-1),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        events=escape,
    )
    if sol.t_events[0].size:  # left the bounded region -> diverged
        return False
    return bool(np.linalg.norm(sol.y[:, -1]) < tol)


# ---------------------------------------------------------------------------
# 1-D Lur'e benchmark (scripts/generate_one_d_dataset.py)
# ---------------------------------------------------------------------------
# x+ = A x + B u + B2 dzn(C2 x + D21 u),  y = C x + D u + D12 dzn(...)
# The dead zone breaks at |x| = 1, above which the autonomous map becomes
# x+ = 2.0 x - 1.1: the true basin of attraction is |x| < 1.1, and the regional
# certificate at alpha = 0.99 covers {|x| <= 1.0}, exactly the linear region.

ONE_D_A, ONE_D_B, ONE_D_B2 = 0.9, 1.0, 1.1
ONE_D_C, ONE_D_C2, ONE_D_D21 = 1.0, 1.0, 0.0
ONE_D_D, ONE_D_D12 = 0.0, 0.0
ONE_D_TS = 1.0
ONE_D_DIVERGE_THRESH = 5.0  # matches the dataset generator
ONE_D_N_AUTONOMOUS = 500  # steps for the autonomous verdict (0.9^500 ~ 1e-23)


def dead_zone(z: float) -> float:
    """Scalar dead zone ``max(|z| - 1, 0) sign(z)``."""
    return max(abs(z) - 1.0, 0.0) * np.sign(z)


def one_d_dt(x, u: float = 0.0) -> np.ndarray:
    """One step of the 1-D Lur'e recurrence."""
    x = float(np.ravel(x)[0])
    w = dead_zone(ONE_D_C2 * x + ONE_D_D21 * u)
    return np.array([ONE_D_A * x + ONE_D_B * u + ONE_D_B2 * w])


def simulate_one_d(
    x0,
    u_seq,
    diverge_thresh: float = ONE_D_DIVERGE_THRESH,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Simulate the 1-D Lur'e system.

    Same contract as :func:`simulate_duffing`: ``(X, y, diverged)`` with ``X`` of
    shape ``(T+1, 1)`` (shorter if divergence triggered early), ``y = X[:-1, 0]``
    and ``diverged`` set when ``|x|`` exceeded ``diverge_thresh``.
    """
    u_seq = np.asarray(u_seq, dtype=float).reshape(-1)
    X = [np.asarray(x0, dtype=float).reshape(1)]
    diverged = False
    for k in range(len(u_seq)):
        x_next = one_d_dt(X[-1], u=float(u_seq[k]))
        X.append(x_next)
        if np.any(np.abs(x_next) > diverge_thresh) or not np.all(np.isfinite(x_next)):
            diverged = True
            break
    X = np.asarray(X)
    y = X[:-1, 0]
    return X, y, diverged


def one_d_converges_to_origin(
    x0,
    n_steps: int = ONE_D_N_AUTONOMOUS,
    blow_up: float = ONE_D_DIVERGE_THRESH,
    tol: float = 1e-3,
) -> bool:
    """True autonomous 1-D plant (``u = 0``) from ``x0``: back to the origin?"""
    X, _, diverged = simulate_one_d(x0, np.zeros(n_steps), diverge_thresh=blow_up)
    if diverged:
        return False
    return bool(np.linalg.norm(X[-1]) < tol)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrueDynamicsSpec:
    name: str
    simulate: Callable
    Ts: float
    state_dim: int
    state_labels: Tuple[str, ...]
    output_labels: Tuple[str, ...]
    metadata: Dict[str, float] = field(default_factory=dict)
    converges_to_origin: Optional[Callable] = None


_REGISTRY: Dict[str, TrueDynamicsSpec] = {
    "duffing": TrueDynamicsSpec(
        name="duffing",
        simulate=simulate_duffing,
        Ts=DUFFING_TS,
        state_dim=2,
        state_labels=("q", "q_dot"),
        output_labels=("q",),
        metadata={
            "delta_d": DUFFING_DELTA_D,
            "u_c": DUFFING_U_C,
            "V_saddle": DUFFING_V_SADDLE,
        },
        converges_to_origin=duffing_converges_to_origin,
    ),
    "one_d": TrueDynamicsSpec(
        name="one_d",
        simulate=simulate_one_d,
        Ts=ONE_D_TS,
        state_dim=1,
        state_labels=("x",),
        output_labels=("x",),
        metadata={
            "A": ONE_D_A,
            "B": ONE_D_B,
            "B2": ONE_D_B2,
            "x_breakpoint": 1.0,   # where the dead zone starts
            "x_basin": 1.1,        # |x| < 1.1 still returns to the origin
        },
        converges_to_origin=one_d_converges_to_origin,
    ),
}


def get_true_dynamics(name: str) -> TrueDynamicsSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown true-dynamics '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_true_dynamics() -> List[str]:
    return sorted(_REGISTRY.keys())
