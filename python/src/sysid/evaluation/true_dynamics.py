"""Reference (ground-truth) dynamics for synthetic benchmark systems.

These are used by post-processing scripts (e.g. ``scripts/post_process.py``) to
compare an identified regionally-stable model against the true unknown system
under conditions that violate the model's input/state regional-stability
constraint. Each registered system exposes a uniform ``simulate(x0, u_seq, **)``
interface, so callers can switch systems by name.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

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


def duffing_V_energy(q, dq):
    """Hamiltonian-like energy. V > V_saddle ⇔ inside basin of attraction."""
    return dq ** 2 / 2.0 - q ** 2 / 2.0 + q ** 4 / 4.0


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
