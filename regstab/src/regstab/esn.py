"""Echo-state-network initialization of the DRN.

Draw random reservoirs θ̂ (every draw satisfies the certificate by construction), roll each
one out on the converging training inputs from x0 = 0, and fit the output layer C by ridge
regression on the states after the washout. Keep the reservoir whose readout has the lowest
validation error. Afterwards all parameters are trained jointly; the ESN is only the
starting point.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional, Sequence, Union

import torch

from .data import Trajectories, masked_mse, pad
from .model import DRN

logger = logging.getLogger(__name__)
Tensor = torch.Tensor


def _states_and_targets(model: DRN, traj: Trajectories, washout: int):
    u, y = pad(traj)
    with torch.no_grad():
        _, x = model.simulate(u)
    keep = torch.isfinite(y).all(-1)
    keep[:, :washout] = False
    return x[keep], y[keep]


def ridge_readout(x: Tensor, y: Tensor, ridge: float) -> Tensor:
    """C minimizing Σ‖y_k - C x_k‖² + λ‖C‖²_F, with λ = ridge · tr(XᵀX)/n.

    The ridge is relative to the state energy because the state scale depends on s (the
    input enters as u/s).
    """
    xtx = x.T @ x
    lam = ridge * torch.trace(xtx) / x.shape[1]
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(xtx + lam * eye, x.T @ y).T


def _draw_scale(scale: Union[float, Sequence[float]], g: torch.Generator) -> float:
    if isinstance(scale, (int, float)):
        return float(scale)
    lo, hi = (math.log(float(v)) for v in scale)
    return math.exp(lo + (hi - lo) * float(torch.rand((), generator=g, dtype=torch.float64)))


@torch.no_grad()
def esn_init(
    model: DRN,
    train: Trajectories,
    washout: int,
    val: Optional[Trajectories] = None,
    restarts: int = 10,
    ridge: float = 1e-8,
    scale: Union[float, Sequence[float]] = (0.01, 1.0),
    seed: int = 0,
) -> Dict[str, object]:
    """Initialize ``model`` in place; returns a summary of the restarts.

    Args:
        train, val: converging trajectories (normalized).
        washout: number of initial steps excluded from the readout fit and the error.
        restarts: number of random reservoirs.
        ridge: relative ridge weight of the readout.
        scale: std of the entries of (X, Y, Z); a ``[lo, hi]`` pair draws it log-uniformly
            per restart.
    """
    g = torch.Generator().manual_seed(seed)
    select = val if val is not None and len(val) else train
    best, best_err, history = None, math.inf, []
    for r in range(restarts):
        sc = _draw_scale(scale, g)
        model.sample_free_parameters(sc, generator=g)
        x, y = _states_and_targets(model, train, washout)
        model.C.copy_(ridge_readout(x, y, ridge))
        u_sel, y_sel = pad(select)
        y_hat = model(u_sel)
        err = float(masked_mse(y_hat[:, washout:], y_sel[:, washout:]))
        history.append({"restart": r, "scale": sc, "mse": err})
        logger.info("ESN restart %d: scale %.3g, %s mse %.4g", r, sc,
                    "val" if select is val else "train", err)
        if math.isfinite(err) and err < best_err:
            best, best_err = copy.deepcopy(model.state_dict()), err
    if best is None:
        raise RuntimeError("every ESN restart produced a non-finite error")
    model.load_state_dict(best)
    chosen = min(history, key=lambda h: h["mse"] if math.isfinite(h["mse"]) else math.inf)
    return {"best_mse": best_err, "best_restart": chosen["restart"],
            "best_scale": chosen["scale"], "restarts": history}
