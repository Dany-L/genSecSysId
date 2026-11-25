"""Loading the trajectory datasets and turning them into padded tensors.

Folder layout (as produced by the dataset generators)::

    <path>/train/*.csv        converging trajectories (unknown x0 -> washout)
    <path>/train_div/*.csv    diverging trajectories (x0 = 0, end at the divergence threshold)
    <path>/validation[_div]/  <path>/test[_div]/

Trajectories can have different lengths. Batches are padded: inputs with 0, targets with
NaN, and :func:`masked_mse` ignores the NaN targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

Tensor = torch.Tensor
SPLITS = ("train", "validation", "test")


@dataclass
class Trajectories:
    """Lists of (T_i, nu) inputs and (T_i, ny) outputs."""

    u: List[np.ndarray] = field(default_factory=list)
    y: List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.u)

    def map(self, fu, fy) -> "Trajectories":
        return Trajectories([fu(a) for a in self.u], [fy(a) for a in self.y])


@dataclass
class Split:
    conv: Trajectories
    div: Trajectories


def load_folder(folder: Path, inputs: Sequence[str], outputs: Sequence[str]) -> Trajectories:
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {folder}")
    traj = Trajectories()
    for f in files:
        df = pd.read_csv(f)
        traj.u.append(df[list(inputs)].to_numpy(dtype=np.float64))
        traj.y.append(df[list(outputs)].to_numpy(dtype=np.float64))
    return traj


def load_dataset(
    path: str, inputs: Sequence[str], outputs: Sequence[str], diverging: bool
) -> Dict[str, Split]:
    """Load train/validation/test, each with its converging and (optionally) diverging part."""
    root = Path(path).expanduser()
    out = {}
    for split in SPLITS:
        conv = load_folder(root / split, inputs, outputs)
        div_dir = root / f"{split}_div"
        if diverging and not div_dir.exists():
            raise FileNotFoundError(f"diverging trajectories requested but {div_dir} is missing")
        div = load_folder(div_dir, inputs, outputs) if diverging else Trajectories()
        out[split] = Split(conv, div)
    return out


class Normalizer:
    """Per-channel scaling fitted on the converging training data.

    ``scale_only`` (default) divides by the standard deviation and keeps the origin, so the
    certificate's equilibrium u = 0, y = 0 stays the physical one. ``standard`` also removes
    the mean. ``none`` is the identity.
    """

    METHODS = ("scale_only", "standard", "none")

    def __init__(self, method: str = "scale_only"):
        if method not in self.METHODS:
            raise ValueError(f"normalization must be one of {self.METHODS}, got {method!r}")
        self.method = method
        self.u_mean = self.u_std = self.y_mean = self.y_std = None

    def fit(self, traj: Trajectories) -> "Normalizer":
        u, y = np.concatenate(traj.u), np.concatenate(traj.y)
        nu, ny = u.shape[1], y.shape[1]
        self.u_mean, self.y_mean = np.zeros(nu), np.zeros(ny)
        self.u_std, self.y_std = np.ones(nu), np.ones(ny)
        if self.method in ("scale_only", "standard"):
            self.u_std, self.y_std = u.std(0), y.std(0)
            self.u_std[self.u_std == 0] = 1.0
            self.y_std[self.y_std == 0] = 1.0
        if self.method == "standard":
            self.u_mean, self.y_mean = u.mean(0), y.mean(0)
        return self

    def u(self, a: np.ndarray) -> np.ndarray:
        return (a - self.u_mean) / self.u_std

    def y(self, a: np.ndarray) -> np.ndarray:
        return (a - self.y_mean) / self.y_std

    def y_inverse(self, a):
        return a * self.y_std + self.y_mean

    def apply(self, traj: Trajectories) -> Trajectories:
        return traj.map(self.u, self.y)

    def state_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        n = cls(d["method"])
        for k in ("u_mean", "u_std", "y_mean", "y_std"):
            setattr(n, k, np.asarray(d[k], dtype=np.float64))
        return n


def windows(traj: Trajectories, length: Optional[int], stride: Optional[int]) -> Trajectories:
    """Cut every trajectory into windows of ``length`` with ``stride`` (default: length).

    ``length=None`` keeps the full trajectories. A trajectory shorter than ``length`` is kept
    whole.
    """
    if length is None:
        return traj
    stride = stride or length
    out = Trajectories()
    for u, y in zip(traj.u, traj.y):
        if len(u) <= length:
            out.u.append(u), out.y.append(y)
            continue
        for start in range(0, len(u) - length + 1, stride):
            out.u.append(u[start : start + length])
            out.y.append(y[start : start + length])
    return out


def pad(traj: Trajectories, idx: Optional[Sequence[int]] = None, device="cpu"):
    """Stack (a subset of) trajectories into (batch, T_max, ·) tensors.

    Inputs are padded with 0 and targets with NaN.
    """
    idx = range(len(traj)) if idx is None else idx
    us = [traj.u[i] for i in idx]
    ys = [traj.y[i] for i in idx]
    T = max(len(a) for a in us)
    U = np.zeros((len(us), T, us[0].shape[1]))
    Y = np.full((len(ys), T, ys[0].shape[1]), np.nan)
    for i, (u, y) in enumerate(zip(us, ys)):
        U[i, : len(u)] = u
        Y[i, : len(y)] = y
    return (torch.as_tensor(U, dtype=torch.float64, device=device),
            torch.as_tensor(Y, dtype=torch.float64, device=device))


def masked_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over the finite target entries."""
    valid = torch.isfinite(target)
    if not valid.any():
        return pred.new_zeros(())
    return ((pred[valid] - target[valid]) ** 2).mean()


def relative_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean over trajectories of Σ_k‖ŷ_k - y_k‖² / Σ_k‖y_k‖² (finite targets only).

    Scale-free, so every diverging trajectory counts about equally, however large its
    output grows before it hits the divergence threshold.
    """
    valid = torch.isfinite(target)
    t = torch.where(valid, target, torch.zeros_like(target))
    err = torch.where(valid, pred - t, torch.zeros_like(pred))
    num = (err**2).flatten(1).sum(1)
    den = (t**2).flatten(1).sum(1)
    has = valid.flatten(1).any(1) & (den > 0)
    if not has.any():
        return pred.new_zeros(())
    return (num[has] / den[has]).mean()


DIVERGING_LOSSES = {"relative": relative_mse, "mse": masked_mse}


def max_input_norm(traj: Trajectories) -> float:
    """max_k ‖u_k‖ over all trajectories."""
    return float(max(np.linalg.norm(u, axis=1).max() for u in traj.u))


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
