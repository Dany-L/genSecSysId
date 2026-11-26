"""Dynamic recurrent network (DRN) in Lur'e form and the LSTM baseline."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .parametrization import free_parameter_shapes, gamma

Tensor = torch.Tensor

CERTIFICATES = ("regional", "global", "none")


def dzn(v: Tensor) -> Tensor:
    """Deadzone: 0 on [-1, 1], v - 1 above, v + 1 below."""
    return v - v.clamp(-1.0, 1.0)


def simulate(
    A: Tensor, B: Tensor, B2: Tensor, C: Tensor, C2: Tensor, D21: Tensor, u: Tensor,
    x0: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Roll out the DRN.

    Args:
        u: inputs (batch, N, nu).
        x0: initial state (batch, nx); zero if None.

    Returns:
        y (batch, N, ny) with y_k = C x_k, and the states x_0..x_{N-1} (batch, N, nx).
    """
    batch, N, _ = u.shape
    nx = A.shape[0]
    x = u.new_zeros(batch, nx) if x0 is None else x0
    # unbind once: indexing [:, k] inside the loop would make every step's backward
    # allocate a full-size gradient, i.e. O(N²) time.
    Bu = (u @ B.T).unbind(1)
    Du = (u @ D21.T).unbind(1)
    AC2 = torch.cat([A, C2], dim=0).T  # one matmul for A x and C2 x
    B2T = B2.T
    xs = []
    for k in range(N):
        xs.append(x)
        ax, c2x = (x @ AC2).split([nx, C2.shape[0]], dim=-1)
        w = dzn(c2x + Du[k])
        x = ax + Bu[k] + w @ B2T
    X = torch.stack(xs, dim=1)
    return X @ C.T, X


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class DRN(nn.Module):
    """Dynamic recurrent network x⁺ = Ax + Bu + B2 w, y = Cx, w = dzn(C2 x + D21 u).

    ``certificate``:
        * ``regional``: (A, B, B2, C2, D21) = Γ(θ̂) with H from z — regional certificate.
        * ``global``: Γ with H = 0 — standard sector condition, globally ISS.
        * ``none``: free matrices, no certificate.

    For the certified variants α = sigmoid(α̂) and s = exp(ŝ) are parameters; whether they
    are trained is set by ``train_alpha`` / ``train_s``.
    """

    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nz: int,
        certificate: str = "regional",
        alpha: float = 0.99,
        s: float = 1.0,
        train_alpha: bool = True,
        train_s: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if certificate not in CERTIFICATES:
            raise ValueError(f"certificate must be one of {CERTIFICATES}, got {certificate!r}")
        if not 0.0 < alpha < 1.0 or s <= 0.0:
            raise ValueError(f"need 0 < alpha < 1 and s > 0, got alpha={alpha}, s={s}")
        self.nu, self.ny, self.nx, self.nz = nu, ny, nx, nz
        self.certificate = certificate
        self.eps = eps
        dt = torch.float64
        self.C = nn.Parameter(torch.randn(ny, nx, dtype=dt) / math.sqrt(nx))
        if self.certified:
            shapes = free_parameter_shapes(nx, nu, nz, regional=certificate == "regional")
            self.theta = nn.ParameterDict(
                {k: nn.Parameter(torch.zeros(*v, dtype=dt)) for k, v in shapes.items()}
            )
            self.alpha_raw = nn.Parameter(torch.tensor(_logit(alpha), dtype=dt), train_alpha)
            self.log_s = nn.Parameter(torch.tensor(math.log(s), dtype=dt), train_s)
            self.sample_free_parameters(scale=0.1)
        else:
            self.theta = nn.ParameterDict(
                {
                    k: nn.Parameter(torch.zeros(*v, dtype=dt))
                    for k, v in {
                        "A": (nx, nx), "B": (nx, nu), "B2": (nx, nz),
                        "C2": (nz, nx), "D21": (nz, nu),
                    }.items()
                }
            )
            self.register_buffer("alpha_raw", torch.tensor(_logit(alpha), dtype=dt))
            self.register_buffer("log_s", torch.tensor(math.log(s), dtype=dt))
            self.sample_free_parameters(scale=0.1)

    @property
    def certified(self) -> bool:
        return self.certificate != "none"

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_raw)

    @property
    def s(self) -> Tensor:
        return torch.exp(self.log_s)

    def matrices(self) -> Dict[str, Tensor]:
        """Model matrices (A, B, B2, C, C2, D21), plus (Lam, H) when certified."""
        if not self.certified:
            return {**dict(self.theta.items()), "C": self.C}
        mats = gamma(
            dict(self.theta.items()), self.nx, self.nu, self.nz, self.alpha, self.s,
            regional=self.certificate == "regional", eps=self.eps,
        )
        mats["C"] = self.C
        return mats

    def simulate(self, u: Tensor, x0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """(y, x) for inputs u (batch, N, nu); see :func:`simulate`."""
        m = self.matrices()
        u = u.to(m["A"].dtype)
        return simulate(m["A"], m["B"], m["B2"], m["C"], m["C2"], m["D21"], u, x0)

    def forward(self, u: Tensor) -> Tensor:
        return self.simulate(u)[0]

    @torch.no_grad()
    def sample_free_parameters(self, scale: float = 0.1, generator: Optional[torch.Generator] = None):
        """Draw a random reservoir; α, s and C are left unchanged.

        Certified: every entry of θ̂ ~ N(0, scale²), except d ~ N(0, 1) (Λ ≈ I) and
        z ~ N(0, 1). The scale of (X, Y, Z) sets the dynamics: small scales give A close to
        α·(rotation), i.e. slow, lightly damped modes.
        Uncertified: the matrices of a random regional model with the same α and s, so all
        three variants start from the same distribution of reservoirs.
        """
        if not self.certified:
            ref = DRN(self.nu, self.ny, self.nx, self.nz, "regional",
                      alpha=float(self.alpha), s=float(self.s), eps=self.eps)
            ref.sample_free_parameters(scale, generator)
            mats = ref.matrices()
            for k in self.theta:
                self.theta[k].copy_(mats[k])
            return
        for k, p in self.theta.items():
            std = 1.0 if k in ("d", "z") else scale
            p.copy_(torch.randn(p.shape, generator=generator, dtype=p.dtype) * std)


class LSTM(nn.Module):
    """LSTM baseline with a linear output layer (no certificate)."""

    certified = False

    def __init__(self, nu: int, ny: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(nu, hidden_size, num_layers, batch_first=True, dtype=torch.float64)
        self.head = nn.Linear(hidden_size, ny, dtype=torch.float64)

    def forward(self, u: Tensor) -> Tensor:
        h, _ = self.lstm(u.to(torch.float64))
        return self.head(h)
