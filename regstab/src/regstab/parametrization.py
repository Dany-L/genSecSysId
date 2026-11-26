"""Direct parametrization Γ: free parameters θ̂ -> DRN matrices with a regional certificate.

Model (paper, eq. ``state-space-model``)::

    x_{k+1} = A x_k + B u_k + B2 w_k
    y_k     = C x_k
    v_k     = C2 x_k + D21 u_k,      w_k = dzn(v_k)

Certificate (Theorem ``regional_iss`` with X = I): for α ∈ (0, 1), s > 0 there exist
Λ ∈ D+ and N = ΛH such that, with K = ΛC2 + N and D̄ = ΛD21,

    F = [[-α²I, 0,      Kᵀ,  Aᵀ ],
         [ 0,  -I/s²,   D̄ᵀ,  Bᵀ ],
         [ K,   D̄,    -2Λ,   B2ᵀ],
         [ A,   B,     B2,  -I  ]]  ≺ 0        and      ‖h_i‖ ≤ 1  (⇔ G_i ⪰ 0).

Then the unit ball is forward invariant for inputs with ‖u_k‖² ≤ s²(1 - α²‖x_k‖²).

Construction (paper, Theorem ``direct-parametrization``; dimensions n = nx, m = nu, q = nz):

1. Λ = diag(exp(d)).
2. U = Cayley(X2, Y2, Z2) ∈ R^{q×(n+m)}, ‖U‖ < 1, and G := [K/α, sD̄] = √(2Λ) U,
   so that 2Λ - GGᵀ = √(2Λ)(I - UUᵀ)√(2Λ) ≻ 0, i.e. J ≻ 0.
3. A factor R_J R_Jᵀ = J is available in closed form,
   R_J = [[diag(αI, I/s), 0], [-G, L_S]] with L_S L_Sᵀ = 2Λ - GGᵀ.
4. V = Cayley(X1, Y1, Z1)ᵀ ∈ R^{n×(n+m+q)}, ‖V‖ < 1, and W = [A, B, B2] = V R_Jᵀ,
   so WᵀW ≺ J, which is F ≺ 0 by a Schur complement.
5. h_i = z_i / √(1 + ‖z_i‖²) (regional) or H = 0 (global), C2 = Λ⁻¹K - H, D21 = Λ⁻¹D̄.

The closed-form R_J avoids a Cholesky factorization of J, whose 1/s² block makes it
ill-conditioned for large s. Γ is evaluated in float64.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


def contraction_shapes(rows: int, cols: int) -> Dict[str, Tuple[int, int]]:
    """Free-parameter shapes for a ``rows × cols`` matrix with spectral norm < 1."""
    c, r = min(rows, cols), max(rows, cols)
    return {"X": (c, c), "Y": (c, c), "Z": (r - c, c)}


def cayley(X: Tensor, Y: Tensor, Z: Tensor, eps: float) -> Tensor:
    """Tall Cayley transform with ‖N‖₂ < 1.

    N = [(I - M)(I + M)⁻¹; -2Z(I + M)⁻¹] with M = XᵀX + Y - Yᵀ + ZᵀZ + εI.
    Since M + Mᵀ = 2(XᵀX + ZᵀZ + εI), NᵀN = I - 4(I+M)⁻ᵀ(XᵀX + εI)(I+M)⁻¹ ≺ I.
    """
    c = Y.shape[-1]
    I = torch.eye(c, dtype=Y.dtype, device=Y.device)
    M = X.T @ X + Y - Y.T + Z.T @ Z + eps * I
    inv = torch.linalg.inv(I + M)
    return torch.cat([(I - M) @ inv, -2.0 * Z @ inv], dim=0)


def contraction(X: Tensor, Y: Tensor, Z: Tensor, rows: int, cols: int, eps: float) -> Tensor:
    """``rows × cols`` matrix with spectral norm < 1 (transposed Cayley for wide shapes)."""
    N = cayley(X, Y, Z, eps)
    return N if rows >= cols else N.T


def free_parameter_shapes(nx: int, nu: int, nz: int, regional: bool) -> Dict[str, Tuple[int, ...]]:
    """Shapes of θ̂ (without the output matrix C, which is free)."""
    n, m, q = nx, nu, nz
    shapes: Dict[str, Tuple[int, ...]] = {"d": (q,)}
    if regional:
        shapes["z"] = (q, n)
    for k, v in contraction_shapes(q, n + m).items():
        shapes[f"{k}2"] = v
    for k, v in contraction_shapes(n, n + m + q).items():
        shapes[f"{k}1"] = v
    return shapes


def gamma(
    p: Dict[str, Tensor],
    nx: int,
    nu: int,
    nz: int,
    alpha: Tensor,
    s: Tensor,
    regional: bool,
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    """Map free parameters to (A, B, B2, C2, D21) and the certificate (Λ, H).

    Args:
        p: free parameters with the shapes of :func:`free_parameter_shapes`.
        alpha: contraction rate in (0, 1) (scalar tensor).
        s: input scaling > 0 (scalar tensor).
        regional: H from ``p["z"]`` if True, H = 0 (standard sector condition) if False.
    """
    n, m, q = nx, nu, nz
    lam = torch.exp(p["d"])
    sq2lam = torch.sqrt(2.0 * lam)[:, None]

    U = contraction(p["X2"], p["Y2"], p["Z2"], q, n + m, eps)  # (q, n+m)
    G = sq2lam * U  # = [K/α, s D̄]
    K = alpha * G[:, :n]
    Dbar = G[:, n:] / s
    Iq = torch.eye(q, dtype=U.dtype, device=U.device)
    L_S = sq2lam * torch.linalg.cholesky(Iq - U @ U.T)  # L_S L_Sᵀ = 2Λ - GGᵀ

    V = contraction(p["X1"], p["Y1"], p["Z1"], n, n + m + q, eps)  # (n, n+m+q)
    V1, V2 = V[:, : n + m], V[:, n + m :]
    A = alpha * V1[:, :n]
    B = V1[:, n:] / s
    B2 = -V1 @ G.T + V2 @ L_S.T

    if regional:
        z = p["z"]
        H = z / torch.sqrt(1.0 + (z**2).sum(dim=1, keepdim=True))
    else:
        H = torch.zeros(q, n, dtype=U.dtype, device=U.device)
    C2 = K / lam[:, None] - H
    D21 = Dbar / lam[:, None]
    return {"A": A, "B": B, "B2": B2, "C2": C2, "D21": D21, "Lam": lam, "H": H}


def certificate_margins(
    A, B, B2, C2, D21, Lam, H, alpha: float, s: float
) -> Dict[str, float]:
    """Check the conditions of Theorem 1 numerically (float64 numpy).

    Returns ``max_eig_F`` (< 0 when F ≺ 0) and ``max_h_norm`` (≤ 1 when every G_i ⪰ 0).
    """
    A, B, B2, C2, D21, H = (np.asarray(t, dtype=np.float64) for t in (A, B, B2, C2, D21, H))
    lam = np.asarray(Lam, dtype=np.float64).reshape(-1)
    n, m, q = A.shape[0], B.shape[1], lam.shape[0]
    L = np.diag(lam)
    K = L @ C2 + L @ H
    Dbar = L @ D21
    F = np.block(
        [
            [-(alpha**2) * np.eye(n), np.zeros((n, m)), K.T, A.T],
            [np.zeros((m, n)), -np.eye(m) / s**2, Dbar.T, B.T],
            [K, Dbar, -2.0 * L, B2.T],
            [A, B, B2, -np.eye(n)],
        ]
    )
    F = 0.5 * (F + F.T)
    return {
        "max_eig_F": float(np.linalg.eigvalsh(F).max()),
        "max_h_norm": float(np.linalg.norm(H, axis=1).max()) if q else 0.0,
    }
