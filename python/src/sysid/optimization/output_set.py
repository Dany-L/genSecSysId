r"""The certified OUTPUT set — the image of the state ellipsoid under ``y = C x``.

The certificate bounds the state to the invariant ellipsoid

.. math::  \mathcal{X} = \{x : x^T P^{-1} x \le s^2\},

and the model reads out ``y_n = C x`` in NORMALIZED output units, which the
normalizer maps to physical units channel by channel, ``y = S y_n`` with
``S = diag(output_std)``. Writing ``x = s P^{1/2} v`` with ``\|v\| \le 1`` the
image is

.. math::  \mathcal{Y} = \{ M v : \|v\|_2 \le 1 \},\quad M = s\,S\,C\,P^{1/2},

i.e. the ellipsoid ``{y : y^T Y y \le 1}`` with ``Y = W^{-1}`` and

.. math::  W = M M^T = s^2\, S\, (C P C^T)\, S .

``W`` is the object; everything reported about the output set is a reading of
it, and every reading below reduces to the SISO formula ``sigma*s*sqrt(CPC^T)``
that predates multi-output support.

Two readings matter:

* **Per output channel.** The support function of :math:`\mathcal{Y}` in
  direction ``e_i`` is the largest ``|y_i|`` the certificate admits,

  .. math::  \bar y_i = \sqrt{W_{ii}} = \sigma_i\, s\, \sqrt{(C P C^T)_{ii}},

  so :attr:`OutputEllipsoid.y_bar_per_output` is the half-width of the tightest
  axis-aligned BOX containing the output set. This is "the max value in each
  direction".

* **The worst direction over the sphere.** ``sqrt(lambda_min(W))`` is the radius
  of the largest ball INSIDE :math:`\mathcal{Y}`, which is what a coverage
  requirement of the form "the certified set reaches level ``y_max`` in every
  direction" binds on — the SDP writes it ``W \succeq y_max^2 I``. It is the
  conservative scalar, and the one :attr:`OutputEllipsoid.y_bar` reports.

The box bound is never smaller than the ball bound
(``y_bar_min <= min_i y_bar_i``), and the two coincide only when ``W`` is a
multiple of the identity — i.e. when the certified output set is a ball.

``W`` is singular whenever ``rank(C P C^T) < ne``, which happens as soon as
``ne > nx``: the state ellipsoid is then mapped into a proper subspace of the
output space and the set is flat in the remaining directions. The per-channel
bounds and ``y_bar`` stay well defined (the flat directions simply give zero),
but ``Y = W^{-1}`` does not exist, so :attr:`OutputEllipsoid.Y` returns
``None`` there rather than a pseudo-inverse that would silently describe a
different set.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class OutputEllipsoid:
    r"""The certified output set ``{y : y^T Y y <= 1}`` in PHYSICAL units.

    Built by :func:`output_ellipsoid`; ``W`` is the shape matrix ``Y^{-1}``.
    """

    #: ``(ne, ne)`` shape matrix ``s^2 S (C P C^T) S``. Symmetric PSD.
    W: np.ndarray
    #: ``(ne,)`` per-channel half-widths ``sqrt(W_ii)`` — the tightest box.
    y_bar_per_output: np.ndarray
    #: ``sqrt(lambda_min(W))`` — the largest ball inside the set.
    y_bar_min: float
    #: ``sqrt(lambda_max(W))`` — the smallest ball containing it.
    y_bar_max: float

    @property
    def y_bar(self) -> float:
        """The scalar certified half-width: the WORST output direction.

        Equal to ``sigma*s*sqrt(C P C^T)`` when ``ne == 1``, so this is the
        drop-in generalization of the old scalar ``y_bar``.
        """
        return self.y_bar_min

    @property
    def ne(self) -> int:
        return int(self.W.shape[0])

    @property
    def Y(self) -> Optional[np.ndarray]:
        """``W^{-1}``, the matrix of ``{y : y^T Y y <= 1}``; ``None`` if flat.

        Singular ``W`` means the output set is degenerate (``ne`` exceeds the
        rank of ``C P C^T``); there is then no ``Y`` describing it and a
        pseudo-inverse would describe a *different*, lower-dimensional set.
        """
        try:
            Y = np.linalg.inv(self.W)
        except np.linalg.LinAlgError:
            return None
        return None if not np.all(np.isfinite(Y)) else 0.5 * (Y + Y.T)

    def covers(self, y_max: float) -> bool:
        """Does the set reach ``y_max`` in EVERY direction (``W >= y_max^2 I``)?"""
        return bool(self.y_bar_min >= float(y_max))

    def to_dict(self) -> Dict[str, Any]:
        """JSON/MLflow-friendly summary (lists, not arrays)."""
        Y = self.Y
        return {
            "y_bar": float(self.y_bar),
            "y_bar_per_output": [float(v) for v in self.y_bar_per_output],
            "y_bar_min": float(self.y_bar_min),
            "y_bar_max": float(self.y_bar_max),
            "W": [[float(v) for v in row] for row in self.W],
            "Y": None if Y is None else [[float(v) for v in row] for row in Y],
        }


def _as_output_std(output_std, ne: int) -> np.ndarray:
    """``output_std`` as a ``(ne,)`` positive vector; a scalar is broadcast.

    The normalizer stores ``(1, 1, ne)``; a directly constructed model may carry
    a plain float. Both are accepted so callers never have to care.
    """
    sigma = np.asarray(output_std, dtype=float).reshape(-1)
    if sigma.size == 1:
        sigma = np.repeat(sigma, ne)
    if sigma.size != ne:
        raise ValueError(
            f"output_std has {sigma.size} entries but the model has ne={ne}; "
            "give one scale per output channel (or a single scalar)."
        )
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError(f"output_std must be finite and positive, got {sigma}")
    return sigma


def output_ellipsoid(C, P, s, output_std) -> OutputEllipsoid:
    r"""The certified output set for ``y = C x`` over ``{x : x^T P^{-1} x <= s^2}``.

    Args:
        C: ``(ne, nx)`` output map, in the model's normalized units.
        P: ``(nx, nx)`` Lyapunov matrix (the ellipsoid is ``x^T P^{-1} x <= s^2``).
        s: certificate scale.
        output_std: per-channel physical output scale, ``(ne,)`` or a scalar.

    Returns:
        :class:`OutputEllipsoid` in PHYSICAL units.
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    P = np.atleast_2d(np.asarray(P, dtype=float))
    s = float(np.asarray(s).reshape(-1)[0])
    ne = C.shape[0]
    sigma = _as_output_std(output_std, ne)

    CPCt = C @ P @ C.T
    CPCt = 0.5 * (CPCt + CPCt.T)  # kill the asymmetry rounding leaves behind
    W = (s ** 2) * (sigma[:, None] * CPCt * sigma[None, :])
    W = 0.5 * (W + W.T)

    # Clip at zero before the square roots: P is only numerically PSD, so a
    # direction the certificate does not extend into can come back at -1e-18.
    diag = np.clip(np.diag(W), 0.0, None)
    eigs = np.clip(np.linalg.eigvalsh(W), 0.0, None)
    return OutputEllipsoid(
        W=W,
        y_bar_per_output=np.sqrt(diag),
        y_bar_min=float(np.sqrt(eigs.min())),
        y_bar_max=float(np.sqrt(eigs.max())),
    )
