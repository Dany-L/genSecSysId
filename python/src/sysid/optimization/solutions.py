"""Typed results for the Lure certificate SDPs.

Every certificate SDP shares the same core solution — the Lyapunov matrix ``P``,
the coupling ``L``, the multiplier ``M = diag(m)`` and the scale ``s`` — so the
solvers in :mod:`sysid.optimization.synthesizer` return these dataclasses instead
of loosely-typed dicts. Each specialization adds the diagnostics that solve
produces (feasibility margins, the ellipsoid volume, the certified output
half-width, …).

All matrices are numpy arrays in the model's *normalized* units; ``s`` is the
scale of the certified invariant ellipsoid ``{x : (1/s²)·xᵀP⁻¹x ≤ 1}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CertificateSolution:
    """The core certificate ``(P, L, M, s)`` shared by every SDP.

    ``_apply_certificate_solution`` writes exactly these four fields back into the
    model, so any solution below can be applied.
    """

    P: np.ndarray
    L: np.ndarray
    M: np.ndarray
    s: float


@dataclass
class MaxSSolution(CertificateSolution):
    """**MaxS** — the max-feasible-``s`` certificate (feasibility ceiling)."""

    max_eig_F: float = 0.0
    locality_min_eigs: List[float] = field(default_factory=list)


@dataclass
class BootstrapSolution(CertificateSolution):
    """**Bootstrap** — MaxS solved with ``D21`` (and optionally ``B``) free.

    Used once, at initialization, when the identity draw lands outside the
    feasible set: the SDP then repairs the *dynamics* alongside the certificate
    rather than only the certificate. ``B`` / ``D21`` carry the solved values, or
    ``None`` when that matrix was held fixed.
    """

    B: Optional[np.ndarray] = None
    D21: Optional[np.ndarray] = None


@dataclass
class CoverageSolution(CertificateSolution):
    """**Coverage** — a fixed-``s`` certificate binding the output requirement.

    ``y_bar`` is the PHYSICAL certified output half-width ``σ·s·√(C P Cᵀ)``
    (``ne == 1`` only, else ``None``).
    """

    y_bar: Optional[float] = None


@dataclass
class CoveragePoint:
    """One point of the coverage s-sweep."""

    s: float
    y_bar: Optional[float]


@dataclass
class CoverageSweepResult:
    """Tightest-coverage sweep: the smallest certified ``ȳ`` reaching ``y_max``."""

    y_f: Optional[float]
    s_f: Optional[float]
    sol: Optional[CoverageSolution]
    sweep: List[CoveragePoint] = field(default_factory=list)


@dataclass
class InitializationReport:
    """Flat, log/mlflow-friendly summary of the certificate established at init.

    :meth:`to_metrics` yields the finite scalar metrics for mlflow under the
    ``initialization/`` namespace.
    """

    volume: float
    s: float
    norm_H: float
    max_eig_F: float
    constraints_satisfied: bool
    y_max: Optional[float] = None
    # Dead-zone activity on the training rollout at init. ``firing_rate`` is the
    # fraction of (step, unit) pairs outside the dead band, ``units_firing`` the
    # fraction of units that ever fire and ``max_abs_z`` the largest |z| reached.
    # firing_rate == 0 means the nonlinearity is inert on the training data: the
    # model is LTI *in that regime*, and — because Δ'(z) = 0 inside the dead band —
    # no gradient reaches B2/C2/D21 through the prediction loss, so training can
    # never escape it.
    firing_rate: Optional[float] = None
    units_firing: Optional[float] = None
    max_abs_z: Optional[float] = None
    # Warm start only: ‖Δθ_i‖/‖θ_i‖ actually applied per parameter, so a sanity
    # run records how far from the reference it started.
    warm_start_offsets: Optional[dict] = None

    def to_metrics(self) -> dict:
        """The report as ``{metric_name: float}`` — finite scalars only (bools
        become 0/1; ``None`` and non-finite values such as an unbounded ``rho``
        are dropped so mlflow never sees them)."""
        raw = {
            "volume": self.volume,
            "s": self.s,
            "norm_H": self.norm_H,
            "max_eig_F": self.max_eig_F,
            "constraints_satisfied": self.constraints_satisfied,
            "y_max": self.y_max,
            "firing_rate": self.firing_rate,
            "units_firing": self.units_firing,
            "max_abs_z": self.max_abs_z,
        }
        for name, value in (self.warm_start_offsets or {}).items():
            raw[f"warm_start_offset/{name}"] = value
        metrics = {}
        for name, value in raw.items():
            if value is None:
                continue
            num = float(int(value) if isinstance(value, bool) else value)
            if np.isfinite(num):
                metrics[name] = num
        return metrics
