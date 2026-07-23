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
class VolumePoint:
    """One point of the MaxVol s-sweep."""

    s: float
    volume: float
    logdet_P: float


@dataclass
class MaxVolSolution(CertificateSolution):
    """**MaxVol** — the max-*volume* invariant-set certificate.

    ``volume = sⁿˣ·√(det P)`` is the certified ellipsoid volume. ``s_feas`` is the
    MaxS feasibility ceiling that brackets the sweep, and ``unbounded_volume`` is
    ``True`` when the model is globally stable (the volume is unbounded and this
    is the MaxS fallback). ``sweep`` records the per-grid-point volumes.
    """

    volume: float = 0.0
    logdet_P: float = 0.0
    max_eig_F: float = 0.0
    locality_min_eigs: List[float] = field(default_factory=list)
    s_feas: Optional[float] = None
    unbounded_volume: bool = False
    sweep: List[VolumePoint] = field(default_factory=list)


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
class CoverageRatio:
    """One evaluation of ``rho = vol(MaxVol)/vol(tightest coverage)`` at a given
    ``C2`` scale factor ``f`` — the search state of the C2 calibration.

    ``rho`` encodes the degenerate ends so a sign test on ``rho - 1`` brackets the
    sweet spot: ``+∞`` while globally stable (grow ``f``), ``0`` when ``y_max`` is
    uncertifiable (shrink ``f``), finite otherwise.
    """

    f: float
    rho: float
    feasible: bool
    max_vol: Optional[MaxVolSolution] = None
    cov_sol: Optional[CoverageSolution] = None
    cov_volume: Optional[float] = None


@dataclass
class InitializationReport:
    """Flat, log/mlflow-friendly summary of the certificate established at init.

    Aggregates the operative MaxVol certificate diagnostics and (when it ran) the
    C2 calibration. :meth:`to_metrics` yields the finite scalar metrics for mlflow
    under the ``initialization/`` namespace.
    """

    volume: float
    s: float
    s_feas: Optional[float]
    norm_H: float
    max_eig_F: float
    unbounded_volume: bool
    constraints_satisfied: bool
    y_bar: Optional[float] = None
    y_max: Optional[float] = None
    coverage_ok: Optional[bool] = None
    calibrated: bool = False
    c2_factor: Optional[float] = None
    rho: Optional[float] = None
    rho_in_band: Optional[bool] = None
    calibration_iterations: Optional[int] = None
    cov_volume: Optional[float] = None

    def to_metrics(self) -> dict:
        """The report as ``{metric_name: float}`` — finite scalars only (bools
        become 0/1; ``None`` and non-finite values such as an unbounded ``rho``
        are dropped so mlflow never sees them)."""
        raw = {
            "volume": self.volume,
            "s": self.s,
            "s_feas": self.s_feas,
            "norm_H": self.norm_H,
            "max_eig_F": self.max_eig_F,
            "unbounded_volume": self.unbounded_volume,
            "constraints_satisfied": self.constraints_satisfied,
            "y_bar": self.y_bar,
            "y_max": self.y_max,
            "coverage_ok": self.coverage_ok,
            "calibrated": self.calibrated,
            "c2_factor": self.c2_factor,
            "rho": self.rho,
            "rho_in_band": self.rho_in_band,
            "calibration_iterations": self.calibration_iterations,
            "cov_volume": self.cov_volume,
        }
        metrics = {}
        for name, value in raw.items():
            if value is None:
                continue
            num = float(int(value) if isinstance(value, bool) else value)
            if np.isfinite(num):
                metrics[name] = num
        return metrics


@dataclass
class CalibrationResult:
    """Result of scaling ``C2`` so the max-volume set just covers the coverage set.

    ``f`` is the winning multiplicative factor on the base ``C2``; ``rho`` is
    ``vol(MaxVol)/vol(tightest coverage)`` at ``f`` (driven toward 1 from above);
    ``in_band`` is ``0 ≤ rho - 1 < eps``.
    """

    f: float
    rho: float
    in_band: bool
    max_vol: MaxVolSolution
    cov_sol: Optional[CoverageSolution]
    cov_volume: Optional[float]
    iterations: int
