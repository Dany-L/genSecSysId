"""Certificate-synthesis optimization for the Lure system.

:class:`LureCertificateSynthesizer` collects every certificate SDP (MaxS, MaxVol,
coverage, feasibility, C2 calibration); the solves return the typed results in
:mod:`sysid.optimization.solutions` instead of loose dicts.
"""

from .solutions import (
    CalibrationResult,
    CertificateSolution,
    CoveragePoint,
    CoverageRatio,
    CoverageSolution,
    CoverageSweepResult,
    InitializationReport,
    MaxSSolution,
    MaxVolSolution,
    TightCertSolution,
    VolumePoint,
)
from .synthesizer import LureCertificateSynthesizer

__all__ = [
    "LureCertificateSynthesizer",
    "CertificateSolution",
    "MaxSSolution",
    "MaxVolSolution",
    "VolumePoint",
    "CoverageSolution",
    "CoveragePoint",
    "CoverageSweepResult",
    "CoverageRatio",
    "CalibrationResult",
    "InitializationReport",
    "TightCertSolution",
]
