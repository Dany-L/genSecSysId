"""Certificate-synthesis optimization for the Lure system.

:class:`LureCertificateSynthesizer` collects every certificate SDP (MaxS, the
initialization bootstrap, coverage, feasibility); the solves return the typed
results in :mod:`sysid.optimization.solutions` instead of loose dicts.
"""

from .solutions import (
    BootstrapSolution,
    CertificateSolution,
    CoveragePoint,
    CoverageSolution,
    CoverageSweepResult,
    InitializationReport,
    MaxSSolution,
)
from .synthesizer import LureCertificateSynthesizer

__all__ = [
    "LureCertificateSynthesizer",
    "CertificateSolution",
    "MaxSSolution",
    "BootstrapSolution",
    "CoverageSolution",
    "CoveragePoint",
    "CoverageSweepResult",
    "InitializationReport",
]
