"""Certificate-synthesis optimization for the Lure system.

:class:`LureCertificateSynthesizer` collects every certificate SDP (MaxS, the
initialization bootstrap, coverage, feasibility); the solves return the typed
results in :mod:`sysid.optimization.solutions` instead of loose dicts.
:mod:`sysid.optimization.output_set` turns a certificate into the output
ellipsoid it certifies, which is where every reported ``ȳ`` comes from.
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
from .output_set import OutputEllipsoid, output_ellipsoid
from .synthesizer import LureCertificateSynthesizer

__all__ = [
    "LureCertificateSynthesizer",
    "OutputEllipsoid",
    "output_ellipsoid",
    "CertificateSolution",
    "MaxSSolution",
    "BootstrapSolution",
    "CoverageSolution",
    "CoveragePoint",
    "CoverageSweepResult",
    "InitializationReport",
]
