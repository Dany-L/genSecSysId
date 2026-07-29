"""Certificate synthesis SDPs for the Lure system.

All the certificate optimization that used to live inline in
``sysid.models.constrained_rnn`` is collected here. :class:`LureCertificateSynthesizer`
holds the (fixed) prediction dynamics as numpy arrays and exposes one method per
SDP, each returning a typed result from :mod:`sysid.optimization.solutions`:

* :meth:`max_s`          — MaxS: the max-feasible-``s`` feasibility ceiling.
* :meth:`max_vol_at_s`   — the convex fixed-``s`` slice (max-det) of MaxVol.
* :meth:`max_vol`        — MaxVol: the max-*volume* invariant-set certificate.
* :meth:`coverage_at_s`  — the fixed-``s`` binding-coverage SDP.
* :meth:`coverage_sweep` — the tightest coverage over an s-grid.
* :meth:`feasibility`    — fixed-``s`` certificate repair (optional hard coverage floor).
* :meth:`tight_cert`     — TightCert: the ρ-pinned re-synthesis solve (``ŝ`` free).
* :meth:`calibrate_c2`   — scale ``C2`` so MaxVol just covers the coverage set.

Every solve is **pure**: it reads the synthesizer's arrays and returns a solution;
nothing here mutates a model. The model keeps thin wrappers that build a
synthesizer from its current parameters and write applied solutions back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

import cvxpy as cp
import numpy as np

from sysid.utils import get_volume_of_ellipsoid

from .solutions import (
    CalibrationResult,
    CertificateSolution,
    CoveragePoint,
    CoverageRatio,
    CoverageSolution,
    CoverageSweepResult,
    MaxSSolution,
    MaxVolSolution,
    TightCertSolution,
    VolumePoint,
)

logger = logging.getLogger(__name__)

EPS = 1e-6


@dataclass
class LureCertificateSynthesizer:
    """Fixed prediction dynamics (θ) for the certificate SDPs, in normalized units.

    Build one with :meth:`from_model`; scale ``C2`` with :meth:`with_c2`. ``L_fixed``
    is the fixed coupling when ``learn_L`` is False (globally stable, L = 0), and
    ``s_fixed`` is the frozen scale in that regime.
    """

    A: np.ndarray
    B: np.ndarray
    B2: np.ndarray
    C: np.ndarray
    C2: np.ndarray
    D21: np.ndarray
    alpha: float
    nx: int
    nz: int
    nd: int
    ne: int
    learn_L: bool
    L_fixed: Optional[np.ndarray]
    s_fixed: float
    output_std: float
    P_current: np.ndarray
    eps: float = EPS

    # ------------------------------------------------------------------ build
    @classmethod
    def from_model(cls, model) -> "LureCertificateSynthesizer":
        """Extract the current (fixed) dynamics from a ``SimpleLure``-like model."""
        def to_np(t):
            return t.cpu().detach().numpy()

        alpha = float(1.0 / (1.0 + np.exp(-to_np(model.tau))))
        return cls(
            A=to_np(model.A),
            B=to_np(model.B),
            B2=to_np(model.B2),
            C=to_np(model.C),
            C2=to_np(model.C2),
            D21=to_np(model.D21),
            alpha=alpha,
            nx=int(model.nx),
            nz=int(model.nz),
            nd=int(model.nd),
            ne=int(model.ne),
            learn_L=bool(model.learn_L),
            L_fixed=None if model.learn_L else to_np(model.L),
            s_fixed=float(to_np(model.s)),
            output_std=float(np.asarray(to_np(model.output_std)).reshape(-1)[0]),
            P_current=to_np(model.P),
        )

    def with_c2(self, C2_new: np.ndarray) -> "LureCertificateSynthesizer":
        """A copy of this synthesizer with ``C2`` replaced (pure — for the sweep)."""
        return replace(self, C2=np.asarray(C2_new))

    def _build_F(self, P, L, M):
        """The shared stability LMI matrix ``F`` (same for every certificate SDP).

        Works with cvxpy variables or numpy arrays for ``P``, ``L``, ``M``.
        """
        return cp.bmat(
            [
                [-(self.alpha ** 2) * P, np.zeros((self.nx, self.nd)), P @ self.C2.T + L.T, P @ self.A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), self.D21.T, self.B.T],
                [self.C2 @ P + L, self.D21, -2 * M, M @ self.B2.T],
                [self.A @ P, self.B, self.B2 @ M, -P],
            ]
        )

    # -------------------------------------------------------------------- MaxS
    def max_s(self, gamma: float = 0.0) -> Optional[MaxSSolution]:
        """MaxS — maximize ``s`` (minimize ``S_hat = 1/s²``) subject to stability +
        locality LMIs. With fixed ``L = 0`` (not ``learn_L``) ``s`` is frozen and
        the locality LMIs are dropped.

        ``gamma > 0`` adds a ``-γ·log det P`` pull to the objective, keeping ``P``
        (and its inverse ``P⁻¹`` in ``H = LP⁻¹`` / ``V = xᵀP⁻¹x``) off zero and
        pulling ``s`` down off the locality ``1/s² = ε`` ceiling — i.e. it slides
        the operative certificate off the large-``s``/small-``P`` extreme toward a
        balanced, better-conditioned point on the ``s↔P`` gauge. ``γ = 0`` (default)
        is pure MaxS, so the feasibility-ceiling semantics are preserved.

        Returns ``None`` if the solver fails."""
        eps = self.eps
        P = cp.Variable((self.nx, self.nx), symmetric=True)
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)
        if self.learn_L:
            L = cp.Variable((self.nz, self.nx))
            S_hat = cp.Variable((1, 1))
        else:
            L = self.L_fixed
            S_hat = None

        F = self._build_F(P, L, M)
        nF = F.shape[0]
        constraints = [F << -eps * np.eye(nF), m >= 0]

        Gs = []
        if self.learn_L:
            for i in range(self.nz):
                li = L[i, :].reshape((1, -1), order="C")
                locality_lmi = cp.bmat([[S_hat, li], [li.T, P]])
                Gs.append(locality_lmi)
                constraints.append(locality_lmi >> eps * np.eye(self.nx + 1))

        base_obj = S_hat if self.learn_L else 0
        obj_expr = base_obj - gamma * cp.log_det(P) if gamma > 0 else base_obj
        objective = cp.Minimize(obj_expr)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.error(f"max-s SDP solver failed: {e}")
            return None
        if problem.status != "optimal":
            logger.error(f"max-s SDP failed with status: {problem.status}")
            return None

        if self.learn_L:
            assert S_hat.value is not None
            S_hat_opt = (
                S_hat.value[0, 0] if hasattr(S_hat.value, "__len__") else float(S_hat.value)
            )
            if S_hat_opt <= 0:
                logger.error(f"max-s SDP returned non-positive S_hat ({S_hat_opt})")
                return None
            s_star = float(np.sqrt(1.0 / S_hat_opt))
            L_val = L.value
        else:
            s_star = self.s_fixed
            L_val = L

        min_eig_diff = float(np.min(np.real(np.linalg.eigvals(self.P_current - P.value))))
        # DEBUG: max_s is called many times inside the C2 calibration bisection;
        # the calibration itself emits the INFO-level progress.
        logger.debug(
            f"max-s SDP solved: s = {s_star:.2f}, "
            f"min eig(P_current - P_opt) = {min_eig_diff:.2e}"
        )
        return MaxSSolution(
            P=P.value,
            L=L_val,
            M=M.value,
            s=s_star,
            max_eig_F=float(np.max(np.real(np.linalg.eigvals(F.value)))),
            locality_min_eigs=[
                float(np.min(np.real(np.linalg.eigvals(g.value)))) for g in Gs
            ],
        )

    # ------------------------------------------------------------------ MaxVol
    def max_vol_at_s(self, s: float) -> Optional[MaxVolSolution]:
        """The convex fixed-``s`` slice of MaxVol: ``max log det P`` subject to the
        stability + (constant-``1/s²``) locality LMIs. Returns ``None`` if
        infeasible / unbounded (globally-stable regime) / the solver fails."""
        eps = self.eps
        s2 = float(s) ** 2
        P = cp.Variable((self.nx, self.nx), symmetric=True)
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)
        L = cp.Variable((self.nz, self.nx)) if self.learn_L else self.L_fixed

        F = self._build_F(P, L, M)
        nF = F.shape[0]
        constraints = [F << -eps * np.eye(nF), m >= 0]

        Gs = []
        if self.learn_L:
            for i in range(self.nz):
                li = L[i, :].reshape((1, -1), order="C")
                locality_lmi = cp.bmat([[np.array([[1.0 / s2]]), li], [li.T, P]])
                Gs.append(locality_lmi)
                constraints.append(locality_lmi >> eps * np.eye(self.nx + 1))

        problem = cp.Problem(cp.Maximize(cp.log_det(P)), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.debug(f"max-vol SDP failed at s={s:.4f}: {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None

        P_val = P.value
        return MaxVolSolution(
            P=P_val,
            L=L.value if self.learn_L else L,
            M=M.value,
            s=float(s),
            volume=float(get_volume_of_ellipsoid(P_val, float(s))),
            logdet_P=float(np.linalg.slogdet(P_val)[1]),
            max_eig_F=float(np.max(np.real(np.linalg.eigvals(F.value)))),
            locality_min_eigs=[
                float(np.min(np.real(np.linalg.eigvals(g.value)))) for g in Gs
            ],
        )

    def max_vol(self, n_grid: int = 25) -> Optional[MaxVolSolution]:
        """MaxVol — the largest-*volume* invariant-set certificate. Sweeps
        ``s ∈ (0, s_feas]`` (``s_feas`` = the MaxS ceiling), keeps the largest
        ``sⁿˣ·√(det P)``. Falls back to the MaxS certificate (``unbounded_volume``)
        in the globally-stable regime where the volume is unbounded. ``None`` if θ
        is infeasible."""
        ceil_sol = self.max_s()
        if ceil_sol is None:
            return None
        s_feas = float(ceil_sol.s)

        s_grid = (
            np.linspace(s_feas / n_grid, s_feas, int(n_grid))
            if self.learn_L else np.array([s_feas])
        )
        sweep = []
        best = None
        for s in s_grid:
            sol = self.max_vol_at_s(float(s))
            if sol is None:
                continue
            sweep.append(VolumePoint(s=sol.s, volume=sol.volume, logdet_P=sol.logdet_P))
            if best is None or sol.volume > best.volume:
                best = sol

        if best is not None:
            best.s_feas = s_feas
            best.unbounded_volume = False
            best.sweep = sweep
            logger.debug(
                f"max-vol SDP solved: volume = {best.volume:.3e} at s = {best.s:.4f} "
                f"(feasibility ceiling s_max = {s_feas:.4f})"
            )
            return best

        # No finite-volume point: globally stable -> volume unbounded. Fall back to
        # the MaxS feasibility-ceiling certificate (largest feasible s).
        logger.warning(
            "max-vol SDP: invariant-set volume is unbounded (globally-stable "
            f"regime); falling back to the MaxS feasibility ceiling at s={s_feas:.4f}."
        )
        return MaxVolSolution(
            P=ceil_sol.P,
            L=ceil_sol.L,
            M=ceil_sol.M,
            s=ceil_sol.s,
            volume=float(get_volume_of_ellipsoid(ceil_sol.P, s_feas)),
            logdet_P=float(np.linalg.slogdet(ceil_sol.P)[1]),
            max_eig_F=ceil_sol.max_eig_F,
            locality_min_eigs=ceil_sol.locality_min_eigs,
            s_feas=s_feas,
            unbounded_volume=True,
            sweep=[],
        )

    # ---------------------------------------------------------------- Coverage
    def coverage_at_s(self, s: float, y_max: float) -> Optional[CoverageSolution]:
        """Fixed-``s`` binding-coverage SDP: minimize ``tr((σs)²·CPCᵀ)`` subject to
        the stability + locality LMIs and the physical coverage floor
        ``(σs)²·CPCᵀ ⪰ y_max²·I``. Returns ``None`` if infeasible / solver fails."""
        eps = self.eps
        s2 = float(s) ** 2
        sigma = self.output_std

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        L = cp.Variable((self.nz, self.nx)) if self.learn_L else self.L_fixed
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)

        F = self._build_F(P, L, M)
        nF = F.shape[0]
        constraints = [F << -eps * np.eye(nF)]

        if self.learn_L:
            for i in range(self.nz):
                li = L[i, :].reshape((1, -1), order="C")
                constraints.append(
                    cp.bmat([[np.array([[1.0 / s2]]), li], [li.T, P]])
                    >> eps * np.eye(self.nx + 1)
                )

        constraints.append(
            (sigma ** 2) * s2 * self.C @ P @ self.C.T - float(y_max) ** 2 * np.eye(self.ne)
            >> eps * np.eye(self.ne)
        )

        problem = cp.Problem(
            cp.Minimize(cp.trace((sigma ** 2) * s2 * self.C @ P @ self.C.T)), constraints
        )
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.debug(f"coverage SDP failed at s={s:.4f}: {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None

        P_val = P.value
        CPCt = self.C @ P_val @ self.C.T
        y_bar = float(sigma * s * np.sqrt(CPCt.item())) if self.ne == 1 else None
        return CoverageSolution(
            P=P_val,
            L=L.value if self.learn_L else L,
            M=M.value,
            s=float(s),
            y_bar=y_bar,
        )

    def coverage_sweep(
        self, y_max: float, n_grid: int = 20, s_min: float = 1.0, s_max: float = 100.0
    ) -> Optional[CoverageSweepResult]:
        """Tightest output coverage ``ȳ_f`` over ``s ∈ [s_min, s_max]``: the
        feasible fixed-``s`` solution with the smallest physical certified
        half-width. ``None`` when no grid point is feasible."""
        s_grid = np.linspace(float(s_min), float(s_max), int(n_grid))
        sweep = [sol for sol in (self.coverage_at_s(float(s), y_max) for s in s_grid) if sol]
        if not sweep:
            return None
        ybars = [c for c in sweep if c.y_bar is not None]
        tight = min(ybars, key=lambda c: c.y_bar) if ybars else None
        return CoverageSweepResult(
            y_f=tight.y_bar if tight is not None else None,
            s_f=tight.s if tight is not None else None,
            sol=tight,
            sweep=[CoveragePoint(s=c.s, y_bar=c.y_bar) for c in sweep],
        )

    # ------------------------------------------------------------- Feasibility
    def feasibility(
        self, s: float, y_max: Optional[float] = None
    ) -> Optional[CertificateSolution]:
        """Fixed-``s`` certificate repair: find P ≻ 0, M ⪰ 0, L satisfying the
        stability + (constant-``1/s²``) locality LMIs, with a well-conditioning
        ``min t`` (‖P‖ ≤ t, ‖M‖ ≤ t). ``None`` if infeasible / solver fails.

        ``y_max`` (physical) adds the **hard coverage floor**
        ``(σ·s)²·C P Cᵀ ⪰ y_max²·I`` so a repair can no longer buy feasibility by
        shrinking the certified output image. At fixed ``s`` this is just another
        LMI in ``P``. ``None`` (default) keeps the historical, floor-free repair —
        the trainer uses it as the second tier when the floor makes the repair
        infeasible."""
        eps = self.eps
        s_hat = 1.0 / float(s) ** 2
        P = cp.Variable((self.nx, self.nx), symmetric=True)
        L = cp.Variable((self.nz, self.nx)) if self.learn_L else self.L_fixed
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)

        F = self._build_F(P, L, M)
        nF = F.shape[0]
        constraints = [F << -eps * np.eye(nF), m >= 0]
        for i in range(self.nz):
            li = L[i, :].reshape((1, -1), order="C")
            constraints.append(
                cp.bmat([[np.array([[s_hat]]), li], [li.T, P]]) >> eps * np.eye(self.nx + 1)
            )

        if y_max is not None and float(y_max) > 0:
            constraints.append(
                (self.output_std ** 2) / s_hat * self.C @ P @ self.C.T
                - float(y_max) ** 2 * np.eye(self.ne)
                >> eps * np.eye(self.ne)
            )

        t = cp.Variable((1, 1))
        constraints += [cp.norm(P) <= t, cp.norm(M) <= t]
        problem = cp.Problem(cp.Minimize(t), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.debug(f"feasibility SDP failed at s={s:.4f}: {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None
        return CertificateSolution(
            P=P.value,
            L=L.value if self.learn_L else L,
            M=M.value,
            s=float(s),
        )

    # ---------------------------------------------------------- TightCert (ρ≈1)
    def tight_cert(
        self, y_max: Optional[float] = None, beta: Optional[float] = 2.0
    ) -> Optional[TightCertSolution]:
        """**TightCert** — re-synthesize the certificate with ``ρ`` pinned into
        ``[1, βⁿˣ]``. The per-epoch solve of the certificate-re-synthesis scheme.

        The key structural fact: substituting ``ŝ = 1/s²`` makes *every* constraint
        jointly linear in ``(P, L, M, ŝ)`` — the stability LMI ``F`` contains no
        ``s`` at all, locality is ``[[ŝ, lᵢ], [lᵢᵀ, P]] ⪰ 0``, and both coverage
        sides come from multiplying through by ``ŝ > 0``::

            (C↓)  (σ·s)²·C P Cᵀ ⪰ y_max²·I    ⟺   σ²·C P Cᵀ ⪰ y_max²·ŝ·I
            (C↑)  (σ·s)²·C P Cᵀ ⪯ β²y_max²·I  ⟺   σ²·C P Cᵀ ⪯ β²y_max²·ŝ·I

        so this is ONE SDP — no s-grid, no bisection. (The *irreducible*
        bilinearity of the coverage problem is ``ŝ·Ỹ`` with a **variable** output
        level ``Ỹ``; here the level is the constant ``y_max``, so it does not
        appear. Likewise the volume objective ``log det P − nx·log ŝ`` would be
        non-convex, which is why the objective below is conditioning, not volume.)

        Objective ``min t`` s.t. ``‖P‖ ≤ t, ‖M‖ ≤ t``: with (C↓) binding,
        shrinking ``‖P‖`` *forces* ``ŝ`` down, i.e. ``s`` **up** — the counter-push
        the ``-log det`` barrier lacks. ``ŝ`` cannot run away because the
        non-homogeneous disturbance blocks of ``F`` bound ``P`` from below.

        Degenerate cases:

        * ``y_max is None`` (or ``β`` unset, or ``ne`` mismatch) — the band drops
          and ``min t`` no longer pins the scale at all, so this falls back to
          :meth:`max_s` (``min ŝ``), the operative certificate today. The returned
          solution has ``band_enforced=False``.
        * ``not learn_L`` — ``s`` is frozen and locality is dropped; ``ŝ`` is held
          at ``1/s_fixed²`` and only ``P, M`` move under the band.

        Note for ``ne > 1``: the two-sided band constrains *every* output
        direction, which forces a near-isotropic certified image. Intended for
        ``ne == 1``; pass ``beta=None`` for a floor-only solve otherwise.
        """
        if y_max is None or float(y_max) <= 0:
            sol = self.max_s()
            if sol is None:
                return None
            return TightCertSolution(
                P=sol.P, L=sol.L, M=sol.M, s=sol.s,
                y_bar=None, rho=None, beta=None, band_enforced=False,
                max_eig_F=sol.max_eig_F,
                norm_P=float(np.linalg.norm(sol.P, ord=2)),
            )

        eps = self.eps
        y_max = float(y_max)
        sigma = self.output_std

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)
        L = cp.Variable((self.nz, self.nx)) if self.learn_L else self.L_fixed
        # ŝ = 1/s² is a decision variable exactly when s is (i.e. with learn_L).
        S_hat = (
            cp.Variable(nonneg=True) if self.learn_L
            else 1.0 / float(self.s_fixed) ** 2
        )

        F = self._build_F(P, L, M)
        nF = F.shape[0]
        constraints = [F << -eps * np.eye(nF), m >= 0]

        if self.learn_L:
            S_block = cp.reshape(S_hat, (1, 1), order="C")
            for i in range(self.nz):
                li = L[i, :].reshape((1, -1), order="C")
                constraints.append(
                    cp.bmat([[S_block, li], [li.T, P]]) >> eps * np.eye(self.nx + 1)
                )

        I_e = np.eye(self.ne)
        CPCt = (sigma ** 2) * self.C @ P @ self.C.T
        constraints.append(CPCt - (y_max ** 2) * S_hat * I_e >> eps * I_e)  # (C↓)
        if beta is not None and float(beta) > 0:
            constraints.append(
                (float(beta) ** 2) * (y_max ** 2) * S_hat * I_e - CPCt >> 0
            )  # (C↑)

        t = cp.Variable((1, 1))
        constraints += [cp.norm(P) <= t, cp.norm(M) <= t]
        problem = cp.Problem(cp.Minimize(t), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.debug(f"TightCert SDP failed (y_max={y_max:.4g}, beta={beta}): {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            logger.debug(
                f"TightCert SDP status={problem.status} (y_max={y_max:.4g}, beta={beta})"
            )
            return None

        P_val = P.value
        s_star = (
            float(1.0 / np.sqrt(float(S_hat.value))) if self.learn_L else float(self.s_fixed)
        )
        CPCt_val = self.C @ P_val @ self.C.T
        lam_min = max(float(np.min(np.linalg.eigvalsh(CPCt_val))), 0.0)
        y_bar = float(sigma * s_star * np.sqrt(lam_min))
        rho = float((y_bar / y_max) ** self.nx)
        logger.debug(
            f"TightCert solved: s={s_star:.4g}, ȳ={y_bar:.4g} (y_max={y_max:.4g}), "
            f"rho={rho:.4g}, ‖P‖={np.linalg.norm(P_val, ord=2):.4g}"
        )
        return TightCertSolution(
            P=P_val,
            L=L.value if self.learn_L else L,
            M=M.value,
            s=s_star,
            y_bar=y_bar,
            rho=rho,
            beta=float(beta) if beta is not None else None,
            band_enforced=True,
            max_eig_F=float(np.max(np.real(np.linalg.eigvals(F.value)))),
            norm_P=float(np.linalg.norm(P_val, ord=2)),
        )

    # ------------------------------------------------------- C2 calibration
    def coverage_ratio_at_c2(self, f: float, y_max: float) -> CoverageRatio:
        """Evaluate ``rho = (ȳ_MaxS / y_max)ⁿˣ`` with ``C2`` scaled by ``f`` (pure —
        via :meth:`with_c2`), where ``ȳ_MaxS = σ·s·√(λ_min(C P Cᵀ))`` is the MaxS
        certificate's own certified output half-width (worst output direction).

        This is exactly ``vol(𝒳_MaxS)/vol(𝒳c)``, with ``𝒳c`` the *minimal covering
        set of the MaxS shape* — the MaxS ellipsoid scaled down (``s' = s·y_max/ȳ``)
        until it just reaches ``y_max`` — since same-shape scaling gives
        ``vol ∝ sⁿˣ``. It needs only :meth:`max_s` (no coverage sweep), is smooth
        and monotone decreasing in ``f``, and ``rho ≥ 1 ⇔ the operative MaxS set
        covers y_max`` (``rho < 1`` ⇔ even the max-s set cannot reach it). So a sign
        test on ``rho - 1`` brackets the tightest covering ``C2``."""
        synth = self.with_c2(self.C2 * float(f))
        cert = synth.max_s()
        if cert is None:
            return CoverageRatio(f=f, rho=0.0, feasible=False)
        cert_vol = float(get_volume_of_ellipsoid(cert.P, cert.s))
        CPCt = synth.C @ cert.P @ synth.C.T
        lam_min = max(float(np.min(np.linalg.eigvalsh(CPCt))), 0.0)
        y_bar = float(synth.output_std * cert.s * np.sqrt(lam_min))  # worst direction
        rho = float((y_bar / y_max) ** synth.nx) if y_max > 0 else float("inf")
        cov_vol = float(cert_vol / rho) if rho > 0 else None
        return CoverageRatio(
            f=f, rho=rho, feasible=True, cert=cert, cert_volume=cert_vol,
            cov_sol=None, cov_volume=cov_vol,
        )

    def calibrate_c2(
        self,
        y_max: float,
        eps: float = 0.05,
        max_iter: int = 30,
        f_min: float = 1e-3,
        f_max: float = 1e3,
    ) -> Optional[CalibrationResult]:
        """Find a scalar factor on ``C2`` with ``0 ≤ rho - 1 < eps`` by geometric
        bisection of the root of ``rho - 1``, where ``rho = (ȳ_MaxS/y_max)ⁿˣ``
        (:meth:`coverage_ratio_at_c2`) is smooth and monotone decreasing in the
        factor. Returns the largest factor that still covers (``rho ≥ 1``),
        converged so ``rho - 1 < eps`` when reachable — the tightest coupling for
        which the operative MaxS set still covers ``y_max``. Pure — the caller
        applies the winning ``C2`` factor and certificate. ``None`` if even the
        base C2 admits no certificate."""
        memo = {}

        def state(f: float) -> CoverageRatio:
            key = float(f)
            if key not in memo:
                memo[key] = self.coverage_ratio_at_c2(key, y_max)
            return memo[key]

        base = state(1.0)
        if base.cert is None and not base.feasible:
            logger.warning("C2 calibration: base C2 admits no certificate; skipping.")
            return None

        # Bracket the root of rho - 1: lo keeps rho >= 1 (covers), hi has rho < 1.
        lo, hi = 1.0, 1.0
        while state(hi).rho >= 1.0 and hi < f_max:
            hi = min(hi * 2.0, f_max)
        while state(lo).rho < 1.0 and lo > f_min:
            lo = max(lo / 2.0, f_min)

        def in_band(st: CoverageRatio) -> bool:
            return np.isfinite(st.rho) and 0.0 <= st.rho - 1.0 < eps

        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            if in_band(state(lo)):
                break
            if hi / lo < 1.0 + 1e-6:  # bracket collapsed
                break
            mid = float(np.sqrt(lo * hi))  # geometric midpoint (f spans decades)
            if state(mid).rho >= 1.0:
                lo = mid
            else:
                hi = mid

        best = state(lo)
        best_in_band = in_band(best)
        if not best_in_band:
            logger.warning(
                f"C2 calibration: could not land rho in [1, 1+{eps}); "
                f"kept closest covering factor f={best.f:.4g} (rho={best.rho})."
            )
        return CalibrationResult(
            f=best.f,
            rho=best.rho,
            in_band=best_in_band,
            cert=best.cert,
            cert_volume=best.cert_volume,
            cov_sol=best.cov_sol,
            cov_volume=best.cov_volume,
            iterations=iterations,
        )
