"""Certificate synthesis SDPs for the Lure system.

All the certificate optimization that used to live inline in
``sysid.models.constrained_rnn`` is collected here. :class:`LureCertificateSynthesizer`
holds the (fixed) prediction dynamics as numpy arrays and exposes one method per
SDP, each returning a typed result from :mod:`sysid.optimization.solutions`:

* :meth:`max_s`          — MaxS: the max-feasible-``s`` feasibility ceiling.
* :meth:`bootstrap`      — MaxS with ``D21`` (and optionally ``B``) free, for init.
* :meth:`project_theta`  — nearest certifiable theta at a FIXED certificate.
* :meth:`coverage_at_s`  — the fixed-``s`` binding-coverage SDP.
* :meth:`coverage_sweep` — the tightest coverage over an s-grid.
* :meth:`feasibility`    — certificate repair at fixed ``s``, or with ``s`` free.

Every solve is **pure**: it reads the synthesizer's arrays and returns a solution;
nothing here mutates a model. The model keeps thin wrappers that build a
synthesizer from its current parameters and write applied solutions back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import cvxpy as cp
import numpy as np

from .solutions import (
    BootstrapSolution,
    CertificateSolution,
    CoveragePoint,
    CoverageSolution,
    CoverageSweepResult,
    MaxSSolution,
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

    def _build_F(self, P, L, M, A=None, B=None, B2=None, C2=None, D21=None):
        """The shared stability LMI matrix ``F`` (same for every certificate SDP).

        Every argument may be a numpy array or a cvxpy expression; the theta
        arguments default to this synthesizer's fixed dynamics.

        ``F`` is **bilinear** in (certificate, theta) — it contains products
        ``P·C2ᵀ``, ``P·Aᵀ`` and ``M·B2ᵀ`` — so exactly one side may vary at a time:

        * :meth:`max_s` / :meth:`coverage_at_s` / :meth:`feasibility` vary the
          certificate (P, L, M) at fixed theta;
        * :meth:`bootstrap` additionally frees B and D21, which appear only in the
          blocks that carry no P or M, so it stays affine;
        * :meth:`project_theta` varies theta at a FIXED certificate, which makes
          every block affine in (A, B, B2, C2, D21).

        Passing cvxpy variables on both sides at once would make the constraint
        non-convex; nothing here does that.
        """
        A = self.A if A is None else A
        B = self.B if B is None else B
        B2 = self.B2 if B2 is None else B2
        C2 = self.C2 if C2 is None else C2
        D21 = self.D21 if D21 is None else D21
        return cp.bmat(
            [
                [-(self.alpha ** 2) * P, np.zeros((self.nx, self.nd)), P @ C2.T + L.T, P @ A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P + L, D21, -2 * M, M @ B2.T],
                [A @ P, B, B2 @ M, -P],
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

    # --------------------------------------------------------------- Bootstrap
    def bootstrap(
        self, learn_B: bool = False, learn_D21: bool = True
    ) -> Optional[BootstrapSolution]:
        """MaxS with ``D21`` (and optionally ``B``) as additional free variables.

        The initialization-only sibling of :meth:`max_s`: same stability +
        locality LMIs and the same ``min 1/s²`` objective, but the input maps are
        solved for instead of held fixed. ``_init_identity`` draws ``D21`` from
        ``N(0, std²)`` with no reference to the data, so the draw usually lands
        outside the feasible set; letting the SDP choose ``D21`` shrinks it and
        lifts the certifiable ``s`` to the scale the data needs, while A, B, C, D,
        D12, B2 and C2 keep exactly the values the identity init gave them.

        ``learn_B`` stays ``False`` by default on purpose: with ``B`` free too the
        SDP drives both ``B`` and ``D21`` to zero — trivially certifiable, but a
        dead model (``e_hat == 0``).

        Returns ``None`` if the solver fails or the problem is infeasible.

        Note this deliberately does NOT impose ``m >= 0`` the way :meth:`max_s`
        does; the bootstrap has always run as a pure feasibility/scale solve over
        ``(P, la, L, s)`` plus the free input maps, and the trained models were
        established with that feasible set.
        """
        eps = self.eps
        P = cp.Variable((self.nx, self.nx), symmetric=True)
        la = cp.Variable((self.nz, 1))
        M = cp.diag(la)
        B = cp.Variable(self.B.shape) if learn_B else self.B
        D21 = cp.Variable(self.D21.shape) if learn_D21 else self.D21

        if self.learn_L:
            L = cp.Variable((self.nz, self.nx))
            S_hat = cp.Variable((1, 1))
        else:
            L = self.L_fixed
            S_hat = None

        F = self._build_F(P, L, M, B=B, D21=D21)
        nF = F.shape[0]
        constraints = [F << -eps * np.eye(nF)]

        # Locality LMIs only in the regional (learn_L) regime — with L fixed at 0
        # the scale is frozen and the constraint is vacuous, matching max_s.
        if self.learn_L:
            for i in range(self.nz):
                li = L[i, :].reshape((1, -1), order="C")
                constraints.append(
                    cp.bmat([[S_hat, li], [li.T, P]]) >> eps * np.eye(self.nx + 1)
                )

        problem = cp.Problem(cp.Minimize(S_hat if self.learn_L else 0), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.error(f"bootstrap SDP solver failed: {e}")
            return None
        if problem.status != "optimal":
            logger.error(f"bootstrap SDP failed with status: {problem.status}")
            return None

        if self.learn_L:
            S_hat_opt = float(np.asarray(S_hat.value).reshape(-1)[0])
            if S_hat_opt <= 0:
                logger.error(f"bootstrap SDP returned non-positive S_hat ({S_hat_opt})")
                return None
            s_star = float(np.sqrt(1.0 / S_hat_opt))
            L_val = L.value
        else:
            s_star = self.s_fixed
            L_val = L

        return BootstrapSolution(
            P=P.value,
            L=L_val,
            M=M.value,
            s=s_star,
            B=B.value if learn_B else None,
            D21=D21.value if learn_D21 else None,
        )

    # --------------------------------------------------------------- Projection
    #: theta blocks that appear in the stability LMI, i.e. the ones a projection
    #: can constrain. C, D and D12 are absent from F — they are output maps and the
    #: certificate says nothing about them — so they are never projected.
    THETA_IN_F = ("A", "B", "B2", "C2", "D21")

    def max_eig_F(self, P, L, M, **theta) -> float:
        """``max eig(F)`` for a given certificate and theta — the stability margin.

        ``< 0`` means the pair satisfies the stability LMI. Keyword arguments
        override individual theta blocks (see :meth:`_build_F`); anything omitted
        comes from this synthesizer. Pure numpy, no solver.
        """
        F = self._build_F(P, L, M, **theta)
        return float(np.max(np.real(np.linalg.eigvals(np.asarray(F.value)))))

    def project_theta(
        self,
        target: dict,
        P: np.ndarray,
        L: np.ndarray,
        M: np.ndarray,
        free: Optional[Sequence[str]] = None,
    ) -> Optional[dict]:
        """Closest theta to ``target`` that the FIXED certificate ``(P, L, M)`` certifies.

        Solves

            min  sum_i ||X_i - target_i||_F^2 / ||target_i||_F^2    over ``free``
            s.t. F(theta; P, L, M) << -eps I

        The per-block normalization makes the objective RELATIVE, so blocks whose
        entries differ by orders of magnitude are moved by comparable fractions
        rather than the smallest-magnitude one absorbing the whole correction.

        With the certificate held fixed every block of ``F`` is affine in
        (A, B, B2, C2, D21) — see :meth:`_build_F` — so this is a convex QP over an
        LMI, i.e. an SDP with a quadratic objective. It is a genuine Euclidean
        projection onto the certified set, so a ``target`` that is already
        certifiable comes back unchanged.

        Why this and not "perturb, then re-solve the certificate": re-solving moves
        the certificate to fit whatever theta the noise produced, and can simply
        fail when the draw lands outside the certifiable set. Projecting instead
        keeps the certificate — the one from the UNDISTURBED reference — and moves
        theta the smallest distance that makes it valid. The result is feasible by
        construction, deterministic, and still the nearest point to the draw.

        Args:
            target: ``{name: array}`` for every block in :attr:`THETA_IN_F`. Blocks
                not in ``free`` are held at these values as constants, so a
                structurally-fixed parameter must be passed at its fixed value.
            P, L, M: the fixed certificate, e.g. from :meth:`max_s` at the
                undisturbed theta. ``s`` is not needed: it enters only the locality
                LMIs, which involve no theta and are therefore unaffected.
            free: which blocks are optimization variables. Defaults to all of
                :attr:`THETA_IN_F`.

        Returns ``{name: array}`` for the free blocks, or ``None`` if the solver
        fails (which here means the certificate admits no theta at all, not that
        the draw was unlucky).
        """
        free = tuple(self.THETA_IN_F if free is None else free)
        unknown = [n for n in free if n not in self.THETA_IN_F]
        if unknown:
            raise ValueError(
                f"project_theta: {unknown} are not in the stability LMI; only "
                f"{list(self.THETA_IN_F)} can be projected (C/D/D12 do not appear in F)."
            )
        missing = [n for n in self.THETA_IN_F if n not in target]
        if missing:
            raise ValueError(f"project_theta: target is missing {missing}.")

        shapes = {
            "A": (self.nx, self.nx), "B": (self.nx, self.nd),
            "B2": (self.nx, self.nz), "C2": (self.nz, self.nx),
            "D21": (self.nz, self.nd),
        }
        blocks, objective_terms = {}, []
        for name in self.THETA_IN_F:
            tgt = np.asarray(target[name], dtype=float)
            if tgt.shape != shapes[name]:
                raise ValueError(
                    f"project_theta: target['{name}'] has shape {tgt.shape}, "
                    f"expected {shapes[name]}."
                )
            if name in free:
                X = cp.Variable(shapes[name])
                blocks[name] = X
                # Weight each block by its own scale so the objective is RELATIVE.
                # Unweighted, ``sum ||X_i - target_i||_F^2`` is measured in absolute
                # units, so the cheapest way to satisfy the LMI is to move whichever
                # block has the smallest entries — on the Duffing reference that is
                # B (entries ~5e-3, vs C2 ~13), which ends up displaced by ~12%
                # while everything else moves ~0.1%. Dividing by ||target_i||_F
                # makes the projection scale-invariant and puts it in the same
                # relative units the perturbation itself uses.
                scale = float(np.linalg.norm(tgt))
                objective_terms.append(
                    cp.sum_squares(X - tgt) / (scale ** 2) if scale > 0
                    else cp.sum_squares(X - tgt)
                )
            else:
                blocks[name] = tgt

        F = self._build_F(P, L, M, **blocks)
        nF = F.shape[0]
        problem = cp.Problem(
            cp.Minimize(cp.sum(objective_terms)),
            [F << -self.eps * np.eye(nF)],
        )
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.error(f"theta projection SDP solver failed: {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            logger.error(f"theta projection SDP failed with status: {problem.status}")
            return None

        out = {n: np.asarray(blocks[n].value, dtype=float) for n in free}
        moved = float(np.sqrt(sum(
            np.linalg.norm(out[n] - np.asarray(target[n], dtype=float)) ** 2 for n in free
        )))
        scale = float(np.sqrt(sum(
            np.linalg.norm(np.asarray(target[n], dtype=float)) ** 2 for n in free
        )))
        logger.debug(
            f"theta projection: moved ||dtheta|| = {moved:.4g} "
            f"({100 * moved / max(scale, 1e-12):.4f}% of ||theta||)"
        )
        return out

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
        half-width. ``None`` when no grid point is feasible.

        ``s_min`` should be the **input floor** ``max_{k,i}‖u_k^{(i)}‖``: below it the
        input condition ``‖u_k‖² ≤ s² − α²x_kᵀP⁻¹x_k`` cannot hold for any ``P``
        (the quadratic form is ``≥ 0``), so a tighter ``ȳ`` found there belongs to a
        certificate that does not admit its own training inputs."""
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
    def feasibility(self, s: Optional[float]) -> Optional[CertificateSolution]:
        """Certificate repair with θ and α held fixed: find P ≻ 0, M ⪰ 0, L (and
        optionally ``s``) satisfying the stability + locality LMIs, with a
        well-conditioning ``min t`` (‖P‖ ≤ t, ‖M‖ ≤ t). ``None`` if infeasible /
        the solver fails.

        ``s`` given pins the scale, so the locality LMIs carry the constant
        ``1/s²`` and only (P, L, M) move — the smallest repair, and the one that
        leaves ``s`` where the gradient and the barrier put it.

        ``s=None`` frees the scale. The substitution ``ŝ = 1/s²`` keeps both LMIs
        *jointly linear* in (P, L, M, ŝ), so this is still one convex solve; the
        recovered ``s = 1/√ŝ``. Use it only when the fixed-``s`` repair is
        infeasible, since a free ``s`` discards what the barrier had learned."""
        eps = self.eps
        free_s = s is None
        P = cp.Variable((self.nx, self.nx), symmetric=True)
        L = cp.Variable((self.nz, self.nx)) if self.learn_L else self.L_fixed
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)
        if free_s:
            S_hat = cp.Variable((1, 1))
            s_block = S_hat
        else:
            s_block = np.array([[1.0 / float(s) ** 2]])

        F = self._build_F(P, L, M)
        nF = F.shape[0]
        constraints = [F << -eps * np.eye(nF), m >= 0]
        if free_s:
            constraints.append(S_hat >> eps)
        for i in range(self.nz):
            li = L[i, :].reshape((1, -1), order="C")
            constraints.append(
                cp.bmat([[s_block, li], [li.T, P]]) >> eps * np.eye(self.nx + 1)
            )

        t = cp.Variable((1, 1))
        constraints += [cp.norm(P) <= t, cp.norm(M) <= t]
        # problem = cp.Problem(cp.Minimize(t), constraints)
        problem = cp.Problem(cp.Minimize([None]), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.debug(f"feasibility SDP failed at s={'free' if free_s else f'{s:.4f}'}: {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None
        if free_s:
            s_opt = float(S_hat.value[0, 0])
            if s_opt <= 0:
                return None
            s_out = float(np.sqrt(1.0 / s_opt))
        else:
            s_out = float(s)
        return CertificateSolution(
            P=P.value,
            L=L.value if self.learn_L else L,
            M=M.value,
            s=s_out,
        )

