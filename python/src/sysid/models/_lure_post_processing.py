"""Post-processing certificate synthesis for SimpleLure.

Split out of ``constrained_rnn.py`` as a mixin, alongside
:mod:`~sysid.models._lure_initialization` and
:mod:`~sysid.models._lure_regularization`, to keep that module focused on the
model/dynamics. This is what runs **after** training, on a fixed θ: the
prediction dynamics (A, B, B2, C, C2, D21, α) are frozen and only the
certificate (P, L, Λ, s) is (re)solved and written back.

* :meth:`post_process` — solve both certificate problems over the trained θ
  (MaxS, and the tightest-coverage sweep) and report them separately; the MaxS
  certificate is the one applied.
* :meth:`solve_output_coverage_certificate` — **MinTrProb**, the binding
  Corollary-1 certificate: the tightest certified output interval that also
  leaves zero input violations on the supplied data.

Every method uses ``self`` and relies on SimpleLure's params/methods (``_synth``,
``_apply_certificate_solution``, ``check_constraints``, ``forward`` /
``forward_unfiltered``, ``get_regularization_input``, ``P, L, la, s, C, y_max,
output_std, ne, nx``) — provided at runtime via the MRO.
"""

import logging
from typing import Optional

import numpy as np
import torch

from sysid.utils import get_volume_of_ellipsoid

logger = logging.getLogger(__name__)


class LurePostProcessingMixin:
    def post_process(
        self,
        y_max: Optional[float] = None,
        n_grid: int = 20,
        s_min: float = 1.0,
        s_max: float = 100.0,
    ) -> dict:
        """Post-process a trained model: solve the two certificate SDPs, report
        them **separately**, and set the model to the *largest invariant set*.

        Everything that shapes the predictions (θ = A, B, B2, C, C2, D21, α) is
        held fixed; only the certificate (P, L, Λ, s) is (re)computed. Two clearly
        separated optimization problems are solved over that fixed θ:

        **Problem 1 — max-feasible-s certificate** (MaxS, :meth:`~sysid.optimization.LureCertificateSynthesizer.max_s`).
        Maximizes the scale ``s`` (minimizes ``1/s²``) subject to the stability +
        locality LMIs only — the largest *regional* certifiable invariant set. It
        is well conditioned (a moderate ``s``, not the tiny-``s``/huge-``P`` corner
        the volume objective falls into) and is the operative certificate written
        back into the model. It does **not** constrain output coverage — the
        coverage floor ``(σ·s)²·C P Cᵀ ⪰ y_max²·I`` is *checked afterwards*
        (``coverage_ok``). Reported: the ellipsoid ``volume`` (``sⁿˣ·√(det P)``),
        the coupling norm ``‖H‖ = ‖L P⁻¹‖``, the scale ``s`` and the certified
        output half-width ``ȳ_c = σ·s·√(C P Cᵀ)`` (physical; ``ne == 1`` only).

        **Problem 2 — tightest coverage** (MinTrProb, :meth:`~sysid.optimization.LureCertificateSynthesizer.coverage_at_s` swept
        over a finite s-grid). The joint problem is bilinear (convex once ``s`` is
        fixed), so it is gridded over ``s ∈ [s_min, s_max]``; reported is the
        smallest feasible certified half-width ``ȳ_f`` — the tightest coverage of
        the demanded ``y_max``. Skipped when ``y_max`` is unset.

        The MaxS solution (ȳ_c) is the one written back into the model; ``ρ`` (the
        volume ratio ``vol(𝒳_MaxS)/vol(𝒳_cov)``) is reported as a tightness
        diagnostic. ``y_max`` is physical; ``None`` falls back to the model's
        stored physical level (which may be unset, in which case Problem 2 and the
        coverage check are skipped).

        Returns a summary dict::

            {
              "success": bool,
              "s_opt": float,                 # operative (MaxS) s == max_s["s"]
              "constraints_satisfied": bool,
              "y_max": Optional[float],       # physical demanded level (or None)
              "max_s": {s, volume, norm_H, y_bar(=ȳ_c), max_eig_F, coverage_ok, rho},
              "coverage":{y_bar(=ȳ_f), s, reason, s_min, s_max, n_grid, sweep},
            }
        """
        # Resolve the (physical) demanded output level.
        if y_max is None and not bool(torch.isnan(self.y_max)):
            y_max = float(self.y_max)
        y_max = float(y_max) if y_max is not None else None

        C = self.C.cpu().detach().numpy()
        sigma = float(self.output_std)

        def _fmt(v):
            return "n/a" if v is None else f"{v:.4f}"

        logger.info("=" * 80)
        logger.info(
            "POST-PROCESSING: (1) max-feasible-s certificate + (2) tightest-coverage sweep"
        )
        logger.info("=" * 80)

        # ------------------------------------------------------------------
        # Problem 1 — MaxS: the largest regional certifiable invariant set
        # (operative, well conditioned — moderate s, not the tiny-s/huge-P corner).
        # ------------------------------------------------------------------
        synth = self._synth()
        max_s_sol = synth.max_s()
        if max_s_sol is None:
            return {"success": False, "status": "max_s_sdp_failed"}

        P_c, L_c, s_c = max_s_sol.P, max_s_sol.L, max_s_sol.s
        vol_c = float(get_volume_of_ellipsoid(P_c, s_c))
        norm_H_c = float(np.linalg.norm(L_c @ np.linalg.inv(P_c), ord=2))
        if self.ne == 1:
            CPCt_c = max(float((C @ P_c @ C.T).item()), 0.0)
            y_c = float(sigma * s_c * np.sqrt(CPCt_c))
            coverage_ok = (
                bool((sigma * s_c) ** 2 * CPCt_c >= y_max ** 2)
                if y_max is not None else None
            )
        else:
            y_c = None
            coverage_ok = None

        logger.info("[Problem 1: MaxS — largest regional invariant set]")
        logger.info(f"  volume   = {vol_c:.3e}")
        logger.info(f"  s        = {_fmt(s_c)}")
        logger.info(f"  ‖H‖      = {_fmt(norm_H_c)}   (H = L P⁻¹)")
        logger.info(f"  ȳ_c      = {_fmt(y_c)}   (σ·s·√(C P Cᵀ))")
        logger.info(f"  max λ(F) = {max_s_sol.max_eig_F:.3e}")
        if coverage_ok is not None:
            logger.info(
                f"  coverage ((σ·s)²·CPCᵀ ≥ y_max²={y_max ** 2:.4g}): "
                f"{'OK' if coverage_ok else 'NOT met'}"
            )

        # ------------------------------------------------------------------
        # Problem 2 — tightest coverage over the s-grid (reported, not applied).
        # ------------------------------------------------------------------
        y_f = s_f = None
        rho = None
        coverage_reason = "y_max_unset"
        coverage_sweep: list = []
        if y_max is not None:
            cov = synth.coverage_sweep(y_max, n_grid=n_grid, s_min=s_min, s_max=s_max)
            if cov is None:
                coverage_reason = "coverage_unreachable"
                logger.warning(
                    f"[Problem 2: coverage] y_max={y_max:.4g} unreachable on the "
                    f"grid s∈[{s_min:g}, {s_max:g}] — this θ cannot certify it."
                )
            else:
                y_f, s_f = cov.y_f, cov.s_f
                coverage_sweep = [{"s": p.s, "y_bar": p.y_bar} for p in cov.sweep]
                coverage_reason = "ok"
                # Tightness ratio of the OPERATIVE (MaxS) certificate: how much its
                # own certified image over-covers y_max, as a volume ratio
                # rho = (ȳ_c/y_max)^nx = vol(𝒳_MaxS)/vol(minimal covering set).
                if y_c is not None and y_max > 0:
                    rho = float((y_c / y_max) ** self.nx)
                logger.info("[Problem 2: coverage — tightest ȳ over the s-grid]")
                logger.info(
                    f"  ȳ_f = {_fmt(y_f)}  at s = {_fmt(s_f)}   "
                    f"(target y_max = {_fmt(y_max)}); ρ = (ȳ_c/y_max)^nx = {_fmt(rho)}"
                )
        else:
            logger.info("[Problem 2: coverage] skipped (y_max unset)")

        # ------------------------------------------------------------------
        # Set the model to the largest regional invariant set (MaxS / ȳ_c).
        # This is the TRUE certificate for the trained theta: the largest set these
        # parameters certify. A *tight* certificate is trivially available (shrink s
        # until ȳ meets y_max) and therefore says nothing about theta, so it is the
        # ceiling — not the tight one — that belongs in the final report.
        # ------------------------------------------------------------------
        self._apply_certificate_solution(max_s_sol)
        constraints_ok = self.check_constraints()
        logger.info(
            f"Applied MaxS certificate to the model. Constraints satisfied: {constraints_ok}"
        )
        logger.info("=" * 80)

        return {
            "success": True,
            "s_opt": s_c,
            "constraints_satisfied": constraints_ok,
            "y_max": y_max,
            "max_s": {
                "s": s_c,
                "volume": vol_c,
                "norm_H": norm_H_c,
                "y_bar": y_c,
                "max_eig_F": float(max_s_sol.max_eig_F),
                "coverage_ok": coverage_ok,
                "rho": rho,
            },
            "coverage": {
                "y_bar": y_f,
                "s": s_f,
                "reason": coverage_reason,
                "s_min": float(s_min),
                "s_max": float(s_max),
                "n_grid": int(n_grid),
                "sweep": coverage_sweep,
            },
        }

    def _count_input_violations(
        self, inputs: torch.Tensor, x0: Optional[torch.Tensor], warmup_steps: int
    ) -> int:
        """Number of trajectories whose *unfiltered* rollout breaches the input
        constraint (any ``c_k > 0``) under the current certificate.

        The certificate is a claim about the raw model on admissible inputs, so
        the check is on the unfiltered dynamics even for ``SimpleLureSafe`` (its
        filter would otherwise hide every violation by construction)."""
        with torch.no_grad():
            if hasattr(self, "forward_unfiltered"):
                _, (x, _), u_applied = self.forward_unfiltered(inputs, x0)
            else:
                _, (x, _), u_applied = self.forward(inputs, x0, warmup_steps=warmup_steps)
            _, c = self.get_regularization_input(
                u_applied, x, return_c=True, warmup_steps=warmup_steps
            )
            viol = (torch.nan_to_num(c, nan=float("-inf")) > 0).any(dim=1)
        return int(viol.sum())

    def solve_output_coverage_certificate(
        self,
        y_max: Optional[float] = None,
        inputs: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        warmup_steps: int = 0,
        n_grid: int = 10,
        s_min: float = 1.0,
        s_max: float = 100.0,
    ) -> dict:
        """**MinTrProb** — the binding-Corollary-1 certificate (see the wiki
        ``binding-output-certificate``).

        Produces the tightest certified output interval ``[-ȳ, ȳ]`` with the
        PHYSICAL ``ȳ = y_max`` that (a) satisfies the Lyapunov + regionality
        LMIs and (b) — when ``inputs`` are supplied — leaves **zero input
        violations** on that data. The scale ``s`` is the lone nonconvex degree
        of freedom, so we solve a convex SDP (:meth:`~sysid.optimization.LureCertificateSynthesizer.coverage_at_s`) at each
        point of a fixed grid ``s ∈ [s_min, s_max]`` (default ``[0.1, 20]``; a
        deliberately simple preset — no MaxS bracket / bisection for now) and
        keep the feasible ones. Among the ``s`` with zero input violations we
        pick the tightest ``ȳ`` (the SDP objective already drives ``ȳ`` to
        ``y_max``); if none clears the violations, we take the fewest-violations
        ``s``. If no grid ``s`` is feasible at all, this θ cannot certify
        ``y_max`` — reported, not hidden.

        ``y_max`` is physical; ``None`` uses the model's stored physical
        ``y_max``. The selected certificate is written back into the model.
        Returns a summary dict with ``success``, ``reason``, ``s``, ``y_bar``
        (physical, the operative certificate), ``y_max`` (physical), ``s_min``,
        ``s_max``, ``n_input_violations`` and the full ``sweep``. It also reports
        the feasibility ceiling + diagnostics (all physical, ne=1 only; ``None``
        for ne>1):

        - ``y_feas`` / ``s_feas`` / ``norm_H_feas``: the MaxS feasibility ceiling
          — fix θ and maximize s (:meth:`~sysid.optimization.LureCertificateSynthesizer.max_s`), giving the largest feasible
          certificate ``ȳ = σ·s_feas·√(C P* C*ᵀ)`` (grid-independent) and its
          coupling norm ``‖H*‖ = ‖L* P*⁻¹‖``. **A large ``s_feas`` with a small
          ``norm_H_feas`` is a strong indication of a globally stable model** (the
          certificate needs no locality restriction). This is the honest
          feasibility ceiling and the global-stability diagnostic.
        - ``y_tight`` / ``s_tight``: the smallest ȳ over ALL feasible grid s
          ignoring input violations — the tight-branch value (≈ ``y_max``). The
          ``y_tight → y_bar`` gap is the conservatism the input constraint forces.
        """
        if y_max is None:
            if self.y_max is None or bool(torch.isnan(self.y_max)):
                return {"success": False, "reason": "y_max_unset"}
            y_max = float(self.y_max)
        y_max = float(y_max)

        # Save the current certificate so a failed search leaves the model as-is.
        saved = {
            "P": self.P.detach().clone(),
            "L": self.L.detach().clone(),
            "la": self.la.detach().clone(),
            "s": self.s.detach().clone(),
        }

        def _restore():
            with torch.no_grad():
                self.P.data.copy_(saved["P"])
                self.L.data.copy_(saved["L"])
                self.la.data.copy_(saved["la"])
                self.s.data.copy_(saved["s"])

        # Fixed-grid sweep over the preset band [s_min, s_max]. Infeasible s
        # (too small for coverage, or too large for regionality) are skipped.
        # The synthesizer is a snapshot of θ, so the mid-loop _apply (used only to
        # count input violations) does not perturb the remaining solves.
        synth = self._synth()
        s_grid = np.linspace(float(s_min), float(s_max), int(n_grid))
        sweep = []
        for s in s_grid:
            sol = synth.coverage_at_s(float(s), y_max)
            if sol is None:
                continue
            n_viol = None
            if inputs is not None:
                self._apply_certificate_solution(sol)
                n_viol = self._count_input_violations(inputs, x0, warmup_steps)
            sweep.append({"s": sol.s, "y_bar": sol.y_bar, "n_violations": n_viol, "sol": sol})

        if not sweep:
            _restore()
            return {
                "success": False,
                "reason": "coverage_unreachable",
                "s_min": float(s_min),
                "s_max": float(s_max),
                "y_max": y_max,
            }

        # ``eligible`` is the band the operative certificate is drawn from: the
        # zero-input-violation grid points (or, if none clear the violations, the
        # fewest-violations ones — matching the old fallback selection). When no
        # inputs are given there is no violation filter, so the whole sweep is
        # eligible.
        if inputs is not None:
            zero_viol = [c for c in sweep if c["n_violations"] == 0]
            violation_free = bool(zero_viol)
            if zero_viol:
                eligible = zero_viol
            else:
                min_viol = min(c["n_violations"] for c in sweep)
                eligible = [c for c in sweep if c["n_violations"] == min_viol]
        else:
            violation_free = None
            eligible = sweep

        # Operative certificate (point 2): always the LARGEST invariant set — the
        # largest-s eligible certificate (the largest certifiable safe region),
        # not the tightest ȳ. When feasibility runs to the top of the grid this is
        # the s_max solution. The tightest coverage value (≈ y_max) is still
        # reported as y_tight, and the grid-independent ceiling as the MaxS y_feas.
        best = max(eligible, key=lambda c: c["s"])

        # Feasibility ceiling via MaxS (point 1): fix θ and maximize s
        # (synth.max_s, pure). This is the grid-independent max-feasible certificate;
        # its output half-width is y_feas = σ·s_feas·√(C P* C*ᵀ). A large s_feas
        # together with a small ‖H*‖ = ‖L* P*⁻¹‖ is a strong indication of a
        # globally stable model — the certificate needs no locality restriction.
        # y_tight is the tight-branch value (smallest ȳ over ALL feasible grid s,
        # ignoring input violations, ≈ y_max); the y_tight→y_bar gap is the
        # conservatism the input constraint forces. All are ne=1 only (None else).
        all_ybars = [c for c in sweep if c["y_bar"] is not None]
        tight = min(all_ybars, key=lambda c: c["y_bar"]) if all_ybars else None
        y_tight = tight["y_bar"] if tight is not None else None
        s_tight = tight["s"] if tight is not None else None

        y_feas = s_feas = norm_H_feas = None
        ceil_sol = synth.max_s()  # pure; depends only on the (fixed) θ
        if ceil_sol is not None and self.ne == 1:
            P_c = ceil_sol.P
            C_np = self.C.cpu().detach().numpy()
            sigma = float(self.output_std)
            CPCt_c = float((C_np @ P_c @ C_np.T).item())
            s_feas = float(ceil_sol.s)
            y_feas = float(sigma * s_feas * np.sqrt(CPCt_c))
            H_c = ceil_sol.L @ np.linalg.inv(P_c)
            norm_H_feas = float(np.linalg.norm(H_c, ord=2))

        self._apply_certificate_solution(best["sol"])
        constraints_ok = self.check_constraints()

        return {
            "success": True,
            "reason": "ok" if (violation_free is not False) else "violations_remain",
            "s": best["s"],
            "y_bar": best["y_bar"],
            "y_max": y_max,
            "y_feas": y_feas,
            "s_feas": s_feas,
            "norm_H_feas": norm_H_feas,
            "y_tight": y_tight,
            "s_tight": s_tight,
            "s_min": float(s_min),
            "s_max": float(s_max),
            "n_input_violations": best["n_violations"],
            "violation_free": violation_free,
            "constraints_satisfied": constraints_ok,
            "sweep": [
                {"s": c["s"], "y_bar": c["y_bar"], "n_violations": c["n_violations"]}
                for c in sweep
            ],
        }
