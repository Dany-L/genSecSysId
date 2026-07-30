"""Parameter initialization + certificate calibration for SimpleLure.

Split out of ``constrained_rnn.py`` as a mixin. Every method uses ``self`` and
relies on SimpleLure's params/methods (``_synth``, ``forward``,
``_apply_certificate_solution``, ``get_regularization_input``,
``solve_output_coverage_certificate``, the structural-constraint helpers) —
provided at runtime via the MRO.
"""

import itertools
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sysid.optimization import InitializationReport
from sysid.utils import get_volume_of_ellipsoid, max_abs_output

from ..data import DataNormalizer

logger = logging.getLogger(__name__)


class LureInitializationMixin:
    def initialize_parameters(
        self,
        train_inputs,
        train_states,
        train_outputs,
        init_config=None,
        normalizer: Optional[DataNormalizer] = None,
    ):
        """Initialize model parameters with the **identity** strategy.

        Sets a stable diagonal A, identity-like C, and configurable random
        B2/C2/D21 (see :meth:`_init_identity`), then establishes the certificate
        (P, L, s) via MinTrProb (step 2 of the algorithm). ESN / N4SID inits were
        removed — ``identity`` is the only supported method.

        Args:
            train_inputs: Training input data (B, N, nd).
            train_states: Training state data (unused by the identity init; kept
                for API symmetry with the loaders).
            train_outputs: Training output data (B, N, ne) — used for y_max.
            init_config: InitializationConfig; ``method`` must be ``'identity'``.
            normalizer: Data normalizer used to scale C/B and derive y_max.

        Returns:
            :class:`~sysid.optimization.solutions.InitializationReport` — the
            established certificate diagnostics (and C2 calibration, when it ran);
            ``to_metrics()`` yields the ``initialization/`` mlflow metrics.
        """
        init_method = (
            getattr(init_config, "method", "identity").lower()
            if init_config is not None else "identity"
        )
        if init_method != "identity":
            raise ValueError(
                f"Unknown initialization method: {init_method!r}. Only 'identity' "
                "is supported (esn/n4sid were removed)."
            )

        logger.info("=" * 80)
        logger.info("INITIALIZATION: Using 'identity' method")
        logger.info("=" * 80)

        # Normalize inputs for the MinTrProb init below (initialize_s_from_conditions
        # expects normalized inputs).
        train_outputs_n = train_outputs
        if normalizer is not None:
            train_inputs = normalizer.transform_inputs(train_inputs)
            if hasattr(normalizer, "transform_outputs"):
                train_outputs_n = normalizer.transform_outputs(train_outputs)

        self._init_identity(normalizer)

        # Common post-initialization
        constraints_ok = self.check_constraints()
        logger.info(f"Initialization complete. Constraints satisfied: {constraints_ok}")
        logger.info("=" * 80)

        # Step 2 of the clean algorithm: establish the certificate (P, L, s) via
        # MinTrProb from the output + input conditions. This also guarantees
        # feasibility, so no separate analysis_problem_init bootstrap is needed
        # on this path. y_max is PHYSICAL (max |raw training output|); output_std
        # relates the model's normalized C/P/s to physical units.
        sigma = normalizer.output_std if normalizer is not None else 1.0
        sigma_scalar = float(np.asarray(sigma).reshape(-1)[0])
        y_max = max_abs_output(train_outputs) if normalizer is not None else None
        C = self.C.detach().cpu().numpy()

        # Calibrate the C2 std so that, under the operative MaxS certificate, BOTH
        # init conditions hold on the training data: (input) zero input-condition
        # violations on the rollout — the trajectory stays in the invariant set, so
        # predictions do not diverge — and (output) the certified image covers
        # y_max. C2 is the plant's nonlinear coupling: too large -> unstable rollout
        # -> exploding predictions (the earlier coverage-only failure); too small ->
        # near-global / over-conservative. The calibration takes the largest C2
        # (tightest region) that keeps both conditions. Requires the training
        # inputs, output_std + physical y_max, ne == 1, learn_L.
        self.set_output_coverage_level(y_max, sigma_scalar)
        calibrate = (
            bool(getattr(init_config, "calibrate_c2_for_coverage", True))
            if init_config is not None else True
        )
        cal = None
        if calibrate and self.learn_L and self.ne == 1 and y_max is not None:
            eps = float(getattr(init_config, "calibrate_c2_eps", 0.05)) if init_config is not None else 0.05
            max_iter = int(getattr(init_config, "calibrate_c2_max_iter", 30)) if init_config is not None else 30
            knobs = list(getattr(init_config, "calibrate_knobs", ["C2", "B2", "D21"])) if init_config is not None else ["C2", "B2", "D21"]
            # _calibrate_nonlinearity logs its own header/per-combo/summary progress
            # and leaves the knob params + certificate at the winning point.
            cal = self._calibrate_nonlinearity(
                train_inputs, y_max, knobs=knobs, eps=eps, max_iter=max_iter,
                rollout_trajectories=int(getattr(init_config, "calibrate_rollout_trajectories", 0)),
                rollout_steps=int(getattr(init_config, "calibrate_rollout_steps", 0)),
            )

        # Operative certificate: MaxS — the largest regional invariant set, well
        # conditioned (moderate s). From the calibration when it ran, else MaxS.
        cert_sol = cal["cert"] if cal is not None else self._synth().max_s()
        if cert_sol is None:
            # No (P, L, s) certifies regional (or, for learn_L=False, global)
            # stability, so training cannot start from a feasible point.
            raise RuntimeError(
                "Initialization failed: the MaxS certificate SDP found no feasible "
                "parameter set for the initialized dynamics. Check the identity "
                "initialization / structural constraints (e.g. A must be stable, "
                "alpha < 1)."
            )

        P_c, L_c, s_c = cert_sol.P, cert_sol.L, cert_sol.s
        volume_c = float(get_volume_of_ellipsoid(P_c, s_c))
        norm_H_c = float(np.linalg.norm(L_c @ np.linalg.inv(P_c), ord=2))
        if self.ne == 1:
            CPCt_c = max(float((C @ P_c @ C.T).item()), 0.0)
            y_c = float(sigma_scalar * s_c * np.sqrt(CPCt_c))
            coverage_ok = (
                bool((sigma_scalar * s_c) ** 2 * CPCt_c >= y_max ** 2)
                if y_max is not None else None
            )
        else:
            y_c = None
            coverage_ok = None

        self._apply_certificate_solution(cert_sol)
        constraints_ok = self.check_constraints()

        # Dead-zone activity of the final initialized model. Reported always (not
        # just when the calibration ran): firing_rate == 0 means the nonlinearity is
        # inert on the training data and, because Δ'(z) = 0 in the dead band, training
        # can never revive it — so this is the number that decides whether the model
        # class is usable at all.
        activity: Dict[str, float] = {}
        if train_inputs is not None:
            try:
                inp = torch.as_tensor(
                    np.asarray(train_inputs), dtype=self.C2.dtype, device=self.C2.device
                )
                if inp.dim() == 2:
                    inp = inp.unsqueeze(-1)
                activity = self.deadzone_activity(inp)
            except Exception as exc:  # diagnostics must never break initialization
                logger.debug(f"Dead-zone activity probe failed: {exc}")

        report = InitializationReport(
            volume=volume_c,
            s=float(s_c),
            norm_H=norm_H_c,
            max_eig_F=float(cert_sol.max_eig_F),
            constraints_satisfied=bool(constraints_ok),
            y_bar=y_c,
            y_max=float(y_max) if y_max is not None else None,
            coverage_ok=coverage_ok,
            calibrated=cal is not None,
            c2_factor=float(cal["factors"]["C2"]) if cal is not None else None,
            b2_factor=float(cal["factors"]["B2"]) if cal is not None else None,
            d21_factor=float(cal["factors"]["D21"]) if cal is not None else None,
            rho=float(cal["rho"]) if cal is not None else None,
            calibration_feasible=bool(cal["feasible"]) if cal is not None else None,
            calibration_iterations=int(cal["n_evals"]) if cal is not None else None,
            n_input_violations=(
                int(cal["n_input_violations"])
                if cal is not None and cal["n_input_violations"] is not None else None
            ),
            firing_rate=activity.get("firing_rate"),
            units_firing=activity.get("units_firing"),
            max_abs_z=activity.get("max_abs_z"),
        )
        logger.info(
            "INITIALIZATION certificate (MaxS): "
            f"volume={report.volume:.3e}, s={report.s:.4f}, ||H||_2={report.norm_H:.4f}, "
            f"y_c={y_c}, y_max={y_max}, coverage_ok={coverage_ok}, rho={report.rho}, "
            f"n_input_violations={report.n_input_violations}, "
            f"calibration_feasible={report.calibration_feasible}, "
            f"constraints_satisfied={constraints_ok}"
        )
        if report.firing_rate is not None:
            logger.info(
                f"INITIALIZATION dead-zone activity: firing_rate="
                f"{100 * report.firing_rate:.3f}% of (step, unit) pairs, "
                f"units_firing={100 * (report.units_firing or 0.0):.0f}%, "
                f"max|z|={report.max_abs_z:.3f} (dead band |z|<=1)"
            )
            if report.firing_rate <= 0.0:
                logger.warning(
                    "INITIALIZATION: the dead zone is INERT on the training data — the "
                    "model is LTI in this regime and, since Δ'(z)=0 inside the band, no "
                    "gradient reaches B2/C2/D21, so training cannot revive it. "
                    "Diagnostic only; the initialization does not optimize for firing."
                )
        self._last_init_report = report
        return report

    def _calibrate_nonlinearity(
        self,
        inputs_n: np.ndarray,
        y_max: float,
        knobs: Optional[List[str]] = None,
        eps: float = 0.05,
        max_iter: int = 30,
        grid: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
        f_min: float = 1e-3,
        f_max: float = 1e3,
        rollout_trajectories: int = 0,
        rollout_steps: int = 0,
    ) -> Optional[dict]:
        """Scale the nonlinearity-shaping maps ``{C2, B2, D21}`` so the operative
        MaxS certificate is the **tightest** one for which BOTH init conditions
        hold, evaluated on the training rollout:

        * **input** — zero input-condition violations
          (``‖u_k‖² ≤ s² − α²·x_kᵀP⁻¹x_k`` at every step,
          :meth:`get_regularization_input`) → the trajectory provably stays in the
          invariant set, so the open-loop predictions do **not** diverge;
        * **output** — the certified image covers the data:
          ``ȳ_MaxS = σ·s·√(λ_min(CPCᵀ)) ≥ y_max``.

        These maps trade off differently: ``C2`` (state→nonlinearity) is the main
        *tightness* knob (larger ``C2`` → smaller ``s`` → tighter, but eventually
        input-infeasible); ``D21`` (input→nonlinearity) sets how hard the input
        excites the nonlinearity, so smaller ``D21`` *relaxes* the input condition
        and lets ``C2`` grow further; ``B2`` (nonlinearity→state) sets the loop
        gain. The objective is ``min ρ = (ȳ_MaxS/y_max)ⁿˣ`` subject to feasibility.

        Search: for **every** combination of the ``{B2, D21}`` factors on a small
        log ``grid``, bisect the largest feasible ``C2`` (``ρ`` is monotone in
        ``C2``) and keep the globally tightest (min-``ρ``, feasible) point. Re-doing
        the C2 bisection *inside* each ``(B2, D21)`` combo is what captures the
        coupling — e.g. only after ``D21`` is lowered does the C2 bisection discover
        that a much larger ``C2`` (tighter) is now input-admissible. A pure
        coordinate descent that line-searches ``D21`` at a fixed small ``C2`` misses
        this and gets stuck. ``knobs=["C2"]`` reduces to the single-knob C2
        bisection (no grid).

        **Cost.** Each candidate is one ``max_s`` SDP (~200 ms) *plus* one full
        rollout of the training set, and the rollout is a Python loop over the
        sequence, so at 60×4000 it dominates (~700 ms, 76 % of the candidate).
        The ``rho`` path evaluates ~275 candidates (36 ``(B2, D21)`` combos, each
        with a C2 bisection) ⇒ ~4 min *per model*, which is what makes a sweep take
        hours. Three levers, in order of effect:

        * ``rollout_trajectories`` / ``rollout_steps`` subsample the calibration
          rollout (0 = use everything). This is a *scale* decision, so a handful of
          trajectories is ample; 5×1000 is ~5× cheaper than 60×4000. Caveat: the
          input-violation count and the firing rate are then estimated on the
          subsample, so keep enough trajectories to contain the input peaks.
        * a candidate that already fails the coverage gate is infeasible whatever
          the rollout says, so its rollout is **skipped** (free; big on the ``rho``
          path, where the bisection spends most evaluations below the coverage
          floor).
        * ``knobs=["C2"]`` reduces the search to a single ~7-evaluation bisection
          instead of 36 combos x a bisection each.

        Needs the (normalised) training inputs for the rollout. Mutates the knob
        params + certificate during the search and leaves them at the winning
        point. Returns a result dict (``factors`` per knob, ``rho``,
        ``firing_rate``, ``feasible``, ``cov_ok``, ``n_input_violations``,
        ``cert``, ``knobs``, ``n_evals``).
        """
        t_start = time.perf_counter()
        device, dtype = self.C2.device, self.C2.dtype
        ALL = ["C2", "B2", "D21"]
        requested = list(knobs) if knobs is not None else list(ALL)
        # canonical order, valid names, skip structurally-constrained / zero maps.
        bases = {k: getattr(self, k).detach().cpu().numpy().copy() for k in ALL}
        active = [
            k for k in ALL
            if k in requested
            and k not in self.structural_constraints
            and float(np.linalg.norm(bases[k])) > 0.0
        ]
        if "C2" not in active and float(np.linalg.norm(bases["C2"])) > 0.0 \
                and "C2" not in self.structural_constraints:
            active = ["C2"] + active  # C2 is the primary tightness knob
        non_c2 = [k for k in active if k != "C2"]

        C_np = self.C.detach().cpu().numpy()
        sigma = float(self.output_std)
        inputs = torch.as_tensor(np.asarray(inputs_n), dtype=dtype, device=device)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)

        # Subsample the calibration rollout — it is ~76% of every candidate's cost.
        n_traj_all, n_step_all = inputs.shape[0], inputs.shape[1]
        if rollout_trajectories and 0 < int(rollout_trajectories) < n_traj_all:
            # Evenly spaced, so the subset keeps the spread of the input amplitudes.
            idx = torch.linspace(
                0, n_traj_all - 1, int(rollout_trajectories), device=device
            ).round().long().unique()
            inputs = inputs[idx]
        if rollout_steps and 0 < int(rollout_steps) < n_step_all:
            inputs = inputs[:, : int(rollout_steps)]
        if inputs.shape[0] != n_traj_all or inputs.shape[1] != n_step_all:
            logger.info(
                f"  calibration rollout subsampled: {n_traj_all}x{n_step_all} -> "
                f"{inputs.shape[0]}x{inputs.shape[1]} "
                f"({(n_traj_all * n_step_all) / (inputs.shape[0] * inputs.shape[1]):.1f}x cheaper; "
                "the input-violation count and firing rate are estimated on this subset), "
                f"max|u_n| kept = {float(inputs.abs().max()):.4g} of {float(torch.as_tensor(np.asarray(inputs_n)).abs().max()):.4g}"
            )
        x0 = torch.zeros((inputs.shape[0], self.nx), dtype=dtype, device=device)
        logger.info(
            f"Nonlinearity calibration: knobs={active}, y_max={y_max:.4g}, "
            f"max|u_n|={float(inputs.abs().max()):.4g}, grid={tuple(float(g) for g in grid)} "
            "— minimizing rho s.t. 0 input violations + coverage."
        )

        memo: Dict[tuple, dict] = {}
        _EMPTY = {"rho": float("inf"), "firing_rate": 0.0,
                  "feasible": False, "cert": None, "cov_ok": False,
                  "n_input_violations": None, "factors": {k: 1.0 for k in ALL}}
        best = dict(_EMPTY)
        # Best-scoring candidate that merely HAS a certificate, ignoring the other
        # gates — a far better fallback than the smallest-C2 corner when nothing is
        # strictly feasible.
        best_any = dict(_EMPTY)

        def eval_at(factors: dict) -> dict:
            nonlocal best, best_any
            key = tuple(round(float(factors[k]), 12) for k in ALL)
            if key in memo:
                return memo[key]
            for k in ALL:
                getattr(self, k).data = torch.tensor(
                    bases[k] * float(factors[k]), device=device, dtype=dtype
                )
            mv = self._synth().max_s()
            if mv is None:
                memo[key] = {"feasible": False, "cert": None, "rho": float("inf"),
                             "firing_rate": 0.0,
                             "cov_ok": False, "n_input_violations": None,
                             "factors": dict(factors)}
                return memo[key]
            # Need P, s applied for the state-dependent input condition c_k.
            self._apply_certificate_solution(mv)
            CPCt = C_np @ mv.P @ C_np.T
            lam_min = max(float(np.min(np.linalg.eigvalsh(CPCt))), 0.0)
            y_bar = float(sigma * mv.s * np.sqrt(lam_min))            # worst output dir
            rho = float((y_bar / y_max) ** self.nx) if y_max > 0 else float("inf")
            cov_ok = bool(y_bar >= y_max)
            if not cov_ok:
                # Already infeasible on a criterion the SDP alone decides — the
                # rollout (the expensive part) cannot change that verdict. On the
                # rho path the bisection spends most of its evaluations here.
                r = {"feasible": False, "cert": mv, "rho": rho,
                     "firing_rate": 0.0, "cov_ok": False, "n_input_violations": None,
                     "factors": dict(factors), "rollout_skipped": True}
                memo[key] = r
                return r
            # Input condition over the FULL rollout (warmup 0): a divergent rollout
            # gives huge/±inf/nan c_k -> map those to a violation (unlike
            # _count_input_violations, which maps nan to -inf).
            with torch.no_grad():
                if hasattr(self, "forward_unfiltered"):
                    e_hat, (x, _), u_applied = self.forward_unfiltered(inputs, x0)
                else:
                    e_hat, (x, _), u_applied = self.forward(inputs, x0, warmup_steps=0)
                _, c = self.get_regularization_input(u_applied, x, return_c=True, warmup_steps=0)
                c = torch.nan_to_num(c, nan=1.0, posinf=1.0, neginf=-1.0)
                n_viol = int((c > 0).any(dim=1).sum())
                # Same rollout, two more numbers: does the dead zone fire, and how
                # well does this candidate predict? Both are free here.
                xs = x.squeeze(-1) if x.dim() == 4 else x
                n = min(xs.shape[1], u_applied.shape[1])
                z = xs[:, :n, :] @ self.C2.T + u_applied[:, :n, :] @ self.D21.T
                firing = float((z.abs() > 1.0).double().mean())
            feasible = bool(cov_ok and n_viol == 0)
            r = {"feasible": feasible, "cert": mv, "rho": rho,
                 "firing_rate": firing, "cov_ok": cov_ok,
                 "n_input_violations": n_viol, "factors": dict(factors)}
            memo[key] = r
            if feasible and r["rho"] < best["rho"]:
                best = dict(r)
            if r["rho"] < best_any["rho"]:
                best_any = dict(r)
            return r

        def bisect_c2(factors: dict) -> dict:
            """Largest feasible C2 factor holding the others (ρ↓ monotone in C2);
            returns the state at that C2. Logs each bisection step at DEBUG."""
            fac = dict(factors)

            def st(fc: float) -> dict:
                fac["C2"] = fc
                return eval_at(fac)

            hi = 1.0
            while st(hi)["feasible"] and hi < f_max:
                hi = min(hi * 2.0, f_max)
            if st(hi)["feasible"]:
                return st(hi)
            lo = hi / 2.0
            while not st(lo)["feasible"] and lo > f_min:
                lo = max(lo / 2.0, f_min)
            if not st(lo)["feasible"]:
                return st(lo)  # nothing feasible along C2 at these (B2, D21)
            for it in range(1, int(max_iter) + 1):
                if hi / lo < 1.0 + eps:
                    break
                mid = float(np.sqrt(lo * hi))
                sm = st(mid)
                logger.debug(
                    f"      C2 bisect it {it}: bracket [{lo:.3g}, {hi:.3g}] mid={mid:.3g} "
                    f"feasible={sm['feasible']} rho={sm['rho']:.4g} viol={sm['n_input_violations']}"
                )
                if sm["feasible"]:
                    lo = mid
                else:
                    hi = mid
            return st(lo)

        if "C2" in active:
            # Grid over the secondary knobs; re-bisect C2 inside each combo so the
            # D21↓→C2↑ coupling is captured. eval_at tracks the global best.
            combos = list(itertools.product(*([grid] * len(non_c2)))) if non_c2 else [()]
            for i_c, combo in enumerate(combos, 1):
                factors = {k: 1.0 for k in ALL}
                for k, g in zip(non_c2, combo):
                    factors[k] = float(g)
                res = bisect_c2(factors)
                combo_str = ", ".join(f"{k}×{factors[k]:.2g}" for k in non_c2) or "(C2 only)"
                s_str = f"{res['cert'].s:.3f}" if res.get("cert") is not None else "n/a"
                logger.info(
                    f"  [{i_c}/{len(combos)}] {combo_str}: C2×{res['factors']['C2']:.3g} "
                    f"-> s={s_str}, rho={res['rho']:.4g}, firing={100*res['firing_rate']:.3f}%, "
                    f"input_viol={res['n_input_violations']}, "
                    f"feasible={res['feasible']}  (running best rho={best['rho']:.4g})"
                )
        else:
            eval_at({k: 1.0 for k in ALL})  # no active knobs -> evaluate the base

        if best["cert"] is None:
            # Nothing satisfied every gate. Prefer the best-SCORING candidate that at
            # least admits a certificate over the smallest-C2 corner: that corner has
            # a huge s and a vacuous rho, and (with a divergent rollout) can even look
            # "firing" and "input-admissible" while being useless.
            if best_any["cert"] is not None:
                logger.warning(
                    "Nonlinearity calibration: no candidate satisfied every gate; "
                    "keeping the best-rho certified candidate instead "
                    f"(rho={best_any['rho']:.5g}, "
                    f"input_viol={best_any['n_input_violations']}, "
                    f"firing={100 * best_any['firing_rate']:.3f}%)."
                )
                best = dict(best_any)
        if best["cert"] is None:
            # Not even a certificate anywhere: fall back to the most-global attempt
            # (smallest C2 -> largest s -> most likely stable + covering), else base.
            logger.warning(
                "Nonlinearity calibration: no (stable + covering) factors found; "
                "falling back to the most-global certificate."
            )
            eval_at({"C2": f_min, "B2": 1.0, "D21": 1.0})
            if best["cert"] is None:
                for k in ALL:
                    getattr(self, k).data = torch.tensor(bases[k], device=device, dtype=dtype)
                mv = self._synth().max_s()
                best = {"feasible": False, "cert": mv, "rho": float("nan"),
                        "firing_rate": float("nan"),
                        "cov_ok": False, "n_input_violations": None,
                        "factors": {k: 1.0 for k in ALL}}

        # Apply the globally best factors + certificate.
        for k in ALL:
            getattr(self, k).data = torch.tensor(
                bases[k] * float(best["factors"][k]), device=device, dtype=dtype
            )
        if best["cert"] is not None:
            self._apply_certificate_solution(best["cert"])
        best["knobs"] = active
        best["n_evals"] = len(memo)
        n_skipped = sum(1 for v in memo.values() if v.get("rollout_skipped"))
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Nonlinearity calibration cost: {elapsed:.1f}s, {len(memo)} candidates "
            f"({n_skipped} rollouts skipped as already-infeasible), "
            f"rollout {inputs.shape[0]}x{inputs.shape[1]}"
        )
        facs = ", ".join(f"{k}×{best['factors'][k]:.3g}" for k in active) or "(none)"
        logger.info(
            f"Nonlinearity calibration DONE: [{facs}] "
            f"rho={best['rho']:.4f}, "
            f"firing={100 * best.get('firing_rate', float('nan')):.3f}%, "
            f"input_viol={best['n_input_violations']}, coverage_ok={best['cov_ok']}, "
            f"feasible={best['feasible']}, n_evals={best['n_evals']}"
        )
        if not best.get("firing_rate"):
            logger.warning(
                "Nonlinearity calibration: the dead zone NEVER fires on the training "
                "rollout — the model is LTI in this regime and, since Δ'(z)=0 inside "
                "the band, no gradient will reach B2/C2/D21. Reported as a diagnostic; "
                "the calibration does not optimize for it."
            )
        return best

    def _resolve_init_spec(
        self,
        name: str,
        shape: tuple,
        default_std: float,
    ) -> torch.Tensor:
        """
        Build initialization tensor for parameter `name` from identity_init config.

        Config (under custom_params['identity_init'][name]):
            {std: float}        -> Gaussian: std * randn(shape)
            {value: [[...]]}    -> Inline fixed start value
            {load_from: "*.npy"}-> Load from file (supports ~ expansion)
            missing             -> Gaussian with default_std
        """
        spec = self._identity_init_cfg.get(name, {}) or {}
        target = getattr(self, name)
        device, dtype = target.device, target.dtype

        if "load_from" in spec:
            path = Path(os.path.expanduser(str(spec["load_from"])))
            if not path.exists():
                raise FileNotFoundError(f"Init file for '{name}' not found: {path}")
            arr = np.load(path)
            tensor = torch.tensor(arr, device=device, dtype=dtype)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"Loaded '{name}' from {path} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(shape)}"
                )
            logger.info(f"  {name}: loaded from {path}")
            return tensor

        if "value" in spec:
            tensor = torch.tensor(spec["value"], device=device, dtype=dtype)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"Fixed init for '{name}' has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(shape)}"
                )
            logger.info(f"  {name}: fixed value from config")
            return tensor

        std = float(spec.get("std", default_std))
        logger.info(f"  {name}: random N(0, {std}^2)")
        return std * torch.randn(*shape, device=device, dtype=dtype)

    def _set_param_data(self, name: str, init_data: torch.Tensor):
        """Assign init_data to parameter `name`, respecting partial constraints."""
        if name in self.structural_constraints:
            self._apply_partial_initialization(name, init_data)
        else:
            getattr(self, name).data = init_data

    def _init_identity(self, normalizer: Optional[DataNormalizer] = None):
        """
        Identity initialization: stable Euler-discretized A, identity-like C,
        configurable random B2/C2/D21.

        Configurable via ``custom_params['identity_init']``. Each entry accepts:
            {std: float}             -> Gaussian random (B2, C2, D21)
            {scale: float}           -> Uniform random magnitude (A's last row only)
            {value: [[...]]}         -> Inline fixed start value
            {load_from: "*.npy"}     -> Load fixed start value from file

        Defaults reproduce the previous behavior (A_scale=1, B2_std=ts,
        C2_std=1, D21_std=1). Respects ``structural_constraints``: only
        learnable parts are touched.
        """
        logger.info("Identity initialization")
        cfg = self._identity_init_cfg

        # --- A: I + ts * A_ct, with last row of A_ct = -scale * U(0,1) ---
        if not self._should_skip_initialization('A'):
            A_spec = cfg.get('A', {}) or {}
            if 'value' in A_spec or 'load_from' in A_spec:
                A_init = self._resolve_init_spec('A', (self.nx, self.nx), default_std=0.0)
            else:
                A_scale = float(A_spec.get('scale', 1.0))
                device, dtype = self.A.device, self.A.dtype
                A_ct = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device, dtype=dtype)
                A_ct[1, :] = -A_scale * torch.rand((1, self.nx), device=device, dtype=dtype)
                A_init = torch.eye(self.nx, device=device, dtype=dtype) + A_ct * self.ts  # Euler discretization
                logger.info(f"  A: scale={A_scale}, |eig|={torch.linalg.eigvals(A_init).abs().tolist()}")
            self._set_param_data('A', A_init)

        # --- B: deterministic input_scale * ts * [0; 1] (override via value/load_from) ---
        if not self._should_skip_initialization('B'):
            B_spec = cfg.get('B', {}) or {}
            if 'value' in B_spec or 'load_from' in B_spec:
                B_init = self._resolve_init_spec('B', (self.nx, self.nd), default_std=0.0)
            else:
                input_scale = 1.0
                if normalizer is not None:
                    input_std = getattr(normalizer, 'input_std', None)
                    if input_std is not None:
                        input_scale = input_std.squeeze()
                B_init = input_scale * self.ts * torch.tensor(
                    [[0.0], [1.0]], device=self.B.device, dtype=self.B.dtype
                )
                # B_init = 0.01*self.ts * torch.tensor(
                #     [[0.0], [1.0]], device=self.B.device, dtype=self.B.dtype
                # )
            self._set_param_data('B', B_init)

        # --- B2, C2, D21: random (configurable std), or C2 by breakpoint placement ---
        if not self._should_skip_initialization('B2'):
            B2_init = self._resolve_init_spec('B2', (self.nx, self.nw), default_std=float(self.ts))
            self._set_param_data('B2', B2_init)

        if not self._should_skip_initialization('C2'):
            C2_init = self._resolve_init_spec('C2', (self.nz, self.nx), default_std=1.0)
            self._set_param_data('C2', C2_init)

        # --- C: identity-like / output_std (override via value/load_from) ---
        if not self._should_skip_initialization('C'):
            C_spec = cfg.get('C', {}) or {}
            if 'value' in C_spec or 'load_from' in C_spec:
                C_init = self._resolve_init_spec('C', (self.ne, self.nx), default_std=0.0)
            else:
                if normalizer is None or getattr(normalizer, 'output_std', None) is None:
                    raise ValueError(
                        "Identity initialization of 'C' requires a normalizer with "
                        "'output_std', or an explicit 'identity_init.C.value' / "
                        "'identity_init.C.load_from' override in custom_params."
                    )
                assert normalizer is not None and normalizer.output_std is not None
                C_init = (1.0 / normalizer.output_std.squeeze()) * torch.tensor(
                    [[1.0, 0.0]], device=self.C.device, dtype=self.C.dtype
                )
                # C_init = 0.01 * torch.tensor(
                #     [[1.0, 0.0]], device=self.C.device, dtype=self.C.dtype
                # )
            self._set_param_data('C', C_init)

        # --- D, D12: zero direct feedthrough ---
        if not self._should_skip_initialization('D'):
            self.D.data = torch.zeros_like(self.D)
        if not self._should_skip_initialization('D12'):
            self.D12.data = torch.zeros_like(self.D12)

        if not self._should_skip_initialization('D21'):
            D21_init = self._resolve_init_spec('D21', (self.nz, self.nd), default_std=1.0)
            self._set_param_data('D21', D21_init)

        logger.info(f"  ||A||={np.linalg.norm(self.A.detach().cpu().numpy()):.4f}")
        logger.info(f"  ||C||={np.linalg.norm(self.C.detach().cpu().numpy()):.4f}")
        logger.info(f"  ||C2||={np.linalg.norm(self.C2.detach().cpu().numpy()):.4f}")

    def initialize_s_from_conditions(
        self,
        train_inputs_n,
        y_max: float,
        warmup_steps: int = 0,
        n_grid: int = 15,
        s_min: float = 0.1,
        s_max: float = 20.0,
    ) -> dict:
        """Initialize ``s`` (and ``P``, ``L``) from the output + input conditions.

        Replaces the cumbersome max-s / heuristic ``s`` initialization with the
        *same sweep used for the final certificate* (MinTrProb,
        :meth:`solve_output_coverage_certificate`): over the preset ``s`` band it
        prefers an ``s`` that satisfies the (physical) output-coverage floor
        **and** leaves zero input violations on the training data, else the
        ``s`` with the fewest input violations.

        When the output level is not yet reachable for this freshly-initialized
        ``theta`` (``coverage_unreachable``), it falls back to plain max-s (the
        fewest-violations feasible certificate); the output-coverage penalty
        then grows the image toward ``y_max`` during training.

        ``y_max`` is the PHYSICAL safe output level; ``self.output_std`` must
        already be set (see :meth:`set_output_coverage_level`).
        ``train_inputs_n`` are the *normalized* training inputs ``(B, N, nd)``.
        Returns the certificate summary dict (``success=False`` when the max-s
        fallback was used).
        """
        self.set_output_coverage_level(y_max)  # output_std left unchanged
        inputs = torch.as_tensor(
            np.asarray(train_inputs_n), dtype=self.P.dtype, device=self.P.device
        )
        res = self.solve_output_coverage_certificate(
            y_max=y_max, inputs=inputs, warmup_steps=warmup_steps,
            n_grid=n_grid, s_min=s_min, s_max=s_max,
        )
        if res["success"]:
            logger.info(
                f"Init s from conditions: s={res['s']:.2f} "
                f"(band [{res['s_min']:.2f}, {res['s_max']:.2f}]), "
                f"y_bar={res['y_bar']:.2f} (y_max={y_max:.2f}), "
                f"output band y_tight={res['y_tight']:.2f} <= y_bar; "
                f"MaxS ceiling y_feas={res['y_feas']:.2f} "
                f"(s_feas={res['s_feas']:.2f}, ||H||={res['norm_H_feas']:.2f}), "
                f"input violations={res['n_input_violations']}, "
                f"violation_free={res['violation_free']}"
            )
            return res

        # Output level not reachable at init -> fall back to MaxS (the largest
        # regional invariant set among feasible certificates). Coverage is then
        # handled by the output-coverage penalty during training.
        logger.warning(
            f"Init s from conditions: output level y_max={y_max:.2f} not "
            f"reachable (reason={res['reason']}); falling back to max-s."
        )
        sol = self._synth().max_s()
        if sol is not None:
            self._apply_certificate_solution(sol)
        else:
            logger.warning("Init s fallback (max-s) also failed; s left unchanged.")
        return res
