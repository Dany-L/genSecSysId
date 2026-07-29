"""Training-time regularization losses + feasibility margins for SimpleLure.

Split out of ``constrained_rnn.py`` as a mixin to keep that module focused on the
model/dynamics. Every method uses ``self`` and relies on SimpleLure's attributes
(``P, L, s, C, C2, tau, la, y_max, output_std, ne, nz, ...``) and methods
(``get_lmis``, ``get_scalar_inequalities``) — provided at runtime via the MRO.
"""

from typing import Dict, Literal, Optional, Tuple, Union, overload

import torch

EPS = 1e-6


class LureRegularizationMixin:
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Feasibility regularization via the log-det interior-point barrier.

        For each LMI ``F ≻ 0`` adds ``-log det F`` and for each scalar
        inequality ``s > 0`` adds ``-log s``. Requires strictly feasible
        parameters (all eigenvalues > 0); the barrier grows to ``+∞`` as any
        constraint approaches its boundary.

        **Constant terms are skipped.** A term whose matrix does not require grad
        contributes nothing but a constant offset to the loss. This matters once
        the certificate is SDP-owned (:meth:`SimpleLure.freeze_certificate`): the
        ``nz`` locality LMIs ``[1/s², l_i; l_iᵀ, P]`` are built from ``s, L, P``
        **only** — no θ — so with the certificate frozen they become constants and
        the barrier reduces to the single stability term ``-log det(-F(θ; κ))``.
        That is also precisely why freezing removes the ``s → 0`` drift: the
        ``∂/∂s ≈ +2·nz/s`` push came from terms that are now constant. (Skipping
        them also saves ``nz`` small log-dets per batch; the reported
        ``reg_feasibility`` value drops the constant offset accordingly.)

        Returns:
            Regularization loss (sum of negative log-determinants).
        """
        # Only skip constants when a gradient is actually being built; under
        # ``no_grad`` (monitoring/eval) nothing requires grad, so the full barrier
        # value is still reported.
        skip_constants = torch.is_grad_enabled()

        feasibility_loss = torch.tensor(0.0, device=self.P.device)
        for f_i in self.get_lmis():
            F = f_i()
            if skip_constants and not F.requires_grad:
                continue  # constant term (frozen certificate) — no gradient
            # feasibility_loss += torch.relu(-torch.logdet(F))
            feasibility_loss += -torch.logdet(F)
        for s_i in self.get_scalar_inequalities():
            val = s_i()
            if skip_constants and torch.is_tensor(val) and not val.requires_grad:
                continue
            # feasibility_loss += torch.relu(-torch.log(val).squeeze())
            feasibility_loss += -torch.log(val).squeeze()

        return feasibility_loss

    def get_feasibility_margins(self) -> Dict[str, float]:
        """
        Per-constraint feasibility margins for monitoring interior-point drift.

        For each LMI ``F ≻ 0`` reports the smallest eigenvalue ``min λ(F)`` —
        the distance to the constraint boundary (``≤ 0`` means infeasible). For
        each scalar inequality ``s > 0`` reports its value. Also returns the
        aggregate ``min_eig`` (smallest margin across all constraints).

        Intended use: watch whether ``min_eig`` climbs steadily over training
        (certificate parameters drifting deeper into the feasible interior,
        pushed by the ``-log det`` barrier) while the input/output violation
        terms worsen — the signature of the barrier tugging against the other
        loss terms. A stationary ``min_eig`` means the barrier is not causing
        drift and the negative barrier value can be left as is.

        Returns:
            Dict mapping ``lmi_<i>_min_eig`` / ``scalar_<j>`` to their margins,
            plus aggregate ``min_eig`` (smallest margin across all constraints).
        """
        margins: Dict[str, float] = {}
        overall_min = float("inf")
        with torch.no_grad():
            for i, f_i in enumerate(self.get_lmis()):
                min_eig = torch.linalg.eigvalsh(f_i()).min().item()
                margins[f"lmi_{i}_min_eig"] = min_eig
                overall_min = min(overall_min, min_eig)
            for j, s_i in enumerate(self.get_scalar_inequalities()):
                val = s_i().squeeze().item()
                margins[f"scalar_{j}"] = val
                overall_min = min(overall_min, val)

        if overall_min != float("inf"):
            margins["min_eig"] = overall_min
        return margins

    @overload
    def get_regularization_input(
        self, u: torch.Tensor, x: torch.Tensor,
        return_c: Literal[False] = ..., warmup_steps: int = ...,
    ) -> torch.Tensor: ...

    @overload
    def get_regularization_input(
        self, u: torch.Tensor, x: torch.Tensor,
        return_c: Literal[True], warmup_steps: int = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def get_regularization_input(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        return_c: bool = False,
        warmup_steps: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute input constraint regularization loss (vectorized).

        Enforces input constraints: ||u_k||^2 <= s^2 - α^2 * (x_k^T * P^(-1) * x_k)
        Where:
        - u_k: input at timestep k, shape (batch, seq_len, n_inputs)
        - x_k: state at timestep k, shape (batch, seq_len, n_states)
        - s: input constraint bound
        - α: sigmoid-gated constraint parameter
        - P: parameter matrix

        This vectorized implementation replaces nested loops for efficiency.

        Args:
            inputs: Input trajectories, shape (batch_size, seq_len, n_inputs)
            states: State trajectories, shape (batch_size, seq_len, n_states, 1) or (batch_size, seq_len, n_states)

        Returns:
            Scalar tensor representing mean squared constraint violation
        """

        _, N, _ = u.shape  # batch size, sequence length, input dimension
        # Handle state tensor shape - squeeze trailing dimension if present
        # (batch, seq_len, state_dim, 1) -> (batch, seq_len, state_dim)
        if x.dim() == 4:
            x = x.squeeze(-1)
        

        # Get parameters
        alpha = 1.0 / (1.0 + torch.exp(-self.tau))  # sigmoid
        s = self.s
        X = torch.linalg.inv(self.P)  # P^(-1)

        # Compute vectorized quantities
        # ||u_k||^2 for all timesteps: (batch, seq_len, n_inputs) -> (batch, seq_len)
        u_norm_sq = (u[:,warmup_steps:N] ** 2).sum(dim=-1)

        # x_k^T * P^(-1) * x_k for all timesteps using einsum
        # states: (batch, seq_len, n_states)
        # X: (n_states, n_states)
        # Result: (batch, seq_len)
        x_quad_form = torch.einsum("bti,ij,btj->bt", x[:,warmup_steps:N,:], X, x[:,warmup_steps:N,:])

        # Compute constraint: c_k = ||u_k||^2 - s^2 + α^2 * (x_k^T * P^(-1) * x_k)
        # Shape: (batch, seq_len)
        eps = 0  # small epsilon for numerical stability
        c = u_norm_sq - s**2 + alpha**2 * x_quad_form + eps

        # Coverage is a worst-case property: the safe set must contain each
        # trajectory's peak excursion, so penalize the largest per-trajectory
        # violation (peak over time) averaged over the batch, rather than the
        # mean over all steps which dilutes the few steps that actually breach s.
        # relu is still needed (after amax): a satisfied trajectory has a
        # negative peak c, and without relu minimizing it would reward pushing c
        # ever more negative (inflating s) even when the constraint already holds.
        # relu is monotone, so relu(max_k c_k) == max_k relu(c_k).
        reg_loss = torch.relu(c.amax(dim=1)).mean()

        if return_c:
            return reg_loss, c

        return reg_loss

    def set_output_coverage_level(self, y_max, output_std=None) -> None:
        """Set the **physical** safe output level ``y_max`` (and, optionally, the
        physical output scale ``output_std`` used to relate the model's
        normalized ``C/P/s`` to physical units).

        ``y_max`` has a physical meaning and is stored unnormalized; the coverage
        machinery divides by ``output_std`` internally. ``None``/``nan`` y_max
        disables the output-coverage penalty. ``output_std=None`` leaves the
        stored scale unchanged. Kept on the model's device/dtype."""
        device, dtype = self.P.device, self.P.dtype
        self.y_max = torch.tensor(
            float(y_max) if y_max is not None else float("nan"), device=device, dtype=dtype
        )
        if output_std is not None:
            self.output_std = torch.tensor(float(output_std), device=device, dtype=dtype)

    def get_regularization_output(self, return_margin: bool = False):
        """Output-coverage penalty — bind Corollary 1 (see the wiki
        ``binding-output-certificate``).

        Enforces the coverage-on-image floor ``(σ·s)² C P Cᵀ ⪰ y_max² I`` in
        PHYSICAL output units (``σ = output_std``): the model's *own* certified
        output set must reach the physical safe level ``y_max``. The penalty is
        ``relu(λ_max(y_max² I − (σ·s)² C P Cᵀ))`` — zero once the physical image
        covers the data envelope in every direction, else the largest remaining
        per-direction deficit (for ``ne = 1`` just
        ``relu(y_max² − (σ·s)²·CPCᵀ)``).

        No-op (returns 0) when ``y_max`` is unset. This is the differentiable
        training surrogate for the exact binding SDP in
        :meth:`solve_output_coverage_certificate`.
        """
        zero = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        if self.y_max is None or bool(torch.isnan(self.y_max)):
            return (zero, zero) if return_margin else zero

        CPCt = self.C @ self.P @ self.C.T  # (ne, ne), symmetric
        deficit = self.y_max**2 * torch.eye(
            self.ne, device=self.P.device, dtype=self.P.dtype
        ) - (self.output_std * self.s)**2 * CPCt
        # Coverage <=> deficit ⪯ 0 <=> lambda_max(deficit) <= 0.
        lam_max = torch.linalg.eigvalsh(deficit)[-1]
        reg_loss = torch.relu(lam_max)

        if return_margin:
            return reg_loss, lam_max
        return reg_loss

    def get_regularization_tightness(self, return_margin: bool = False):
        """Output-*tightness* penalty — pull the certified output image DOWN onto
        ``y_max`` (the complement of :meth:`get_regularization_output`).

        Coverage (that method) is the one-sided *floor* ``(σs)²CPCᵀ ⪰ y_max²I``;
        tightness is the one-sided *ceiling*: it penalizes OVER-coverage
        ``relu(λ_max((σs)²CPCᵀ − y_max²I))`` so the certified half-width
        ``ȳ = σ·s·√(CPCᵀ)`` sits *just* at ``y_max`` instead of far above it. The
        two one-sided terms sandwich ``ȳ`` at ``y_max``; because this one is zero
        the moment ``ȳ ≤ y_max`` it can never *under*-cover on its own.

        ``C P Cᵀ`` is **detached**, so the gradient flows only to the certificate
        scale ``s`` — the free, gradient-owned knob. That keeps tightness from
        distorting the learned output map ``C`` (which is on the prediction path)
        or fighting the fixed-``s`` SDP repair on ``P``: the operative lever is
        simply "shrink ``s`` toward the coverage floor", which needs no per-epoch
        bisection and does not depend on ``C2`` (learned during training).

        No-op (returns 0) when ``y_max`` is unset.
        """
        zero = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        if self.y_max is None or bool(torch.isnan(self.y_max)):
            return (zero, zero) if return_margin else zero

        # Detach C P Cᵀ so only s carries gradient (largest output gain for ne>1).
        CPCt = (self.C @ self.P @ self.C.T).detach()  # (ne, ne), symmetric
        lam_max_CPCt = torch.linalg.eigvalsh(CPCt)[-1]
        # excess > 0 <=> the certified image over-shoots y_max in some direction.
        excess = (self.output_std * self.s) ** 2 * lam_max_CPCt - self.y_max ** 2
        reg_loss = torch.relu(excess)

        if return_margin:
            return reg_loss, excess
        return reg_loss

    def get_regularization_activity(
        self,
        w: torch.Tensor,
        w_star: float,
        warmup_steps: int = 0,
        return_activity: bool = False,
    ):
        """Dead-zone activity penalty — prevent the degenerate *linear collapse*.

        For the dead-zone nonlinearity ``w = Δ(z) = z − hardtanh(z)`` the rollout
        follows the pure LTI part exactly when ``w = 0`` (pre-activations stay in
        the dead band ``|z| ≤ 1``). That degenerate "linear collapse" is a poor
        fit for nonlinear data (and trivially globally stable). This penalty
        rewards the dead-zone to *fire* (``w ≠ 0``), pushing the model into its
        nonlinear regime.

        Caveat — this is **not** a certificate-level anti-global mechanism. Firing
        the nonlinearity does *not* by itself make the global (``H = 0``)
        certificate infeasible: ``H = 0`` means the *global* sector condition is
        used, and tanh/dzn satisfy sector ``[0, 1]`` **globally**, so a model can
        be globally absolutely stable with a fully active nonlinearity
        (``w ≠ 0``). Hence ``H = 0`` does **not** imply a linear model. Making
        ``H = 0`` genuinely infeasible requires shaping the *linear* dynamics so
        the global-sector LMI fails — a separate lever. This term only removes the
        ``w ≡ 0`` collapse; it is a behavioral heuristic that *correlates* with,
        but does not guarantee, a non-global model.

        Penalty ``relu(w_star − a)`` with ``a = ⟨‖w_k‖₂⟩`` the mean per-step
        activation norm over the (warmup-skipped) rollout; zero once the mean
        activity reaches the target ``w_star``. ``w_star ≤ 0`` disables it
        (no-op). It reads the rollout, not the certificate, so it does not
        directly touch P, L, s.

        Note: only meaningful for the ``dzn`` (dead-zone) activation, where
        ``w = 0`` ⇔ the linear regime. For ``sat``/``tanh`` (linear near 0,
        saturating for large ``z``) small ``w`` is *not* the linear regime, so
        the sign of "activity" is different — do not enable it there unchanged.

        Args:
            w: nonlinearity output ``(B, N, nw)`` from ``forward``.
            w_star: target mean activation norm (the hinge threshold).
            warmup_steps: leading steps skipped (match the prediction loss).
            return_activity: also return the scalar mean activity ``a``.
        """
        zero = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        if w_star is None or float(w_star) <= 0.0:
            return (zero, zero) if return_activity else zero

        N = w.shape[1]
        w_active = w[:, warmup_steps:N, :]
        # Mean over batch and time of the per-step L2 activation norm.
        activity = torch.linalg.vector_norm(w_active, dim=-1).mean()
        reg_loss = torch.relu(float(w_star) - activity)

        if return_activity:
            return reg_loss, activity
        return reg_loss

    def get_regularization_H(self, h_star, return_norm: bool = False):
        """Anti-global-certificate penalty — push the coupling ``H = L P⁻¹``
        away from zero.

        The regionality certificate collapses to the *global* sector condition
        exactly when ``H = L P⁻¹ = 0`` (no locality restriction — a globally
        absolutely stable, typically near-linear model). Unlike
        :meth:`get_regularization_activity`, which only removes the ``w ≡ 0``
        rollout collapse and does *not* make the ``H = 0`` certificate infeasible,
        this term acts directly on the certificate parameters (L, P), so it
        directly discourages the global certificate. It hinges on the coupling
        norm::

            relu(h_star − ‖H‖_F)

        zero once ``‖H‖_F ≥ h_star`` (the target coupling magnitude), else the
        remaining deficit; minimizing it grows ‖H‖. The Frobenius norm is
        computed with a tiny additive floor so the gradient is finite (rather
        than NaN) at the ``H = 0`` cone point — training escapes zero as soon as
        the SDP init / prediction loss make ``L`` nonzero.

        ``h_star ≤ 0`` disables it (no-op). Requires ``learn_L`` (``H`` is
        identically zero and non-trainable otherwise, so the term would be a
        constant no-op there too).

        Args:
            h_star: target coupling norm ‖H‖_F (the hinge threshold).
            return_norm: also return the scalar ‖H‖_F.
        """
        zero = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        if h_star is None or float(h_star) <= 0.0 or not self.learn_L:
            return (zero, zero) if return_norm else zero

        H = self.L @ torch.linalg.inv(self.P)
        # Frobenius norm with a small floor: sqrt(0) has a NaN gradient, so the
        # floor keeps the gradient finite at H = 0 without changing the value
        # meaningfully once H is nonzero.
        norm_H = torch.sqrt((H ** 2).sum() + EPS ** 2)
        reg_loss = torch.relu(float(h_star) - norm_H)

        if return_norm:
            return reg_loss, norm_H
        return reg_loss
