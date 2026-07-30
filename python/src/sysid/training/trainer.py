"""Main trainer class for RNN-based system identification."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]

from ..evaluation.evaluator import Evaluator
from ..models.base import BaseRNN
from ..models.constrained_rnn import SimpleLure
from ..utils import plot_predictions, plot_safe_set_trajectories, get_volume_of_ellipsoid


class Trainer:
    """Trainer for RNN models."""

    def __init__(
        self,
        model: BaseRNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        output_dir: str = "outputs",
        model_dir: str = "models",
        log_dir: str = "logs",
        gradient_clip_value: Optional[float] = None,
        regularization_weight: float = 0.0,
        decay_regularization_weight: bool = False,
        regularization_decay_factor: float = 0.5,
        min_regularization_weight: float = 1e-7,
        checkpoint_frequency: int = 10,
        early_stopping_patience: int = 50,
        mlflow_tracking: bool = True,
        log_gradients: bool = True,
        warmup_steps: int = 0,
        input_regularization_weight: float = 0.01,
        output_regularization_weight: float = 0.0,
        tightness_regularization_weight: float = 0.0,
        activity_regularization_weight: float = 0.0,
        activity_target: float = 0.0,
        h_regularization_weight: float = 0.0,
        h_target: float = 0.0,
        output_std: float = 1.0,
        train_div_loader: Optional[DataLoader] = None,
        val_div_loader: Optional[DataLoader] = None,
        freeze_certificate: bool = False,
        freeze_alpha: bool = True,
        repair_enforce_coverage: bool = False,
        resynthesize_certificate: bool = False,
        resynthesis_every: int = 0,
        resynthesis_beta: float = 2.0,
        resynthesis_beta_min: float = 1.0,
        resynthesis_beta_decay: float = 0.9,
        resynthesis_beta_grow: float = 1.5,
        resynthesis_guard: bool = True,
        resynthesis_target_mid: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to train on
            output_dir: Directory for outputs
            model_dir: Directory for saved models
            log_dir: Directory for logs
            gradient_clip_value: Gradient clipping value
            regularization_weight: Initial weight for custom regularization
            decay_regularization_weight: Whether to decay reg weight with LR
            regularization_decay_factor: Factor to decay reg weight (interior point method)
            min_regularization_weight: Minimum threshold for reg weight early stopping (default: 1e-7)
            checkpoint_frequency: Save checkpoint every N epochs
            early_stopping_patience: Patience for early stopping
            mlflow_tracking: Whether to use MLflow tracking
            log_gradients: Whether to log gradient statistics
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_div_loader = train_div_loader
        self.val_div_loader = val_div_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        # Directories
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.gradient_clip_value = gradient_clip_value
        self.regularization_weight = regularization_weight
        self.initial_regularization_weight = regularization_weight  # Store initial value
        self.decay_regularization_weight = decay_regularization_weight
        self.regularization_decay_factor = regularization_decay_factor
        self.min_regularization_weight = min_regularization_weight
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        self.warmup_steps = warmup_steps  # Number of steps to skip before computing loss
        self.input_regularization_weight = input_regularization_weight  # Weight for input constraint regularization
        self.initial_input_regularization_weight = input_regularization_weight  # Store initial value
        # Output-coverage regularization (bind Corollary 1): pushes the certified
        # output image s^2 C P C^T to reach the physical safe data level y_max.
        self.output_regularization_weight = output_regularization_weight
        self.initial_output_regularization_weight = output_regularization_weight
        # Output-tightness regularization: pulls the certified half-width y_bar DOWN
        # onto y_max (complement of the coverage floor). NOT decayed — tightness
        # must hold all through training.
        self.tightness_regularization_weight = tightness_regularization_weight
        # Dead-zone activity regularization: pushes mean ||w|| up to activity_target
        # so the nonlinearity fires, preventing the degenerate linear collapse
        # (w == 0 -> pure LTI rollout). NOTE: it does not by itself force a
        # non-global model (see get_regularization_activity). NOT decayed on purpose
        # (must hold all through training), so no initial_* / decay entry.
        self.activity_regularization_weight = float(activity_regularization_weight)
        self.activity_target = float(activity_target)
        # Anti-global-certificate regularization: pushes ||H||_F (H = L P^-1) up
        # to h_target so the certificate stays LOCAL (H = 0 is the global sector
        # condition). Acts on the certificate params directly, unlike the activity
        # term. NOT decayed on purpose (must hold all through training).
        self.h_regularization_weight = float(h_regularization_weight)
        self.h_target = float(h_target)
        # Physical output scale (relates normalized C/P/s to physical y_max);
        # used only by the _init_output_coverage_level fallback.
        self.output_std = float(output_std)

        # --- Certificate ownership (see the wiki note training/certificate-resynthesis)
        # theta <- SGD, (P, L, la, s) <- the SDPs. `freeze_certificate` removes the
        # certificate from autograd (killing the barrier's s -> 0 drift, which has
        # no counterweight since s is not in the prediction loss);
        # `resynthesize_certificate` re-solves it from the current theta once per
        # epoch with the coverage band as a HARD constraint, so rho stays ~1.
        self.freeze_certificate_flag = bool(freeze_certificate)
        self.freeze_alpha = bool(freeze_alpha)
        self.repair_enforce_coverage = bool(repair_enforce_coverage)
        self.resynthesize_certificate_flag = bool(resynthesize_certificate)
        # <= 0 disables the cadence: re-synthesize only when rho leaves the band.
        self.resynthesis_every = int(resynthesis_every)
        self.resynthesis_beta = float(resynthesis_beta)
        self.resynthesis_beta_min = float(resynthesis_beta_min)
        self.resynthesis_beta_decay = float(resynthesis_beta_decay)
        self.resynthesis_beta_grow = float(resynthesis_beta_grow)
        self.resynthesis_guard = bool(resynthesis_guard)
        # Restore rho to the geometric middle of [1, beta^nx] instead of its
        # lower edge, so the drift test is not a knife edge.
        self.resynthesis_target_mid = bool(resynthesis_target_mid)
        # Counters / cached guard batch
        self.resynthesis_applied = 0
        self.resynthesis_rejected = 0
        self.resynthesis_failed = 0
        self.epoch_coverage_repair_fallbacks = 0
        self._guard_batch: Optional[tuple] = None

        # Rollback tracking
        self.rollback_count = 0
        self.epoch_rollback_count = 0

        # Logging
        self.mlflow_tracking = mlflow_tracking
        self.log_gradients = log_gradients

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0  # Track which epoch had the best validation loss
        self.patience_counter = 0
        self.train_losses: list[float] = []
        self.train_pred_losses: list[float] = []
        self.train_reg_feasibility: list[float] = []
        self.train_reg_inputs: list[float] = []
        self.val_losses: list[float] = []
        # Diverging-trajectory metric histories (populated when *_div loaders are provided)
        self.train_div_losses: list[float] = []
        self.val_div_losses: list[float] = []

        # Scheduler (can be set later)
        self.scheduler = None

        # Derive the (physical) safe output level y_max from the training targets
        # and hand it to the model for the output-coverage penalty.
        self._init_output_coverage_level()

        # Hand the certificate to the SDPs (must happen after the model is on the
        # device; safe after the optimizer was built — optimizer.step() skips
        # parameters whose grad is None).
        if self.freeze_certificate_flag and hasattr(self.model, "freeze_certificate"):
            self.model.freeze_certificate(freeze_alpha=self.freeze_alpha)
        if self.resynthesize_certificate_flag:
            y_max = getattr(self.model, "y_max", None)
            has_y_max = y_max is not None and not bool(torch.isnan(y_max))
            logging.info(
                "Certificate re-synthesis: enabled "
                f"({'cadence off — rho-triggered only' if self.resynthesis_every <= 0 else f'every {self.resynthesis_every} epoch(s)'}"
                f", beta0={self.resynthesis_beta}, "
                f"guard={'on' if self.resynthesis_guard else 'off'})"
                + ("" if has_y_max else " — y_max unset, TightCert degenerates to MaxS "
                   "and the rho trigger is unavailable (cadence only)")
            )

    def _init_output_coverage_level(self):
        """Set the model's PHYSICAL safe output level ``y_max`` (fallback path).

        No-op unless something needs ``y_max`` — output-coverage or tightness
        regularization, the hard coverage floor in the repair, or per-epoch
        re-synthesis — and skipped when the model already has ``y_max`` set: in the
        normal pipeline ``initialize_parameters`` sets both ``y_max`` and
        ``output_std`` from the raw data + normalizer, so this only fires for
        directly-constructed / loaded models. The loader yields *normalized*
        targets, so ``max |e| · output_std`` is the physical ``y_max``.
        """
        needs_y_max = (
            self.output_regularization_weight > 0
            or self.tightness_regularization_weight > 0
            or self.repair_enforce_coverage
            or self.resynthesize_certificate_flag
        )
        if not needs_y_max:
            return
        if not hasattr(self.model, "set_output_coverage_level"):
            return
        y_max = getattr(self.model, "y_max", None)
        if y_max is not None and not bool(torch.isnan(y_max)):
            return  # already set by the caller (e.g. initialize_parameters)

        peak_n = 0.0
        for batch in self.train_loader:
            e = batch[1]
            e = e[torch.isfinite(e)]  # ignore NaN padding
            if e.numel() > 0:
                peak_n = max(peak_n, float(e.abs().max()))
        if peak_n > 0.0:
            y_max_phys = peak_n * self.output_std
            self.model.set_output_coverage_level(y_max_phys, self.output_std)
            logging.info(f"Output-coverage level y_max set from training data: {y_max_phys:.6f}")

    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler

    def compute_gradient_stats(self) -> Dict[str, float]:
        """
        Compute gradient norms for each model parameter.

        Returns:
            Dictionary with gradient norms: {param_name: grad_norm}
        """
        stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm().item()
                # Use forward slashes for nested module names (MLflow compatible)
                stats[f"grad_norm/{name}"] = grad_norm

        return stats

    def plot_trajectories(self, normalizer=None, name="initial_trajectories"):
        """
        Plot initial model predictions before training as a reference.
        Uses validation data and saves plots as MLflow artifacts.

        Args:
            normalizer: Data normalizer for denormalization (optional)
        """

        # Create temporary evaluator
        temp_output_dir = self.output_dir / "predictions"
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        evaluator = Evaluator(model=self.model, device=self.device, output_dir=str(temp_output_dir), warmup_steps=self.warmup_steps)

        # Evaluate on validation set
        results = evaluator.evaluate(
            test_loader=self.val_loader,
            normalizer=normalizer,
            print_results=False,
            save_files=False,  # Don't save prediction files during training
        )

        e_hat = results["e_hat"]
        e = results["e"]
        d = results.get("inputs", None)
        x = results.get("x", None)
        c = results.get("c", None)

        # Select sample indices: always include sequence 0, plus 2 random sequences
        num_sequences = e_hat.shape[0]
        sample_indices = [0]  # Always include sequence 0

        if num_sequences > 1:
            # Select 2 random sequences (excluding sequence 0)
            other_indices = list(range(1, num_sequences))
            num_random = min(2, len(other_indices))
            random_indices = np.random.choice(
                other_indices, size=num_random, replace=False
            ).tolist()
            sample_indices.extend(random_indices)

        if self.model.nx == 2:
            fig, ax, _, _ = plot_safe_set_trajectories(
                P=self.model.P.cpu().detach().numpy(),
                L=self.model.L.cpu().detach().numpy(),
                s=self.model.s.cpu().detach().numpy(),
                x_traj=x,
                c=c,
                warmup_steps=self.warmup_steps,
                horizon=50,
            )
            ellipse_plot_path = temp_output_dir / f"ellipse-{name}.png"
            fig.savefig(ellipse_plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            # Plots are uploaded in bulk by train.py via log_artifacts at the
            # end of training (avoids the predictions/ vs outputs/predictions/
            # duplicate in MLflow).

        # Generate plot
        plot_path = temp_output_dir / f"{name}.png"
        plot_predictions(
            output_dir=evaluator.output_dir,
            e_hat=e_hat,
            e=e,
            d=d,
            sample_indices=sample_indices,
            save_path=str(plot_path),
            warmup_steps=self.warmup_steps,
        )

    def plot_trajectories_div(self, normalizer=None, name="initial_trajectories_div"):
        """Plot model predictions on diverging validation trajectories.

        Mirrors plot_trajectories() but iterates val_div_loader (batch_size=1,
        variable length) and runs the model with warmup_steps=0. Trajectories
        are NaN-padded to a common length only for the plotting call so the
        existing plot_predictions can index them; matplotlib skips NaN points.
        """
        if self.val_div_loader is None:
            return

        output_dir = self.output_dir / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_e_hat = []
        all_e = []
        all_d = []
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_div_loader:
                if len(batch) == 3:
                    d, e, x0 = batch
                    x0 = None
                else:
                    d, e = batch
                    x0 = None
                d = d.to(self.device)
                e = e.to(self.device)
                e_hat, _, _ = self.model(d, x0, warmup_steps=0)
                e_hat_np = e_hat.cpu().numpy()
                e_np = e.cpu().numpy()
                d_np = d.cpu().numpy()
                if normalizer is not None:
                    e_hat_np = normalizer.inverse_transform_outputs(e_hat_np)
                    e_np = normalizer.inverse_transform_outputs(e_np)
                    d_np = normalizer.inverse_transform_inputs(d_np)
                # Each batch is (1, T_i, n) — unwrap to (T_i, n).
                all_e_hat.append(e_hat_np[0])
                all_e.append(e_np[0])
                all_d.append(d_np[0])

        if not all_e_hat:
            return

        # NaN-pad to a common length so plot_predictions can index by sample.
        max_len = max(a.shape[0] for a in all_e_hat)

        def _pad(arrays, n_cols):
            out = np.full((len(arrays), max_len, n_cols), np.nan)
            for i, a in enumerate(arrays):
                out[i, : a.shape[0], :] = a
            return out

        e_hat_padded = _pad(all_e_hat, all_e_hat[0].shape[1])
        e_padded = _pad(all_e, all_e[0].shape[1])
        d_padded = _pad(all_d, all_d[0].shape[1])

        num_seq = len(all_e_hat)
        sample_indices = [0]
        if num_seq > 1:
            other_indices = list(range(1, num_seq))
            num_random = min(2, len(other_indices))
            random_indices = np.random.choice(
                other_indices, size=num_random, replace=False
            ).tolist()
            sample_indices.extend(random_indices)

        plot_path = output_dir / f"{name}.png"
        plot_predictions(
            output_dir=output_dir,
            e_hat=e_hat_padded,
            e=e_padded,
            d=d_padded,
            sample_indices=sample_indices,
            save_path=str(plot_path),
            warmup_steps=0,
        )
        # Plot uploaded in bulk by train.py at end of training (no duplicate).

    def decay_regularization(self):
        """
        Decay regularization weights (Interior Point Method).

        In interior point methods for convex optimization, the barrier parameter
        is reduced as we approach the solution. Here we decay both the feasibility
        regularization weight and the input-constraint regularization weight
        whenever the learning rate is reduced.
        """
        if not self.decay_regularization_weight:
            return

        if self.regularization_weight > 0:
            old_weight = self.regularization_weight
            self.regularization_weight *= self.regularization_decay_factor
            # Ensure we don't go below minimum threshold
            if self.regularization_weight < self.min_regularization_weight:
                self.regularization_weight = self.min_regularization_weight
            logging.info(
                f"Regularization weight decayed: {old_weight:.6e} → {self.regularization_weight:.6e}"
            )

        if self.input_regularization_weight > 0:
            old_input_weight = self.input_regularization_weight
            self.input_regularization_weight *= self.regularization_decay_factor
            # Ensure we don't go below minimum threshold
            if self.input_regularization_weight < self.min_regularization_weight:
                self.input_regularization_weight = self.min_regularization_weight
            logging.info(
                f"Input regularization weight decayed: {old_input_weight:.6e} → {self.input_regularization_weight:.6e}"
            )

        if self.output_regularization_weight > 0:
            old_output_weight = self.output_regularization_weight
            self.output_regularization_weight *= self.regularization_decay_factor
            if self.output_regularization_weight < self.min_regularization_weight:
                self.output_regularization_weight = self.min_regularization_weight
            logging.info(
                f"Output regularization weight decayed: {old_output_weight:.6e} → {self.output_regularization_weight:.6e}"
            )

    def _get_guard_batch(self):
        """One cached training batch ``(d, x0)`` for the re-synthesis accept guard."""
        if self._guard_batch is None:
            for batch in self.train_loader:
                d = batch[0].to(self.device)
                x0 = (
                    batch[2].to(device=self.device, dtype=d.dtype)
                    if len(batch) == 3 and batch[2] is not None else None
                )
                self._guard_batch = (d, x0)
                break
        return self._guard_batch

    def _resynthesize_certificate(self, epoch: int) -> Dict[str, Any]:
        """Per-epoch certificate re-synthesis — the epoch-boundary step of the
        ownership scheme (wiki: ``training/certificate-resynthesis``).

        1. Read the **free** drift monitor ``ρ = (ȳ/y_max)ⁿˣ`` (no SDP: a couple of
           small matrix products on parameters already in memory).
        2. Re-solve TightCert only when ``ρ`` has left the band ``[1, βⁿˣ]`` or the
           cadence fires. Without ``y_max`` there is no ``ρ``, so the cadence is
           the only trigger and TightCert degenerates to MaxS.
        3. Accept the new certificate unless it increases the input-condition
           violation count on the cached guard batch.
        4. Anneal ``β``: widen after an all-rollback epoch (give the model slack
           back), otherwise tighten geometrically toward ``β_min``.

        Returns the metrics for this epoch (empty when re-synthesis is off).
        """
        if not self.resynthesize_certificate_flag or not isinstance(self.model, SimpleLure):
            return {}

        metrics: Dict[str, Any] = {"beta": self.resynthesis_beta}
        rho_before = self.model.coverage_ratio()
        if rho_before is not None:
            metrics["rho"] = rho_before
            metrics["rho_before"] = rho_before  # kept even when the solve overwrites `rho`

        # Alarm at the band EDGE, restore to the band MIDDLE. The solve's objective
        # (min ‖P‖ subject to the coverage floor) otherwise lands ρ *on* the lower
        # edge, so the next epoch's drift crosses it whatever the cadence is —
        # measured: ρ = 1.029 against a band of [1, 1.1025], i.e. 2.9 % of headroom,
        # and the drift test fired on 51 of 100 epochs. Solving against a slightly
        # inflated y_max puts ρ at the geometric middle instead, leaving room on
        # both sides. Note ρ < 1 means the certified image no longer covers the data,
        # so the LOWER edge gets no tolerance — only the target moves.
        band_hi = self.resynthesis_beta ** self.model.nx
        out_of_band = rho_before is not None and not (1.0 <= rho_before <= band_hi)
        # `resynthesis_every <= 0` disables the cadence entirely (event-driven only).
        cadence = self.resynthesis_every > 0 and (epoch % self.resynthesis_every) == 0
        if out_of_band or cadence:
            guard = self._get_guard_batch() if self.resynthesis_guard else None
            y_max_target = None
            if self.resynthesis_target_mid:
                y_max = getattr(self.model, "y_max", None)
                if y_max is not None and not bool(torch.isnan(y_max)):
                    # ȳ target = √β · y_max  ⇒  ρ_target = β^(nx/2), the geometric
                    # middle of [1, β^nx]. The band handed to the SDP shrinks to
                    # [√β, β] · y_max, which stays inside the requested band.
                    y_max_target = float(y_max) * float(np.sqrt(self.resynthesis_beta))
            result = self.model.resynthesize_certificate(
                y_max=y_max_target,
                beta=float(np.sqrt(self.resynthesis_beta)) if y_max_target else self.resynthesis_beta,
                guard_inputs=guard[0] if guard is not None else None,
                guard_x0=guard[1] if guard is not None else None,
                warmup_steps=self.warmup_steps,
            )
            metrics["resynth_trigger"] = 1.0 if out_of_band else 0.5  # drift vs cadence
            if not result["success"]:
                self.resynthesis_failed += 1
                logging.warning(
                    f"Epoch {epoch}: certificate re-synthesis failed "
                    f"({result['reason']}); keeping the current certificate."
                )
            elif result["applied"]:
                self.resynthesis_applied += 1
                rho_after = self.model.coverage_ratio()
                if rho_after is not None:
                    metrics["rho"] = rho_after
                    metrics["rho_after"] = rho_after
                metrics["s_resynth"] = result["s"]
                if result.get("norm_P") is not None:
                    metrics["norm_P"] = result["norm_P"]
                logging.debug(
                    f"Epoch {epoch}: certificate re-synthesized — s={result['s']:.4g}, "
                    f"rho={result['rho'] if result['rho'] is None else round(result['rho'], 4)}"
                    f" (band [1, {band_hi:.3g}], beta={self.resynthesis_beta:.3g})"
                )
            else:
                self.resynthesis_rejected += 1
        metrics["resynth_applied"] = float(self.resynthesis_applied)
        metrics["resynth_rejected"] = float(self.resynthesis_rejected)
        metrics["resynth_failed"] = float(self.resynthesis_failed)

        # Anneal beta. It is the band that *drives* tightening, so it cannot be
        # left to move only when the band is violated: with a loose start nothing
        # would ever trigger, so beta would never shrink and the certificate would
        # stay loose forever. Nor should it march on the clock — that squeezes rho
        # out of the band by itself and manufactures a re-solve every epoch.
        # The rule is therefore: tighten on a HEALTHY epoch (rho stayed in the band
        # and the model is not rolling back) — the band is then demonstrably looser
        # than it needs to be. Hold it still on an epoch that already needed a
        # re-solve; widen after an all-rollback epoch.
        if self.epoch_rollback_count >= max(len(self.train_loader), 1):
            self.resynthesis_beta *= self.resynthesis_beta_grow
        elif not out_of_band:
            self.resynthesis_beta = max(
                self.resynthesis_beta * self.resynthesis_beta_decay,
                self.resynthesis_beta_min,
            )
        return metrics

    def _repair_certificate(self) -> bool:
        """Two-tier fixed-``s`` certificate repair. ``False`` ⇒ the caller rolls back.

        Tier 1 repairs P, L, Λ **with** the hard coverage floor
        ``(σ·s)²·C P Cᵀ ⪰ y_max²·I``, so a repair cannot buy feasibility by
        shrinking the certified output image. Tier 2 (only if tier 1 is
        infeasible) drops the floor: the certificate is then feasible but may
        under-cover, which is preferable to stalling the whole epoch — at
        ρ ≈ 1 the floor-constrained repair fails often, and the epoch-boundary
        re-synthesis restores coverage with ``s`` free anyway.

        With ``repair_enforce_coverage=False`` (or no ``y_max``) tier 1 *is* the
        historical floor-free repair and tier 2 never runs.
        """
        if self.model.feasibility_problem(enforce_coverage=self.repair_enforce_coverage):
            return True
        if not self.repair_enforce_coverage:
            return False
        # Tier 2: retry without the coverage floor.
        if self.model.feasibility_problem(enforce_coverage=False):
            self.epoch_coverage_repair_fallbacks += 1
            logging.debug(
                "Repair infeasible with the coverage floor; fell back to the "
                "floor-free repair (certificate may under-cover until the next "
                "re-synthesis)."
            )
            return True
        return False

    def reduce_lr_on_rollback(self, factor: float = 0.5):
        """
        Reduce learning rate when rollbacks occur frequently.
        This helps when the optimizer step is too large for the constrained space.

        Args:
            factor: Factor to multiply learning rate by (default: 0.5)
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            param_group["lr"] *= factor
            new_lr = param_group["lr"]
            logging.info(f"Learning rate reduced due to rollbacks: {old_lr:.6e} → {new_lr:.6e}")

    def _train_diverging_epoch(self) -> float:
        """One pass over the diverging-trajectory loader.

        Each batch has batch_size=1 so trajectories of different lengths
        do not need to be stacked. Loss is computed from t=0 with no
        warmup skipping (these trajectories start from x0=0 so no
        transient needs to be discarded). Equal-sum semantics with the
        converging pass: each diverging batch produces its own optimizer
        step, so the per-epoch gradient effectively adds ∇pred_div on top
        of ∇pred_conv.

        Returns the average diverging prediction loss over the epoch.
        """
        total_pred_loss_div = 0.0
        num_batches = 0

        assert self.train_div_loader is not None  # caller guards this
        for batch in self.train_div_loader:
            if len(batch) == 3:
                d, e, x0 = batch
            else:
                d, e = batch
                x0 = None
            d = d.to(self.device)
            e = e.to(self.device)
            if x0 is not None:
                x0 = x0.to(device=self.device, dtype=d.dtype)

            self.optimizer.zero_grad()
            e_hat, (x, w), _ = self.model(d, x0=x0, warmup_steps=0)
            # NO warmup slicing — diverging trajectories often die before
            # warmup_steps would expire, and they share x0 with the model
            # so there is no transient to discard.
            pred_loss_div = self.loss_fn(e_hat, e)

            if self.regularization_weight > 0:
                # feasibility loss
                reg_feasibility_loss = self.model.get_regularization_loss()
                # reg_feasibility_loss = torch.tensor(0.0)
                reg_feasibility_value = reg_feasibility_loss.item()

                loss = pred_loss_div + self.regularization_weight * reg_feasibility_loss
            else:
                loss = pred_loss_div
                
            loss.backward()

            if self.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_value
                )

            # Mirror the converging loop's SimpleLure rollback safety so
            # diverging-batch updates also respect feasibility.
            if isinstance(self.model, SimpleLure):
                saved_state = {
                    name: param.data.clone()
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                }

            self.optimizer.step()

            if isinstance(self.model, SimpleLure):
                if not self.model.check_constraints() and self.regularization_weight > 0:
                    b_feasible = self._repair_certificate()
                    if not b_feasible:
                        logging.warning(
                            "Diverging batch: Feasibility SDP failed, rolling back parameters"
                        )
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if param.requires_grad and name in saved_state:
                                    param.data.copy_(saved_state[name])
                        self.rollback_count += 1
                        self.epoch_rollback_count += 1

            total_pred_loss_div += pred_loss_div.item()
            num_batches += 1

        return total_pred_loss_div / max(num_batches, 1)

    def train_epoch(self) -> Dict[str, Any]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training loss, prediction loss, regularization loss, and gradient statistics
        """
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_reg_loss = 0.0
        total_reg_feasibility = 0.0
        total_reg_parametric = 0.0
        total_reg_inputs = 0.0
        total_reg_output = 0.0
        total_reg_tightness = 0.0
        total_reg_activity = 0.0
        total_reg_H = 0.0
        num_batches = 0

        # Reset epoch rollback counter
        self.epoch_rollback_count = 0
        # Reset the per-epoch count of repairs that had to drop the coverage floor
        self.epoch_coverage_repair_fallbacks = 0

        # Accumulate gradient stats over epoch
        epoch_grad_stats: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch (states may be None)
            if len(batch) == 3:
                d, e, x0 = batch  # d: input, e: output, x: states (optional)
                x0 = None
            else:
                d, e = batch
                x0 = None
            d = d.to(self.device)
            e = e.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            e_hat, (x, w), _ = self.model(d, x0=x0, warmup_steps=self.warmup_steps)
            # e_hat = self.model(d, x0=x0)  # e_hat: predicted output

            # Compute prediction loss (skip warmup steps).
            # NaN positions in e (padded trajectories) are already handled by
            # MaskedLoss, so slicing [:, n:, :] is safe even when some sequences
            # are shorter than n — those positions are NaN and get ignored.
            pred_loss = self.loss_fn(e_hat[:, self.warmup_steps:, :], e[:, self.warmup_steps:, :])

            # Add custom regularization
            reg_loss_value = 0.0
            reg_feasibility_value = 0.0
            reg_input_value = 0.0
            reg_output_value = 0.0
            reg_tightness_value = 0.0
            reg_activity_value = 0.0
            reg_H_value = 0.0
            if self.regularization_weight > 0:
                # feasibility loss
                reg_feasibility_loss = self.model.get_regularization_loss()
                # reg_feasibility_loss = torch.tensor(0.0)
                reg_feasibility_value = reg_feasibility_loss.item()

                # Input constraint regularization (vectorized, moved to model)
                reg_input_loss = self.model.get_regularization_input(d, x)
                # reg_input_loss = torch.tensor(0.0)
                reg_input_value = reg_input_loss.item()

                loss = pred_loss + self.regularization_weight * reg_feasibility_loss + self.input_regularization_weight * reg_input_loss

                # Output-coverage regularization (bind Corollary 1): push the
                # certified output image to reach the physical safe level y_max.
                # No-op when the weight is 0 or y_max is unset.
                if self.output_regularization_weight > 0:
                    reg_output_loss = self.model.get_regularization_output()
                    reg_output_value = reg_output_loss.item()
                    loss = loss + self.output_regularization_weight * reg_output_loss

                # Output-tightness regularization: pull the certified half-width
                # y_bar DOWN onto y_max (penalize over-coverage), so the certificate
                # stays tight. C P C^T is detached inside the term, so it moves only
                # the scale s. No-op when weight/y_max is 0/unset.
                if self.tightness_regularization_weight > 0 and hasattr(
                    self.model, "get_regularization_tightness"
                ):
                    reg_tightness_loss = self.model.get_regularization_tightness()
                    reg_tightness_value = reg_tightness_loss.item()
                    loss = loss + self.tightness_regularization_weight * reg_tightness_loss

                # Dead-zone activity regularization: push mean ||w|| up to
                # activity_target so the nonlinearity fires, preventing the
                # degenerate linear collapse (w == 0 -> pure LTI rollout). Uses the
                # rollout w from the forward pass above. No-op when weight/target
                # is 0 or the model doesn't implement it.
                if self.activity_regularization_weight > 0 and hasattr(
                    self.model, "get_regularization_activity"
                ):
                    reg_activity_loss = self.model.get_regularization_activity(
                        w, self.activity_target, warmup_steps=self.warmup_steps
                    )
                    reg_activity_value = reg_activity_loss.item()
                    loss = loss + self.activity_regularization_weight * reg_activity_loss

                # Anti-global-certificate regularization: push ||H||_F (H = L P^-1)
                # up to h_target so the certificate stays local (H = 0 is the
                # global sector condition). Acts on the certificate params (L, P).
                # No-op when weight/target is 0 or the model doesn't implement it.
                if self.h_regularization_weight > 0 and hasattr(
                    self.model, "get_regularization_H"
                ):
                    reg_H_loss = self.model.get_regularization_H(self.h_target)
                    reg_H_value = reg_H_loss.item()
                    loss = loss + self.h_regularization_weight * reg_H_loss
            else:
                loss = pred_loss

            # Backward pass
            loss.backward()  # Retain graph for potential second backward pass if needed

            # Compute gradient statistics (before clipping) if logging enabled
            if self.log_gradients:
                grad_stats = self.compute_gradient_stats()

                # Accumulate stats (compute epoch average later)
                for key, value in grad_stats.items():
                    if key not in epoch_grad_stats:
                        epoch_grad_stats[key] = []
                    epoch_grad_stats[key].append(value)

            # Gradient clipping
            if self.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

            # Save parameter state before update (for constrained models)
            if isinstance(self.model, SimpleLure):
                saved_state = {
                    name: param.data.clone()
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                }

            # Update weights
            self.optimizer.step()
            # Check if constraints are satisfied (for constrained models)
            if isinstance(self.model, SimpleLure):
                if not self.model.check_constraints() and self.regularization_weight > 0:
                    # Constraints violated - repair P, L, M at the current s (fixed-s
                    # Feasibility, two-tier); roll back if no feasible cert exists.
                    b_feasible = self._repair_certificate()

                    if not b_feasible:
                        # SDP failed - roll back to previous parameters
                        logging.warning(
                            f"Batch {batch_idx}: Feasibility SDP failed, rolling back parameters"
                        )

                        # Restore saved parameters
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if param.requires_grad and name in saved_state:
                                    param.data.copy_(saved_state[name])

                        # Track rollbacks
                        self.rollback_count += 1
                        self.epoch_rollback_count += 1

                        logging.info(
                            f"Batch {batch_idx}: Parameters rolled back successfully (total: {self.rollback_count})"
                        )
                    # else:
                    # logging.info(f"Batch {batch_idx}: Feasibility SDP succeeded, parameters updated")

            # Update metrics
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_reg_feasibility += reg_feasibility_value
            total_reg_inputs += reg_input_value
            total_reg_output += reg_output_value
            total_reg_tightness += reg_tightness_value
            total_reg_activity += reg_activity_value
            total_reg_H += reg_H_value
            num_batches += 1

        # Average loss
        avg_loss = total_loss / num_batches
        avg_pred_loss = total_pred_loss / num_batches
        avg_reg_feasibility = total_reg_feasibility / num_batches
        avg_reg_inputs = total_reg_inputs / num_batches
        avg_reg_output = total_reg_output / num_batches
        avg_reg_tightness = total_reg_tightness / num_batches
        avg_reg_activity = total_reg_activity / num_batches
        avg_reg_H = total_reg_H / num_batches

        # Average gradient statistics over epoch
        avg_grad_stats = {key: np.mean(values) for key, values in epoch_grad_stats.items()}

        # Second pass over diverging trajectories (variable length, batch_size=1,
        # no warmup skipping). Loss is reported separately as `pred_loss_div`.
        pred_loss_div = None
        if self.train_div_loader is not None:
            pred_loss_div = self._train_diverging_epoch()

        return {
            "loss": avg_loss,
            "pred_loss": avg_pred_loss,
            "pred_loss_div": pred_loss_div,
            "reg_feasibility": avg_reg_feasibility,
            "reg_input": avg_reg_inputs,
            "reg_output": avg_reg_output,
            "reg_tightness": avg_reg_tightness,
            "reg_activity": avg_reg_activity,
            "reg_H": avg_reg_H,
            "rollback_count": self.epoch_rollback_count,
            **avg_grad_stats,
        }

    def validate(self) -> Dict[str, Optional[float]]:
        """Validate the model.

        Returns a dict with keys:
            val_loss: validation loss on converging trajectories (warmup
                applied — this is the metric used for early stopping /
                model selection, preserving the legacy behavior).
            val_loss_div: validation loss on diverging trajectories
                (no warmup applied); None when no val_div_loader is set.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch (states may be None)
                if len(batch) == 3:
                    d, e, x0 = batch  # d: input, e: output, x0: initial states (optional)
                    x0 = None
                else:
                    d, e = batch
                    x0 = None

                d = d.to(self.device)
                e = e.to(self.device)

                # Forward pass
                e_hat, _, _ = self.model(d, x0, warmup_steps=self.warmup_steps)

                # Compute loss
                loss = self.loss_fn(e_hat[:, self.warmup_steps:, :], e[:, self.warmup_steps:, :])

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Diverging validation: full sequence from t=0, no warmup skipping.
        val_loss_div: Optional[float] = None
        if self.val_div_loader is not None:
            div_total = 0.0
            div_batches = 0
            with torch.no_grad():
                for batch in self.val_div_loader:
                    if len(batch) == 3:
                        d, e, x0 = batch
                        x0 = None
                    else:
                        d, e = batch
                        x0 = None
                    d = d.to(self.device)
                    e = e.to(self.device)
                    e_hat, _, _ = self.model(d, x0, warmup_steps=0)
                    loss = self.loss_fn(e_hat, e)
                    div_total += loss.item()
                    div_batches += 1
            val_loss_div = div_total / max(div_batches, 1)

        return {"val_loss": avg_loss, "val_loss_div": val_loss_div}

    def train(self, max_epochs: int, normalizer=None) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            max_epochs: Maximum number of epochs
            normalizer: Data normalizer for plotting (optional)

        Returns:
            Training history
        """
        print(f"Starting training for {max_epochs} epochs")
        print(f"Model has {self.model.count_parameters()} trainable parameters")

        # Plot initial trajectories before training
        self.plot_trajectories(name="initial_trajectories")
        self.plot_trajectories_div(name="initial_trajectories_div")

        # Epoch-level progress bar
        pbar = tqdm(range(max_epochs), desc="Training Progress")
        epoch_times = []
        train_start_time = time.perf_counter()

        for epoch in pbar:
            self.current_epoch = epoch
            epoch_start = time.perf_counter()

            # Train (returns dict with loss and gradient stats)
            train_results = self.train_epoch()
            train_loss = train_results["loss"]
            train_pred_loss = train_results["pred_loss"]
            train_pred_loss_div = train_results.get("pred_loss_div")
            train_reg_feasibility = train_results["reg_feasibility"]
            epoch_rollback_count = train_results.get("rollback_count", 0)
            grad_stats = {
                k: v
                for k, v in train_results.items()
                if k not in [
                    "loss",
                    "pred_loss",
                    "pred_loss_div",
                    "reg_feasibility",
                    "reg_input",
                    "reg_output",
                    "reg_activity",
                    "reg_H",
                    "rollback_count",
                ]
            }

            # Epoch boundary: re-solve the certificate for the current theta so it
            # stays the tightest one this theta admits (rho ~ 1) instead of drifting.
            # Runs before validation/logging so every metric below reflects the
            # certificate the model actually carries into the next epoch.
            resynth_metrics = self._resynthesize_certificate(epoch)

            self.train_losses.append(train_loss)
            self.train_pred_losses.append(train_pred_loss)
            self.train_reg_feasibility.append(train_reg_feasibility)
            self.train_reg_inputs.append(train_results["reg_input"])
            if train_pred_loss_div is not None:
                self.train_div_losses.append(train_pred_loss_div)

            # Validate
            val_results = self.validate()
            val_loss = val_results["val_loss"]
            val_loss_div = val_results["val_loss_div"]
            assert val_loss is not None  # validate() always returns a converging val_loss
            self.val_losses.append(val_loss)

            # Synchronize CUDA so the wall-clock time reflects completed GPU work,
            # not just queued kernels.
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            epoch_time_sec = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time_sec)
            if val_loss_div is not None:
                self.val_div_losses.append(val_loss_div)
            # print(f'Epoch {epoch}: constraints satisfied={self.model.check_constraints()}')

            # Feasibility margins (min eigenvalue per LMI + scalar inequalities).
            # Diagnostic for interior-point drift: if min_eig climbs steadily while
            # the input/output violation terms worsen, the -log det barrier is
            # tugging the certificate params deeper into the interior.
            feas_margins = (
                self.model.get_feasibility_margins()
                if hasattr(self.model, "get_feasibility_margins")
                else {}
            )

            # Get scheduler patience info if using ReduceLROnPlateau
            scheduler_patience_info = ""
            if self.scheduler is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler_patience_info = (
                    f"{self.scheduler.num_bad_epochs}/{self.scheduler.patience}"
                )

            # Update progress bar with current metrics
            progress_metrics = {
                "train_loss": f"{train_loss:.4f}",
                "pred": f"{train_pred_loss:.4f}",
                "feas": f"{train_reg_feasibility:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "best_val": f"{self.best_val_loss:.4f}",
                "constraints": f"{self.model.check_constraints()}",
                "epoch_s": f"{epoch_time_sec:.2f}",
            }
            if "min_eig" in feas_margins:
                progress_metrics["min_eig"] = f"{feas_margins['min_eig']:.2e}"
            if "rho" in resynth_metrics:
                progress_metrics["rho"] = f"{resynth_metrics['rho']:.3g}"
            if train_pred_loss_div is not None:
                progress_metrics["pred_div"] = f"{train_pred_loss_div:.4f}"
            if val_loss_div is not None:
                progress_metrics["val_div"] = f"{val_loss_div:.4f}"
            if scheduler_patience_info:
                progress_metrics["scheduler_patience"] = scheduler_patience_info
            if self.log_gradients and grad_stats:
                # Compute total gradient norm from individual parameter norms
                total_grad_norm = np.sqrt(
                    sum(v**2 for k, v in grad_stats.items() if k.startswith("grad_norm/"))
                )
                progress_metrics["grad_norm"] = f"{total_grad_norm:.2e}"
            pbar.set_postfix(progress_metrics)

            # Log to MLflow
            if self.mlflow_tracking:
                # Loss metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_pred_loss", train_pred_loss, step=epoch)
                mlflow.log_metric("train_reg_feasibility", train_results["reg_feasibility"], step=epoch)
                # Feasibility margins: min eig per LMI + scalar inequalities (drift monitor)
                for margin_name, margin_value in feas_margins.items():
                    mlflow.log_metric(f"feas_margin/{margin_name}", margin_value, step=epoch)
                mlflow.log_metric("train_reg_input", train_results["reg_input"], step=epoch)
                if self.output_regularization_weight > 0:
                    mlflow.log_metric("train_reg_output", train_results["reg_output"], step=epoch)
                    mlflow.log_metric(
                        "output_regularization_weight",
                        self.output_regularization_weight,
                        step=epoch,
                    )
                if self.tightness_regularization_weight > 0:
                    mlflow.log_metric("train_reg_tightness", train_results["reg_tightness"], step=epoch)
                    # Tightness margin: relu-arg (output_std*s)^2 lambda_max(CPC^T) - y_max^2.
                    # ~0 means the certified image sits right at y_max (tight); >0 over-covers.
                    if hasattr(self.model, "get_regularization_tightness"):
                        with torch.no_grad():
                            _, tight_margin = self.model.get_regularization_tightness(
                                return_margin=True
                            )
                        mlflow.log_metric("output_tightness_margin", float(tight_margin), step=epoch)
                if self.activity_regularization_weight > 0:
                    mlflow.log_metric("train_reg_activity", train_results["reg_activity"], step=epoch)
                    # Coverage margin: lambda_max(y_max^2 I - (output_std*s)^2 CPC^T).
                    # <=0 means the certified output image covers the safe level.
                    if hasattr(self.model, "get_regularization_output"):
                        with torch.no_grad():
                            _, cov_margin = self.model.get_regularization_output(return_margin=True)
                        mlflow.log_metric("output_coverage_margin", float(cov_margin), step=epoch)
                if self.h_regularization_weight > 0:
                    mlflow.log_metric("train_reg_H", train_results["reg_H"], step=epoch)
                    # Current coupling norm ||H||_F (H = L P^-1): watch it climb
                    # toward h_target as the anti-global term takes effect.
                    if hasattr(self.model, "get_regularization_H"):
                        with torch.no_grad():
                            _, norm_H = self.model.get_regularization_H(
                                self.h_target if self.h_target > 0 else 1.0,
                                return_norm=True,
                            )
                        mlflow.log_metric("norm_H", float(norm_H), step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                if train_pred_loss_div is not None:
                    mlflow.log_metric("train_pred_loss_div", train_pred_loss_div, step=epoch)
                if val_loss_div is not None:
                    mlflow.log_metric("val_loss_div", val_loss_div, step=epoch)
                mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)
                mlflow.log_metric("epoch_time_sec", epoch_time_sec, step=epoch)
                if self.regularization_weight > 0:
                    mlflow.log_metric(
                        "regularization_weight", self.regularization_weight, step=epoch
                    )
                if self.input_regularization_weight > 0:
                    mlflow.log_metric(
                        "input_regularization_weight",
                        self.input_regularization_weight,
                        step=epoch,
                    )

                # Gradient statistics (if enabled)
                if self.log_gradients:
                    for stat_name, stat_value in grad_stats.items():
                        mlflow.log_metric(stat_name, stat_value, step=epoch)

                # Certificate re-synthesis diagnostics (rho, beta, applied/rejected/
                # failed counters, the re-synthesized s and ||P||).
                for name, value in resynth_metrics.items():
                    mlflow.log_metric(f"resynthesis/{name}", float(value), step=epoch)

                # Dead-zone activity: firing_rate == 0 means the nonlinearity is inert
                # on the data, i.e. the model is LTI in this regime and no gradient
                # reaches B2/C2/D21 (Delta'(z) = 0 in the band) — an absorbing state
                # that only the initialization can avoid. Watch it from epoch 0.
                if hasattr(self.model, "deadzone_activity"):
                    guard = self._get_guard_batch()
                    if guard is not None:
                        act = self.model.deadzone_activity(
                            guard[0], guard[1], warmup_steps=self.warmup_steps
                        )
                        for name, value in act.items():
                            mlflow.log_metric(f"deadzone/{name}", float(value), step=epoch)
                if self.repair_enforce_coverage:
                    mlflow.log_metric(
                        "coverage_repair_fallbacks",
                        float(self.epoch_coverage_repair_fallbacks),
                        step=epoch,
                    )

                if isinstance(self.model, SimpleLure):
                    alpha = 1/(1+ np.exp(-self.model.tau.cpu().detach().numpy()))
                    mlflow.log_metric("s", self.model.s.item(), step=epoch)
                    mlflow.log_metric("alpha", alpha, step=epoch)
                    s = float(self.model.s.cpu().detach().numpy())
                    P = self.model.P.cpu().detach().numpy()
                    vol_X = get_volume_of_ellipsoid(P, s)
                    mlflow.log_metric("vol_X", vol_X, step=epoch)
                    logging.debug(f"Epoch {epoch}: s={s:.6f}, ||P|| = {np.linalg.norm(P):.6f}, vol(Xc)={vol_X:.6e}, alpha={alpha:.6f}")

            # Plot trajectories and ellipse periodically (at checkpoint frequency)
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.plot_trajectories(name=f"epoch_{epoch}", normalizer=normalizer)
                self.plot_trajectories_div(
                    name=f"epoch_{epoch}_div", normalizer=normalizer
                )

            # Learning rate scheduling
            prev_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Decay regularization weight when learning rate is reduced (Interior Point Method)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr < prev_lr:
                self.decay_regularization()

            # If ALL batches rolled back this epoch, reduce learning rate and regularization
            if epoch_rollback_count >= len(self.train_loader):  # 100% of batches rolled back
                pbar.write(
                    f"\n⚠ All batches rolled back ({epoch_rollback_count}/{len(self.train_loader)}), reducing LR and regularization"
                )
                self.reduce_lr_on_rollback(
                    factor=self.optimizer.param_groups[0].get("lr_reduction_factor", 0.5)
                )
                self.decay_regularization()

            # Log rollback count to MLflow
            if self.mlflow_tracking:
                mlflow.log_metric("rollback_count", epoch_rollback_count, step=epoch)
                mlflow.log_metric("total_rollbacks", self.rollback_count, step=epoch)

            # Save checkpoint
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Early stopping checks
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch  # Track the best epoch
                self.patience_counter = 0
                # print(self.model.check_constraints())
                self.save_checkpoint("best_model.pt")
                pbar.write(f"✓ Epoch {epoch}: New best model (val_loss={val_loss:.6f})")
            else:
                self.patience_counter += 1
                # if self.patience_counter >= self.early_stopping_patience:
                # pbar.write(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                # break

            # Early stopping based on regularization weight threshold
            if (
                self.decay_regularization_weight
                and self.min_regularization_weight > 0
                and self.regularization_weight <= self.min_regularization_weight
            ):
                pbar.write(
                    f"\n⚠ Early stopping: Regularization weight reached minimum threshold ({self.min_regularization_weight:.2e})"
                )
                pbar.write(f"   Training has converged after {epoch + 1} epochs")
                break

        # Close progress bar
        pbar.close()

        total_train_time = time.perf_counter() - train_start_time
        if epoch_times:
            mean_epoch = float(np.mean(epoch_times))
            median_epoch = float(np.median(epoch_times))
            # Use last-half mean to exclude warmup epochs (SDP init, JIT, cache fills).
            steady_epoch = float(np.mean(epoch_times[len(epoch_times) // 2 :]))
            print("\n=== Timing summary ===")
            print(f"Device:                 {self.device}")
            print(f"Epochs run:             {len(epoch_times)}")
            print(f"Total wall time (s):    {total_train_time:.2f}")
            print(f"Mean epoch (s):         {mean_epoch:.3f}")
            print(f"Median epoch (s):       {median_epoch:.3f}")
            print(f"Mean epoch, last half:  {steady_epoch:.3f}")
            if self.mlflow_tracking:
                mlflow.log_metric("total_train_time_sec", total_train_time)
                mlflow.log_metric("mean_epoch_time_sec", mean_epoch)
                mlflow.log_metric("median_epoch_time_sec", median_epoch)
                mlflow.log_metric("steady_state_epoch_time_sec", steady_epoch)

        # Save final model
        self.save_checkpoint("final_model.pt")

        # Store ellipse parameters for SimpleLure models
        if isinstance(self.model, SimpleLure):
            X = np.linalg.inv(self.model.P.cpu().detach().numpy())
            H = self.model.L.cpu().detach().numpy() @ X
            s = self.model.s.cpu().detach().numpy()
            max_norm_x0 = self.model.max_norm_x0
            #     finally:
            #         plt.close(fig)

        # Save training history
        history = {
            "train_losses": self.train_losses,
            "train_pred_losses": self.train_pred_losses,
            "train_reg_feasibility": self.train_reg_feasibility,
            "val_losses": self.val_losses,
            "train_div_losses": self.train_div_losses,
            "val_div_losses": self.val_div_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "final_epoch": self.current_epoch,
            "epoch_times_sec": epoch_times,
            "total_time": total_train_time,
        }

        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.model_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "train_pred_losses": self.train_pred_losses,
            "train_reg_feasibility": self.train_reg_feasibility,
            "train_reg_inputs": self.train_reg_inputs,
            "val_losses": self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        # Save model parameters as .mat file for best model
        if "best" in filename:
            self.save_parameters_mat(filename.replace(".pt", "_params.mat"))

        # Checkpoint files are uploaded in bulk by train.py via
        # log_artifacts(run_model_dir, "models") at the end of training, so we
        # don't log_artifact here — that would create a duplicate model/ folder
        # alongside the models/ one from the bulk upload.

    def save_parameters_mat(self, filename: str):
        """
        Save model parameters as MATLAB .mat file.

        Args:
            filename: Name of the .mat file (e.g., 'best_model_params.mat')
        """
        mat_path = self.model_dir / filename

        # Extract model parameters from state_dict and convert to numpy
        params_dict = {}
        for name, param in self.model.state_dict().items():
            # Convert parameter name to MATLAB-compatible format (replace dots with underscores)
            mat_name = name.replace(".", "_")
            # Convert tensor to numpy array
            params_dict[mat_name] = param.cpu().numpy()

        # Add metadata
        params_dict["_metadata"] = {
            "epoch": self.current_epoch,
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": self.best_epoch,
            "model_type": self.model.__class__.__name__,
        }

        # Save to .mat file
        savemat(mat_path, params_dict)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint.get("best_epoch", 0)  # Use .get() for backward compatibility
        self.train_losses = checkpoint.get("train_losses", [])
        self.train_pred_losses = checkpoint.get("train_pred_losses", [])
        self.train_reg_feasibility = checkpoint.get("train_reg_feasibility", [])
        self.train_reg_inputs = checkpoint.get("train_reg_inputs", [])
        self.val_losses = checkpoint.get("val_losses", [])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
