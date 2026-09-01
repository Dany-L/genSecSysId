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
        solve_max_s_on_violation: bool = False,
        activity_regularization_weight: float = 0.0,
        activity_target: float = 0.0,
        h_regularization_weight: float = 0.0,
        h_target: float = 0.0,
        output_std: float = 1.0,
        train_div_loader: Optional[DataLoader] = None,
        val_div_loader: Optional[DataLoader] = None,
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
        # Re-solve MaxS after an epoch whose training data breached the input
        # condition — see _maybe_maximize_s and TrainingConfig.
        self.solve_max_s_on_violation = solve_max_s_on_violation
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
        # Physical output scale — relates the model's normalized C/P/s to the
        # physical y_max it records for reporting.
        self.output_std = float(output_std)

        # Cached batch for the per-epoch dead-zone diagnostic
        self._diag_batch: Optional[tuple] = None

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

        # Record the (physical) data level y_max on the model. Nothing in training
        # constrains the certificate to reach it — the model class provably cannot
        # (see the wiki note certificate-synthesis/ellipsoidal-conservatism) — it is
        # kept so rho = (y_bar/y_max)^nx can be REPORTED per epoch and at post-process.
        self._record_output_level()

        # Input floor u_max = sup_k ||u_k||^2 over the training inputs, so every solve
        # that chooses s respects s >= sqrt(u_max) (necessary for the input condition).
        self._init_input_bound()

    def _record_output_level(self):
        """Store the PHYSICAL data level ``y_max`` on the model (fallback path).

        Diagnostic only. Skipped when the model already has ``y_max``: in the
        normal pipeline ``initialize_parameters`` sets both ``y_max`` and
        ``output_std`` from the raw data + normalizer, so this only fires for
        directly-constructed / loaded models. The loader yields *normalized*
        targets, so ``max |e| · output_std`` is the physical ``y_max``.
        """
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
            logging.info(f"Data output level y_max recorded (reporting only): {y_max_phys:.6f}")

    def _init_input_bound(self):
        """Set the model's input floor ``u_max = sup_k ‖u_k‖²`` from the loader.

        The input condition forces ``s² ≥ ‖u_k‖²`` for every sample regardless of
        ``P`` (the quadratic form is non-negative), so this is the smallest scale
        the data admits. Skipped when the model already has one (initialization
        normally sets it) or does not support it."""
        if not hasattr(self.model, "set_input_bound"):
            return
        u = getattr(self.model, "u_max", None)
        if u is not None and not bool(torch.isnan(u)):
            return
        peak = 0.0
        for batch in self.train_loader:
            d = batch[0]
            d = d[torch.isfinite(d)].reshape(-1, d.shape[-1]) if d.numel() else d
            if d.numel():
                peak = max(peak, float((d ** 2).sum(dim=-1).max()))
        if peak > 0.0:
            self.model.set_input_bound(peak)
            logging.info(
                f"Input floor u_max = {peak:.6g} from the training inputs -> s >= {peak ** 0.5:.4g}"
            )

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

    def _diagnostic_batch(self):
        """One cached training batch ``(d, x0)`` for the per-epoch dead-zone report."""
        if self._diag_batch is None:
            for batch in self.train_loader:
                d = batch[0].to(self.device)
                x0 = (
                    batch[2].to(device=self.device, dtype=d.dtype)
                    if len(batch) == 3 and batch[2] is not None else None
                )
                self._diag_batch = (d, x0)
                break
        return self._diag_batch

    def _repair_certificate(self) -> bool:
        """Certificate repair after a step that broke the LMIs. ``False`` ⇒ roll back.

        θ and α stay exactly where the optimizer put them; only the certificate
        (P, L, Λ, and ``s`` if needed) is re-solved — see
        :meth:`SimpleLure.feasibility_problem`. This is the whole constraint
        machinery: barrier during the step, repair-or-rollback after it.
        """
        return self.model.feasibility_problem()

    def _maybe_maximize_s(self, epoch: int) -> Optional[float]:
        """Re-solve MaxS (once per epoch) if the training data breaches the input
        condition. Returns the new ``s``, or ``None`` if nothing was solved.

        Two steps, so the SDP is decoupled from the mini-batching:

        1. Scan *all* training batches (no grad) for the peak input-constraint
           margin ``c_k = ||u_k||^2 - s^2 + alpha^2 V(x_k)``. That says whether
           the currently certified set still covers the whole training set.
        2. Only if it is breached somewhere (``max_k c_k > 0``) solve MaxS once.
           The SDP fixes theta and optimizes (P, L, Lambda, s), so a single solve
           re-certifies an enlarged ``s`` for the entire dataset.

        Why it matters: ``s`` is learnable and the log-barrier only ever pushes it
        DOWN. Nothing pushes back, so without this step ``s`` decays below the
        input floor ``sqrt(u_max)``, every optimizer step then lands outside the
        feasible set, and the per-batch repair SDP fires on every batch and mostly
        fails into a rollback. Restored from cf9cb54, where it ran on 820 of 1500
        epochs and held ``s`` near the floor with zero rollbacks.

        No-op for non-SimpleLure models.
        """
        if not isinstance(self.model, SimpleLure):
            return None

        self.model.eval()
        try:
            c_max = float("-inf")
            with torch.no_grad():
                for batch in self.train_loader:
                    d = batch[0].to(self.device)
                    _, (x, _), _ = self.model(d, x0=None, warmup_steps=self.warmup_steps)
                    _, c = self.model.get_regularization_input(d, x, return_c=True)
                    # NaN positions come from padded trajectories; ignore them.
                    c_max = max(c_max, float(torch.nan_to_num(c, nan=float("-inf")).max()))

            if c_max <= 0:
                return None

            sol = self.model._synth().max_s()
            if sol is None:
                logging.warning(
                    f"Epoch {epoch}: input condition violated (max c={c_max:.6f}) "
                    "but the MaxS SDP failed; leaving s unchanged."
                )
                return None

            s_before = float(self.model.s)
            self.model._apply_certificate_solution(sol)
            new_s = float(self.model.s)
            logging.info(
                f"Epoch {epoch}: input condition violated (max c={c_max:.6f}), "
                f"solved MaxS, s {s_before:.6f} -> {new_s:.6f}"
            )
            if self.mlflow_tracking:
                mlflow.log_metric("max_s_solved", 1, step=epoch)
                mlflow.log_metric("max_s_value", new_s, step=epoch)
            return new_s
        finally:
            self.model.train()

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
        total_reg_feasibility = 0.0
        total_reg_inputs = 0.0
        total_reg_activity = 0.0
        total_reg_H = 0.0
        num_batches = 0

        # Reset epoch rollback counter
        self.epoch_rollback_count = 0

        # Accumulate gradient stats over epoch
        epoch_grad_stats: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # WASHOUT INITIALIZATION — x0 is discarded on purpose.
            #
            # The loader carries the recorded initial state, but at deployment
            # there is no access to it, so the model must be trained the way it
            # will be used: rolled out from x0 = 0 and allowed to synchronize to
            # the input. ``warmup_steps`` then excludes the washout transient from
            # the loss, which is why it has to be long enough for the transient to
            # decay (for the Duffing reference rho(A) = 0.9937, so 500 steps leaves
            # ~4% of it). Feeding the true x0 here would train against information
            # the deployed model does not have.
            #
            # Diverging trajectories are recorded from x0 = 0 anyway, so the same
            # line in _train_diverging_epoch is a no-op rather than a washout.
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
            reg_feasibility_value = 0.0
            reg_input_value = 0.0
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
            total_reg_activity += reg_activity_value
            total_reg_H += reg_H_value
            num_batches += 1

        # Average loss
        avg_loss = total_loss / num_batches
        avg_pred_loss = total_pred_loss / num_batches
        avg_reg_feasibility = total_reg_feasibility / num_batches
        avg_reg_inputs = total_reg_inputs / num_batches
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
                    "reg_activity",
                    "reg_H",
                    "rollback_count",
                ]
            }

            # Coverage ratio rho = (y_bar/y_max)^nx — REPORTED, never enforced.
            # A couple of matrix products on parameters already in memory, so it is
            # free; None when y_max is unset.
            rho = (
                self.model.coverage_ratio()
                if hasattr(self.model, "coverage_ratio") else None
            )

            self.train_losses.append(train_loss)
            self.train_pred_losses.append(train_pred_loss)
            self.train_reg_feasibility.append(train_reg_feasibility)
            self.train_reg_inputs.append(train_results["reg_input"])
            if train_pred_loss_div is not None:
                self.train_div_losses.append(train_pred_loss_div)

            # Repair s: if the training inputs breached the input condition this
            # epoch, re-solve MaxS so the certified set covers them again. Runs
            # before validation so val reflects the updated certificate.
            if self.solve_max_s_on_violation:
                self._maybe_maximize_s(epoch)

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
            if rho is not None:
                progress_metrics["rho"] = f"{rho:.3g}"
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
                if self.activity_regularization_weight > 0:
                    mlflow.log_metric("train_reg_activity", train_results["reg_activity"], step=epoch)
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
                if rho is not None:
                    mlflow.log_metric("rho", float(rho), step=epoch)
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

                # Dead-zone activity: firing_rate == 0 means the nonlinearity is inert
                # on the data, i.e. the model is LTI in this regime and no gradient
                # reaches B2/C2/D21 (Delta'(z) = 0 in the band) — an absorbing state
                # that only the initialization can avoid. Watch it from epoch 0.
                if hasattr(self.model, "deadzone_activity"):
                    diag = self._diagnostic_batch()
                    if diag is not None:
                        act = self.model.deadzone_activity(
                            diag[0], diag[1], warmup_steps=self.warmup_steps
                        )
                        for name, value in act.items():
                            mlflow.log_metric(f"deadzone/{name}", float(value), step=epoch)
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
