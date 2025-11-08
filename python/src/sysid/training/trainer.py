"""Main trainer class for RNN-based system identification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from typing import Optional, Dict, Any
import mlflow
from scipy.io import savemat

from ..models.base import BaseRNN
from .losses import get_loss_function
from .optimizers import get_optimizer, get_scheduler


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
        checkpoint_frequency: int = 10,
        early_stopping_patience: int = 50,
        mlflow_tracking: bool = True,
        log_gradients: bool = True,
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
            checkpoint_frequency: Save checkpoint every N epochs
            early_stopping_patience: Patience for early stopping
            mlflow_tracking: Whether to use MLflow tracking
            log_gradients: Whether to log gradient statistics
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        
        # Logging
        self.mlflow_tracking = mlflow_tracking
        self.log_gradients = log_gradients
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0  # Track which epoch had the best validation loss
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        # Scheduler (can be set later)
        self.scheduler = None
    
    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler
    
    def compute_gradient_stats(self) -> Dict[str, float]:
        """
        Compute comprehensive gradient statistics.
        
        Returns:
            Dictionary with gradient statistics:
            - grad_norm_total: Global gradient norm (L2)
            - grad_norm_max: Maximum gradient norm across parameters
            - grad_norm_min: Minimum gradient norm across parameters
            - grad_value_max: Largest gradient value (absolute)
            - grad_value_min: Smallest gradient value (absolute)
            - grad_mean: Mean gradient value
            - grad_std: Standard deviation of gradients
            - dead_params_ratio: Ratio of parameters with zero gradient
            - grad_param_ratio_mean: Mean ratio of gradient norm to parameter norm
            - grad_param_ratio_max: Max ratio of gradient norm to parameter norm
        """
        stats = {}
        
        # Collect all gradients and parameters
        all_grads = []
        all_params = []
        param_grad_norms = []
        param_norms = []
        dead_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                param_val = param.detach()
                
                # Collect flattened gradients
                all_grads.append(grad.flatten())
                all_params.append(param_val.flatten())
                
                # Per-parameter norms
                grad_norm = grad.norm().item()
                param_norm = param_val.norm().item()
                
                param_grad_norms.append(grad_norm)
                param_norms.append(param_norm)
                
                # Check for dead parameters
                if grad_norm < 1e-10:
                    dead_params += 1
                total_params += 1
        
        if len(all_grads) > 0:
            # Concatenate all gradients
            all_grads_tensor = torch.cat(all_grads)
            all_params_tensor = torch.cat(all_params)
            
            # Global statistics
            stats['grad_norm_total'] = all_grads_tensor.norm().item()
            stats['grad_value_max'] = all_grads_tensor.abs().max().item()
            stats['grad_value_min'] = all_grads_tensor.abs().min().item()
            stats['grad_mean'] = all_grads_tensor.mean().item()
            stats['grad_std'] = all_grads_tensor.std().item()
            
            # Per-parameter statistics
            if len(param_grad_norms) > 0:
                stats['grad_norm_max'] = max(param_grad_norms)
                stats['grad_norm_min'] = min(param_grad_norms)
                
                # Gradient-to-parameter ratio (indicates update size)
                grad_param_ratios = []
                for g_norm, p_norm in zip(param_grad_norms, param_norms):
                    if p_norm > 1e-10:  # Avoid division by zero
                        grad_param_ratios.append(g_norm / p_norm)
                
                if len(grad_param_ratios) > 0:
                    stats['grad_param_ratio_mean'] = np.mean(grad_param_ratios)
                    stats['grad_param_ratio_max'] = max(grad_param_ratios)
            
            # Dead parameters
            stats['dead_params_ratio'] = dead_params / total_params if total_params > 0 else 0.0
        
        return stats
    
    def decay_regularization(self):
        """
        Decay regularization weight (Interior Point Method).
        
        In interior point methods for convex optimization, the barrier parameter
        is reduced as we approach the solution. Here we decay the regularization
        weight whenever the learning rate is reduced.
        """
        if self.decay_regularization_weight and self.regularization_weight > 0:
            old_weight = self.regularization_weight
            self.regularization_weight *= self.regularization_decay_factor
            print(f"  Regularization weight decayed: {old_weight:.6e} → {self.regularization_weight:.6e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training loss and gradient statistics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Accumulate gradient stats over epoch
        epoch_grad_stats = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch (states may be None)
            if len(batch) == 3:
                d, e, x0 = batch  # d: input, e: output, x: states (optional)
            else:
                d, e = batch
                x0 = None
            d = d.to(self.device)
            e = e.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            e_hat = self.model(d, x0=x0)  # e_hat: predicted output
            
            # Compute loss
            loss = self.loss_fn(e_hat, e)
            
            # Add custom regularization
            if self.regularization_weight > 0:
                reg_loss = self.model.get_regularization_loss()
                loss = loss + self.regularization_weight * reg_loss
            
            # Backward pass
            loss.backward()
            
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
        
        # Average loss
        avg_loss = total_loss / num_batches
        
        # Average gradient statistics over epoch
        avg_grad_stats = {
            key: np.mean(values) for key, values in epoch_grad_stats.items()
        }
        
        return {'loss': avg_loss, **avg_grad_stats}
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch (states may be None)
                if len(batch) == 3:
                    d, e, x = batch  # d: input, e: output, x: states (optional)
                else:
                    d, e = batch
                    x = None
                
                d = d.to(self.device)
                e = e.to(self.device)
                
                # Forward pass
                e_hat = self.model(d)  # e_hat: predicted output
                
                # Compute loss
                loss = self.loss_fn(e_hat, e)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, max_epochs: int) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            max_epochs: Maximum number of epochs
            
        Returns:
            Training history
        """
        print(f"Starting training for {max_epochs} epochs")
        print(f"Model has {self.model.count_parameters()} trainable parameters")
        
        # Epoch-level progress bar
        pbar = tqdm(range(max_epochs), desc="Training Progress")
        
        for epoch in pbar:
            self.current_epoch = epoch
            
            # Train (returns dict with loss and gradient stats)
            train_results = self.train_epoch()
            train_loss = train_results['loss']
            grad_stats = {k: v for k, v in train_results.items() if k != 'loss'}
            
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update progress bar with current metrics
            progress_metrics = {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "best_val": f"{self.best_val_loss:.4f}",
                "patience": f"{self.patience_counter}/{self.early_stopping_patience}",
                "constraints": f"{self.model.check_constraints()}"
            }
            if self.log_gradients and grad_stats:
                progress_metrics["grad_norm"] = f"{grad_stats.get('grad_norm_total', 0):.2e}"
            pbar.set_postfix(progress_metrics)
            
            # Log to MLflow
            if self.mlflow_tracking:
                # Loss metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)
                if self.regularization_weight > 0:
                    mlflow.log_metric("regularization_weight", self.regularization_weight, step=epoch)
                
                # Gradient statistics (if enabled)
                if self.log_gradients:
                    for stat_name, stat_value in grad_stats.items():
                        mlflow.log_metric(f"grad/{stat_name}", stat_value, step=epoch)
            
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
            
            # Update dual penalty coefficient if using dual method
            if hasattr(self.model, 'update_dual_penalty') and hasattr(self.model, 'regularization_method'):
                if self.model.regularization_method == "dual":
                    constraints_satisfied = self.model.check_constraints()
                    self.model.update_dual_penalty(constraints_satisfied)
                    
                    # Log dual penalty and constraint violation to MLflow
                    if self.mlflow_tracking:
                        mlflow.log_metric("dual_penalty", self.model.dual_penalty.item(), step=epoch)
                        if hasattr(self.model, 'get_constraint_violation'):
                            violation = self.model.get_constraint_violation()
                            mlflow.log_metric("constraint_violation", violation, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch  # Track the best epoch
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                pbar.write(f"✓ Epoch {epoch}: New best model (val_loss={val_loss:.6f})")
            else:
                self.patience_counter += 1
                # if self.patience_counter >= self.early_stopping_patience:
                    # pbar.write(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                    # break
        
        # Close progress bar
        pbar.close()
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "final_epoch": self.current_epoch,
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
            "val_losses": self.val_losses,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save model parameters as .mat file for best model
        if "best" in filename:
            self.save_parameters_mat(filename.replace(".pt", "_params.mat"))
        
        # Log model to MLflow (fixes: artifact_path deprecation + signature warnings)
        if self.mlflow_tracking and "best" in filename:
            try:
                # Get a sample batch from train_loader for input example
                sample_batch = next(iter(self.train_loader))
                if isinstance(sample_batch, (tuple, list)):
                    if len(sample_batch) == 3:
                        sample_input, _, _ = sample_batch  # (input, output, initial_state)
                    elif len(sample_batch) >= 2:
                        sample_input, _ = sample_batch  # (input, output)
                    else:
                        sample_input = sample_batch[0]
                else:
                    sample_input = sample_batch
                
                # Move to correct device and get a single sample
                sample_input = sample_input[:1].to(self.device)
                
                # Log model with input example (auto-generates signature)
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    name="model",
                    input_example=sample_input.cpu().numpy()
                )
            except Exception as e:
                # Fallback: log without input example if something goes wrong
                print(f"Warning: Could not create input example for model signature: {e}")
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    name="model"
                )
    
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
            mat_name = name.replace('.', '_')
            # Convert tensor to numpy array
            params_dict[mat_name] = param.cpu().numpy()
        
        # Add metadata
        params_dict['_metadata'] = {
            'epoch': self.current_epoch,
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': self.best_epoch,
            'model_type': self.model.__class__.__name__
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
        self.val_losses = checkpoint.get("val_losses", [])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
