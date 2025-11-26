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
import logging
import copy
import matplotlib.pyplot as plt

# Set default dtype to float64 to match numpy's default and avoid type clashes
# torch.set_default_dtype(torch.float64)

from ..models.base import BaseRNN
from ..models.constrained_rnn import SimpleLure
from .losses import get_loss_function
from .optimizers import get_optimizer, get_scheduler
from ..evaluation.evaluator import Evaluator
from ..utils import plot_ellipse_and_parallelogram
from ..utils import plot_predictions

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
        self.train_losses = []
        self.train_pred_losses = []
        self.train_reg_losses = []
        self.val_losses = []
        
        # Scheduler (can be set later)
        self.scheduler = None
    
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
        temp_output_dir = self.output_dir / f'predictions'
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator = Evaluator(
            model=self.model,
            device=self.device,
            output_dir=str(temp_output_dir)
        )
        
        # Evaluate on validation set
        # try:
        results = evaluator.evaluate(
            test_loader=self.val_loader,
            normalizer=normalizer,
            print_results=False,
            save_files=False  # Don't save prediction files during training
        )
        
        e_hat = results['e_hat']
        e = results['e']
        d = results.get('inputs', None)
        
        # Select sample indices: always include sequence 0, plus 2 random sequences
        num_sequences = e_hat.shape[0]
        sample_indices = [0]  # Always include sequence 0
        
        if num_sequences > 1:
            # Select 2 random sequences (excluding sequence 0)
            other_indices = list(range(1, num_sequences))
            num_random = min(2, len(other_indices))
            random_indices = np.random.choice(other_indices, size=num_random, replace=False).tolist()
            sample_indices.extend(random_indices)
        
        # Generate plot
        plot_path = temp_output_dir / f'{name}.png'
        plot_predictions(output_dir=evaluator.output_dir,
            e_hat=e_hat,
            e=e,
            d=d,
            sample_indices=sample_indices,
            save_path=str(plot_path)
        )
        
        # Log to MLflow
        if self.mlflow_tracking:
            mlflow.log_artifact(str(plot_path), artifact_path="predictions")
            # print(f"✓ Trajectory plot logged to MLflow")
        
        # print(f"✓ Trajectory plot saved: {plot_path}")
            
        # except Exception as e:
        #     print(f"⚠ Failed to generate initial trajectory plot: {e}")
        #     logging.warning(f"Failed to generate initial trajectory plot: {e}")
    
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
            # Ensure we don't go below minimum threshold
            if self.regularization_weight < self.min_regularization_weight:
                self.regularization_weight = self.min_regularization_weight
            print(f"  Regularization weight decayed: {old_weight:.6e} → {self.regularization_weight:.6e}")
    
    def reduce_lr_on_rollback(self, factor: float = 0.5):
        """
        Reduce learning rate when rollbacks occur frequently.
        This helps when the optimizer step is too large for the constrained space.
        
        Args:
            factor: Factor to multiply learning rate by (default: 0.5)
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] *= factor
            new_lr = param_group['lr']
            print(f"  Learning rate reduced due to rollbacks: {old_lr:.6e} → {new_lr:.6e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training loss, prediction loss, regularization loss, and gradient statistics
        """
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        # Reset epoch rollback counter
        self.epoch_rollback_count = 0
        
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
            
            # Compute prediction loss
            pred_loss = self.loss_fn(e_hat, e)
            loss = pred_loss
            
            # Add custom regularization
            reg_loss_value = 0.0
            if self.regularization_weight > 0:
                reg_loss = self.model.get_regularization_loss()
                reg_loss_value = reg_loss.item()
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
            
            # Save parameter state before update (for constrained models)
            if isinstance(self.model, SimpleLure):
                saved_state = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
            
            # Update weights
            self.optimizer.step()
            # logging.info(f"B:{batch_idx}: s= {self.model.s.detach().numpy():.4f}, alpha= {self.model.alpha.detach().numpy():.4f}")

            # Check if constraints are satisfied (for constrained models)
            if isinstance(self.model, SimpleLure):
                if not self.model.check_constraints():
                    # Constraints violated - try to solve feasibility SDP
                    # logging.info(f"Batch {batch_idx}: Constraints violated after parameter update, attempting feasibility SDP...")
                    # if self.model.alpha > 1:
                    #     self.model.alpha.data = torch.tensor(0.99)
                    
                    # logging.info(f"B:{batch_idx}: s= {self.model.s.detach().numpy():.4f}, alpha= {self.model.alpha.detach().numpy():.4f}")
                    b_feasible = self.model.analysis_problem()
                    
                    if not b_feasible:
                        # SDP failed - roll back to previous parameters
                        logging.warning(f"Batch {batch_idx}: Feasibility SDP failed, rolling back parameters")
                        
                        # Restore saved parameters
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if param.requires_grad and name in saved_state:
                                    param.data.copy_(saved_state[name])
                        
                        # Track rollbacks
                        self.rollback_count += 1
                        self.epoch_rollback_count += 1
                        
                        logging.info(f"Batch {batch_idx}: Parameters rolled back successfully (total: {self.rollback_count})")
                    # else:
                        # logging.info(f"Batch {batch_idx}: Feasibility SDP succeeded, parameters updated")




            # Update metrics
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_reg_loss += reg_loss_value
            num_batches += 1
        
        # Average loss
        avg_loss = total_loss / num_batches
        avg_pred_loss = total_pred_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        # Average gradient statistics over epoch
        avg_grad_stats = {
            key: np.mean(values) for key, values in epoch_grad_stats.items()
        }
        
        return {
            'loss': avg_loss,
            'pred_loss': avg_pred_loss,
            'reg_loss': avg_reg_loss,
            'rollback_count': self.epoch_rollback_count,
            **avg_grad_stats
        }
    
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
                e_hat = self.model(d, x)  # e_hat: predicted output
                
                # Compute loss
                loss = self.loss_fn(e_hat, e)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
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
        self.plot_trajectories(normalizer=normalizer, name="initial_trajectories")
        
        # Epoch-level progress bar
        pbar = tqdm(range(max_epochs), desc="Training Progress")
        
        for epoch in pbar:
            self.current_epoch = epoch
            
            # Train (returns dict with loss and gradient stats)
            train_results = self.train_epoch()
            train_loss = train_results['loss']
            train_pred_loss = train_results['pred_loss']
            train_reg_loss = train_results['reg_loss']
            epoch_rollback_count = train_results.get('rollback_count', 0)
            grad_stats = {k: v for k, v in train_results.items() if k not in ['loss', 'pred_loss', 'reg_loss', 'rollback_count']}
            
            self.train_losses.append(train_loss)
            self.train_pred_losses.append(train_pred_loss)
            self.train_reg_losses.append(train_reg_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Get scheduler patience info if using ReduceLROnPlateau
            scheduler_patience_info = ""
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_patience_info = f"{self.scheduler.num_bad_epochs}/{self.scheduler.patience}"
            
            # Update progress bar with current metrics
            progress_metrics = {
                "train_loss": f"{train_loss:.4f}",
                "pred": f"{train_pred_loss:.4f}",
                "reg": f"{train_reg_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "best_val": f"{self.best_val_loss:.4f}",
                "constraints": f"{self.model.check_constraints()}"
            }
            if scheduler_patience_info:
                progress_metrics["scheduler_patience"] = scheduler_patience_info
            if self.log_gradients and grad_stats:
                # Compute total gradient norm from individual parameter norms
                total_grad_norm = np.sqrt(sum(v**2 for k, v in grad_stats.items() if k.startswith('grad_norm/')))
                progress_metrics["grad_norm"] = f"{total_grad_norm:.2e}"
            pbar.set_postfix(progress_metrics)
            
            # Log to MLflow
            if self.mlflow_tracking:
                # Loss metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_pred_loss", train_pred_loss, step=epoch)
                mlflow.log_metric("train_reg_loss", train_reg_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)
                if self.regularization_weight > 0:
                    mlflow.log_metric("regularization_weight", self.regularization_weight, step=epoch)
                
                # Gradient statistics (if enabled)
                if self.log_gradients:
                    for stat_name, stat_value in grad_stats.items():
                        mlflow.log_metric(stat_name, stat_value, step=epoch)

                if isinstance(self.model, SimpleLure):
                    mlflow.log_metric("s", self.model.s.item(), step=epoch)
                    mlflow.log_metric("alpha", self.model.alpha.item(), step=epoch)
            
            # Plot trajectories and ellipse periodically (at checkpoint frequency)
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.plot_trajectories(name=f"epoch_{epoch}", normalizer=normalizer)
                
                if isinstance(self.model, SimpleLure) and self.mlflow_tracking:
                    X = np.linalg.inv(self.model.P.cpu().detach().numpy())
                    H = self.model.L.cpu().detach().numpy() @ X
                    s = self.model.s.cpu().detach().numpy()
                    max_norm_x0 = self.model.max_norm_x0
                    fig, ax = plot_ellipse_and_parallelogram(X, H, s, max_norm_x0)
                    
                    try:
                        # Log directly to MLflow without saving to output_dir
                        mlflow.log_figure(fig, f'plots/ellipse_epoch_{epoch}.png')
                    except Exception as e:
                        logging.warning(f"Failed to log ellipse plot: {e}")
                    finally:
                        plt.close(fig)
            
            
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
                pbar.write(f"\n⚠ All batches rolled back ({epoch_rollback_count}/{len(self.train_loader)}), reducing LR and regularization")
                self.reduce_lr_on_rollback(factor=self.optimizer.param_groups[0].get('lr_reduction_factor', 0.5))
                self.decay_regularization()
                if isinstance(self.model, SimpleLure):
                    self.model.reset_s()
                    pbar.write(f"   Reset s escape local minimum")

                
            # Log rollback count to MLflow
            if self.mlflow_tracking:
                mlflow.log_metric("rollback_count", epoch_rollback_count, step=epoch)
                mlflow.log_metric("total_rollbacks", self.rollback_count, step=epoch)
            
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
            
            # Early stopping checks
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
            
            # Early stopping based on regularization weight threshold
            if self.decay_regularization_weight and self.min_regularization_weight > 0 and self.regularization_weight <= self.min_regularization_weight:
                pbar.write(f"\n⚠ Early stopping: Regularization weight reached minimum threshold ({self.min_regularization_weight:.2e})")
                pbar.write(f"   Training has converged after {epoch + 1} epochs")
                break
        
        # Close progress bar
        pbar.close()
        
        # Save final model
        self.save_checkpoint("final_model.pt")

        # Plot and log final ellipse for SimpleLure models
        if isinstance(self.model, SimpleLure):
            X = np.linalg.inv(self.model.P.cpu().detach().numpy())
            H = self.model.L.cpu().detach().numpy() @ X
            s = self.model.s.cpu().detach().numpy()
            max_norm_x0 = self.model.max_norm_x0
            fig, ax = plot_ellipse_and_parallelogram(X, H, s, max_norm_x0)
            
            # Save and log to MLflow
            if self.mlflow_tracking:
                try:
                    # Log directly to MLflow without saving to output_dir
                    mlflow.log_figure(fig, 'plots/final_ellipse.png')
                    logging.info("Logged final ellipse plot to MLflow artifacts")
                except Exception as e:
                    logging.warning(f"Failed to log final ellipse plot: {e}")
                finally:
                    plt.close(fig)
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "train_pred_losses": self.train_pred_losses,
            "train_reg_losses": self.train_reg_losses,
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
            "train_pred_losses": self.train_pred_losses,
            "train_reg_losses": self.train_reg_losses,
            "val_losses": self.val_losses,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save model parameters as .mat file for best model
        if "best" in filename:
            self.save_parameters_mat(filename.replace(".pt", "_params.mat"))
        
        # Only log model to MLflow for final model (not every best model update)
        # This avoids slowdown from logging the model every time validation improves
        if self.mlflow_tracking and "final" in filename:
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
        self.train_pred_losses = checkpoint.get("train_pred_losses", [])
        self.train_reg_losses = checkpoint.get("train_reg_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
