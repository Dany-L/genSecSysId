"""Main trainer class for RNN-based system identification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Dict, Any
import mlflow

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
        checkpoint_frequency: int = 10,
        early_stopping_patience: int = 50,
        mlflow_tracking: bool = True,
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
            regularization_weight: Weight for custom regularization
            checkpoint_frequency: Save checkpoint every N epochs
            early_stopping_patience: Patience for early stopping
            mlflow_tracking: Whether to use MLflow tracking
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
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        
        # Logging
        self.mlflow_tracking = mlflow_tracking
        
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
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.loss_fn(outputs, targets)
            
            # Add custom regularization
            if self.regularization_weight > 0:
                reg_loss = self.model.get_regularization_loss()
                loss = loss + self.regularization_weight * reg_loss
            
            # Backward pass
            loss.backward()
            
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
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.loss_fn(outputs, targets)
                
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
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Log to MLflow
            if self.mlflow_tracking:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch  # Track the best epoch
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print(f"New best model saved (val_loss={val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
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
        
        # Log model to MLflow (fixes: artifact_path deprecation + signature warnings)
        if self.mlflow_tracking and "best" in filename:
            try:
                # Get a sample batch from train_loader for input example
                sample_batch = next(iter(self.train_loader))
                if isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 2:
                    sample_input, _ = sample_batch
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
