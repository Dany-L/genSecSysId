"""Hyperparameter optimization using Optuna."""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import mlflow
import numpy as np
import optuna
import torch
import yaml

from sysid.config import Config
from sysid.data import create_dataloaders
from sysid.data.direct_loader import load_split_data
from sysid.models import create_model
from sysid.training import Trainer, get_loss_function, get_optimizer, get_scheduler
from sysid.utils import get_device, set_seed

torch.set_default_dtype(torch.float64)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config: Config, data_cache: Dict[str, Any]) -> float:
    """
    Optuna objective function to minimize validation loss.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration (loaded from YAML)
        data_cache: Cached data to avoid reloading
        
    Returns:
        Best validation loss
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    nw = trial.suggest_int("nw", 5, 20)
    # For SimpleLure with pad_state=True (default), nx must equal nw
    # So we only optimize nw and set nx=nw
    nx = nw
    regularization_weight = trial.suggest_float("regularization_weight", 1e-4, 1e-1, log=True)
    
    # Optional: suggest more hyperparameters
    if base_config.model.model_type == "crnn":
        if hasattr(base_config.model, 'custom_params') and base_config.model.custom_params:
            learn_L = trial.suggest_categorical("learn_L", [True, False])
        else:
            learn_L = True
            
    # Create modified config
    config = base_config.copy() if hasattr(base_config, 'copy') else base_config
    
    # Update config with trial suggestions
    config.optimizer.learning_rate = learning_rate
    config.optimizer.weight_decay = weight_decay
    config.data.batch_size = batch_size
    config.model.nw = nw
    config.model.nx = nx
    config.training.regularization_weight = regularization_weight
    
    if config.model.model_type == "crnn" and hasattr(config.model, 'custom_params'):
        if config.model.custom_params is None:
            config.model.custom_params = {}
        config.model.custom_params['learn_L'] = learn_L
    
    # Reduce epochs for faster trials
    config.training.max_epochs = min(100, config.training.max_epochs)
    config.training.early_stopping_patience = 20
    
    try:
        # Get device
        device = get_device(config.training.device)
        
        # Create temporary directories for this trial (won't be saved)
        temp_dir = tempfile.mkdtemp()
        temp_output_dir = os.path.join(temp_dir, "outputs")
        temp_model_dir = os.path.join(temp_dir, "models")
        temp_log_dir = os.path.join(temp_dir, "logs")
        os.makedirs(temp_output_dir, exist_ok=True)
        os.makedirs(temp_model_dir, exist_ok=True)
        os.makedirs(temp_log_dir, exist_ok=True)
        
        # Create data loaders with new batch size
        # Don't use states during trials to avoid dimension mismatches with varying nx
        train_loader, val_loader, _, normalizer = create_dataloaders(
            train_inputs=data_cache['train_inputs'],
            train_outputs=data_cache['train_outputs'],
            train_states=None,  # Skip states during trials
            val_inputs=data_cache['val_inputs'],
            val_outputs=data_cache['val_outputs'],
            val_states=None,  # Skip states during trials
            batch_size=batch_size,
            sequence_length=config.data.sequence_length,
            normalize=config.data.normalize,
            normalization_method=config.data.normalization_method,
            shuffle=config.data.shuffle,
        )
        
        # Create model
        nd = data_cache['train_inputs'].shape[2]
        ne = data_cache['train_outputs'].shape[2]
        
        # For CRNN models, use SimpleLure directly
        if config.model.model_type == "crnn":
            from sysid.models import SimpleLure
            model = SimpleLure(
                nd=nd,
                ne=ne,
                nx=nx,
                nw=nw,
                activation=config.model.activation,
                custom_params=getattr(config.model, 'custom_params', None),
                delta=data_cache['delta'],
                max_norm_x0=data_cache['max_norm_x0'],
            )
        else:
            # For other models, use factory with updated config
            model = create_model(config, delta=data_cache['delta'], max_norm_x0=data_cache['max_norm_x0'])
        
        # Initialize model parameters (lightweight version)
        if hasattr(model, 'initialize_parameters'):
            logger.info("Initializing model parameters...")
            # Don't pass states during trials to avoid dimension mismatches
            # The model will use default initialization values
            model.initialize_parameters(
                train_inputs=data_cache['train_inputs'][:2],  # Use subset for speed
                train_states=None,  # Skip states during trial initialization
                train_outputs=data_cache['train_outputs'][:2],
                n_restarts=1,  # Reduced for speed
                data_dir=None,  # Skip N4SID loading for trials
            )
        
        # Create trainer
        # Call get_optimizer with individual parameters (not config)
        optimizer = get_optimizer(
            model.parameters(),
            optimizer_type=config.optimizer.optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=getattr(config.optimizer, 'betas', (0.9, 0.999)),
            momentum=getattr(config.optimizer, 'momentum', 0.9),
        )
        scheduler = None
        if config.optimizer.use_scheduler:
            scheduler = get_scheduler(
                optimizer,
                scheduler_type=config.optimizer.scheduler_type,
                patience=config.optimizer.scheduler_patience,
                factor=config.optimizer.scheduler_factor,
            )
        loss_function = get_loss_function(config.training.loss_type)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_function,  # Use loss_fn not loss_function
            optimizer=optimizer,
            device=str(device),
            output_dir=temp_output_dir,  # Use temp directory
            model_dir=temp_model_dir,  # Use temp directory
            log_dir=temp_log_dir,  # Use temp directory
            gradient_clip_value=config.training.gradient_clip_value,
            regularization_weight=0.0,  # Disable during trials
            decay_regularization_weight=False,
            regularization_decay_factor=1.0,
            checkpoint_frequency=10,
            early_stopping_patience=config.training.early_stopping_patience,
            mlflow_tracking=False,  # Disable MLflow for trials
            log_gradients=False,  # Disable for speed
        )
        
        # Set scheduler if one was created
        if scheduler is not None:
            trainer.set_scheduler(scheduler)
        
        # Train model
        history = trainer.train(max_epochs=config.training.max_epochs)
        
        # Return best validation loss
        best_val_loss = min(history['val_loss'])
        
        # Report intermediate values for pruning
        for epoch, val_loss in enumerate(history['val_loss']):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        # Return high loss to indicate failure
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to base config file")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--study-name", type=str, default="crnn-optimization", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Hyperparameter Optimization with Optuna")
    logger.info("=" * 80)
    
    # Load base configuration
    config_path = Path(args.config)
    if config_path.suffix in [".yaml", ".yml"]:
        config = Config.from_yaml(args.config)
    elif config_path.suffix == ".json":
        config = Config.from_json(args.config)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    logger.info(f"Base config: {config_path}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Device: {config.training.device}")
    
    # Set seed
    set_seed(config.seed)
    
    # Load data once (cached for all trials)
    logger.info("Loading data...")
    data_config = config.data
    train_path = Path(os.path.expanduser(data_config.train_path))
    
    if train_path.is_dir():
        result = load_split_data(
            data_dir=str(train_path),
            input_col=getattr(data_config, "input_col", ["d"]),
            output_col=getattr(data_config, "output_col", ["e"]),
            state_col=getattr(data_config, "state_col", None),
            pattern=getattr(data_config, "pattern", "*.csv"),
            load_test=False,
        )
        (train_inputs, train_outputs, val_inputs, val_outputs, _, _,
         train_states, val_states, _) = result
    else:
        raise ValueError("Only directory-based data loading supported for optimization")
    
    logger.info(f"Train: {train_inputs.shape}, Val: {val_inputs.shape}")
    
    # Compute delta and max_norm_x0
    if train_states is not None:
        max_norm_x0 = np.max(np.linalg.norm(train_states[:, 0, :], 2, axis=1), axis=0)
    else:
        max_norm_x0 = 0.1
    delta = np.max(np.abs(train_inputs), axis=(0, 1))
    
    # Cache data
    data_cache = {
        'train_inputs': train_inputs,
        'train_outputs': train_outputs,
        'val_inputs': val_inputs,
        'val_outputs': val_outputs,
        'train_states': train_states,
        'val_states': val_states,
        'delta': delta,
        'max_norm_x0': max_norm_x0,
    }
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    logger.info(f"Study name: {args.study_name}")
    if args.storage:
        logger.info(f"Storage: {args.storage}")
        logger.info("Dashboard: optuna-dashboard <storage-url>")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, data_cache),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # Print results
    logger.info("=" * 80)
    logger.info("Optimization Complete!")
    logger.info("=" * 80)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value (val_loss): {study.best_trial.value:.6f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save best config
    output_dir = Path(os.path.expanduser(config.output_dir)) if hasattr(config, 'output_dir') else Path("results/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_config_path = output_dir / f"best_config_{args.study_name}.yaml"
    
    # Create config dict with best parameters
    best_params = study.best_trial.params
    config_dict = {
        'data': {
            'train_path': data_config.train_path,
            'input_col': getattr(data_config, 'input_col', ['d']),
            'output_col': getattr(data_config, 'output_col', ['e']),
            'normalize': data_config.normalize,
            'normalization_method': data_config.normalization_method,
            'batch_size': best_params['batch_size'],
            'sequence_length': data_config.sequence_length,
            'shuffle': data_config.shuffle,
        },
        'model': {
            'model_type': config.model.model_type,
            'nw': best_params['nw'],
            'nx': best_params['nw'],  # nx=nw for SimpleLure with pad_state=True
            'activation': config.model.activation,
        },
        'optimizer': {
            'optimizer_type': config.optimizer.optimizer_type,
            'learning_rate': best_params['learning_rate'],
            'weight_decay': best_params['weight_decay'],
            'use_scheduler': config.optimizer.use_scheduler,
            'scheduler_type': config.optimizer.scheduler_type,
        },
        'training': {
            'max_epochs': config.training.max_epochs,
            'regularization_weight': best_params['regularization_weight'],
            'device': config.training.device,
        },
    }
    
    if 'learn_L' in best_params:
        if 'custom_params' not in config_dict['model']:
            config_dict['model']['custom_params'] = {}
        config_dict['model']['custom_params']['learn_L'] = best_params['learn_L']
    
    with open(best_config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Best config saved to: {best_config_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
