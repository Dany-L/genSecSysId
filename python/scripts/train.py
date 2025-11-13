"""Main training script."""

import argparse
from pathlib import Path
import logging
import sys
from datetime import datetime
import json
import mlflow
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch

from sysid.config import Config
from sysid.data import create_dataloaders, DataLoader
from sysid.data.direct_loader import load_split_data
from sysid.models import create_model, SimpleLure
from sysid.training import Trainer, get_loss_function, get_optimizer, get_scheduler
from sysid.utils import set_seed, get_device, print_model_summary



torch.set_default_dtype(torch.float64)




def setup_console_logging() -> logging.Logger:
    """Setup console-only logging (before run_id is known)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger(__name__)


def setup_file_logging(log_dir: Path, log_prefix: str) -> logging.Logger:
    """Setup logging with both file and console output (after run_id is known)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file_path}")
    
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train RNN for system identification")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        config = Config.from_yaml(args.config)
    elif config_path.suffix == ".json":
        config = Config.from_json(args.config)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Derive directories. If config.root_dir is provided, derive model/output/log dirs
    # as: <root>/models/<model_type>, <root>/outputs/<model_type>, <root>/logs/<model_type>
    if getattr(config, 'root_dir', None):
        base = Path(os.path.expanduser(config.root_dir))
        model_dir = base / "models" / config.model.model_type
        output_dir = base / "outputs" / config.model.model_type
        log_dir = base / "logs" / config.model.model_type
    else:
        output_dir = Path(os.path.expanduser(config.output_dir))
        model_dir = Path(os.path.expanduser(config.model_dir))
        log_dir = Path(os.path.expanduser(config.log_dir))

    # Setup basic console-only logging until we have run_id
    logger = setup_console_logging()
    
    logger.info("=" * 70)
    logger.info("Training RNN for System Identification")
    logger.info("=" * 70)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Model type: {config.model.model_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model directory: {model_dir}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")
    
    # Get device
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    print("Loading data...")
    data_config = config.data
    
    try:
        # Auto-detect loading method based on path
        train_path = Path(os.path.expanduser(data_config.train_path))
        
        if train_path.is_dir():
            # Primary method: Load directly from CSV folder structure
            logger.info("Loading directly from CSV folder structure...")
            logger.info(f"Data directory: {data_config.train_path}")
            
            # Get state column if provided
            state_col = getattr(data_config, 'state_col', None)
            if state_col and len(state_col) == 0:  # Empty list means no state
                state_col = None
            
            # Load from train/validation subfolders (skip test for training)
            result = load_split_data(
                data_dir=str(train_path),
                input_col=getattr(data_config, 'input_col', ['d']),
                output_col=getattr(data_config, 'output_col', ['e']),
                state_col=state_col,
                pattern=getattr(data_config, 'pattern', '*.csv'),
                load_test=False  # Don't load test data during training
            )
            
            # Unpack (test_* will be None)
            train_inputs, train_outputs, val_inputs, val_outputs, _, _, train_states, val_states, _ = result
            if train_states is not None:
                logger.info(f"State information loaded: train_states={train_states.shape}")
            
        elif train_path.suffix == '.csv':
            # Fallback: Load from single CSV files (requires train_path, val_path, test_path)
            logger.info("Loading from single CSV files...")
            train_inputs, train_outputs = DataLoader.load_from_csv(
                data_config.train_path,
                delimiter=","
            )
            val_inputs, val_outputs = DataLoader.load_from_csv(
                data_config.val_path,
                delimiter=","
            )
            test_inputs = test_outputs = None
            if hasattr(data_config, 'test_path'):
                test_inputs, test_outputs = DataLoader.load_from_csv(
                    data_config.test_path,
                    delimiter=","
                )
        else:
            raise ValueError(
                f"Unsupported data format: {data_config.train_path}\n"
                f"Use either:\n"
                f"  1. Folder path (recommended): 'data/prepared' with train/test/validation subfolders\n"
                f"  2. Single CSV file: 'data/train.csv' (also requires val_path and test_path)"
            )
        
        logger.info(f"Train data loaded: inputs={train_inputs.shape}, outputs={train_outputs.shape}")
        logger.info(f"Validation data loaded: inputs={val_inputs.shape}, outputs={val_outputs.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.exception("Full traceback:")
        raise
    
    print(f"Train data: {train_inputs.shape}, {train_outputs.shape}")
    print(f"Validation data: {val_inputs.shape}, {val_outputs.shape}")

    max_norm_x0 = np.max(np.linalg.norm(train_states[:,0,:],2, axis=1), axis=0) # only consider x0 not the trajectory
    delta = np.max(np.abs(train_inputs), axis = (0,1))
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, _, normalizer = create_dataloaders(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_states=train_states,
        val_inputs=val_inputs,
        val_outputs=val_outputs,
        val_states=val_states,
        batch_size=data_config.batch_size,
        sequence_length=data_config.sequence_length,
        normalize=data_config.normalize,
        normalization_method=data_config.normalization_method,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(f"Batch size: {data_config.batch_size}")
    logger.info(f"Normalization: {data_config.normalization_method if data_config.normalize else 'None'}")

    if normalizer is not None:
        delta = normalizer.transform_inputs(delta).squeeze()
    

    
    # Create model
    logger.info("Creating model...")
    print("Creating model...")
    model = create_model(config, delta, max_norm_x0)
    init_fig = None
    if isinstance(model, SimpleLure):
        init_fig = model.initialize_parameters(train_inputs, train_states, train_outputs)
    print_model_summary(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = get_loss_function(config.training.loss_type)
    logger.info(f"Loss function: {config.training.loss_type}")
    
    # Create optimizer
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type=config.optimizer.optimizer_type,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        momentum=config.optimizer.momentum,
    )
    logger.info(f"Optimizer: {config.optimizer.optimizer_type}")
    logger.info(f"Learning rate: {config.optimizer.learning_rate}")
    logger.info(f"Weight decay: {config.optimizer.weight_decay}")
    
    # Create learning rate scheduler
    scheduler = None
    if config.optimizer.use_scheduler:
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=config.optimizer.scheduler_type,
            patience=config.optimizer.scheduler_patience,
            factor=config.optimizer.scheduler_factor,
        )
        logger.info(f"LR Scheduler: {config.optimizer.scheduler_type}")
        logger.info(f"Scheduler patience: {config.optimizer.scheduler_patience}")
        logger.info(f"Scheduler factor: {config.optimizer.scheduler_factor}")
    
    # Setup MLflow
    logger.info("Setting up MLflow...")
    if config.mlflow.tracking_uri:
        try:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
            logger.info(f"MLflow tracking URI: {config.mlflow.tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to connect to MLflow server: {e}")
            logger.warning("Falling back to local file-based tracking")
            mlflow.set_tracking_uri(None)  # Use local file tracking
    else:
        # Use local file-based tracking (./mlruns directory)
        logger.info("Using local file-based MLflow tracking (./mlruns)")
    
    mlflow.set_experiment(config.mlflow.experiment_name)
    logger.info(f"MLflow experiment: {config.mlflow.experiment_name}")
    
    with mlflow.start_run(run_name=config.mlflow.run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Update directories to include run_id for better organization
        run_model_dir = Path(model_dir) / run_id
        run_output_dir = Path(output_dir) / run_id
        run_log_dir = Path(log_dir) / run_id
        run_model_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        run_log_dir.mkdir(parents=True, exist_ok=True)

        # Setup full logging (console + file) now that we have run_id
        logger = setup_file_logging(run_log_dir, "training")
        logger.info(f"MLflow run ID: {run_id}")
        logger.info(f"Experiment: {config.mlflow.experiment_name}")

        logger.info(f"Model directory: {run_model_dir}")
        logger.info(f"Output directory: {run_output_dir}")
        logger.info(f"Log directory: {run_log_dir}")

        # Save normalizer
        if normalizer is not None:
            normalizer_path = Path(run_model_dir) / "normalizer.json"
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            normalizer.save(str(normalizer_path))
            logger.info(f"Normalizer saved to {normalizer_path}")
            print(f"Normalizer saved to {normalizer_path}")
        
        # Log config
        mlflow.log_params({
            "model_type": config.model.model_type,
            "hidden_size": config.model.nw,
            "activation": config.model.activation,
            "num_layers": config.model.num_layers,
            "learning_rate": config.optimizer.learning_rate,
            "batch_size": config.data.batch_size,
            "max_epochs": config.training.max_epochs,
            "custom_parameters": str(getattr(config.model, 'custom_params', None)),
        })
        
        # Save config
        config_save_path = run_output_dir / "config.yaml"
        config.save_yaml(str(config_save_path))
        mlflow.log_artifact(str(config_save_path))
        logger.info(f"Config saved to {config_save_path}")
        
        # Log initialization plot if available
        if init_fig is not None:
            try:
                plot_path = run_output_dir / 'init_ellipse.png'
                init_fig.savefig(plot_path, bbox_inches='tight')
                mlflow.log_artifact(str(plot_path), artifact_path='plots')
                logger.info(f"Logged initialization plot to MLflow artifacts")
            except Exception as e:
                logger.warning(f"Failed to log initialization plot: {e}")
            finally:
                plt.close(init_fig)
        
        # Save run_id for later use (evaluation/analysis)
        run_info_path = run_model_dir / "run_info.json"
        run_info = {
            "run_id": run_id,
            "experiment_name": config.mlflow.experiment_name,
            "run_name": config.mlflow.run_name,
        }
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        logger.info(f"Run info saved to {run_info_path}")
        
        # Create trainer (use run-specific directories)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            output_dir=str(run_output_dir),
            model_dir=str(run_model_dir),
            log_dir=log_dir,
            gradient_clip_value=config.training.gradient_clip_value,
            regularization_weight=config.training.regularization_weight if config.training.use_custom_regularization else 0.0,
            decay_regularization_weight=getattr(config.training, 'decay_regularization_weight', False) if config.training.use_custom_regularization else False,
            regularization_decay_factor=getattr(config.training, 'regularization_decay_factor', 0.5),
            checkpoint_frequency=config.training.checkpoint_frequency,
            early_stopping_patience=config.training.early_stopping_patience,
            mlflow_tracking=True,
            log_gradients=getattr(config.training, 'log_gradients', True),
        )
        
        if scheduler is not None:
            trainer.set_scheduler(scheduler)
        
        # Train
        logger.info("=" * 70)
        logger.info("Starting training...")
        logger.info(f"Max epochs: {config.training.max_epochs}")
        logger.info(f"Early stopping patience: {config.training.early_stopping_patience}")
        logger.info(f"Gradient clipping: {config.training.gradient_clip_value}")
        logger.info(f"Gradient logging: {getattr(config.training, 'log_gradients', True)}")
        if config.training.use_custom_regularization:
            logger.info("Custom regularization: enabled")
            logger.info(f"  Initial weight: {config.training.regularization_weight}")
            if getattr(config.training, 'decay_regularization_weight', False):
                logger.info(f"  Decay with LR: enabled")
                logger.info(f"  Decay factor: {getattr(config.training, 'regularization_decay_factor', 0.5)}")
            else:
                logger.info(f"  Decay with LR: disabled")
        logger.info("=" * 70)
        print("\nStarting training...")
        
        try:
            history = trainer.train(max_epochs=config.training.max_epochs, normalizer=normalizer)
            
            logger.info("=" * 70)
            logger.info("Training completed successfully!")
            logger.info("=" * 70)
            logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")
            logger.info(f"Best epoch: {history['best_epoch']}")
            logger.info(f"Final epoch: {history['final_epoch']}")
            logger.info(f"Total training time: {history.get('total_time', 'N/A')}")
            
            print("\nTraining completed!")
            print(f"Best validation loss: {history['best_val_loss']:.6f}")
            print(f"Final epoch: {history['final_epoch']}")
            
            # Log final metrics
            mlflow.log_metric("best_val_loss", history["best_val_loss"])
            mlflow.log_metric("best_epoch", history["best_epoch"])
            mlflow.log_metric("final_epoch", history["final_epoch"])
            
            # Log model artifacts (using run-specific directories)
            mlflow.log_artifacts(str(run_model_dir), "models")
            mlflow.log_artifacts(str(run_output_dir), "outputs")
            
            logger.info("Artifacts logged to MLflow")
            logger.info(f"Models saved in: {run_model_dir}")
            logger.info(f"Outputs saved in: {run_output_dir}")
            logger.info(f"Logs saved in: {log_dir}")
            logger.info(f"Run ID: {run_id} - Use this for evaluation/analysis")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.exception("Full traceback:")
            raise


if __name__ == "__main__":
    main()
