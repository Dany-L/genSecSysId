"""Main training script."""

import argparse
from pathlib import Path
import logging
import sys
from datetime import datetime
import json
import mlflow
import os

from sysid.config import Config
from sysid.data import create_dataloaders, DataLoader
from sysid.data.direct_loader import load_split_data
from sysid.models import SimpleRNN, LSTM, GRU, SimpleLure
from sysid.training import Trainer, get_loss_function, get_optimizer, get_scheduler
from sysid.utils import set_seed, get_device, print_model_summary


def setup_logging(output_dir: Path, experiment_name: str) -> logging.Logger:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Experiment: {experiment_name}")
    
    return logger


def create_model(config: Config):
    """Create model from configuration."""
    model_config = config.model
    data_config = config.data
    
    if model_config.model_type == "rnn":
        model = SimpleRNN(
            input_size=len(data_config.input_col),
            hidden_size=model_config.nw,
            output_size=len(data_config.output_col),
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            activation=model_config.activation,
        )
    elif model_config.model_type == "lstm":
        model = LSTM(
            input_size=len(data_config.input_col),
            hidden_size=model_config.nw,
            output_size=len(data_config.output_col),
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
        )
    elif model_config.model_type == "gru":
        model = GRU(
            input_size=len(data_config.input_col),
            hidden_size=model_config.nw,
            output_size=len(data_config.output_col),
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
        )
    elif model_config.model_type == "crnn":
        model = SimpleLure(
            nd=len(data_config.input_col),
            ne=len(data_config.output_col),
            nw=model_config.nw,
            nx=model_config.nx,
            activation=model_config.activation,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")
    
    return model


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
    
    output_dir = os.path.expanduser(config.output_dir)
    model_dir = os.path.expanduser(config.model_dir)
    log_dir = os.path.expanduser(config.log_dir)
    # Setup logging
    logger = setup_logging(Path(os.path.expanduser(output_dir)), config.mlflow.experiment_name)
    
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
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, _, normalizer = create_dataloaders(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        val_inputs=val_inputs,
        val_outputs=val_outputs,
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
    
    # Save normalizer
    if normalizer is not None:
        normalizer_path = Path(model_dir) / "normalizer.json"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        normalizer.save(str(normalizer_path))
        logger.info(f"Normalizer saved to {normalizer_path}")
        print(f"Normalizer saved to {normalizer_path}")
    
    # Create model
    logger.info("Creating model...")
    print("Creating model...")
    model = create_model(config)
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
        run_model_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model directory: {run_model_dir}")
        logger.info(f"Output directory: {run_output_dir}")
        
        # Log config
        mlflow.log_params({
            "model_type": config.model.model_type,
            "hidden_size": config.model.nw,
            "activation": config.model.activation,
            "num_layers": config.model.num_layers,
            "learning_rate": config.optimizer.learning_rate,
            "batch_size": config.data.batch_size,
            "max_epochs": config.training.max_epochs,
        })
        
        # Save config
        config_save_path = run_output_dir / "config.yaml"
        config.save_yaml(str(config_save_path))
        mlflow.log_artifact(str(config_save_path))
        logger.info(f"Config saved to {config_save_path}")
        
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
            logger.info(f"Custom regularization: enabled")
            logger.info(f"  Initial weight: {config.training.regularization_weight}")
            if getattr(config.training, 'decay_regularization_weight', False):
                logger.info(f"  Decay with LR: enabled (Interior Point Method)")
                logger.info(f"  Decay factor: {getattr(config.training, 'regularization_decay_factor', 0.5)}")
            else:
                logger.info(f"  Decay with LR: disabled")
        logger.info("=" * 70)
        print("\nStarting training...")
        
        try:
            history = trainer.train(max_epochs=config.training.max_epochs)
            
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
