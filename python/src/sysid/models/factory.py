"""Model factory for creating and loading models."""

import torch
from pathlib import Path
from typing import Optional, Union

from .rnn import SimpleRNN, LSTM, GRU
from .constrained_rnn import SimpleLure
from ..config import Config, ModelConfig


def create_model(config: Config):
    """
    Create model from configuration.
    
    Args:
        config: Full configuration object
        
    Returns:
        Initialized model
    """
    model_config = config.model
    data_config = config.data
    
    # Determine input/output sizes from data config
    input_size = len(data_config.input_col) if data_config.input_col else 1
    output_size = len(data_config.output_col) if data_config.output_col else 1
    
    if model_config.model_type == "rnn":
        model = SimpleRNN(
            input_size=input_size,
            hidden_size=model_config.nw,
            output_size=output_size,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            activation=model_config.activation,
        )
    elif model_config.model_type == "lstm":
        model = LSTM(
            input_size=input_size,
            hidden_size=model_config.nw,
            output_size=output_size,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
        )
    elif model_config.model_type == "gru":
        model = GRU(
            input_size=input_size,
            hidden_size=model_config.nw,
            output_size=output_size,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
        )
    elif model_config.model_type == "crnn":
        model = SimpleLure(
            nd=input_size,
            ne=output_size,
            nw=model_config.nw,
            nx=model_config.nx,
            activation=model_config.activation,
            custom_params=model_config.custom_params,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")
    
    return model


def load_model(
    checkpoint_path: Union[str, Path],
    config: Config,
    device: Optional[str] = None,
    eval_mode: bool = True
):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt or .pth file)
        config: Configuration object (used to recreate model architecture)
        device: Device to load model to ('cuda', 'cpu', 'mps', etc.). If None, uses CPU.
        eval_mode: If True, sets model to eval mode. Set to False for continued training.
        
    Returns:
        Loaded model
        
    Example:
        >>> config = Config.from_yaml("config.yaml")
        >>> model = load_model("models/best_model.pt", config, device="cuda")
    """
    # Set device
    if device is None:
        device = "cpu"
    
    # Create model from config
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint with optimizer, epoch, etc.
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        # Just state dict
        model.load_state_dict(checkpoint)
    else:
        # Assume it's a state dict directly
        model.load_state_dict(checkpoint)
    
    # Set to eval mode if requested
    if eval_mode:
        model.eval()
    
    return model


def save_model(
    model,
    save_path: Union[str, Path],
    optimizer=None,
    epoch: Optional[int] = None,
    metadata: Optional[dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model to save
        save_path: Path where to save the checkpoint
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        metadata: Optional dictionary with additional metadata to save
        
    Example:
        >>> save_model(model, "models/checkpoint.pt", optimizer=optimizer, epoch=10)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if metadata is not None:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, save_path)
