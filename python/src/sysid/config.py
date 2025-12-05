"""Configuration management for the system identification package."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    train_path: str
    val_path: Optional[str] = None  # Not required for folder loading
    test_path: Optional[str] = None  # Not required for folder loading

    # Direct folder loading parameters
    input_col: list = None  # Column name(s) for input - supports MIMO
    output_col: list = None  # Column name(s) for output - supports MIMO
    state_col: list = None  # Column name(s) for state (optional)
    pattern: str = "*.csv"  # File pattern for folder loading

    # Preprocessing
    normalize: bool = True
    normalization_method: str = "minmax"  # or "standard"
    batch_size: int = 32
    sequence_length: Optional[int] = None  # None = use full sequences
    shuffle: bool = True
    num_workers: int = 0

    def __post_init__(self):
        """Set default column names if none provided."""
        if self.input_col is None:
            self.input_col = ["d"]
        if self.output_col is None:
            self.output_col = ["e"]
        if self.state_col is None:
            self.state_col = []  # Empty list means no state columns


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_type: str = "rnn"  # "rnn", "lstm", "gru", or custom
    nw: int = 64
    nx: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    activation: str = "tanh"
    # Custom parameters for specific models
    custom_params: Optional[Dict[str, Any]] = None


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    optimizer_type: str = "adam"  # "adam", "sgd", "rmsprop"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    momentum: float = 0.9  # for SGD
    betas: tuple = (0.9, 0.999)  # for Adam

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # "step", "exponential", "reduce_on_plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training."""

    max_epochs: int = 1000
    early_stopping_patience: int = 50
    checkpoint_frequency: int = 10  # save every N epochs
    gradient_clip_value: Optional[float] = 1.0

    # Loss function
    loss_type: str = "mse"  # "mse", "mae", "huber"

    # Regularization (Interior Point Method for LMI constraints)
    use_custom_regularization: bool = False
    regularization_weight: float = 0.01
    decay_regularization_weight: bool = True  # Decay reg weight with learning rate
    regularization_decay_factor: float = 0.5  # Same as scheduler_factor by default
    min_regularization_weight: float = 1e-7  # Early stopping threshold for reg weight

    # Gradient monitoring
    log_gradients: bool = True  # Log gradient statistics to MLflow

    # Device
    device: str = "cuda"  # "cuda", "cpu", "mps"

    # Logging
    log_interval: int = 10  # log every N batches


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "sysid_training"
    run_name: Optional[str] = None
    log_models: bool = True
    log_artifacts: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    # Base metrics (always computed, but can be excluded from logging)
    metrics: list = None  # List of metrics to compute and log

    # Available metrics:
    # - mse: Mean Squared Error
    # - rmse: Root Mean Squared Error
    # - mae: Mean Absolute Error
    # - r2: R-squared score
    # - nrmse: Normalized RMSE
    # - max_error: Maximum absolute error

    # For sequence predictions, also available:
    # - <metric>_avg: Average over all time steps
    # - <metric>_final: Metric at final time step

    def __post_init__(self):
        """Set default metrics if none provided."""
        if self.metrics is None:
            # Default: all available metrics
            self.metrics = ["mse", "rmse", "mae", "r2", "nrmse", "max_error"]


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    evaluation: EvaluationConfig = None

    # Paths
    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"
    # Optional root directory: when set, model/output/log dirs are derived from it
    # as: <root>/models/<model_type>, <root>/outputs/<model_type>, <root>/logs/<model_type>
    root_dir: Optional[str] = None

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Initialize evaluation config if not provided."""
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Handle evaluation config
        eval_config = None
        if "evaluation" in config_dict:
            eval_config = EvaluationConfig(**config_dict["evaluation"])

        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            mlflow=MLflowConfig(**config_dict.get("mlflow", {})),
            evaluation=eval_config,
            output_dir=config_dict.get("output_dir", "outputs"),
            model_dir=config_dict.get("model_dir", "models"),
            log_dir=config_dict.get("log_dir", "logs"),
            root_dir=config_dict.get("root_dir", None),
            seed=config_dict.get("seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
            "mlflow": asdict(self.mlflow),
            "evaluation": asdict(self.evaluation) if self.evaluation else None,
            "output_dir": self.output_dir,
            "model_dir": self.model_dir,
            "log_dir": self.log_dir,
            "root_dir": self.root_dir,
            "seed": self.seed,
        }

    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str):
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
