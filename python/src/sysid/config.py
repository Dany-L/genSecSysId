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
    root_dir: Optional[str] = None  # Root directory for relative paths

    # Direct folder loading parameters
    input_col: list = None  # Column name(s) for input - supports MIMO
    input_cols: list = None  # Alias for input_col
    output_col: list = None  # Column name(s) for output - supports MIMO
    output_cols: list = None  # Alias for output_col
    state_col: list = None  # Column name(s) for state (optional)
    pattern: str = "*.csv"  # File pattern for folder loading

    # Preprocessing
    normalize: bool = True
    normalization: Optional[str] = None  # Alias for normalize (if "minmax" or "standard")
    normalization_method: str = "minmax"  # or "standard"
    batch_size: int = 32
    sequence_length: Optional[int] = None  # None = use full sequences
    sequence_stride: Optional[int] = None  # None = auto (non-overlap for concatenated data)
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
class InitializationConfig:
    """Configuration for model parameter initialization."""

    method: str = "esn"  # "esn", "n4sid", or "identity"
    # ESN-specific parameters
    esn_n_restarts: int = 5  # Number of random reservoirs to try
    # Identity initialization uses α=0.99, A=0.9I, C2=Rand(-1,1), C=[I,0], B2=D=D12=0


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_type: str = "rnn"  # "rnn", "lstm", "gru", or custom
    type: Optional[str] = None  # Alias for model_type
    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    activation: str = "tanh"
    # Legacy aliases for backward compatibility
    nw: Optional[int] = None  # alias for hidden_size
    nx: Optional[int] = None  # alias for input_size (unused, kept for compatibility)
    nd: Optional[int] = None  # input dimension (constrained models)
    ne: Optional[int] = None  # output dimension (constrained models)
    # Custom parameters for specific models
    custom_params: Optional[Dict[str, Any]] = None
    # Initialization configuration
    initialization: InitializationConfig = None

    def __post_init__(self):
        """Set default initialization if none provided."""
        if self.initialization is None:
            self.initialization = InitializationConfig()
        # Support type as alias for model_type
        if self.type is not None:
            self.model_type = self.type
        # Support legacy nw/nx parameters
        if self.nw is not None:
            self.hidden_size = self.nw
        if self.nx is not None and self.nx != self.input_size:
            # nx was previously used differently, log warning if differs
            pass
        # Support constrained model dimensions
        if self.nd is not None and self.input_size == 1:  # Only override if not explicitly set
            self.input_size = self.nd
        if self.ne is not None and self.output_size == 1:  # Only override if not explicitly set
            self.output_size = self.ne


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
    metrics: Optional[list] = None  # List of metrics to compute and log
    metrics_to_log: Optional[list] = None  # Alias for metrics
    
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
        """Handle both metrics and metrics_to_log field names."""
        if self.metrics_to_log is not None and self.metrics is None:
            self.metrics = self.metrics_to_log
        elif self.metrics_to_log is None and self.metrics is not None:
            self.metrics_to_log = self.metrics

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
    # Set to None to disable seeding (allows getting different results on each run for variance estimation)
    # Set to an integer (e.g., 42) for reproducible results
    seed: Optional[int] = None

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
        """Create Config from dictionary, properly instantiating nested dataclasses."""
        # Normalize field names for each config section
        
        # Handle data config with field name mappings
        data_dict = config_dict.get("data", {}).copy()
        if "input_cols" in data_dict and "input_col" not in data_dict:
            data_dict["input_col"] = data_dict.pop("input_cols")
        if "output_cols" in data_dict and "output_col" not in data_dict:
            data_dict["output_col"] = data_dict.pop("output_cols")
        if "normalization" in data_dict and "normalize" not in data_dict:
            # Map normalization value to normalize if it's a boolean indicator
            norm_val = data_dict.pop("normalization")
            if isinstance(norm_val, bool):
                data_dict["normalize"] = norm_val
        
        # Handle model config with nested initialization config
        model_dict = config_dict.get("model", {}).copy()
        if "initialization" in model_dict and isinstance(model_dict["initialization"], dict):
            model_dict["initialization"] = InitializationConfig(**model_dict["initialization"])
        
        # Handle optimizer config with field name mappings
        optimizer_dict = config_dict.get("optimizer", {}).copy()
        # Training config often contains optimizer settings, so merge them
        training_dict = config_dict.get("training", {}).copy()
        if "learning_rate" in training_dict and "learning_rate" not in optimizer_dict:
            optimizer_dict["learning_rate"] = training_dict.pop("learning_rate")
        if "optimizer" in training_dict and "optimizer_type" not in optimizer_dict:
            optimizer_dict["optimizer_type"] = training_dict.pop("optimizer")
        
        # Handle training config with field name mappings
        if "epochs" in training_dict:
            training_dict["max_epochs"] = training_dict.pop("epochs")
        if "loss_function" in training_dict and "loss_type" not in training_dict:
            training_dict["loss_type"] = training_dict.pop("loss_function")
        
        # Handle evaluation config
        eval_config = None
        if "evaluation" in config_dict:
            eval_dict = config_dict["evaluation"].copy()
            eval_config = EvaluationConfig(**eval_dict)

        return cls(
            data=DataConfig(**data_dict),
            model=ModelConfig(**model_dict),
            optimizer=OptimizerConfig(**optimizer_dict),
            training=TrainingConfig(**training_dict),
            mlflow=MLflowConfig(**config_dict.get("mlflow", {})),
            evaluation=eval_config,
            output_dir=config_dict.get("output_dir", "outputs"),
            model_dir=config_dict.get("model_dir", "models"),
            log_dir=config_dict.get("log_dir", "logs"),
            root_dir=config_dict.get("root_dir", None),
            seed=config_dict.get("seed", None),
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
