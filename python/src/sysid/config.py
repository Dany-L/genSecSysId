"""Configuration management for the system identification package."""

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


def _known_fields(config_cls, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keys ``config_cls`` actually accepts, warning about the rest.

    Lets old configs load even when they still carry keys from removed features
    (e.g. ``esn_n_restarts`` after the ESN init was dropped): the stale keys are
    ignored with a warning instead of raising ``TypeError``.
    """
    allowed = {f.name for f in fields(config_cls)}
    unknown = [k for k in values if k not in allowed]
    if unknown:
        logger.warning(
            "Ignoring unknown config field(s) in '%s': %s "
            "(not part of %s — likely leftovers from a removed/renamed feature).",
            section, ", ".join(sorted(unknown)), config_cls.__name__,
        )
    return {k: v for k, v in values.items() if k in allowed}


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
    train_sequence_length: Optional[int] = None  # Sequence length for training only. None = use full sequences. Validation/test always use full sequences.
    sequence_stride: Optional[int] = None  # None = auto (non-overlap for concatenated data)
    shuffle: bool = True
    num_workers: int = 0
    sampling_time: float = 0.01

    # Diverging trajectory support. When enabled, the loader additionally reads
    # train_div/, validation_div/, test_div/ sibling folders. Diverging
    # trajectories have variable length, start from x0=0, and are used with
    # batch_size=diverging_batch_size and no warmup skipping in the loss.
    use_diverging_trajectories: bool = False
    diverging_batch_size: int = 1

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

    method: str = "identity"  # only "identity" is supported (esn/n4sid removed)
    # Identity initialization uses α=0.99, A=0.9I, C2=Rand(-1,1), C=[I,0], B2=D=D12=0

    # Initialize s (and P, L) with the output+input condition sweep (MinTrProb)
    # instead of plain max-s: pick an s that meets the output-coverage floor AND
    # leaves zero input violations; else the s with the fewest violations;
    # falling back to max-s only when the output level is not yet reachable.
    # Set False for the legacy max-s initialization.
    init_s_from_conditions: bool = True
    # Number of s grid points for the initialization sweep (kept small; init runs once).
    init_s_grid_size: int = 15
    # Upper end of the s sweep band (simple preset; see solve_output_coverage_certificate).
    init_s_max: float = 20.0

    # Calibrate the C2 std at init so BOTH conditions hold under the operative
    # MaxS certificate: (input) the training rollout has zero input-condition
    # violations (||u_k||^2 <= s^2 - alpha^2 x_k^T P^-1 x_k) -> the trajectory
    # stays in the invariant set so predictions do not diverge; and (output) the
    # certified image covers y_max. Searches a scalar factor on the initialized C2
    # (bisection, relative tol calibrate_c2_eps): both hold for small C2 (large s)
    # and fail as C2 grows, so it keeps the LARGEST C2 (tightest region) that is
    # still stable and covering. Disable to keep the config C2 std unchanged.
    calibrate_c2_for_coverage: bool = True
    calibrate_c2_eps: float = 0.05
    calibrate_c2_max_iter: int = 30
    # Which nonlinearity-shaping maps the calibration may scale. C2 (state->NL) is
    # the primary tightness knob (bisected); D21 (input->NL) and B2 (NL->state) are
    # grid-searched — smaller D21 relaxes the input condition and lets C2 grow, so
    # jointly they reach a much tighter input-admissible certificate. ["C2"] falls
    # back to the single-knob C2 calibration.
    calibrate_knobs: List[str] = field(default_factory=lambda: ["C2", "B2", "D21"])


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
    # For SimpleLure models, supports:
    #   - structural_constraints: dict specifying fixed or partially learnable parameters
    #     Format: {param_name: constraint_spec}
    #     Where param_name is one of: A, B, B2, C, D, D12, C2, D21, D22
    #     And constraint_spec is either:
    #       1. Fully fixed: {fixed: true, value: <scalar or array>}
    #          Example: {fixed: true, value: [[1, 0]]}
    #       2. Partially learnable rows: {learnable_rows: [indices], fixed_value: <scalar>}
    #          Example: {learnable_rows: [1], fixed_value: 0.0}  # Only row 1 learnable
    #       3. Partially learnable cols: {learnable_cols: [indices], fixed_value: <scalar>}
    #          Example: {learnable_cols: [0, 2], fixed_value: 0.0}  # Only cols 0,2 learnable
    #     Notes:
    #       - Fixed parameters have requires_grad=False and keep their fixed value
    #       - Partially learnable parameters use gradient masking to zero non-learnable elements
    #       - Initialization methods respect constraints (fixed params not modified)
    #       - Fully backward compatible (configs without constraints work unchanged)
    #   - pad_state: bool, optional (default: False)
    #     Pads state dimension to match nz (nonlinearity state size)
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
    early_stopping_patience: int = 1000
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

    # Input constraint regularization weight
    input_regularization_weight: float = 0.01  # Weight for input constraint loss

    # Output-coverage regularization weight. Penalizes s^2 C P C^T below the
    # (normalized) safe level (bind Corollary 1): pushes the certified output
    # image to reach the physical data level y_max. 0.0 disables it (default).
    output_regularization_weight: float = 0.0

    # Output-TIGHTNESS regularization weight. The complement of the coverage term:
    # penalizes relu(lambda_max((output_std*s)^2 C P C^T - y_max^2 I)) (C P C^T
    # detached, so only the scale s carries gradient), pulling the certified
    # output half-width y_bar = output_std*s*sqrt(CPC^T) DOWN onto y_max so the
    # certificate stays tight (y_bar ~ y_max) instead of over-conservative. Use
    # alongside output_regularization_weight to sandwich y_bar at y_max. Like the
    # activity/H terms it is NOT decayed (tightness must hold all through
    # training). 0.0 disables it (default).
    tightness_regularization_weight: float = 0.0

    # Dead-zone activity regularization. Penalizes relu(activity_target - mean||w||)
    # on the rollout so the dead-zone nonlinearity fires, preventing the degenerate
    # linear collapse (w == 0 -> pure LTI rollout) and pushing the model into its
    # nonlinear regime. NOTE: this does NOT by itself make the global (H=0)
    # certificate infeasible -- tanh/dzn are globally sector-bounded, so a model can
    # be globally absolutely stable with an active nonlinearity; it is a behavioral
    # heuristic that correlates with, not guarantees, a non-global model. 0.0
    # disables it (default). Only meaningful for the 'dzn' activation. Unlike the
    # other reg terms this one is NOT decayed (it must hold all through training),
    # regardless of decay_regularization_weight.
    activity_regularization_weight: float = 0.0
    activity_target: float = 0.0  # w_star: target mean ||w|| (the hinge threshold)

    # Anti-global-certificate regularization. Penalizes relu(h_target - ||H||_F)
    # with H = L P^-1, pushing the coupling away from zero so the certificate
    # stays LOCAL (H = 0 is the global sector condition -> a globally stable,
    # typically near-linear model). Acts directly on the certificate params
    # (L, P), unlike activity regularization which only touches the rollout. 0.0
    # disables it (default). Requires learn_L. Like the activity term it is NOT
    # decayed (it must hold all through training).
    h_regularization_weight: float = 0.0
    h_target: float = 0.0  # h_star: target coupling norm ||H||_F (hinge threshold)

    # Gradient monitoring
    log_gradients: bool = True  # Log gradient statistics to MLflow

    # Warmup steps (initial transient period to skip when computing loss)
    warmup_steps: int = 0  # Number of warmup steps before computing loss

    # Device
    device: str = "cuda"  # "cuda", "cpu", "mps"


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
        """Reconcile the metrics/metrics_to_log aliases, then apply defaults."""
        # Accept either field name as the source of truth.
        if self.metrics_to_log is not None and self.metrics is None:
            self.metrics = self.metrics_to_log
        elif self.metrics_to_log is None and self.metrics is not None:
            self.metrics_to_log = self.metrics

        # Default when neither was provided.
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

    # Logging verbosity: one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    # DEBUG surfaces per-step SDP diagnostics (e.g. MaxS / feasibility solves).
    # The --debug CLI flag overrides this to DEBUG for a single run.
    log_level: str = "INFO"

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
            model_dict["initialization"] = InitializationConfig(
                **_known_fields(InitializationConfig, "model.initialization", model_dict["initialization"])
            )
        
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
            eval_config = EvaluationConfig(**_known_fields(EvaluationConfig, "evaluation", eval_dict))

        return cls(
            data=DataConfig(**_known_fields(DataConfig, "data", data_dict)),
            model=ModelConfig(**_known_fields(ModelConfig, "model", model_dict)),
            optimizer=OptimizerConfig(**_known_fields(OptimizerConfig, "optimizer", optimizer_dict)),
            training=TrainingConfig(**_known_fields(TrainingConfig, "training", training_dict)),
            mlflow=MLflowConfig(**_known_fields(MLflowConfig, "mlflow", config_dict.get("mlflow", {}))),
            evaluation=eval_config,
            output_dir=config_dict.get("output_dir", "outputs"),
            model_dir=config_dict.get("model_dir", "models"),
            log_dir=config_dict.get("log_dir", "logs"),
            root_dir=config_dict.get("root_dir", None),
            seed=config_dict.get("seed", None),
            log_level=config_dict.get("log_level", "INFO"),
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
            "log_level": self.log_level,
        }

    def save_yaml(self, path: str):
        """Save configuration to YAML file using a safe-load-compatible representation.

        Tuples (e.g. ``OptimizerConfig.betas``) are converted to lists so the
        resulting file contains no ``!!python/...`` tags and can be read back
        with ``yaml.safe_load``. ``yaml.safe_dump`` is used as a belt-and-braces
        check — it raises if any non-trivial Python type sneaks into the dict.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(
                _to_safe_yaml(self.to_dict()), f, default_flow_style=False, sort_keys=False
            )

    def save_json(self, path: str):
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _to_safe_yaml(obj):
    """Recursively convert tuples to lists so a dict is safe_dump-able."""
    if isinstance(obj, tuple):
        return [_to_safe_yaml(x) for x in obj]
    if isinstance(obj, list):
        return [_to_safe_yaml(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_safe_yaml(v) for k, v in obj.items()}
    return obj


class _SafeLoaderWithTuple(yaml.SafeLoader):
    """SafeLoader with one extra constructor for legacy ``!!python/tuple`` tags.

    Older per-run YAMLs were written by ``yaml.dump`` and contain a
    ``!!python/tuple`` tag for ``OptimizerConfig.betas``. We don't want to
    grant the full unsafe loader (``yaml.full_load`` would happily
    instantiate *any* tagged Python object), so we extend SafeLoader with
    a single explicit constructor that maps the tuple tag to a list.
    """


def _construct_python_tuple_as_list(loader, node):
    return loader.construct_sequence(node)


_SafeLoaderWithTuple.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple_as_list
)


def resolve_run_artifacts(
    run_id: str,
    data_root: str = "~/genSecSysId-Data",
) -> Tuple["Config", Path, Optional[Path], Optional[Dict[str, Any]]]:
    """Resolve an MLflow training-run id to all per-run artefacts on disk.

    train.py writes the run files to a standard layout:
        <root>/outputs/<model_type>/<run_id>/config.yaml
        <root>/models/<model_type>/<run_id>/best_model.pt
        <root>/models/<model_type>/<run_id>/normalizer.json
        <root>/models/<model_type>/<run_id>/run_info.json

    Newly written per-run YAMLs are safe_load-compatible (Config.save_yaml
    converts tuples to lists). Older runs were written by yaml.dump and
    contain ``!!python/tuple`` for OptimizerConfig.betas — we read those
    with a SafeLoader subclass that adds *only* a tuple constructor, so
    no arbitrary Python objects can be instantiated even if data_root
    points at untrusted YAML.

    Args:
        run_id: MLflow run id.
        data_root: Base directory containing outputs/ and models/. The
            <model_type> subfolder is discovered automatically.

    Returns:
        config:          Config object reconstructed from the run YAML.
        model_path:      Path to best_model.pt (raises if missing).
        normalizer_path: Path to normalizer.json, or None if not saved.
        run_info:        Dict from run_info.json, or None if not saved.
    """
    base = Path(data_root).expanduser()
    matches = list(base.glob(f"outputs/*/{run_id}/config.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No config.yaml found for run_id={run_id} under {base / 'outputs'}/*/"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple configs match run_id={run_id}: {[str(p) for p in matches]}"
        )
    config_path = matches[0]
    model_type = config_path.parent.parent.name
    with open(config_path) as f:
        cfg_dict = yaml.load(f, Loader=_SafeLoaderWithTuple)
    config = Config.from_dict(cfg_dict)

    run_dir = base / "models" / model_type / run_id
    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    normalizer_path = run_dir / "normalizer.json"
    if not normalizer_path.exists():
        normalizer_path = None

    run_info = None
    run_info_path = run_dir / "run_info.json"
    if run_info_path.exists():
        with open(run_info_path) as f:
            run_info = json.load(f)

    return config, model_path, normalizer_path, run_info


def resolve_run_artifacts_mlflow(
    run_id: str,
    tracking_uri: str,
) -> Tuple["Config", Path, Optional[Path], Optional[Dict[str, Any]]]:
    """Resolve an MLflow run id to its artefacts on a (possibly remote) server.

    The remote counterpart to :func:`resolve_run_artifacts`: rather than reading
    a local ``data_root`` layout, it points MLflow at ``tracking_uri`` and
    downloads the per-run artefacts train.py logs (into MLflow's local artifact
    cache):
        outputs/config.yaml
        models/best_model.pt
        models/normalizer.json
        models/run_info.json

    The config YAML is parsed with the same restricted SafeLoader subclass used
    for local runs (only ``!!python/tuple`` is recognised, mapped to a list), so
    a tampered config can't construct arbitrary Python objects.

    Args:
        run_id: MLflow run id.
        tracking_uri: MLflow tracking URI, e.g. ``http://host/`` or
            ``file:///path/to/mlruns``.

    Returns:
        The same 4-tuple as :func:`resolve_run_artifacts`:
        ``(config, model_path, normalizer_path, run_info)``.
    """
    import mlflow  # imported lazily so sysid.config has no hard mlflow dep

    log = logging.getLogger(__name__)
    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"Downloading artefacts for run {run_id} from {tracking_uri}")
    outputs_dir = Path(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="outputs")
    )
    models_dir = Path(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="models")
    )

    config_path = outputs_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found in MLflow artifacts (outputs/) for run_id={run_id}"
        )
    with open(config_path) as f:
        cfg_dict = yaml.load(f, Loader=_SafeLoaderWithTuple)
    config = Config.from_dict(cfg_dict)

    model_path = models_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"best_model.pt not found in MLflow artifacts (models/) for run_id={run_id}"
        )

    normalizer_path = models_dir / "normalizer.json"
    if not normalizer_path.exists():
        normalizer_path = None

    run_info = None
    run_info_path = models_dir / "run_info.json"
    if run_info_path.exists():
        with open(run_info_path) as f:
            run_info = json.load(f)

    return config, model_path, normalizer_path, run_info


def setup_mlflow_tracking(
    config: "Config",
    override_uri: Optional[str] = None,
) -> None:
    """Configure MLflow tracking URI and experiment from a Config.

    Mirrors scripts/train.py: prefer the configured tracking URI, fall back
    to local file-based tracking if the remote isn't reachable. Then sets
    the experiment so subsequent ``mlflow.start_run(...)`` calls land in
    the right place.

    Args:
        config: A Config with ``mlflow.tracking_uri`` and
            ``mlflow.experiment_name``.
        override_uri: Optional CLI override taking precedence over the
            config's tracking URI.
    """
    import mlflow  # imported lazily so sysid.config has no hard mlflow dep

    log = logging.getLogger(__name__)
    uri = override_uri if override_uri is not None else config.mlflow.tracking_uri
    if uri:
        try:
            mlflow.set_tracking_uri(uri)
            log.info(f"MLflow tracking URI: {uri}")
        except Exception as e:
            log.warning(f"Failed to connect to MLflow server: {e}")
            log.warning("Falling back to local file-based tracking")
            mlflow.set_tracking_uri(None)
    else:
        log.info("Using local file-based MLflow tracking (./mlruns)")
    mlflow.set_experiment(config.mlflow.experiment_name)
    log.info(f"MLflow experiment: {config.mlflow.experiment_name}")
