import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, overload

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

from sysid.utils import max_abs_output, plot_ellipse_and_parallelogram, torch_bmat

from .base import DznActivation, LureSystem, LureSystemClass, LureSystemSafe
from ..data import DataNormalizer

logger = logging.getLogger(__name__)

EPS = 1e-6


class SimpleLure(nn.Module):
    """Simple Lure system model."""

    def __init__(
        self,
        nd: int,
        ne: int,
        nx: int,
        nw: int,
        activation: str,  # saturation nonlinearity
        custom_params: Optional[dict] = None,
        delta: float = 0.1,
        max_norm_x0: float = 1.0,
        ts: float = 0.1,
    ):
        """
        Initialize the Simple Lure system.

        """
        super().__init__()
        nz = nw
        
        # Check if state padding is enabled (default: True)
        pad_state = custom_params.get("pad_state", False) if custom_params is not None else False
        
        # Store original dataset state dimension
        self.nx_data = nx
        # Optionally pad state dimension to match nz
        self.nx = nz if pad_state else nx
        self.nd = nd
        self.ne = ne
        self.nw = nw
        self.nz = nz
        self.pad_state = pad_state
        self.ts = ts

        # Register delta and max_norm_x0 as buffers (saved with model, not trainable)
        self.register_buffer("delta", torch.tensor(delta))
        self.register_buffer("max_norm_x0_buffer", torch.tensor(max_norm_x0))
        self.max_norm_x0 = max_norm_x0  # Keep as attribute for compatibility

        # Safe output level y_max = max_{i,k} |y_k^(i)| in PHYSICAL units (it has
        # a physical meaning, so it is stored unnormalized). The model's C/P/s
        # live in normalized output units, so the output-coverage machinery
        # scales the certified image up to physical by ``output_std`` (the
        # coverage floor is (output_std·s)²·CPCᵀ ⪰ y_max²). y_max NaN = unset ->
        # the output-coverage penalty is a no-op. Both are persistent=False: set
        # from the data each run, never break loading old checkpoints that
        # predate these buffers.
        self.register_buffer("y_max", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("output_std", torch.tensor(1.0), persistent=False)

        self.P = nn.Parameter(torch.eye(self.nx))  # Lyapunov matrix
        if custom_params is not None:
            learn_L = custom_params.get("learn_L", True)
            self._identity_init_cfg = custom_params.get("identity_init", {}) or {}
        else:
            learn_L = True
            self._identity_init_cfg = {}

        self.learn_L = learn_L

        # Parse structural constraints
        self.structural_constraints = self._parse_structural_constraints(custom_params)

        alpha_0 = 0.9999
        if learn_L:
            self.L = nn.Parameter(torch.zeros((nz, nx)))  # Coupling matrix
            # self.alpha = nn.Parameter(torch.tensor(0.9999), requires_grad=True)
            self.tau = nn.Parameter(torch.tensor(np.log(alpha_0/(1-alpha_0))), requires_grad=True)  # unconstrained parameter for alpha
            self.s = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            # self.s = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        else:
            # Register as buffer so .to(device) moves it with the module
            self.register_buffer("L", torch.zeros((nz, nx)))
            # self.alpha = nn.Parameter(torch.tensor(0.9999), requires_grad=False)
            self.tau = nn.Parameter(torch.tensor(np.log(alpha_0/(1-alpha_0))), requires_grad=False)  # 
            self.s = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.la = nn.Parameter(torch.ones(nz))
        # self.M = torch.diag(self.la)

        # Create system matrices with structural constraints
        self.A = self._create_constrained_parameter(
            'A', (self.nx, self.nx), 
            self.structural_constraints.get('A')
        )
        self.B = self._create_constrained_parameter(
            'B', (self.nx, nd),
            self.structural_constraints.get('B')
        )
        self.B2 = self._create_constrained_parameter(
            'B2', (self.nx, nw),
            self.structural_constraints.get('B2')
        )

        self.C = self._create_constrained_parameter(
            'C', (ne, self.nx),
            self.structural_constraints.get('C')
        )
        self.D = self._create_constrained_parameter(
            'D', (ne, nd),
            self.structural_constraints.get('D')
        )
        self.D12 = self._create_constrained_parameter(
            'D12', (ne, nw),
            self.structural_constraints.get('D12')
        )

        self.C2 = self._create_constrained_parameter(
            'C2', (nz, self.nx),
            self.structural_constraints.get('C2')
        )
        self.D21 = self._create_constrained_parameter(
            'D21', (nz, nd),
            self.structural_constraints.get('D21')
        )
        self.D22 = self._create_constrained_parameter(
            'D22', (nz, nw),
            self.structural_constraints.get('D22')
        )

        Delta: nn.Module
        if activation == "sat":
            Delta = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        elif activation == "dzn":
            Delta = DznActivation()
        elif activation == "tanh":
            Delta = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type: {activation}")

        # for p in [self.A, self.B, self.C, self.D]:
        #     p.requires_grad = False

        self.lure = self._build_lure(
            LureSystemClass(
                A=self.A,
                B=self.B,
                C=self.C,
                D=self.D,
                B2=self.B2,
                C2=self.C2,
                D12=self.D12,
                D21=self.D21,
                D22=self.D22,
                Delta=Delta,
            )
        )

        # Register gradient masks for partially constrained parameters
        self._register_gradient_masks()
        
        # Log structural constraints information
        self._log_structural_constraints()

        # self.initialize_parameters()

    # Buffers/parameters retired from the state_dict but tolerated when loading
    # older checkpoints. ``dual_penalty`` was a buffer of the removed dual
    # constrained-learning method; checkpoints saved before its removal still
    # carry it.
    _LEGACY_STATE_KEYS: Tuple[str, ...] = ("dual_penalty",)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load a state dict, silently dropping retired keys (see
        ``_LEGACY_STATE_KEYS``) so pre-existing checkpoints keep loading.

        Everything else is still loaded strictly — unknown or missing keys
        other than the retired ones raise as usual.
        """
        filtered = {
            k: v for k, v in state_dict.items() if k not in self._LEGACY_STATE_KEYS
        }
        return super().load_state_dict(filtered, *args, **kwargs)

    def _parse_structural_constraints(self, custom_params: Optional[dict]) -> dict:
        """
        Parse and validate structural constraints from custom_params.
        
        Args:
            custom_params: Dictionary containing model-specific parameters
            
        Returns:
            dict: Validated constraint specifications mapping {param_name: constraint_spec}
                  constraint_spec can be:
                  - {'fixed': True, 'value': <value>}
                  - {'learnable_rows': [...], 'fixed_value': <val>}
                  - {'learnable_cols': [...], 'fixed_value': <val>}
                  - None (fully learnable)
        
        Raises:
            ValueError: If constraints are invalid
        """
        if custom_params is None:
            return {}
        
        constraints = custom_params.get('structural_constraints', {})
        if not constraints:
            return {}
        
        # Validate each constraint spec
        valid_params = ['A', 'B', 'B2', 'C', 'D', 'D12', 'C2', 'D21', 'D22']
        validated_constraints = {}
        
        for param_name, spec in constraints.items():
            # Check if it's a valid parameter name
            if param_name not in valid_params:
                logger.warning(
                    f"Unknown parameter '{param_name}' in structural_constraints - ignoring"
                )
                continue
            
            # Validate fixed parameters
            if spec.get('fixed', False):
                if 'value' not in spec:
                    raise ValueError(
                        f"Parameter '{param_name}' marked as fixed but no value provided. "
                        f"Please specify 'value' in the constraint."
                    )
            
            # Validate learnable_rows/cols
            if 'learnable_rows' in spec and 'learnable_cols' in spec:
                raise ValueError(
                    f"Parameter '{param_name}' cannot have both 'learnable_rows' and "
                    f"'learnable_cols'. Please specify only one."
                )
            
            validated_constraints[param_name] = spec
        
        return validated_constraints
    
    def _create_constrained_parameter(
        self, 
        name: str, 
        shape: tuple, 
        constraint_spec: Optional[dict]
    ) -> nn.Parameter:
        """
        Create a parameter with structural constraints applied.
        
        Args:
            name: Parameter name (e.g., 'A', 'B2', 'C')
            shape: Parameter shape tuple
            constraint_spec: Constraint specification from config
        
        Returns:
            nn.Parameter with appropriate requires_grad and initial value
            
        Raises:
            ValueError: If constraint specification is invalid
        """
        if constraint_spec is None:
            # Fully learnable (default behavior)
            return nn.Parameter(torch.zeros(shape, dtype=torch.float64))
        
        if constraint_spec.get('fixed', False):
            # Fully fixed parameter
            value = constraint_spec['value']
            
            # Convert value to tensor of correct shape
            if isinstance(value, (int, float)):
                # Scalar: broadcast to full shape
                tensor = torch.full(shape, float(value), dtype=torch.float64)
            elif isinstance(value, list):
                # List or nested list: convert to tensor
                tensor = torch.tensor(value, dtype=torch.float64)
                if tensor.shape != shape:
                    raise ValueError(
                        f"Shape mismatch for fixed parameter '{name}': "
                        f"expected {shape}, got {tensor.shape}. "
                        f"Please check the 'value' specification in your config."
                    )
            else:
                raise ValueError(
                    f"Invalid value type for parameter '{name}': {type(value)}. "
                    f"Expected scalar or list."
                )
            
            # Create non-trainable parameter
            return nn.Parameter(tensor, requires_grad=False)
        
        # Partially learnable: validate indices
        if 'learnable_rows' in constraint_spec:
            learnable_rows = constraint_spec['learnable_rows']
            max_row = shape[0] - 1
            
            for row_idx in learnable_rows:
                if row_idx < 0 or row_idx > max_row:
                    raise ValueError(
                        f"Invalid row index {row_idx} for parameter '{name}' "
                        f"with shape {shape}. Valid range: [0, {max_row}]"
                    )
        
        if 'learnable_cols' in constraint_spec:
            learnable_cols = constraint_spec['learnable_cols']
            max_col = shape[1] - 1 if len(shape) > 1 else 0
            
            for col_idx in learnable_cols:
                if col_idx < 0 or col_idx > max_col:
                    raise ValueError(
                        f"Invalid column index {col_idx} for parameter '{name}' "
                        f"with shape {shape}. Valid range: [0, {max_col}]"
                    )
        
        # Create parameter initialized to fixed_value
        fixed_value = constraint_spec.get('fixed_value', 0.0)
        param = nn.Parameter(torch.full(shape, float(fixed_value), dtype=torch.float64))
        
        # Store constraint info for gradient masking
        if not hasattr(self, '_parameter_constraints'):
            self._parameter_constraints = {}
        self._parameter_constraints[name] = constraint_spec
        
        return param
    
    def _create_gradient_mask(
        self, 
        name: str, 
        shape: tuple, 
        constraint_spec: dict
    ) -> Optional[torch.Tensor]:
        """
        Create gradient mask tensor from constraint specification.
        
        Args:
            name: Parameter name
            shape: Parameter shape
            constraint_spec: Constraint specification
            
        Returns:
            Mask tensor (0 for fixed elements, 1 for learnable) or None if no masking needed
        """
        if constraint_spec.get('fixed', False):
            return None  # Fully fixed, no gradient anyway
        
        mask = torch.ones(shape, dtype=torch.float64)
        
        if 'learnable_rows' in constraint_spec:
            learnable_rows = constraint_spec['learnable_rows']
            # Zero out all rows first
            mask.zero_()
            # Enable learnable rows
            for row_idx in learnable_rows:
                mask[row_idx, :] = 1.0
        
        elif 'learnable_cols' in constraint_spec:
            learnable_cols = constraint_spec['learnable_cols']
            # Zero out all columns first
            mask.zero_()
            # Enable learnable columns
            for col_idx in learnable_cols:
                mask[:, col_idx] = 1.0
        
        elif 'learnable_elements' in constraint_spec:
            learnable_elements = constraint_spec['learnable_elements']
            mask.zero_()
            for (i, j) in learnable_elements:
                mask[i, j] = 1.0
        
        else:
            return None  # No masking needed
        
        return mask
    
    def _register_gradient_masks(self):
        """
        Register gradient hooks for partially constrained parameters.
        
        For each parameter with partial constraints (learnable_rows/cols), 
        registers a hook that zeros out gradients for non-learnable elements.
        """
        if not hasattr(self, '_parameter_constraints') or not self._parameter_constraints:
            return
        
        param_map = {
            'A': self.A, 'B': self.B, 'B2': self.B2,
            'C': self.C, 'D': self.D, 'D12': self.D12,
            'C2': self.C2, 'D21': self.D21, 'D22': self.D22,
        }
        
        for name, constraint_spec in self._parameter_constraints.items():
            if name not in param_map:
                continue
            
            param = param_map[name]
            
            # Skip fully fixed parameters
            if not param.requires_grad:
                continue
            
            # Create gradient mask
            mask = self._create_gradient_mask(name, param.shape, constraint_spec)
            
            if mask is None:
                continue  # No masking needed
            
            # Register hook
            def make_hook(mask_tensor):
                def hook(grad):
                    if grad is None:
                        return None
                    return grad * mask_tensor
                return hook
            
            param.register_hook(make_hook(mask))
            
            # Log mask info
            num_learnable = int(mask.sum().item())
            num_total = int(mask.numel())
            logger.info(
                f"  Registered gradient mask for '{name}': "
                f"{num_learnable}/{num_total} elements learnable "
                f"({100*num_learnable/num_total:.1f}%)"
            )
    
    def _is_parameter_fixed(self, name: str) -> bool:
        """
        Check if a parameter is fully fixed (not trainable at all).
        
        Args:
            name: Parameter name
            
        Returns:
            True if parameter is fully fixed, False otherwise
        """
        if not hasattr(self, 'structural_constraints'):
            return False
        
        if name not in self.structural_constraints:
            return False
        
        return self.structural_constraints[name].get('fixed', False)
    
    def _should_skip_initialization(self, name: str) -> bool:
        """
        Check if parameter initialization should be skipped.
        
        Args:
            name: Parameter name
            
        Returns:
            True for fully fixed parameters (they keep their fixed values),
            False for partially or fully learnable parameters
        """
        return self._is_parameter_fixed(name)
    
    def _apply_partial_initialization(self, name: str, init_data: torch.Tensor):
        """
        Apply initialization data to a partially constrained parameter.
        
        Only updates the learnable portions, keeps fixed portions at fixed_value.
        
        Args:
            name: Parameter name
            init_data: Initialization data tensor
        """
        if not hasattr(self, 'structural_constraints'):
            # No constraints, apply directly
            param = getattr(self, name)
            param.data = init_data
            return
        
        if name not in self.structural_constraints:
            # No constraints on this parameter, apply directly
            param = getattr(self, name)
            param.data = init_data
            return
        
        constraint_spec = self.structural_constraints[name]
        param = getattr(self, name)
        
        if 'learnable_rows' in constraint_spec:
            # Update only learnable rows
            learnable_rows = constraint_spec['learnable_rows']
            for row_idx in learnable_rows:
                param.data[row_idx, :] = init_data[row_idx, :]
        
        elif 'learnable_cols' in constraint_spec:
            # Update only learnable columns
            learnable_cols = constraint_spec['learnable_cols']
            for col_idx in learnable_cols:
                param.data[:, col_idx] = init_data[:, col_idx]
        
        elif 'learnable_elements' in constraint_spec:
            # Update only specific elements
            learnable_elements = constraint_spec['learnable_elements']
            for (i, j) in learnable_elements:
                param.data[i, j] = init_data[i, j]
        else:
            # No partial constraint, apply directly
            param.data = init_data
    
    def _log_structural_constraints(self):
        """Log structural constraints information."""
        if not hasattr(self, 'structural_constraints') or not self.structural_constraints:
            logger.info("No structural constraints specified - all parameters fully learnable")
            return
        
        logger.info("=" * 80)
        logger.info("STRUCTURAL CONSTRAINTS ACTIVE")
        logger.info("=" * 80)
        
        for name, spec in self.structural_constraints.items():
            if spec.get('fixed', False):
                value_str = str(spec['value'])
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                logger.info(f"  {name}: FULLY FIXED to {value_str}")
            elif 'learnable_rows' in spec:
                rows = spec['learnable_rows']
                fixed_val = spec.get('fixed_value', 0.0)
                logger.info(
                    f"  {name}: Partially learnable - only rows {rows} "
                    f"(fixed rows set to {fixed_val})"
                )
            elif 'learnable_cols' in spec:
                cols = spec['learnable_cols']
                fixed_val = spec.get('fixed_value', 0.0)
                logger.info(
                    f"  {name}: Partially learnable - only cols {cols} "
                    f"(fixed cols set to {fixed_val})"
                )
            elif 'learnable_elements' in spec:
                elements = spec['learnable_elements']
                logger.info(f"  {name}: Partially learnable - specific elements {elements}")
        
        logger.info("=" * 80)

    def initialize_parameters(
        self,
        train_inputs,
        train_states,
        train_outputs,
        init_config=None,
        normalizer: Optional[DataNormalizer] = None,
    ):
        """Initialize model parameters with the **identity** strategy.

        Sets a stable diagonal A, identity-like C, and configurable random
        B2/C2/D21 (see :meth:`_init_identity`), then establishes the certificate
        (P, L, s) via MinTrProb (step 2 of the algorithm). ESN / N4SID inits were
        removed — ``identity`` is the only supported method.

        Args:
            train_inputs: Training input data (B, N, nd).
            train_states: Training state data (unused by the identity init; kept
                for API symmetry with the loaders).
            train_outputs: Training output data (B, N, ne) — used for y_max.
            init_config: InitializationConfig; ``method`` must be ``'identity'``.
            normalizer: Data normalizer used to scale C/B and derive y_max.
        """
        init_method = (
            getattr(init_config, "method", "identity").lower()
            if init_config is not None else "identity"
        )
        if init_method != "identity":
            raise ValueError(
                f"Unknown initialization method: {init_method!r}. Only 'identity' "
                "is supported (esn/n4sid were removed)."
            )

        logger.info("=" * 80)
        logger.info("INITIALIZATION: Using 'identity' method")
        logger.info("=" * 80)

        # Normalize inputs for the MinTrProb init below (initialize_s_from_conditions
        # expects normalized inputs).
        if normalizer is not None:
            train_inputs = normalizer.transform_inputs(train_inputs)

        self._init_identity(normalizer)

        # Common post-initialization
        constraints_ok = self.check_constraints()
        logger.info(f"Initialization complete. Constraints satisfied: {constraints_ok}")
        logger.info("=" * 80)

        # Step 2 of the clean algorithm: establish the certificate (P, L, s) via
        # MinTrProb from the output + input conditions. This also guarantees
        # feasibility, so no separate analysis_problem_init bootstrap is needed
        # on this path. y_max is PHYSICAL (max |raw training output|); output_std
        # relates the model's normalized C/P/s to physical units.
        init_s = bool(getattr(init_config, "init_s_from_conditions", True)) if init_config is not None else True
        if init_s and self.learn_L and normalizer is not None:
            y_max = max_abs_output(train_outputs)
            output_std = float(np.asarray(normalizer.output_std).reshape(-1)[0])
            self.set_output_coverage_level(y_max, output_std)
            n_grid = int(getattr(init_config, "init_s_grid_size", 15))
            s_max = float(getattr(init_config, "init_s_max", 20.0))
            self.initialize_s_from_conditions(train_inputs, y_max, n_grid=n_grid, s_max=s_max)
        elif not constraints_ok:
            # No MinTrProb init (learn_L=False, or disabled): fall back to a
            # feasibility bootstrap so training starts from a valid certificate.
            b_feasible = self.analysis_problem_init(learn_B=False, learn_D21=True)
            if not b_feasible:
                raise ValueError("Initialization did not satisfy constraints and problem is infeasible. Please check your initialization method and structural constraints.")



    def _resolve_init_spec(
        self,
        name: str,
        shape: tuple,
        default_std: float,
    ) -> torch.Tensor:
        """
        Build initialization tensor for parameter `name` from identity_init config.

        Config (under custom_params['identity_init'][name]):
            {std: float}        -> Gaussian: std * randn(shape)
            {value: [[...]]}    -> Inline fixed start value
            {load_from: "*.npy"}-> Load from file (supports ~ expansion)
            missing             -> Gaussian with default_std
        """
        spec = self._identity_init_cfg.get(name, {}) or {}
        target = getattr(self, name)
        device, dtype = target.device, target.dtype

        if "load_from" in spec:
            path = Path(os.path.expanduser(str(spec["load_from"])))
            if not path.exists():
                raise FileNotFoundError(f"Init file for '{name}' not found: {path}")
            arr = np.load(path)
            tensor = torch.tensor(arr, device=device, dtype=dtype)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"Loaded '{name}' from {path} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(shape)}"
                )
            logger.info(f"  {name}: loaded from {path}")
            return tensor

        if "value" in spec:
            tensor = torch.tensor(spec["value"], device=device, dtype=dtype)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"Fixed init for '{name}' has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(shape)}"
                )
            logger.info(f"  {name}: fixed value from config")
            return tensor

        std = float(spec.get("std", default_std))
        logger.info(f"  {name}: random N(0, {std}^2)")
        return std * torch.randn(*shape, device=device, dtype=dtype)

    def _set_param_data(self, name: str, init_data: torch.Tensor):
        """Assign init_data to parameter `name`, respecting partial constraints."""
        if name in self.structural_constraints:
            self._apply_partial_initialization(name, init_data)
        else:
            getattr(self, name).data = init_data

    def _init_identity(self, normalizer: Optional[DataNormalizer] = None):
        """
        Identity initialization: stable Euler-discretized A, identity-like C,
        configurable random B2/C2/D21.

        Configurable via ``custom_params['identity_init']``. Each entry accepts:
            {std: float}             -> Gaussian random (B2, C2, D21)
            {scale: float}           -> Uniform random magnitude (A's last row only)
            {value: [[...]]}         -> Inline fixed start value
            {load_from: "*.npy"}     -> Load fixed start value from file

        Defaults reproduce the previous behavior (A_scale=1, B2_std=ts,
        C2_std=1, D21_std=1). Respects ``structural_constraints``: only
        learnable parts are touched.
        """
        logger.info("Identity initialization")
        cfg = self._identity_init_cfg

        # --- A: I + ts * A_ct, with last row of A_ct = -scale * U(0,1) ---
        if not self._should_skip_initialization('A'):
            A_spec = cfg.get('A', {}) or {}
            if 'value' in A_spec or 'load_from' in A_spec:
                A_init = self._resolve_init_spec('A', (self.nx, self.nx), default_std=0.0)
            else:
                A_scale = float(A_spec.get('scale', 1.0))
                device, dtype = self.A.device, self.A.dtype
                A_ct = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device, dtype=dtype)
                A_ct[1, :] = -A_scale * torch.rand((1, self.nx), device=device, dtype=dtype)
                A_init = torch.eye(self.nx, device=device, dtype=dtype) + A_ct * self.ts  # Euler discretization
                logger.info(f"  A: scale={A_scale}, |eig|={torch.linalg.eigvals(A_init).abs().tolist()}")
            self._set_param_data('A', A_init)

        # --- B: deterministic input_scale * ts * [0; 1] (override via value/load_from) ---
        if not self._should_skip_initialization('B'):
            B_spec = cfg.get('B', {}) or {}
            if 'value' in B_spec or 'load_from' in B_spec:
                B_init = self._resolve_init_spec('B', (self.nx, self.nd), default_std=0.0)
            else:
                input_scale = 1.0
                if normalizer is not None:
                    input_std = getattr(normalizer, 'input_std', None)
                    if input_std is not None:
                        input_scale = input_std.squeeze()
                B_init = input_scale * self.ts * torch.tensor(
                    [[0.0], [1.0]], device=self.B.device, dtype=self.B.dtype
                )
                # B_init = 0.01*self.ts * torch.tensor(
                #     [[0.0], [1.0]], device=self.B.device, dtype=self.B.dtype
                # )
            self._set_param_data('B', B_init)

        # --- B2, C2, D21: random (configurable std) ---
        if not self._should_skip_initialization('B2'):
            B2_init = self._resolve_init_spec('B2', (self.nx, self.nw), default_std=float(self.ts))
            self._set_param_data('B2', B2_init)

        if not self._should_skip_initialization('C2'):
            C2_init = self._resolve_init_spec('C2', (self.nz, self.nx), default_std=1.0)
            self._set_param_data('C2', C2_init)

        # --- C: identity-like / output_std (override via value/load_from) ---
        if not self._should_skip_initialization('C'):
            C_spec = cfg.get('C', {}) or {}
            if 'value' in C_spec or 'load_from' in C_spec:
                C_init = self._resolve_init_spec('C', (self.ne, self.nx), default_std=0.0)
            else:
                if normalizer is None or getattr(normalizer, 'output_std', None) is None:
                    raise ValueError(
                        "Identity initialization of 'C' requires a normalizer with "
                        "'output_std', or an explicit 'identity_init.C.value' / "
                        "'identity_init.C.load_from' override in custom_params."
                    )
                assert normalizer is not None and normalizer.output_std is not None
                C_init = (1.0 / normalizer.output_std.squeeze()) * torch.tensor(
                    [[1.0, 0.0]], device=self.C.device, dtype=self.C.dtype
                )
                # C_init = 0.01 * torch.tensor(
                #     [[1.0, 0.0]], device=self.C.device, dtype=self.C.dtype
                # )
            self._set_param_data('C', C_init)

        # --- D, D12: zero direct feedthrough ---
        if not self._should_skip_initialization('D'):
            self.D.data = torch.zeros_like(self.D)
        if not self._should_skip_initialization('D12'):
            self.D12.data = torch.zeros_like(self.D12)

        if not self._should_skip_initialization('D21'):
            D21_init = self._resolve_init_spec('D21', (self.nz, self.nd), default_std=1.0)
            self._set_param_data('D21', D21_init)

        logger.info(f"  ||A||={np.linalg.norm(self.A.detach().cpu().numpy()):.4f}")
        logger.info(f"  ||C||={np.linalg.norm(self.C.detach().cpu().numpy()):.4f}")
        logger.info(f"  ||C2||={np.linalg.norm(self.C2.detach().cpu().numpy()):.4f}")

    def _feasibility_sdp(self, s: float) -> Optional[dict]:
        """**Feasibility** — fixed-``s`` certificate feasibility SDP. Pure — does
        NOT mutate the model.

        Given the (fixed) prediction parameters, ``alpha`` and a fixed ``s``,
        find certificate variables P ≻ 0, M = diag(m) ⪰ 0, L satisfying the
        stability LMI ``F ⪯ -εI`` and the locality LMIs
        ``[1/s², l_i; l_i^T, P] ⪰ εI`` (with ``1/s²`` a *constant*, so this is a
        convex feasibility SDP — ``s`` is not optimized). A small size objective
        ``min t`` with ``‖P‖ ≤ t, ‖M‖ ≤ t`` keeps the solution well conditioned.

        Returns ``{P, L, M}`` (numpy) or ``None`` if infeasible / the solver
        fails. This is the within-epoch repair: ``s`` stays where gradient +
        the input/output penalties put it; only P, M, L are repaired.
        """
        A = self.A.cpu().detach().numpy()
        B = self.B.cpu().detach().numpy()
        B2 = self.B2.cpu().detach().numpy()
        C2 = self.C2.cpu().detach().numpy()
        D21 = self.D21.cpu().detach().numpy()
        alpha = 1 / (1 + np.exp(-self.tau.cpu().detach().numpy()))
        s_hat = 1.0 / float(s) ** 2

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        L = cp.Variable((self.nz, self.nx))
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)

        F = cp.bmat(
            [
                [-(alpha**2) * P, np.zeros((self.nx, self.nd)), P @ C2.T + L.T, P @ A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P + L, D21, -2 * M, M @ B2.T],
                [A @ P, B, B2 @ M, -P],
            ]
        )
        nF = F.shape[0]
        constraints = [F << -EPS * np.eye(nF), m >= 0]
        for i in range(self.nz):
            li = L[i, :].reshape((1, -1), order="C")
            constraints.append(
                cp.bmat([[np.array([[s_hat]]), li], [li.T, P]]) >> EPS * np.eye(self.nx + 1)
            )

        # Well-conditioning objective (keep P, M bounded).
        t = cp.Variable((1, 1))
        constraints += [cp.norm(P) <= t, cp.norm(M) <= t]

        problem = cp.Problem(cp.Minimize(t), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.debug(f"feasibility SDP failed at s={s:.4f}: {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None
        return {"P": P.value, "L": L.value, "M": M.value, "s": float(s)}

    def feasibility_problem(self) -> bool:
        """Repair the certificate at the **current** (fixed) ``s`` — the
        within-epoch step 3.1.

        When a gradient update breaks the LMIs, solve :meth:`_feasibility_sdp`
        at ``self.s`` (unchanged) for feasible P, M, L and write them back.
        Returns ``True`` on success; ``False`` when no feasible P, M, L exists
        at that ``s`` (→ the trainer rolls the step back). ``s`` is owned by
        gradient + the input/output penalties and is never modified here.
        """
        s = float(self.s.cpu().detach().numpy())
        sol = self._feasibility_sdp(s)
        if sol is None:
            return False
        # Write back P, L, Λ only — s stays fixed.
        device, dtype = self.P.device, self.P.dtype
        self.P.data = torch.tensor(sol["P"], device=device, dtype=dtype)
        self.L.data = torch.tensor(sol["L"], device=device, dtype=dtype)
        self.la.data = torch.tensor(np.diag(sol["M"]), device=device, dtype=dtype)
        return True

    def analysis_problem_init(self, learn_B: bool= False, learn_D21: bool = False) -> bool:

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        la = cp.Variable((self.nz, 1))
        M = cp.diag(la)
        A = self.A.cpu().detach().numpy()
        if learn_B:
            B = cp.Variable(self.B.shape)
        else:
            B = self.B.cpu().detach().numpy()
        if learn_D21:
            D21 = cp.Variable(self.D21.shape)
        else:
            D21 = self.D21.cpu().detach().numpy()
        B2 = self.B2.cpu().detach().numpy()
        C2 = self.C2.cpu().detach().numpy()
        alpha = 1/(1+ np.exp(-self.tau.cpu().detach().numpy()))
        if self.learn_L:
            s_hat = cp.Variable((1,1))
            # s_hat = np.array([[1/self.s.cpu().detach().numpy()**2]])
        else:
            s = self.s

        multiplier_constraints = []
        if self.learn_L:
            L = cp.Variable((self.nz, self.nx))
        else:
            L = self.L.cpu().detach().numpy()

        for li in L:  # type: ignore[attr-defined]  # cvxpy Variable is iterable at runtime
            if self.learn_L:
                li = li.reshape((1, -1), "C")
                multiplier_constraints.append(
                    cp.bmat(
                        [
                            [s_hat, li],
                            [li.T, P],
                        ]
                    )
                    >> EPS * np.eye(self.nx + 1)
                )
            else:
                li = li.reshape((1, -1))
                multiplier_constraints.append(
                    cp.bmat(
                        [
                            [np.array([[1 / s**2]]), li],
                            [li.T, P],
                        ]
                    )
                    >> EPS * np.eye(self.nx + 1)
                )

        F = cp.bmat(
            [
                [-(alpha**2) * P, np.zeros((self.nx, self.nd)), P @ C2.T + L.T, P @ A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P + L, D21, -2 * M, M @ B2.T],
                [A @ P, B, B2 @ M, -P],
            ]
        )

        t = cp.Variable((1,1))

        size_constraints = [
            cp.norm(P) <= t,
            cp.norm(M) <= t,
        ]

        nF = F.shape[0]
        if self.learn_L:
            problem = cp.Problem(
                cp.Minimize(s_hat),
                [
                    F << -EPS * np.eye(nF), 
                    *multiplier_constraints, 
                ],
            )
            # problem = cp.Problem(
            #     cp.Minimize(None),
            #     [
            #         F << -EPS * np.eye(nF), 
            #         *multiplier_constraints, 
            #     ],
            # )
        else:
            problem = cp.Problem(
                cp.Minimize([None]),
                [
                    F << -EPS * np.eye(nF), 
                    # *multiplier_constraints, 
                ],
            )
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception:
            return False  # SDP failed due to solver error
        if not problem.status == "optimal":
            return False  # SDP failed to find feasible solution    
        logger.info(f"SDP analysis problem solved: {problem.status}")

        device, dtype = self.P.device, self.P.dtype
        if self.learn_L:
            s = 1/torch.sqrt(torch.tensor(s_hat.value, device=device, dtype=dtype).squeeze())
            self.s.data = s
            logger.info(f"  Initial s from SDP: {s.item():.6f}")
        self.P.data = torch.tensor(P.value, device=device, dtype=dtype)
        self.la.data = torch.tensor(np.diag(M.value), device=device, dtype=dtype)
        # self.M.data = torch.tensor(M.value)
        if self.learn_L:
            self.L.data = torch.tensor(L.value, device=device, dtype=dtype)
        if learn_B:
            self.B.data = torch.tensor(B.value, device=device, dtype=dtype)
        if learn_D21:
            self.D21.data = torch.tensor(D21.value, device=device, dtype=dtype)


        return True  # SDP successfully found feasible solution

    def get_lmis(self):
        lmi_list = []
        """Construct the LMI for stability constraint."""
        alpha = 1/(1+ torch.exp(-self.tau))
        M = torch.diag(self.la)
        def stability_lmi() -> torch.Tensor:
            device = self.P.device
            dtype = self.P.dtype
            F = torch_bmat(
                [
                    [
                        -alpha**2 * self.P,
                        torch.zeros((self.nx, self.nd), device=device, dtype=dtype),
                        self.P @ self.C2.T + self.L.T,
                        self.P @ self.A.T,
                    ],
                    [torch.zeros((self.nd, self.nx), device=device, dtype=dtype), -torch.eye(self.nd, device=device, dtype=dtype), self.D21.T, self.B.T],
                    [self.C2 @ self.P + self.L, self.D21, -2 * M, M @ self.B2.T],
                    [self.A @ self.P, self.B, self.B2 @ M, -self.P],
                ]
            )
            return -0.5 * (F + F.T)  # to ensure symmetry

        lmi_list.append(stability_lmi)

        """Construct the LMIs for locality constraints."""
        for l_i in self.L:
            l_i = l_i.reshape(1, -1)

            def locality_lmi_i(l_i=l_i) -> torch.Tensor:
                R = torch_bmat([[(1 / self.s**2).reshape(1, 1), l_i], [l_i.T, self.P]])
                return 0.5 * (R + R.T)

            lmi_list.append(locality_lmi_i)

        return lmi_list

    def get_scalar_inequalities(self):
        inequalities = []
        """Construct scalar inequalities for positivity of la."""

        def s_positive() -> torch.Tensor:
            return self.s
        
        inequalities.append(s_positive)



        def input_size_condition() -> torch.Tensor:
            return -(self.delta**2 - (1 - self.alpha**2) * self.s**2) + 1e-3  # small margin

        # inequalities.append(input_size_condition)

        def alpha_smaller_one() -> torch.Tensor:
            return 1.0 - self.alpha

        # inequalities.append(alpha_smaller_one)

        def alpha_positive() -> torch.Tensor:
            return self.alpha

        # inequalities.append(alpha_positive)

        return inequalities

    def check_constraints(self) -> bool:
        """Check if the Lure system constraints are satisfied."""
        with torch.no_grad():
            for lmi in self.get_lmis():
                _, info = torch.linalg.cholesky_ex(lmi())
                if info > 0:
                    return False

            for inequality in self.get_scalar_inequalities():
                if inequality() < 0:
                    return False
        return True

    def _build_lure(self, sys: LureSystemClass) -> LureSystem:
        """Construct the inner Lure dynamics. Subclasses override to swap in
        a filtered variant (see ``SimpleLureSafe``)."""
        return LureSystem(sys)

    def _prepare_x0(self, x0: Optional[torch.Tensor], B: int) -> torch.Tensor:
        if x0 is None:
            return torch.zeros(size=(B, self.nx, 1), device=self.P.device, dtype=self.P.dtype)
        # Handle padding if pad_state is enabled and x0 comes from dataset (nx_data dim)
        if self.pad_state and x0.shape[1] == self.nx_data:
            x0_padded = torch.zeros(B, self.nx, 1, device=x0.device, dtype=x0.dtype)
            if x0.ndim == 2:
                x0_padded[:, :self.nx_data, 0] = x0
            else:
                x0_padded[:, :self.nx_data, :] = x0
            return x0_padded
        return x0

    def _run_lure(
        self,
        ds: torch.Tensor,
        x0: torch.Tensor,
        warmup_steps: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Dispatch to the inner Lure dynamics. Subclasses can override to
        inject the safe-set arguments without re-implementing ``forward``."""
        return self.lure(d=ds, x0=x0)

    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        warmup_steps: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run the constrained RNN.

        Args:
            d: input ``(B, seq_len, nd)``.
            x0: initial state ``(B, nx, 1)`` or ``(B, nx_data)`` (padded if
                ``pad_state``). Defaults to zeros.
            warmup_steps: number of leading steps the safety filter is
                bypassed for (only used by ``SimpleLureSafe``).

        Returns:
            ``(e_hat, (x, w), d_applied)`` with shapes
                ``e_hat``: ``(B, seq_len, ne)``
                ``x``:     ``(B, seq_len + 1, nx)`` — full state trajectory
                ``w``:     ``(B, seq_len, nw)``
                ``d_applied``: ``(B, seq_len, nd)`` — equal to ``d`` for the
                plain class; the filtered input for ``SimpleLureSafe``.
        """
        B, N, nd = d.shape
        assert self.lure._nd == nd
        x0 = self._prepare_x0(x0, B)
        ds = d.reshape(shape=(B, N, nd, 1))

        es_hat, (x_seq, w_seq), ds_applied = self._run_lure(ds, x0, warmup_steps)

        e_hat = es_hat.reshape(B, N, self.lure._ne)
        x = x_seq.reshape(B, N + 1, self.nx)
        w = w_seq.reshape(B, N, self.lure._nw)
        d_applied = ds_applied.reshape(B, N, nd)
        return e_hat, (x, w), d_applied

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_system_matrices(self):
        """
        Freeze A, B, C, D (and related) system matrices for post-processing.
        Only P and L remain trainable for constraint optimization.

        This is useful for post-processing where you want to keep the learned
        dynamics fixed but optimize the Lyapunov certificate.
        """
        # Freeze linear system matrices
        self.A.requires_grad = False
        self.B.requires_grad = False
        self.B2.requires_grad = False
        self.C.requires_grad = False
        self.D.requires_grad = False
        self.D12.requires_grad = False
        self.C2.requires_grad = False
        self.D21.requires_grad = False
        self.D22.requires_grad = False

        # Freeze stability parameters
        # self.alpha.requires_grad = False
        self.tau.requires_grad = False
        self.s.requires_grad = False

        # Keep P and L trainable (if L is learnable)
        self.P.requires_grad = True
        if self.learn_L:
            self.L.requires_grad = True

        logger.info("Froze system matrices A, B, C, D. P and L remain trainable.")

    def unfreeze_all_parameters(self):
        """Unfreeze all parameters for normal training."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all parameters.")

    def get_frozen_parameters_info(self) -> dict:
        """
        Get information about which parameters are frozen/trainable.

        Returns:
            Dictionary with parameter names and their trainable status
        """
        param_info = {}
        for name, param in self.named_parameters():
            param_info[name] = {
                "shape": tuple(param.shape),
                "requires_grad": param.requires_grad,
                "num_elements": param.numel(),
            }
        return param_info

    def post_process(self) -> dict:
        """
        Post-process the model by solving an SDP to find optimal P and L
        while keeping system matrices (A, B, C, D) fixed.

        This solves the following SDP:
        - Decision variables: P (Lyapunov), L (coupling), m (multipliers), S_hat (optional)
        - Constraints: Main LMI for stability, locality LMIs, positive definiteness
        - Objective: minimize S_hat (minimize s) or feasibility

        Args:
            optimize_s: If True, optimize for minimum s. If False, keep s fixed.
            eps: Small positive constant for strict inequalities (default: 1e-3)

        Returns:
            Dictionary with results including:
                - success: bool, whether SDP was solved successfully
                - P_opt: Optimized Lyapunov matrix
                - L_opt: Optimized coupling matrix
                - s_opt: Optimized sector bound
                - max_eig_F: Maximum eigenvalue of F matrix
                - summary: Dictionary with comparison metrics
        """
        import cvxpy as cp
        import numpy as np

        logger.info("=" * 80)
        logger.info(
            "POST-PROCESSING: Solving SDP for optimal s with P, L and M as decision variables"
        )
        logger.info("=" * 80)

        # Extract current parameters
        A = self.A.cpu().detach().numpy()
        B = self.B.cpu().detach().numpy()
        B2 = self.B2.cpu().detach().numpy()
        C2 = self.C2.cpu().detach().numpy()
        D21 = self.D21.cpu().detach().numpy()
        alpha = 1/(1+ np.exp(-self.tau.cpu().detach().numpy()))
        # alpha = self.alpha.cpu().detach().numpy()
        s_original = self.s.cpu().detach().numpy()
        # L_original = self.L.cpu().detach().numpy() if self.learn_L else None

        P_original = self.P.cpu().detach().numpy()
        L_original = self.L.cpu().detach().numpy() if self.learn_L else None  # Currently unused
        H_original = L_original @ np.linalg.inv(P_original) if self.learn_L else None

        logger.info(f"Current alpha = {alpha:.6f}, s = {s_original:.6f}")

        # Max-s SDP (MaxS, _max_s_sdp): fix the prediction
        # parameters, optimize P, L, M (=Λ) and s, then write the solution back.
        sol = self._max_s_sdp()
        if sol is None:
            return {"success": False, "status": "max_s_sdp_failed"}

        P_star = sol["P"]
        L_star = sol["L"]
        s_star = sol["s"]
        max_eig_F = sol["max_eig_F"]

        self._apply_certificate_solution(sol)

        logger.info("✓ max-s SDP solved successfully")
        logger.info(f"Max eigenvalue of F: {max_eig_F:.6e}")
        for min_eig_Gi in sol["locality_min_eigs"]:
            if min_eig_Gi < 0:
                logger.warning(f"Locality LMI violated: min eigenvalue = {min_eig_Gi:.6e}")
            else:
                logger.info(f"Locality LMI satisfied: min eigenvalue = {min_eig_Gi:.6e}")

        # calculate output bound
        C = self.C.cpu().detach().numpy()
        X_star = np.linalg.inv(P_star)

        # Y_tilde = cp.Variable((self.ne,self.ne), symmetric=True)
        Y = cp.Variable((self.ne, self.ne), symmetric=True)

        constraints = [
            Y >> EPS * np.eye(self.ne),
            X_star / s_star**2 - C.T @ Y @ C >> EPS * np.eye(self.nx),
        ]
        # constraints.append(E >> EPS * np.eye(self.nx + self.ne))
        # constraints.append(cp.bmat([
        #     [Y_tilde, C @ P, D],
        #     [(C@P).T, P/s**2, np.zeros((nx,nd))],
        #     [D.T, np.zeros((nd,nx)), 1/(s**2*(1-alpha**2))]
        # ]) >> EPS * np.eye(nx+ne+nd))

        objective = cp.Maximize(cp.lambda_min(Y))
        problem = cp.Problem(objective, constraints)
        logger.info(f"Solving output SDP with {len(constraints)} constraints using MOSEK...")

        try:
            problem.solve(solver=cp.MOSEK, verbose=False, accept_unknown=True)
        except Exception as e:
            logger.error(f"SDP solver failed: {e}")
            # return {"success": False, "error": str(e)}

        # Check solution status
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"SDP failed with status: {problem.status}")
            y_bar_n = -1
            # return {"success": False, "status": problem.status}
        else:
            # Y = np.linalg.inv(Y_tilde.value)
            Y_star = Y.value
            assert Y_star is not None
            if self.ne == 1:
                y_bar_n = np.sqrt(1/Y_star[0,0])
            else:
                y_bar_n = -1 # needs to be handled differently
            logger.info(f'Normalized output range {y_bar_n}')

        logger.info(f"✓ output SDP solved successfully: {problem.status}")

        # for ne=1 we can directly calculate y_bar
        if self.ne == 1:
            y_bar_n_exact = float(s_star * np.sqrt((C @ P_star @ C.T).item()))
            logger.info(f'Exact normalized output range {y_bar_n_exact}')

        # norm H
        H = L_star @ np.linalg.inv(P_star)
        norm_H = np.linalg.norm(H, ord=2)
        logger.info(f"Norm of H = {norm_H:.6f}")

        # Verify constraints
        constraints_satisfied = self.check_constraints()

        summary = {
            "original": {
                "s": float(s_original),
                "max_eig_P": float(np.max(np.linalg.eigvals(P_original))),
                "min_eig_P": float(np.min(np.linalg.eigvals(P_original))),
                "norm_P": float(np.linalg.norm(P_original, ord="fro")),
                "norm_L": float(np.linalg.norm(L_original, ord="fro")) if L_original is not None else 0.0,
                "norm_H": float(np.linalg.norm(H_original, ord="fro")) if H_original is not None else 0.0,
            },
            "optimized": {
                "s": float(s_star),
                "max_eig_P": float(np.max(np.linalg.eigvals(P_star))),
                "min_eig_P": float(np.min(np.linalg.eigvals(P_star))),
                "norm_P": float(np.linalg.norm(P_star, ord="fro")),
                "max_eig_F": float(max_eig_F),
                "norm_H": float(norm_H),
                "norm_L": float(np.linalg.norm(L_star, ord="fro")),
                "y_bar_n": float(y_bar_n)
            },
        }

        # Log results
        logger.info("─" * 80)
        logger.info(f"Original s:      {summary['original']['s']:.6f}")
        logger.info(f"Optimized s:     {summary['optimized']['s']:.6f}")
        # logger.info(f"Max eig(F):      {max_eig_F:.6e}")
        logger.info(f"Constraints OK:  {constraints_satisfied}")
        logger.info("=" * 80)

        return {
            "success": True,
            "P_opt": P_star,
            "L_opt": L_star,
            "s_opt": s_star,
            "max_eig_F": max_eig_F,
            "constraints_satisfied": constraints_satisfied,
            "summary": summary,
        }

    def _max_s_sdp(self) -> Optional[dict]:
        """**MaxS** — the max-feasible-``s`` SDP for the current (fixed)
        prediction parameters. Pure — does NOT mutate the model.

        Fixes everything that influences the predictions (A, B, B2, C2, D21 and
        alpha) and optimizes the certificate variables P, L, M (=Λ) and
        ``S_hat = 1/s^2``. Objective: ``minimize S_hat`` (i.e. maximize ``s``)
        subject to the stability LMI ``F ⪯ -εI`` and the locality LMIs
        ``[S_hat, l_i; l_i^T, P] ⪰ εI``. ``s`` **is** an optimization variable.

        Used to bracket the sweep in :meth:`solve_output_coverage_certificate`
        (its ``s_max``) and as the ``post_process`` diagnostic.

        Returns a dict with the numpy solution (``P``, ``L``, ``M``, ``s``,
        ``max_eig_F``, ``locality_min_eigs``) or ``None`` if the solver fails.
        """
        A = self.A.cpu().detach().numpy()
        B = self.B.cpu().detach().numpy()
        B2 = self.B2.cpu().detach().numpy()
        C2 = self.C2.cpu().detach().numpy()
        D21 = self.D21.cpu().detach().numpy()
        alpha = 1 / (1 + np.exp(-self.tau.cpu().detach().numpy()))

        # Decision variables: P, L, M=diag(m), S_hat = 1/s^2.
        P_current = self.P.cpu().detach().numpy()
        P = cp.Variable((self.nx, self.nx), symmetric=True)
        L = cp.Variable((self.nz, self.nx))
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)
        S_hat = cp.Variable((1, 1))

        # Stability LMI: F <= -eps*I (does not depend on s).
        F = cp.bmat(
            [
                [-(alpha**2) * P, np.zeros((self.nx, self.nd)), P @ C2.T + L.T, P @ A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P + L, D21, -2 * M, M @ B2.T],
                [A @ P, B, B2 @ M, -P],
            ]
        )
        nF = F.shape[0]
        # M = diag(m) is the IQC/sector multiplier and must be ⪰ 0. This is in
        # fact already implied by F ⪯ -εI (the -2·diag(m) block sits on F's
        # diagonal, forcing m_i > ε/2), but we state it explicitly for
        # correctness and robustness to future formulation changes.
        constraints = [F << -EPS * np.eye(nF), m >= 0]

        # Locality LMIs: [S_hat, l_i; l_i^T, P] >= eps*I for each row of L.
        Gs = []
        for i in range(self.nz):
            li = L[i, :].reshape((1, -1), order="C")
            locality_lmi = cp.bmat([[S_hat, li], [li.T, P]])
            Gs.append(locality_lmi)
            constraints.append(locality_lmi >> EPS * np.eye(self.nx + 1))

        problem = cp.Problem(cp.Minimize(S_hat), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.error(f"max-s SDP solver failed: {e}")
            return None
        if problem.status != "optimal":
            logger.error(f"max-s SDP failed with status: {problem.status}")
            return None

        assert S_hat.value is not None
        S_hat_opt = S_hat.value[0, 0] if hasattr(S_hat.value, "__len__") else float(S_hat.value)
        if S_hat_opt <= 0:
            logger.error(f"max-s SDP returned non-positive S_hat ({S_hat_opt})")
            return None
        s_star = float(np.sqrt(1.0 / S_hat_opt))

        min_eig_diff = float(np.min(np.real(np.linalg.eigvals(P_current - P.value))))
        logger.debug(
            f"max-s SDP solved: s = {s_star:.6f}, "
            f"min eig(P_current - P_opt) = {min_eig_diff:.6e}"
        )

        return {
            "P": P.value,
            "L": L.value,
            "M": M.value,
            "s": s_star,
            "max_eig_F": float(np.max(np.real(np.linalg.eigvals(F.value)))),
            "locality_min_eigs": [
                float(np.min(np.real(np.linalg.eigvals(g.value)))) for g in Gs
            ],
        }

    def _apply_certificate_solution(self, sol: dict) -> None:
        """Write a :meth:`_max_s_sdp` solution back into the model (P, L, la, s)."""
        device, dtype = self.P.device, self.P.dtype
        self.P.data = torch.tensor(sol["P"], device=device, dtype=dtype)
        self.L.data = torch.tensor(sol["L"], device=device, dtype=dtype)
        self.la.data = torch.tensor(np.diag(sol["M"]), device=device, dtype=dtype)
        self.s.data = torch.tensor(sol["s"], device=device, dtype=dtype)

    def _coverage_sdp(self, s: float, y_max: float) -> Optional[dict]:
        """Fixed-``s`` convex SDP for the binding-Corollary-1 certificate.

        ``y_max`` is the **physical** safe output level and stays physical. The
        model's C/P/s live in normalized output units, so the certified image is
        scaled up to physical by ``σ = output_std``; the coverage floor is then a
        statement about the physical certified half-width
        ``ȳ = σ·s·√(CPCᵀ) ≥ y_max``. With ``s`` fixed the only nonconvex term
        (``P/s²``) becomes a constant scaling, so this is a genuine convex SDP:

            min tr((σ·s)²·C P Cᵀ)
            s.t.  (5a) F ⪯ -εI,  P ≻ 0,  M = diag(m) ⪰ 0
                  (5b) [1/s², lⁱ; (lⁱ)ᵀ, P] ⪰ εI     (locality, 1/s² constant)
                  (cov) (σ·s)²·C P Cᵀ ⪰ y_max²·I     (coverage-on-image, physical)

        Returns the numpy solution (``P``, ``L``, ``M``, ``s``, ``y_bar`` — the
        PHYSICAL certified half-width ``σ·s·√(CPCᵀ)``) or ``None`` if infeasible
        / the solver fails.
        """
        A = self.A.cpu().detach().numpy()
        B = self.B.cpu().detach().numpy()
        B2 = self.B2.cpu().detach().numpy()
        C = self.C.cpu().detach().numpy()
        C2 = self.C2.cpu().detach().numpy()
        D21 = self.D21.cpu().detach().numpy()
        alpha = 1 / (1 + np.exp(-self.tau.cpu().detach().numpy()))
        s2 = float(s) ** 2

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        L = cp.Variable((self.nz, self.nx))
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)

        F = cp.bmat(
            [
                [-(alpha**2) * P, np.zeros((self.nx, self.nd)), P @ C2.T + L.T, P @ A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P + L, D21, -2 * M, M @ B2.T],
                [A @ P, B, B2 @ M, -P],
            ]
        )
        nF = F.shape[0]
        constraints = [F << -EPS * np.eye(nF)]

        for i in range(self.nz):
            li = L[i, :].reshape((1, -1), order="C")
            constraints.append(
                cp.bmat([[np.array([[1.0 / s2]]), li], [li.T, P]])
                >> EPS * np.eye(self.nx + 1)
            )

        # Coverage-on-image (bind Corollary 1) in PHYSICAL units: the physical
        # certified image (σ·s)²·CPCᵀ must reach y_max² (σ = output_std). y_max
        # stays physical; σ scales the normalized model image up to physical.
        # Uses ne (output dim) — the dimension of C P Cᵀ.
        sigma = float(self.output_std)
        constraints.append(
            (sigma ** 2) * s2 * C @ P @ C.T - float(y_max) ** 2 * np.eye(self.ne)
            >> EPS * np.eye(self.ne)
        )

        problem = cp.Problem(cp.Minimize(cp.trace((sigma ** 2) * s2 * C @ P @ C.T)), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.debug(f"coverage SDP failed at s={s:.4f}: {e}")
            return None
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None

        P_val = P.value
        CPCt = C @ P_val @ C.T
        # Physical certified half-width: y_bar = output_std * s * sqrt(CPCᵀ).
        y_bar = float(sigma * s * np.sqrt(CPCt.item())) if self.ne == 1 else None
        return {
            "P": P_val,
            "L": L.value,
            "M": M.value,
            "s": float(s),
            "y_bar": y_bar,
        }

    def _count_input_violations(
        self, inputs: torch.Tensor, x0: Optional[torch.Tensor], warmup_steps: int
    ) -> int:
        """Number of trajectories whose *unfiltered* rollout breaches the input
        constraint (any ``c_k > 0``) under the current certificate.

        The certificate is a claim about the raw model on admissible inputs, so
        the check is on the unfiltered dynamics even for ``SimpleLureSafe`` (its
        filter would otherwise hide every violation by construction)."""
        with torch.no_grad():
            if hasattr(self, "forward_unfiltered"):
                _, (x, _), u_applied = self.forward_unfiltered(inputs, x0)
            else:
                _, (x, _), u_applied = self.forward(inputs, x0, warmup_steps=warmup_steps)
            _, c = self.get_regularization_input(
                u_applied, x, return_c=True, warmup_steps=warmup_steps
            )
            viol = (torch.nan_to_num(c, nan=float("-inf")) > 0).any(dim=1)
        return int(viol.sum())

    def solve_output_coverage_certificate(
        self,
        y_max: Optional[float] = None,
        inputs: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        warmup_steps: int = 0,
        n_grid: int = 10,
        s_min: float = 0.1,
        s_max: float = 20.0,
    ) -> dict:
        """**MinTrProb** — the binding-Corollary-1 certificate (see the wiki
        ``binding-output-certificate``).

        Produces the tightest certified output interval ``[-ȳ, ȳ]`` with the
        PHYSICAL ``ȳ = y_max`` that (a) satisfies the Lyapunov + regionality
        LMIs and (b) — when ``inputs`` are supplied — leaves **zero input
        violations** on that data. The scale ``s`` is the lone nonconvex degree
        of freedom, so we solve a convex SDP (:meth:`_coverage_sdp`) at each
        point of a fixed grid ``s ∈ [s_min, s_max]`` (default ``[0.1, 20]``; a
        deliberately simple preset — no MaxS bracket / bisection for now) and
        keep the feasible ones. Among the ``s`` with zero input violations we
        pick the tightest ``ȳ`` (the SDP objective already drives ``ȳ`` to
        ``y_max``); if none clears the violations, we take the fewest-violations
        ``s``. If no grid ``s`` is feasible at all, this θ cannot certify
        ``y_max`` — reported, not hidden.

        ``y_max`` is physical; ``None`` uses the model's stored physical
        ``y_max``. The selected certificate is written back into the model.
        Returns a summary dict with ``success``, ``reason``, ``s``, ``y_bar``
        (physical), ``y_max`` (physical), ``s_min``, ``s_max``,
        ``n_input_violations`` and the full ``sweep``.
        """
        if y_max is None:
            if self.y_max is None or bool(torch.isnan(self.y_max)):
                return {"success": False, "reason": "y_max_unset"}
            y_max = float(self.y_max)
        y_max = float(y_max)

        # Save the current certificate so a failed search leaves the model as-is.
        saved = {
            "P": self.P.detach().clone(),
            "L": self.L.detach().clone(),
            "la": self.la.detach().clone(),
            "s": self.s.detach().clone(),
        }

        def _restore():
            with torch.no_grad():
                self.P.data.copy_(saved["P"])
                self.L.data.copy_(saved["L"])
                self.la.data.copy_(saved["la"])
                self.s.data.copy_(saved["s"])

        # Fixed-grid sweep over the preset band [s_min, s_max]. Infeasible s
        # (too small for coverage, or too large for regionality) are skipped.
        s_grid = np.linspace(float(s_min), float(s_max), int(n_grid))
        sweep = []
        for s in s_grid:
            sol = self._coverage_sdp(float(s), y_max)
            if sol is None:
                continue
            n_viol = None
            if inputs is not None:
                self._apply_certificate_solution(sol)
                n_viol = self._count_input_violations(inputs, x0, warmup_steps)
            sweep.append({"s": sol["s"], "y_bar": sol["y_bar"], "n_violations": n_viol, "sol": sol})

        if not sweep:
            _restore()
            return {
                "success": False,
                "reason": "coverage_unreachable",
                "s_min": float(s_min),
                "s_max": float(s_max),
                "y_max": y_max,
            }

        # Selection. Prefer zero-violation candidates; among the eligible ones
        # pick the tightest ȳ (then smallest s). ȳ is None for ne>1 (no scalar
        # tightness) -> fall back to smallest s.
        def _tightness_key(cand):
            yb = cand["y_bar"]
            return (yb if yb is not None else float("inf"), cand["s"])

        if inputs is not None:
            eligible = [c for c in sweep if c["n_violations"] == 0]
            violation_free = bool(eligible)
            if eligible:
                best = min(eligible, key=_tightness_key)
            else:
                # No s clears the violations: take the fewest-violations s
                # (then tightest).
                best = min(sweep, key=lambda c: (c["n_violations"], _tightness_key(c)))
        else:
            violation_free = None
            best = min(sweep, key=_tightness_key)

        self._apply_certificate_solution(best["sol"])
        constraints_ok = self.check_constraints()

        return {
            "success": True,
            "reason": "ok" if (violation_free is not False) else "violations_remain",
            "s": best["s"],
            "y_bar": best["y_bar"],
            "y_max": y_max,
            "s_min": float(s_min),
            "s_max": float(s_max),
            "n_input_violations": best["n_violations"],
            "violation_free": violation_free,
            "constraints_satisfied": constraints_ok,
            "sweep": [
                {"s": c["s"], "y_bar": c["y_bar"], "n_violations": c["n_violations"]}
                for c in sweep
            ],
        }

    def initialize_s_from_conditions(
        self,
        train_inputs_n,
        y_max: float,
        warmup_steps: int = 0,
        n_grid: int = 15,
        s_min: float = 0.1,
        s_max: float = 20.0,
    ) -> dict:
        """Initialize ``s`` (and ``P``, ``L``) from the output + input conditions.

        Replaces the cumbersome max-s / heuristic ``s`` initialization with the
        *same sweep used for the final certificate* (MinTrProb,
        :meth:`solve_output_coverage_certificate`): over the preset ``s`` band it
        prefers an ``s`` that satisfies the (physical) output-coverage floor
        **and** leaves zero input violations on the training data, else the
        ``s`` with the fewest input violations.

        When the output level is not yet reachable for this freshly-initialized
        ``theta`` (``coverage_unreachable``), it falls back to plain max-s (the
        fewest-violations feasible certificate); the output-coverage penalty
        then grows the image toward ``y_max`` during training.

        ``y_max`` is the PHYSICAL safe output level; ``self.output_std`` must
        already be set (see :meth:`set_output_coverage_level`).
        ``train_inputs_n`` are the *normalized* training inputs ``(B, N, nd)``.
        Returns the certificate summary dict (``success=False`` when the max-s
        fallback was used).
        """
        self.set_output_coverage_level(y_max)  # output_std left unchanged
        inputs = torch.as_tensor(
            np.asarray(train_inputs_n), dtype=self.P.dtype, device=self.P.device
        )
        res = self.solve_output_coverage_certificate(
            y_max=y_max, inputs=inputs, warmup_steps=warmup_steps,
            n_grid=n_grid, s_min=s_min, s_max=s_max,
        )
        if res["success"]:
            logger.info(
                f"Init s from conditions: s={res['s']:.6f} "
                f"(band [{res['s_min']:.6f}, {res['s_max']:.6f}]), "
                f"y_bar={res['y_bar']} (y_max={y_max}), "
                f"input violations={res['n_input_violations']}, "
                f"violation_free={res['violation_free']}"
            )
            return res

        # Output level not reachable at init -> fall back to MaxS (fewest input
        # violations among feasible certificates). Coverage is then handled by
        # the output-coverage penalty during training.
        logger.warning(
            f"Init s from conditions: output level y_max={y_max:.6f} not "
            f"reachable (reason={res['reason']}); falling back to max-s."
        )
        sol = self._max_s_sdp()
        if sol is not None:
            self._apply_certificate_solution(sol)
        else:
            logger.warning("Init s fallback (max-s) also failed; s left unchanged.")
        return res

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Feasibility regularization via the log-det interior-point barrier.

        For each LMI ``F ≻ 0`` adds ``-log det F`` and for each scalar
        inequality ``s > 0`` adds ``-log s``. Requires strictly feasible
        parameters (all eigenvalues > 0); the barrier grows to ``+∞`` as any
        constraint approaches its boundary.

        Returns:
            Regularization loss (sum of negative log-determinants).
        """
        feasibility_loss = torch.tensor(0.0, device=self.P.device)
        for f_i in self.get_lmis():
            feasibility_loss += -torch.logdet(f_i())
        for s_i in self.get_scalar_inequalities():
            feasibility_loss += -torch.log(s_i()).squeeze()

        return feasibility_loss

    def get_feasibility_margins(self) -> Dict[str, float]:
        """
        Per-constraint feasibility margins for monitoring interior-point drift.

        For each LMI ``F ≻ 0`` reports the smallest eigenvalue ``min λ(F)`` —
        the distance to the constraint boundary (``≤ 0`` means infeasible). For
        each scalar inequality ``s > 0`` reports its value. Also returns the
        aggregate ``min_eig`` (smallest margin across all constraints).

        Intended use: watch whether ``min_eig`` climbs steadily over training
        (certificate parameters drifting deeper into the feasible interior,
        pushed by the ``-log det`` barrier) while the input/output violation
        terms worsen — the signature of the barrier tugging against the other
        loss terms. A stationary ``min_eig`` means the barrier is not causing
        drift and the negative barrier value can be left as is.

        Returns:
            Dict mapping ``lmi_<i>_min_eig`` / ``scalar_<j>`` to their margins,
            plus aggregate ``min_eig`` (smallest margin across all constraints).
        """
        margins: Dict[str, float] = {}
        overall_min = float("inf")
        with torch.no_grad():
            for i, f_i in enumerate(self.get_lmis()):
                min_eig = torch.linalg.eigvalsh(f_i()).min().item()
                margins[f"lmi_{i}_min_eig"] = min_eig
                overall_min = min(overall_min, min_eig)
            for j, s_i in enumerate(self.get_scalar_inequalities()):
                val = s_i().squeeze().item()
                margins[f"scalar_{j}"] = val
                overall_min = min(overall_min, val)

        if overall_min != float("inf"):
            margins["min_eig"] = overall_min
        return margins

    @overload
    def get_regularization_input(
        self, u: torch.Tensor, x: torch.Tensor,
        return_c: Literal[False] = ..., warmup_steps: int = ...,
    ) -> torch.Tensor: ...

    @overload
    def get_regularization_input(
        self, u: torch.Tensor, x: torch.Tensor,
        return_c: Literal[True], warmup_steps: int = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def get_regularization_input(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        return_c: bool = False,
        warmup_steps: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute input constraint regularization loss (vectorized).

        Enforces input constraints: ||u_k||^2 <= s^2 - α^2 * (x_k^T * P^(-1) * x_k)
        Where:
        - u_k: input at timestep k, shape (batch, seq_len, n_inputs)
        - x_k: state at timestep k, shape (batch, seq_len, n_states)
        - s: input constraint bound
        - α: sigmoid-gated constraint parameter
        - P: parameter matrix

        This vectorized implementation replaces nested loops for efficiency.

        Args:
            inputs: Input trajectories, shape (batch_size, seq_len, n_inputs)
            states: State trajectories, shape (batch_size, seq_len, n_states, 1) or (batch_size, seq_len, n_states)

        Returns:
            Scalar tensor representing mean squared constraint violation
        """

        _, N, _ = u.shape  # batch size, sequence length, input dimension
        # Handle state tensor shape - squeeze trailing dimension if present
        # (batch, seq_len, state_dim, 1) -> (batch, seq_len, state_dim)
        if x.dim() == 4:
            x = x.squeeze(-1)
        

        # Get parameters
        alpha = 1.0 / (1.0 + torch.exp(-self.tau))  # sigmoid
        s = self.s
        X = torch.linalg.inv(self.P)  # P^(-1)

        # Compute vectorized quantities
        # ||u_k||^2 for all timesteps: (batch, seq_len, n_inputs) -> (batch, seq_len)
        u_norm_sq = (u[:,warmup_steps:N] ** 2).sum(dim=-1)

        # x_k^T * P^(-1) * x_k for all timesteps using einsum
        # states: (batch, seq_len, n_states)
        # X: (n_states, n_states)
        # Result: (batch, seq_len)
        x_quad_form = torch.einsum("bti,ij,btj->bt", x[:,warmup_steps:N,:], X, x[:,warmup_steps:N,:])

        # Compute constraint: c_k = ||u_k||^2 - s^2 + α^2 * (x_k^T * P^(-1) * x_k)
        # Shape: (batch, seq_len)
        eps = 0  # small epsilon for numerical stability
        c = u_norm_sq - s**2 + alpha**2 * x_quad_form + eps

        # Coverage is a worst-case property: the safe set must contain each
        # trajectory's peak excursion, so penalize the largest per-trajectory
        # violation (peak over time) averaged over the batch, rather than the
        # mean over all steps which dilutes the few steps that actually breach s.
        # relu is still needed (after amax): a satisfied trajectory has a
        # negative peak c, and without relu minimizing it would reward pushing c
        # ever more negative (inflating s) even when the constraint already holds.
        # relu is monotone, so relu(max_k c_k) == max_k relu(c_k).
        reg_loss = torch.relu(c.amax(dim=1)).mean()

        if return_c:
            return reg_loss, c

        return reg_loss

    def set_output_coverage_level(self, y_max, output_std=None) -> None:
        """Set the **physical** safe output level ``y_max`` (and, optionally, the
        physical output scale ``output_std`` used to relate the model's
        normalized ``C/P/s`` to physical units).

        ``y_max`` has a physical meaning and is stored unnormalized; the coverage
        machinery divides by ``output_std`` internally. ``None``/``nan`` y_max
        disables the output-coverage penalty. ``output_std=None`` leaves the
        stored scale unchanged. Kept on the model's device/dtype."""
        device, dtype = self.P.device, self.P.dtype
        self.y_max = torch.tensor(
            float(y_max) if y_max is not None else float("nan"), device=device, dtype=dtype
        )
        if output_std is not None:
            self.output_std = torch.tensor(float(output_std), device=device, dtype=dtype)

    def get_regularization_output(self, return_margin: bool = False):
        """Output-coverage penalty — bind Corollary 1 (see the wiki
        ``binding-output-certificate``).

        Enforces the coverage-on-image floor ``(σ·s)² C P Cᵀ ⪰ y_max² I`` in
        PHYSICAL output units (``σ = output_std``): the model's *own* certified
        output set must reach the physical safe level ``y_max``. The penalty is
        ``relu(λ_max(y_max² I − (σ·s)² C P Cᵀ))`` — zero once the physical image
        covers the data envelope in every direction, else the largest remaining
        per-direction deficit (for ``ne = 1`` just
        ``relu(y_max² − (σ·s)²·CPCᵀ)``).

        No-op (returns 0) when ``y_max`` is unset. This is the differentiable
        training surrogate for the exact binding SDP in
        :meth:`solve_output_coverage_certificate`.
        """
        zero = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        if self.y_max is None or bool(torch.isnan(self.y_max)):
            return (zero, zero) if return_margin else zero

        CPCt = self.C @ self.P @ self.C.T  # (ne, ne), symmetric
        deficit = self.y_max**2 * torch.eye(
            self.ne, device=self.P.device, dtype=self.P.dtype
        ) - (self.output_std * self.s)**2 * CPCt
        # Coverage <=> deficit ⪯ 0 <=> lambda_max(deficit) <= 0.
        lam_max = torch.linalg.eigvalsh(deficit)[-1]
        reg_loss = torch.relu(lam_max)

        if return_margin:
            return reg_loss, lam_max
        return reg_loss


class SimpleLureSafe(SimpleLure):
    """SimpleLure with the safety input filter wired into the forward pass.

    The filter clamps each input ``d_k`` to keep the closed-loop state inside
    the learned safe set ``{x : (1/s²) xᵀ P⁻¹ x ≤ 1}``. The safe-set parameters
    are derived on-the-fly from the model's own learnable ``P``, ``s``, ``tau``.
    """

    def _build_lure(self, sys: LureSystemClass) -> LureSystem:
        return LureSystemSafe(sys)

    def _run_lure(
        self,
        ds: torch.Tensor,
        x0: torch.Tensor,
        warmup_steps: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        X = torch.linalg.inv(self.P)
        alpha = 1.0 / (1.0 + torch.exp(-self.tau))
        return self.lure(
            d=ds, x0=x0, X=X, s=self.s, alpha=alpha, warmup_steps=warmup_steps,
        )

    def forward_unfiltered(
        self,
        d: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Diagnostic: run the dynamics with the safety filter bypassed.

        Used by post-processing to visualize what trajectories *would* do
        without the filter, so the constraint margin ``c`` from
        ``get_regularization_input`` reflects raw (unprotected) behavior.

        Returns the same ``(e_hat, (x, w), d)`` tuple as ``forward``.
        """
        B, N, nd = d.shape
        assert self.lure._nd == nd
        x0 = self._prepare_x0(x0, B)
        ds = d.reshape(shape=(B, N, nd, 1))

        es_hat, (x_seq, w_seq), ds_applied = LureSystem.forward(self.lure, d=ds, x0=x0)

        e_hat = es_hat.reshape(B, N, self.lure._ne)
        x = x_seq.reshape(B, N + 1, self.nx)
        w = w_seq.reshape(B, N, self.lure._nw)
        d_applied = ds_applied.reshape(B, N, nd)
        return e_hat, (x, w), d_applied
