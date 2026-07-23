import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, overload

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

from sysid.optimization import (
    CertificateSolution,
    InitializationReport,
    LureCertificateSynthesizer,
)
from sysid.utils import (
    max_abs_output,
    plot_ellipse_and_parallelogram,
    torch_bmat,
)

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

        Returns:
            :class:`~sysid.optimization.solutions.InitializationReport` — the
            established certificate diagnostics (and C2 calibration, when it ran);
            ``to_metrics()`` yields the ``initialization/`` mlflow metrics.
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
        sigma = normalizer.output_std if normalizer is not None else 1.0
        sigma_scalar = float(np.asarray(sigma).reshape(-1)[0])
        y_max = max_abs_output(train_outputs) if normalizer is not None else None
        C = self.C.detach().cpu().numpy()

        # Calibrate the C2 std so the max-volume invariant set *just* covers the
        # coverage set: find a C2 factor with 0 <= rho - 1 < eps, where
        # rho = vol(MaxVol) / vol(tightest coverage). A globally-stable init has an
        # unbounded set (rho = ∞); growing C2 turns the model regional and shrinks
        # the set onto the y_max requirement. Requires output_std + a physical
        # y_max and the regional regime (ne == 1, learn_L).
        self.set_output_coverage_level(y_max, sigma_scalar)
        calibrate = (
            bool(getattr(init_config, "calibrate_c2_for_coverage", True))
            if init_config is not None else True
        )
        cal = None
        if calibrate and self.learn_L and self.ne == 1 and y_max is not None:
            eps = float(getattr(init_config, "calibrate_c2_eps", 0.05)) if init_config is not None else 0.05
            max_iter = int(getattr(init_config, "calibrate_c2_max_iter", 30)) if init_config is not None else 30
            cal = self._synth().calibrate_c2(y_max, eps=eps, max_iter=max_iter)
            if cal is not None:
                # calibrate_c2 is pure; apply the winning factor to the model's C2
                # so the returned certificate matches the model that will train.
                self.C2.data = self.C2.data * float(cal.f)
                logger.info(
                    f"C2 calibration: f={cal.f:.4g}, rho={cal.rho:.4f} "
                    f"(target 1 <= rho < 1+{eps}), in_band={cal.in_band}, "
                    f"iters={cal.iterations}, cov_volume={cal.cov_volume}"
                )

        max_vol_sol = cal.max_vol if cal is not None else self._synth().max_vol()
        if max_vol_sol is None:
            # The max-vol SDP is infeasible/failed for the initialized dynamics:
            # there is no (P, L, s) certifying regional (or, for learn_L=False,
            # global) stability, so training cannot start from a feasible point.
            raise RuntimeError(
                "Initialization failed: the max-volume certificate SDP (MaxVol) "
                "found no feasible parameter set for the initialized dynamics. "
                "Check the identity initialization / structural constraints "
                "(e.g. A must be stable, alpha < 1)."
            )

        P_c, L_c, s_c = max_vol_sol.P, max_vol_sol.L, max_vol_sol.s
        norm_H_c = float(np.linalg.norm(L_c @ np.linalg.inv(P_c), ord=2))
        if self.ne == 1:
            CPCt_c = max(float((C @ P_c @ C.T).item()), 0.0)
            y_c = float(sigma_scalar * s_c * np.sqrt(CPCt_c))
            coverage_ok = (
                bool((sigma_scalar * s_c) ** 2 * CPCt_c >= y_max ** 2)
                if y_max is not None else None
            )
        else:
            y_c = None
            coverage_ok = None

        self._apply_certificate_solution(max_vol_sol)
        constraints_ok = self.check_constraints()

        report = InitializationReport(
            volume=float(max_vol_sol.volume),
            s=float(s_c),
            s_feas=max_vol_sol.s_feas,
            norm_H=norm_H_c,
            max_eig_F=float(max_vol_sol.max_eig_F),
            unbounded_volume=bool(max_vol_sol.unbounded_volume),
            constraints_satisfied=bool(constraints_ok),
            y_bar=y_c,
            y_max=float(y_max) if y_max is not None else None,
            coverage_ok=coverage_ok,
            calibrated=cal is not None,
            c2_factor=float(cal.f) if cal is not None else None,
            rho=float(cal.rho) if cal is not None else None,
            rho_in_band=bool(cal.in_band) if cal is not None else None,
            calibration_iterations=int(cal.iterations) if cal is not None else None,
            cov_volume=cal.cov_volume if cal is not None else None,
        )
        logger.info(
            "INITIALIZATION certificate (MaxVol): "
            f"volume={report.volume:.3e}, s={report.s:.4f}, ||H||_2={report.norm_H:.4f}, "
            f"unbounded_volume={report.unbounded_volume}, y_c={y_c}, y_max={y_max}, "
            f"coverage_ok={coverage_ok}, rho={report.rho}, "
            f"rho_in_band={report.rho_in_band}, constraints_satisfied={constraints_ok}"
        )
        self._last_init_report = report
        return report

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

    def _synth(self) -> LureCertificateSynthesizer:
        """Build a certificate synthesizer from the current (fixed) dynamics.

        All the certificate SDPs (MaxS, MaxVol, coverage, feasibility, C2
        calibration) live on :class:`~sysid.optimization.LureCertificateSynthesizer`
        and return typed results from :mod:`sysid.optimization.solutions`. This is
        the single seam between the model and that optimization layer.
        """
        return LureCertificateSynthesizer.from_model(self)

    def feasibility_problem(self) -> bool:
        """Repair the certificate at the **current** (fixed) ``s`` — the
        within-epoch step 3.1.

        When a gradient update breaks the LMIs, solve the fixed-``s`` feasibility
        SDP (:meth:`LureCertificateSynthesizer.feasibility`) at ``self.s``
        (unchanged) for feasible P, M, L and write them back. Returns ``True`` on
        success; ``False`` when no feasible P, M, L exists at that ``s`` (→ the
        trainer rolls the step back). ``s`` is owned by gradient + the
        input/output penalties and is never modified here.
        """
        s = float(self.s.cpu().detach().numpy())
        sol = self._synth().feasibility(s)
        if sol is None:
            return False
        # Write back P, L, Λ only — s stays fixed.
        device, dtype = self.P.device, self.P.dtype
        self.P.data = torch.tensor(sol.P, device=device, dtype=dtype)
        if self.learn_L:
            self.L.data = torch.tensor(sol.L, device=device, dtype=dtype)
        self.la.data = torch.tensor(np.diag(sol.M), device=device, dtype=dtype)
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
            logger.info(f"  Initial s from SDP: {s.item():.2f}")
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
        
        # inequalities.append(s_positive)



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

    def post_process(
        self,
        y_max: Optional[float] = None,
        n_grid: int = 20,
        s_min: float = 1.0,
        s_max: float = 100.0,
    ) -> dict:
        """Post-process a trained model: solve the two certificate SDPs, report
        them **separately**, and set the model to the *largest invariant set*.

        Everything that shapes the predictions (θ = A, B, B2, C, C2, D21, α) is
        held fixed; only the certificate (P, L, Λ, s) is (re)computed. Two clearly
        separated optimization problems are solved over that fixed θ:

        **Problem 1 — max-volume certificate** (MaxVol, :meth:`~sysid.optimization.LureCertificateSynthesizer.max_vol`).
        Maximizes the *volume* of the certified invariant ellipsoid,
        ``sⁿˣ·√(det P)`` (not just the scale ``s``, which is all MaxS maximized),
        subject to the stability + locality LMIs only. The joint objective is
        bilinear (``s`` and ``P`` couple through ``P ⪰ s²·lⁱ(lⁱ)ᵀ``) but convex at
        fixed ``s``, so it is swept over ``s ∈ (0, s_max]`` (``s_max`` = the MaxS
        feasibility ceiling). It does **not** constrain output coverage — the
        coverage floor ``(σ·s)²·C P Cᵀ ⪰ y_max²·I`` is instead *checked afterwards*
        (``coverage_ok``). This gives the largest-volume certifiable invariant set.
        Reported quantities: the ellipsoid ``volume``, the coupling norm
        ``‖H‖ = ‖L P⁻¹‖``, the scale ``s`` and the certified output half-width
        ``ȳ_c = σ·s·√(C P Cᵀ)`` (physical; ``ne == 1`` only, else ``None``).

        **Problem 2 — tightest coverage** (MinTrProb, :meth:`~sysid.optimization.LureCertificateSynthesizer.coverage_at_s` swept
        over a finite s-grid). The joint problem is bilinear (convex once ``s`` is
        fixed), so it is gridded over ``s ∈ [s_min, s_max]``; reported is the
        smallest feasible certified half-width ``ȳ_f`` — the tightest coverage of
        the demanded ``y_max``. Skipped when ``y_max`` is unset.

        Because the goal here is a *large* invariant set, the MaxVol solution (ȳ_c)
        is the one written back into the model. ``y_max`` is physical; ``None``
        falls back to the model's stored physical level (which may be unset, in
        which case Problem 2 and the coverage check are skipped).

        Returns a summary dict::

            {
              "success": bool,
              "s_opt": float,                 # operative (MaxVol) s == max_vol["s"]
              "constraints_satisfied": bool,
              "y_max": Optional[float],       # physical demanded level (or None)
              "max_vol": {s, volume, s_feas, norm_H, y_bar(=ȳ_c), max_eig_F,
                          coverage_ok},
              "coverage":{y_bar(=ȳ_f), s, reason, s_min, s_max, n_grid, sweep},
            }
        """
        # Resolve the (physical) demanded output level.
        if y_max is None and not bool(torch.isnan(self.y_max)):
            y_max = float(self.y_max)
        y_max = float(y_max) if y_max is not None else None

        C = self.C.cpu().detach().numpy()
        sigma = float(self.output_std)

        def _fmt(v):
            return "n/a" if v is None else f"{v:.4f}"

        logger.info("=" * 80)
        logger.info(
            "POST-PROCESSING: (1) max-volume certificate + (2) tightest-coverage sweep"
        )
        logger.info("=" * 80)

        # ------------------------------------------------------------------
        # Problem 1 — MaxVol: the largest-volume certifiable invariant set
        # (operative). Sweeps s ∈ (0, s_max] and keeps the largest sⁿˣ·√(det P).
        # ------------------------------------------------------------------
        synth = self._synth()
        max_vol_sol = synth.max_vol(n_grid=n_grid)
        if max_vol_sol is None:
            return {"success": False, "status": "max_vol_sdp_failed"}

        P_c, L_c, s_c = max_vol_sol.P, max_vol_sol.L, max_vol_sol.s
        vol_c, s_feas_c = max_vol_sol.volume, max_vol_sol.s_feas
        norm_H_c = float(np.linalg.norm(L_c @ np.linalg.inv(P_c), ord=2))
        if self.ne == 1:
            CPCt_c = max(float((C @ P_c @ C.T).item()), 0.0)
            y_c = float(sigma * s_c * np.sqrt(CPCt_c))
            coverage_ok = (
                bool((sigma * s_c) ** 2 * CPCt_c >= y_max ** 2)
                if y_max is not None else None
            )
        else:
            y_c = None
            coverage_ok = None

        logger.info("[Problem 1: MaxVol — largest-volume invariant set]")
        logger.info(f"  volume   = {vol_c:.3e}")
        logger.info(f"  s        = {_fmt(s_c)}   (feasibility ceiling s_max = {_fmt(s_feas_c)})")
        logger.info(f"  ‖H‖      = {_fmt(norm_H_c)}   (H = L P⁻¹)")
        logger.info(f"  ȳ_c      = {_fmt(y_c)}   (σ·s·√(C P Cᵀ))")
        logger.info(f"  max λ(F) = {max_vol_sol.max_eig_F:.3e}")
        if coverage_ok is not None:
            logger.info(
                f"  coverage ((σ·s)²·CPCᵀ ≥ y_max²={y_max ** 2:.4g}): "
                f"{'OK' if coverage_ok else 'NOT met'}"
            )

        # ------------------------------------------------------------------
        # Problem 2 — tightest coverage over the s-grid (reported, not applied).
        # ------------------------------------------------------------------
        y_f = s_f = None
        coverage_reason = "y_max_unset"
        coverage_sweep: list = []
        if y_max is not None:
            cov = synth.coverage_sweep(y_max, n_grid=n_grid, s_min=s_min, s_max=s_max)
            if cov is None:
                coverage_reason = "coverage_unreachable"
                logger.warning(
                    f"[Problem 2: coverage] y_max={y_max:.4g} unreachable on the "
                    f"grid s∈[{s_min:g}, {s_max:g}] — this θ cannot certify it."
                )
            else:
                y_f, s_f = cov.y_f, cov.s_f
                coverage_sweep = [{"s": p.s, "y_bar": p.y_bar} for p in cov.sweep]
                coverage_reason = "ok"
                logger.info("[Problem 2: coverage — tightest ȳ over the s-grid]")
                logger.info(
                    f"  ȳ_f = {_fmt(y_f)}  at s = {_fmt(s_f)}   "
                    f"(target y_max = {_fmt(y_max)})"
                )
        else:
            logger.info("[Problem 2: coverage] skipped (y_max unset)")

        # ------------------------------------------------------------------
        # Set the model to the LARGEST-VOLUME invariant set (MaxVol / ȳ_c).
        # ------------------------------------------------------------------
        self._apply_certificate_solution(max_vol_sol)
        constraints_ok = self.check_constraints()
        logger.info(
            f"Applied MaxVol certificate to the model. Constraints satisfied: {constraints_ok}"
        )
        logger.info("=" * 80)

        return {
            "success": True,
            "s_opt": s_c,
            "constraints_satisfied": constraints_ok,
            "y_max": y_max,
            "max_vol": {
                "s": s_c,
                "volume": vol_c,
                "s_feas": s_feas_c,
                "norm_H": norm_H_c,
                "y_bar": y_c,
                "max_eig_F": float(max_vol_sol.max_eig_F),
                "coverage_ok": coverage_ok,
            },
            "coverage": {
                "y_bar": y_f,
                "s": s_f,
                "reason": coverage_reason,
                "s_min": float(s_min),
                "s_max": float(s_max),
                "n_grid": int(n_grid),
                "sweep": coverage_sweep,
            },
        }

    def _apply_certificate_solution(self, sol: CertificateSolution) -> None:
        """Write a certificate solution back into the model (P, L, la, s).

        Accepts any :class:`~sysid.optimization.solutions.CertificateSolution`
        (MaxVol operative, MaxS, coverage, feasibility) — all share ``P``, ``L``,
        ``M``, ``s``.
        """
        device, dtype = self.P.device, self.P.dtype
        self.P.data = torch.tensor(sol.P, device=device, dtype=dtype)
        if self.learn_L:
            self.L.data = torch.tensor(sol.L, device=device, dtype=dtype)
        self.la.data = torch.tensor(np.diag(sol.M), device=device, dtype=dtype)
        self.s.data = torch.tensor(sol.s, device=device, dtype=dtype)

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
        s_min: float = 1.0,
        s_max: float = 100.0,
    ) -> dict:
        """**MinTrProb** — the binding-Corollary-1 certificate (see the wiki
        ``binding-output-certificate``).

        Produces the tightest certified output interval ``[-ȳ, ȳ]`` with the
        PHYSICAL ``ȳ = y_max`` that (a) satisfies the Lyapunov + regionality
        LMIs and (b) — when ``inputs`` are supplied — leaves **zero input
        violations** on that data. The scale ``s`` is the lone nonconvex degree
        of freedom, so we solve a convex SDP (:meth:`~sysid.optimization.LureCertificateSynthesizer.coverage_at_s`) at each
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
        (physical, the operative certificate), ``y_max`` (physical), ``s_min``,
        ``s_max``, ``n_input_violations`` and the full ``sweep``. It also reports
        the feasibility ceiling + diagnostics (all physical, ne=1 only; ``None``
        for ne>1):

        - ``y_feas`` / ``s_feas`` / ``norm_H_feas``: the MaxS feasibility ceiling
          — fix θ and maximize s (:meth:`~sysid.optimization.LureCertificateSynthesizer.max_s`), giving the largest feasible
          certificate ``ȳ = σ·s_feas·√(C P* C*ᵀ)`` (grid-independent) and its
          coupling norm ``‖H*‖ = ‖L* P*⁻¹‖``. **A large ``s_feas`` with a small
          ``norm_H_feas`` is a strong indication of a globally stable model** (the
          certificate needs no locality restriction). This is the honest
          feasibility ceiling and the global-stability diagnostic.
        - ``y_tight`` / ``s_tight``: the smallest ȳ over ALL feasible grid s
          ignoring input violations — the tight-branch value (≈ ``y_max``). The
          ``y_tight → y_bar`` gap is the conservatism the input constraint forces.
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
        # The synthesizer is a snapshot of θ, so the mid-loop _apply (used only to
        # count input violations) does not perturb the remaining solves.
        synth = self._synth()
        s_grid = np.linspace(float(s_min), float(s_max), int(n_grid))
        sweep = []
        for s in s_grid:
            sol = synth.coverage_at_s(float(s), y_max)
            if sol is None:
                continue
            n_viol = None
            if inputs is not None:
                self._apply_certificate_solution(sol)
                n_viol = self._count_input_violations(inputs, x0, warmup_steps)
            sweep.append({"s": sol.s, "y_bar": sol.y_bar, "n_violations": n_viol, "sol": sol})

        if not sweep:
            _restore()
            return {
                "success": False,
                "reason": "coverage_unreachable",
                "s_min": float(s_min),
                "s_max": float(s_max),
                "y_max": y_max,
            }

        # ``eligible`` is the band the operative certificate is drawn from: the
        # zero-input-violation grid points (or, if none clear the violations, the
        # fewest-violations ones — matching the old fallback selection). When no
        # inputs are given there is no violation filter, so the whole sweep is
        # eligible.
        if inputs is not None:
            zero_viol = [c for c in sweep if c["n_violations"] == 0]
            violation_free = bool(zero_viol)
            if zero_viol:
                eligible = zero_viol
            else:
                min_viol = min(c["n_violations"] for c in sweep)
                eligible = [c for c in sweep if c["n_violations"] == min_viol]
        else:
            violation_free = None
            eligible = sweep

        # Operative certificate (point 2): always the LARGEST invariant set — the
        # largest-s eligible certificate (the largest certifiable safe region),
        # not the tightest ȳ. When feasibility runs to the top of the grid this is
        # the s_max solution. The tightest coverage value (≈ y_max) is still
        # reported as y_tight, and the grid-independent ceiling as the MaxS y_feas.
        best = max(eligible, key=lambda c: c["s"])

        # Feasibility ceiling via MaxS (point 1): fix θ and maximize s
        # (synth.max_s, pure). This is the grid-independent max-feasible certificate;
        # its output half-width is y_feas = σ·s_feas·√(C P* C*ᵀ). A large s_feas
        # together with a small ‖H*‖ = ‖L* P*⁻¹‖ is a strong indication of a
        # globally stable model — the certificate needs no locality restriction.
        # y_tight is the tight-branch value (smallest ȳ over ALL feasible grid s,
        # ignoring input violations, ≈ y_max); the y_tight→y_bar gap is the
        # conservatism the input constraint forces. All are ne=1 only (None else).
        all_ybars = [c for c in sweep if c["y_bar"] is not None]
        tight = min(all_ybars, key=lambda c: c["y_bar"]) if all_ybars else None
        y_tight = tight["y_bar"] if tight is not None else None
        s_tight = tight["s"] if tight is not None else None

        y_feas = s_feas = norm_H_feas = None
        ceil_sol = synth.max_s()  # pure; depends only on the (fixed) θ
        if ceil_sol is not None and self.ne == 1:
            P_c = ceil_sol.P
            C_np = self.C.cpu().detach().numpy()
            sigma = float(self.output_std)
            CPCt_c = float((C_np @ P_c @ C_np.T).item())
            s_feas = float(ceil_sol.s)
            y_feas = float(sigma * s_feas * np.sqrt(CPCt_c))
            H_c = ceil_sol.L @ np.linalg.inv(P_c)
            norm_H_feas = float(np.linalg.norm(H_c, ord=2))

        self._apply_certificate_solution(best["sol"])
        constraints_ok = self.check_constraints()

        return {
            "success": True,
            "reason": "ok" if (violation_free is not False) else "violations_remain",
            "s": best["s"],
            "y_bar": best["y_bar"],
            "y_max": y_max,
            "y_feas": y_feas,
            "s_feas": s_feas,
            "norm_H_feas": norm_H_feas,
            "y_tight": y_tight,
            "s_tight": s_tight,
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
                f"Init s from conditions: s={res['s']:.2f} "
                f"(band [{res['s_min']:.2f}, {res['s_max']:.2f}]), "
                f"y_bar={res['y_bar']:.2f} (y_max={y_max:.2f}), "
                f"output band y_tight={res['y_tight']:.2f} <= y_bar; "
                f"MaxS ceiling y_feas={res['y_feas']:.2f} "
                f"(s_feas={res['s_feas']:.2f}, ||H||={res['norm_H_feas']:.2f}), "
                f"input violations={res['n_input_violations']}, "
                f"violation_free={res['violation_free']}"
            )
            return res

        # Output level not reachable at init -> fall back to MaxVol (the largest
        # invariant set among feasible certificates). Coverage is then handled by
        # the output-coverage penalty during training.
        logger.warning(
            f"Init s from conditions: output level y_max={y_max:.2f} not "
            f"reachable (reason={res['reason']}); falling back to max-vol."
        )
        sol = self._synth().max_vol()
        if sol is not None:
            self._apply_certificate_solution(sol)
        else:
            logger.warning("Init s fallback (max-vol) also failed; s left unchanged.")
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
            # feasibility_loss += torch.relu(-torch.logdet(f_i()))
            feasibility_loss += -torch.logdet(f_i())
        for s_i in self.get_scalar_inequalities():
            # feasibility_loss += torch.relu(-torch.log(s_i()).squeeze())
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

    def get_regularization_activity(
        self,
        w: torch.Tensor,
        w_star: float,
        warmup_steps: int = 0,
        return_activity: bool = False,
    ):
        """Dead-zone activity penalty — prevent the degenerate *linear collapse*.

        For the dead-zone nonlinearity ``w = Δ(z) = z − hardtanh(z)`` the rollout
        follows the pure LTI part exactly when ``w = 0`` (pre-activations stay in
        the dead band ``|z| ≤ 1``). That degenerate "linear collapse" is a poor
        fit for nonlinear data (and trivially globally stable). This penalty
        rewards the dead-zone to *fire* (``w ≠ 0``), pushing the model into its
        nonlinear regime.

        Caveat — this is **not** a certificate-level anti-global mechanism. Firing
        the nonlinearity does *not* by itself make the global (``H = 0``)
        certificate infeasible: ``H = 0`` means the *global* sector condition is
        used, and tanh/dzn satisfy sector ``[0, 1]`` **globally**, so a model can
        be globally absolutely stable with a fully active nonlinearity
        (``w ≠ 0``). Hence ``H = 0`` does **not** imply a linear model. Making
        ``H = 0`` genuinely infeasible requires shaping the *linear* dynamics so
        the global-sector LMI fails — a separate lever. This term only removes the
        ``w ≡ 0`` collapse; it is a behavioral heuristic that *correlates* with,
        but does not guarantee, a non-global model.

        Penalty ``relu(w_star − a)`` with ``a = ⟨‖w_k‖₂⟩`` the mean per-step
        activation norm over the (warmup-skipped) rollout; zero once the mean
        activity reaches the target ``w_star``. ``w_star ≤ 0`` disables it
        (no-op). It reads the rollout, not the certificate, so it does not
        directly touch P, L, s.

        Note: only meaningful for the ``dzn`` (dead-zone) activation, where
        ``w = 0`` ⇔ the linear regime. For ``sat``/``tanh`` (linear near 0,
        saturating for large ``z``) small ``w`` is *not* the linear regime, so
        the sign of "activity" is different — do not enable it there unchanged.

        Args:
            w: nonlinearity output ``(B, N, nw)`` from ``forward``.
            w_star: target mean activation norm (the hinge threshold).
            warmup_steps: leading steps skipped (match the prediction loss).
            return_activity: also return the scalar mean activity ``a``.
        """
        zero = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        if w_star is None or float(w_star) <= 0.0:
            return (zero, zero) if return_activity else zero

        N = w.shape[1]
        w_active = w[:, warmup_steps:N, :]
        # Mean over batch and time of the per-step L2 activation norm.
        activity = torch.linalg.vector_norm(w_active, dim=-1).mean()
        reg_loss = torch.relu(float(w_star) - activity)

        if return_activity:
            return reg_loss, activity
        return reg_loss

    def get_regularization_H(self, h_star, return_norm: bool = False):
        """Anti-global-certificate penalty — push the coupling ``H = L P⁻¹``
        away from zero.

        The regionality certificate collapses to the *global* sector condition
        exactly when ``H = L P⁻¹ = 0`` (no locality restriction — a globally
        absolutely stable, typically near-linear model). Unlike
        :meth:`get_regularization_activity`, which only removes the ``w ≡ 0``
        rollout collapse and does *not* make the ``H = 0`` certificate infeasible,
        this term acts directly on the certificate parameters (L, P), so it
        directly discourages the global certificate. It hinges on the coupling
        norm::

            relu(h_star − ‖H‖_F)

        zero once ``‖H‖_F ≥ h_star`` (the target coupling magnitude), else the
        remaining deficit; minimizing it grows ‖H‖. The Frobenius norm is
        computed with a tiny additive floor so the gradient is finite (rather
        than NaN) at the ``H = 0`` cone point — training escapes zero as soon as
        the SDP init / prediction loss make ``L`` nonzero.

        ``h_star ≤ 0`` disables it (no-op). Requires ``learn_L`` (``H`` is
        identically zero and non-trainable otherwise, so the term would be a
        constant no-op there too).

        Args:
            h_star: target coupling norm ‖H‖_F (the hinge threshold).
            return_norm: also return the scalar ‖H‖_F.
        """
        zero = torch.zeros((), device=self.P.device, dtype=self.P.dtype)
        if h_star is None or float(h_star) <= 0.0 or not self.learn_L:
            return (zero, zero) if return_norm else zero

        H = self.L @ torch.linalg.inv(self.P)
        # Frobenius norm with a small floor: sqrt(0) has a NaN gradient, so the
        # floor keeps the gradient finite at H = 0 without changing the value
        # meaningfully once H is nonzero.
        norm_H = torch.sqrt((H ** 2).sum() + EPS ** 2)
        reg_loss = torch.relu(float(h_star) - norm_H)

        if return_norm:
            return reg_loss, norm_H
        return reg_loss


class SimpleLureSafe(SimpleLure):
    """SimpleLure with the safety input filter wired into the forward pass.

    The filter clamps each input ``d_k`` to keep the closed-loop state inside
    the learned safe set ``{x : (1/s²) xᵀ P⁻¹ x ≤ 1}``. The safe-set parameters
    are derived on-the-fly from the model's own learnable ``P``, ``s``, ``tau``.
    """

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
