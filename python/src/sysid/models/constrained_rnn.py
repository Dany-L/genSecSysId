import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

from sysid.utils import plot_ellipse_and_parallelogram, torch_bmat

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
        delta: np.float64 = 0.1,
        max_norm_x0: np.float64 = 1.0,
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

        self.P = nn.Parameter(torch.eye(self.nx))  # Lyapunov matrix
        if custom_params is not None:
            learn_L = custom_params.get("learn_L", True)
            self.regularization_method = custom_params.get(
                "regularization_method", "interior_point"
            )
            self.dual_penalty_init = custom_params.get("dual_penalty_init", 1.0)
            self.dual_penalty_growth = custom_params.get("dual_penalty_growth", 1.1)
            self.dual_penalty_shrink = custom_params.get("dual_penalty_shrink", 0.9)
            self.l_nonzero_weight = custom_params.get("l_nonzero_weight", 0.0)
        else:
            learn_L = True
            self.regularization_method = "interior_point"
            self.dual_penalty_init = 1.0
            self.dual_penalty_growth = 1.1
            self.dual_penalty_shrink = 0.9
            self.l_nonzero_weight = 0.0

        self.learn_L = learn_L

        # Dual penalty coefficient (not a parameter, updated manually)
        self.register_buffer("dual_penalty", torch.tensor(self.dual_penalty_init))

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
            self.L = torch.zeros((nz, nx))  # Coupling matrix, not learnable
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

    def _parse_structural_constraints(self, custom_params: dict) -> dict:
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
        data_dir: Optional[str] = None,
        normalizer: Optional[DataNormalizer] = None
    ):
        """
        Initialize model parameters using the specified method.

        Supports three initialization strategies:
        1. ESN (Echo State Network): Random reservoirs with least-squares fitting
        2. N4SID: Load from n4sid_params.mat file if available
        3. Identity: Predefined diagonal A, random C2, identity-like C

        Args:
            train_inputs: Training input data (B, N, nd)
            train_states: Training state data (B, N, nx) or None
            train_outputs: Training output data (B, N, ne)
            init_config: InitializationConfig object specifying the method.
                        If None, defaults to ESN with 5 restarts.
            data_dir: Directory to search for n4sid_params.mat
            normalizer: Data normalizer to use for scaling inputs/outputs
        """
        # Set defaults
        if init_config is None:
            init_method = "esn"
            n_restarts = 5
        else:
            init_method = getattr(init_config, "method", "esn").lower()
            n_restarts = getattr(init_config, "esn_n_restarts", 5)

        logger.info("=" * 80)
        logger.info(f"INITIALIZATION: Using '{init_method}' method")
        logger.info("=" * 80)

        # set initial according to (u^k)^T u^k <= s^2 - alpha^2 * V(x^k)
        # The maximum s follow by setting x^k = 0 and then getting (u^k)^T u^k <= s^2 for all k, where u^k is the input at time k
        if normalizer is not None:
            train_inputs = normalizer.transform_inputs(train_inputs)
        u_squared_norm = np.sum(train_inputs**2, axis=2)
        max_input_n = np.max(u_squared_norm)
        # self.s.data = torch.tensor(np.sqrt(normalizer.inverse_transform_inputs(max_input_n))).squeeze()
        # self.s.data = torch.tensor(4.0)

        if init_method == "n4sid":
            success = self._init_n4sid(train_inputs, train_states, train_outputs, data_dir)
            if not success:
                logger.warning("N4SID initialization failed or file not found. Falling back to ESN.")
                self._init_esn(train_inputs, train_states, train_outputs, n_restarts)
        elif init_method == "identity":
            self._init_identity(normalizer)
        elif init_method == "esn":
            self._init_esn(train_inputs, train_states, train_outputs, n_restarts)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # Common post-initialization
        constraints_ok = self.check_constraints()
        logger.info(f"Initialization complete. Constraints satisfied: {constraints_ok}")
        logger.info("=" * 80)
        if not constraints_ok:
            b_feasible = self.analysis_problem_init(learn_B=False, learn_D21=True)
            if not b_feasible:
                raise ValueError("Initialization did not satisfy constraints and problem is infeasible. Please check your initialization method and structural constraints.")
            # self.analysis_problem_init(learn_B=True, learn_D21=True)

        

    def _init_identity(self, normalizer: Optional[DataNormalizer] = None):
        """
        Identity initialization: predefined simple linear system.

        Sets:
        - α = 0.99
        - A = 0.9I (stable, diagonal)
        - C2 = Rand(-1,1) (random measurement matrix)
        - C = [I, 0] (identity-like output matrix, padded if nx > ne)
        - B2 = D = D12 = 0 (no direct feedthrough, all nonlinearity from C2)
        - B = 0 (can be learned during training)

        This provides a simple stable starting point.
        
        Respects structural constraints: only updates learnable parameters/elements.
        """
        logger.info("Identity initialization: α=0.99, A=0.9I, random C2, identity-like C")

        # self.alpha.data = torch.tensor(0.9999)
        
        # A matrix - only update if not fully fixed
        if not self._should_skip_initialization('A'):
            A_ct = torch.tensor([[0,1.0], [0.0,0.0]])
            A_ct[1,:] = -torch.rand((1, self.nx))
            A_dt = torch.eye(self.nx) + A_ct * self.ts # euler discretizatioin
            A_init = A_dt
            # torch.nn.init.normal_(A_init, std=1)
            # A_init = torch.tensor([
            #     [ 1.   ,  0.05 ],
            #     [-0.05 ,  0.985]
            # ])
            logger.info(f'Absolute eigenvalues of A_init: {torch.linalg.eigvals(A_init).abs()}')
            if 'A' in self.structural_constraints:
                self._apply_partial_initialization('A', A_init)
            else:
                self.A.data = A_init
        
        if not self._should_skip_initialization('B'):
            # B_init = torch.tensor([
            #     [0.0],
            #     [0.0039662]
            # ])
            input_scale = 1.0
            if normalizer is not None:
                input_std = getattr(normalizer, 'input_std', None)
                if input_std is not None:
                    input_scale = input_std.squeeze()
            B_init = input_scale * self.ts * torch.tensor([
                [0.0],
                [1.0]
            ])
            if 'B' in self.structural_constraints:
                self._apply_partial_initialization('B', B_init)
            else:
                self.B.data = B_init
        
        if not self._should_skip_initialization('B2'):
            # B2_init = torch.zeros(self.nx, self.nw)
            B2_init = self.ts * torch.randn((self.nx, self.nw))
            self.B2.data = B2_init
            # self.B2.data = torch.zeros(self.nx, self.nw)
            # self.B2.data = torch.randn(self.nx, self.nw) *0.1
            # self.B2.data = torch.tensor([
            #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            #     [
            #         0.00556457, 0.01755953, 0.04252413, 0.07508705, 0.11689463, 0.16913675, 0.22928246, 0.30034822, 0.37948621, 0.46870539, 0.56760693, 0.67407641, 0.79379072, 0.91629349, 1.05828194, 1.19480956, 1.36282576, 1.50314181, 1.73255    , 0.        
            #     ]
            # ])
            # self.B2.data += torch.randn(self.nx, self.nw) * 0.01 
            # self.B2.data = torch.tensor(np.random.uniform(-1.1, 1.1, size=(self.nx, self.nw)))
        
        # C2 matrix - random initialization
        if not self._should_skip_initialization('C2'):
            C2_init = torch.randn(self.nz, self.nx)
            # C2_init = torch.tensor(np.random.uniform(-1, 1, size=(self.nz, self.nx)))
            # C2_init = torch.randn(self.nz, self.nx)*0.9
            # C2_init = torch.tensor([
            #     [4.,         0.        ],
            #     [2.,         0.        ],
            #     [1.33333333, 0.        ],
            #     [1.,         0.        ],
            #     [0.8,        0.        ],
            #     [0.66666667, 0.        ],
            #     [0.57142857, 0.        ],
            #     [0.5,        0.        ],
            #     [0.44444444, 0.        ],
            #     [0.4,        0.        ],
            #     [0.36363636, 0.        ],
            #     [0.33333333, 0.        ],
            #     [0.30769231, 0.        ],
            #     [0.28571429, 0.        ],
            #     [0.26666667, 0.        ],
            #     [0.25,       0.        ],
            #     [0.23529412, 0.        ],
            #     [0.22222222, 0.        ],
            #     [0.21052632, 0.        ],
            #     [0.2,        0.        ]
            # ])
            # C2_init += torch.randn(self.nz, self.nx) * 0.01 
            if 'C2' in self.structural_constraints:
                self._apply_partial_initialization('C2', C2_init)
            else:
                self.C2.data = C2_init

        if not self._should_skip_initialization('C'):
            # C_init_tensor = torch.tensor([
            #     [6.58489445, 0.0        ]
            # ])
            C_init_tensor = 1/normalizer.output_std.squeeze()*torch.tensor([
                [1.0, 0.0        ]
            ])
            if 'C' in self.structural_constraints:
                self._apply_partial_initialization('C', C_init_tensor)
            else:
                self.C.data = C_init_tensor

        # D matrix - initialize to zeros
        if not self._should_skip_initialization('D'):
            self.D.data = torch.zeros_like(self.D)
        
        # D12 matrix - initialize to zeros
        if not self._should_skip_initialization('D12'):
            self.D12.data = torch.zeros_like(self.D12)

        # Random D21 for measurement noise
        if not self._should_skip_initialization('D21'):
            D21_init = torch.randn(self.nz, self.nd)
            if 'D21' in self.structural_constraints:
                self._apply_partial_initialization('D21', D21_init)
            else:
                self.D21.data = D21_init

        logger.info(f"  ||A||={np.linalg.norm(self.A.detach().numpy()):.4f}")
        logger.info(f"  ||C||={np.linalg.norm(self.C.detach().numpy()):.4f}")
        logger.info(f"  ||C2||={np.linalg.norm(self.C2.detach().numpy()):.4f}")

    def _init_n4sid(self, train_inputs, train_states, train_outputs, data_dir: Optional[str] = None) -> bool:
        """
        Initialize from N4SID system identification results.

        Loads A, B, C, D from n4sid_params.mat. If N4SID state dimension < nx,
        pads matrices (A top-left block, B top rows, C left columns).
        Runs SDP to find feasible P and L.

        Note: Respects structural constraints - fixed parameters are not modified.

        Args:
            train_inputs: Training inputs (B, N, nd)
            train_states: Training states (B, N, nx_data)
            train_outputs: Training outputs (B, N, ne)
            data_dir: Directory containing n4sid_params.mat

        Returns:
            True if successful, False otherwise
        """
        if data_dir is None:
            return False

        from scipy.io import loadmat

        mat_path = Path(os.path.expanduser(data_dir)) / "n4sid_params.mat"
        if not mat_path.exists():
            logger.info(f"N4SID file not found at {mat_path}")
            return False

        logger.info(f"Loading N4SID parameters from {mat_path}")

        try:
            mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

            # Support both direct keys and nested struct
            def _extract(key):
                if key in mat:
                    return np.atleast_2d(np.array(mat[key], dtype=np.float64))
                for v in mat.values():
                    if hasattr(v, key):
                        return np.atleast_2d(np.array(getattr(v, key), dtype=np.float64))
                raise KeyError(f"Key '{key}' not found in {mat_path}")

            A_n4 = _extract("A")
            B_n4 = _extract("B")
            C_n4 = _extract("C")
            D_n4 = _extract("D")

            def _maybe_transpose_to_match(mat: np.ndarray, target_shape: tuple, name: str) -> np.ndarray:
                if mat.shape == target_shape:
                    return mat
                if mat.T.shape == target_shape:
                    logger.info(f"Transposing N4SID {name} from {mat.shape} to {mat.T.shape}")
                    return mat.T
                return mat

            B_n4 = _maybe_transpose_to_match(B_n4, (A_n4.shape[0], self.nd), "B")
            C_n4 = _maybe_transpose_to_match(C_n4, (self.ne, A_n4.shape[0]), "C")
            D_n4 = _maybe_transpose_to_match(D_n4, (self.ne, self.nd), "D")

            # Validate
            if A_n4.shape[0] != A_n4.shape[1]:
                raise ValueError(f"N4SID A must be square, got {A_n4.shape}")
            if A_n4.shape[0] > self.nx:
                raise ValueError(
                    f"N4SID state dim ({A_n4.shape[0]}) > model nx ({self.nx})"
                )
            if B_n4.shape[1] != self.nd or C_n4.shape[0] != self.ne:
                raise ValueError("N4SID input/output dimensions mismatch")
            if D_n4.shape != (self.ne, self.nd):
                raise ValueError(f"N4SID D has wrong shape: {D_n4.shape}")

            # Pad if needed
            n4_nx = A_n4.shape[0]
            A_init = np.zeros((self.nx, self.nx), dtype=np.float64)
            A_init[:n4_nx, :n4_nx] = A_n4

            B_init = np.zeros((self.nx, self.nd), dtype=np.float64)
            B_init[:B_n4.shape[0], :] = B_n4

            C_init = np.zeros((self.ne, self.nx), dtype=np.float64)
            C_init[:, :C_n4.shape[1]] = C_n4

            if n4_nx < self.nx:
                logger.info(
                    f"Padding N4SID state dim {n4_nx} → {self.nx} "
                    "(A top-left, B top rows, C left cols)"
                )

            # Set parameters (skip fixed parameters)
            if not self._should_skip_initialization('A'):
                self.A.data = torch.tensor(A_init)
            if not self._should_skip_initialization('B'):
                self.B.data = torch.tensor(B_init)
            if not self._should_skip_initialization('C'):
                self.C.data = torch.tensor(C_init)
            if not self._should_skip_initialization('B2'):
                self.B2.data = torch.zeros_like(self.B2)
            if not self._should_skip_initialization('C2'):
                self.C2.data = torch.tensor(np.random.randn(self.nz, self.nx))
            if not self._should_skip_initialization('D21'):
                self.D21.data = torch.tensor(np.random.randn(self.nz, self.nd) * 0.01)

            logger.info(f"N4SID loaded: ||A||={np.linalg.norm(A_init):.4f}, "
                       f"||B||={np.linalg.norm(B_init):.4f}, "
                       f"||C||={np.linalg.norm(C_init):.4f}")

            # Run SDP for feasibility
            self.analysis_problem_init(learn_B=False, learn_D21=True)

            # Refit C, D, D12 with new B, D21
            self._refit_output_matrices(train_inputs, train_states, train_outputs)
            return True

        except Exception as e:
            logger.warning(f"N4SID initialization failed: {e}")
            return False

    def _init_esn(
        self,
        train_inputs,
        train_states,
        train_outputs,
        n_restarts: int = 5,
    ):
        """
        Echo State Network initialization: Random reservoirs + Least Squares.

        For each restart:
        1. Sample random A (spectral radius ≈ α), B, C2, D21
        2. Simulate to get x, w states
        3. Fit C, D, D12 via least squares (masking NaN-padded rows)
        4. Keep the best (lowest training MSE on valid entries)

        Then run SDP once on the best reservoir to find feasible B, D21, P, L.
        Finally, refit C, D, D12 with the SDP-updated B, D21.

        Note: Respects structural constraints - fixed parameters are not modified.

        Args:
            train_inputs: (B, N, nd)
            train_states: (B, N, nx_data) or None
            train_outputs: (B, N, ne)
            n_restarts: Number of random reservoirs to try
        """
        logger.info(f"ESN initialization with {n_restarts} random restarts")

        Bs, N, _ = train_inputs.shape
        alpha_val = 0.9
        delta_val = float(self.delta.item())

        # Prepare data
        x0s = torch.zeros(Bs, self.nx, 1)
        if train_states is not None:
            x0s_data = torch.tensor(train_states[:, 0, :].reshape(Bs, self.nx_data, 1))
            x0s[:, :self.nx_data, :] = x0s_data
        else:
            x0s = torch.randn(Bs, self.nx, 1) * self.max_norm_x0

        ds = torch.tensor(train_inputs.reshape(Bs, N, self.nd, 1))
        y_target = torch.tensor(train_outputs.reshape(Bs * N, self.ne))
        valid_target_rows = torch.isfinite(y_target).all(dim=1)

        best_mse = float("inf")
        best_reservoir = None

        for trial in range(n_restarts):
            # Random A: spectral radius = alpha_val
            A_rand = np.random.randn(self.nx, self.nx)
            rho = np.max(np.abs(np.linalg.eigvals(A_rand)))
            A_rand = (alpha_val / max(rho, 1e-8)) * A_rand

            # B2 = 0: keep linear (nonlinearity optimized during training)
            B2_rand = np.zeros((self.nx, self.nw))

            # Random B: scaled by input amplitude
            B_rand = np.random.randn(self.nx, self.nd) * (delta_val / max(np.sqrt(self.nx), 1.0))

            # Random C2: scaled to keep pre-activation ~ O(1)
            C2_rand = np.random.randn(self.nz, self.nx) / np.sqrt(self.nx)

            # Small D21
            D21_rand = np.random.randn(self.nz, self.nd) * 0.01

            # Set temporarily (skip fixed parameters)
            if not self._is_parameter_fixed('A'):
                self.A.data = torch.tensor(A_rand)
            if not self._is_parameter_fixed('B'):
                self.B.data = torch.tensor(B_rand)
            if not self._is_parameter_fixed('B2'):
                self.B2.data = torch.tensor(B2_rand)
            if not self._is_parameter_fixed('C2'):
                self.C2.data = torch.tensor(C2_rand)
            if not self._is_parameter_fixed('D21'):
                self.D21.data = torch.tensor(D21_rand)

            # Simulate
            with torch.no_grad():
                _, (xs, ws) = self.lure.forward(x0=x0s, d=ds, return_states=True)

            # Least squares: fit C, D, D12 on valid rows only
            x_flat = xs[:, :N, :, :].squeeze(-1).reshape(Bs * N, self.nx)
            w_flat = ws.squeeze(-1).reshape(Bs * N, self.nw)
            u_flat = ds.squeeze(-1).reshape(Bs * N, self.nd)
            regr = torch.cat([x_flat, u_flat, w_flat], dim=1)
            valid_rows = valid_target_rows & torch.isfinite(regr).all(dim=1)

            if int(valid_rows.sum()) <= regr.shape[1]:
                logger.info(f"  Trial {trial + 1}/{n_restarts}: skipped (not enough valid rows)")
                continue

            regr_valid = regr[valid_rows]
            y_valid = y_target[valid_rows]
            sol = torch.linalg.lstsq(regr_valid, y_valid).solution
            y_hat = regr_valid @ sol
            mse = float(torch.mean((y_hat - y_valid) ** 2))

            logger.info(f"  Trial {trial + 1}/{n_restarts}: MSE = {mse:.6e}")

            if mse < best_mse:
                best_mse = mse
                best_reservoir = dict(
                    A=A_rand.copy(),
                    B=B_rand.copy(),
                    B2=B2_rand.copy(),
                    C2=C2_rand.copy(),
                    D21=D21_rand.copy(),
                )

        if best_reservoir is None:
            raise ValueError(
                "ESN initialization failed: no trial had enough finite rows for least squares"
            )

        logger.info(f"Best reservoir MSE: {best_mse:.6e}")

        # Set best reservoir (skip fixed parameters)
        if not self._is_parameter_fixed('A'):
            self.A.data = torch.tensor(best_reservoir["A"])
        if not self._is_parameter_fixed('B'):
            self.B.data = torch.tensor(best_reservoir["B"])
        if not self._is_parameter_fixed('B2'):
            self.B2.data = torch.tensor(best_reservoir["B2"])
        if not self._is_parameter_fixed('C2'):
            self.C2.data = torch.tensor(best_reservoir["C2"])
        if not self._is_parameter_fixed('D21'):
            self.D21.data = torch.tensor(best_reservoir["D21"])

        # Run SDP for feasibility
        logger.info("Running SDP for feasibility...")
        self.analysis_problem_init(learn_B=True, learn_D21=True)

        # Refit C, D, D12
        self._refit_output_matrices(train_inputs, train_states, train_outputs)

    def _refit_output_matrices(self, train_inputs, train_states, train_outputs):
        """
        Re-simulate with current A, B, B2, C2, D21 and fit output matrices C, D, D12.

        This is called after SDP initialization to refine output matrices with
        the updated system matrices.

        Note: Respects structural constraints - fixed output matrices are not modified.
        """
        Bs, N, _ = train_inputs.shape

        # Prepare data
        x0s = torch.zeros(Bs, self.nx, 1)
        if train_states is not None:
            x0s_data = torch.tensor(train_states[:, 0, :].reshape(Bs, self.nx_data, 1))
            x0s[:, :self.nx_data, :] = x0s_data
        else:
            x0s = torch.randn(Bs, self.nx, 1) * self.max_norm_x0

        ds = torch.tensor(train_inputs.reshape(Bs, N, self.nd, 1))
        y_flat = torch.tensor(train_outputs.reshape(Bs * N, self.ne))

        # Simulate
        with torch.no_grad():
            _, (xs, ws) = self.lure.forward(x0=x0s, d=ds, return_states=True)

            x_flat = xs[:, :N, :, :].squeeze(-1).reshape(Bs * N, self.nx)
            w_flat = ws.squeeze(-1).reshape(Bs * N, self.nw)
            u_flat = ds.squeeze(-1).reshape(Bs * N, self.nd)
            regression_matrix = torch.cat([x_flat, u_flat, w_flat], dim=1)
            valid_rows = torch.isfinite(y_flat).all(dim=1) & torch.isfinite(
                regression_matrix
            ).all(dim=1)

            if int(valid_rows.sum()) <= regression_matrix.shape[1]:
                raise ValueError(
                    "Refit failed: not enough valid rows for least squares"
                )

            regression_valid = regression_matrix[valid_rows]
            y_valid = y_flat[valid_rows]
            solution = torch.linalg.lstsq(regression_valid, y_valid).solution

            # Only update learnable output matrices
            if not self._should_skip_initialization('C'):
                self.C.data = solution[: self.nx, :].T
            if not self._should_skip_initialization('D'):
                self.D.data = solution[self.nx : self.nx + self.nd, :].T
            if not self._should_skip_initialization('D12'):
                self.D12.data = solution[self.nx + self.nd :, :].T

            final_mse = float(torch.mean((regression_valid @ solution - y_valid) ** 2))
            logger.info(f"Output matrix refit MSE: {final_mse:.6e}")
            logger.info(f"  ||C||={torch.norm(self.C).item():.6f}")
            logger.info(f"  ||D||={torch.norm(self.D).item():.6f}")
            logger.info(f"  ||D12||={torch.norm(self.D12).item():.6f}")


    def analysis_problem(self, learn_B_and_D21: bool = False) -> bool:

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        la = cp.Variable((self.nz, 1))
        M = cp.diag(la)
        # s_hat = cp.Variable((1,1))
        if learn_B_and_D21:
            B = cp.Variable(self.B.shape)
            D21 = cp.Variable(self.D21.shape)
        else:
            B = self.B.cpu().detach().numpy()
            D21 = self.D21.cpu().detach().numpy()
        A = self.A.cpu().detach().numpy()
        B2 = self.B2.cpu().detach().numpy()
        C2 = self.C2.cpu().detach().numpy()
        alpha = 1/(1+ np.exp(-self.tau.cpu().detach().numpy()))
        # alpha = self.alpha.cpu().detach().numpy()
        s = self.s.cpu().detach().numpy()
        # S_hat = cp.Variable((1,1))
        # if alpha >= 1:
        #     self.alpha.data = torch.tensor(0.99)
        # if alpha < 0.9:

        # if self.get_scalar_inequalities()[0]() < 0 and self.learn_L:
        #     logger.info("Condition on inputs is not satisfied")
        #     self.reset_s()
        #

        multiplier_constraints = []
        if self.learn_L:
            L = cp.Variable((self.nz, self.nx))
        else:
            L = self.L.cpu().detach().numpy()

        for li in L:
            if self.learn_L:
                li = li.reshape((1, -1), "C")
            else:
                li = li.reshape((1, -1))
            multiplier_constraints.append(
                cp.bmat(
                    [
                        [np.array([[1 / s**2]]), li],
                        [li.T, P],
                    ]
                )
                # cp.bmat(
                #     [
                #         [S_hat, li],
                #         [li.T, P],
                #     ]
                # )
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

        # init_constraints = [-P + self.max_norm_x0**2/s**2 * np.eye(self.nx)<< 0]

        nF = F.shape[0]
        problem = cp.Problem(cp.Minimize([None]), [F << -EPS * np.eye(nF), *multiplier_constraints])
        try:
            problem.solve(solver=cp.MOSEK)
        except Exception:
            return False  # SDP failed due to solver error
        if not problem.status == "optimal":
            return False  # SDP failed to find feasible solution
        # logger.info(f"SDP analysis problem solved: {problem.status}")

        # self.s.data = 1/torch.sqrt(torch.tensor(S_hat.value).squeeze())
        self.P.data = torch.tensor(P.value)
        # self.M.data = torch.tensor(M.value)
        self.la.data = torch.tensor(np.diag(M.value))
        if self.learn_L:
            self.L.data = torch.tensor(L.value)
        if learn_B_and_D21:
            self.B.data = torch.tensor(B.value)
            self.D21.data = torch.tensor(D21.value)

        # self.s.data = torch.tensor(np.sqrt(1/s_hat.value).squeeze())

        return True  # SDP successfully found feasible solution

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
        # s = self.s.cpu().detach().numpy()
        s_hat = cp.Variable((1,1))

        multiplier_constraints = []
        if self.learn_L:
            L = cp.Variable((self.nz, self.nx))
        else:
            L = self.L.cpu().detach().numpy()

        for li in L:
            if self.learn_L:
                li = li.reshape((1, -1), "C")
            else:
                li = li.reshape((1, -1))
            multiplier_constraints.append(
                # cp.bmat(
                #     [
                #         [np.array([[1 / s**2]]), li],
                #         [li.T, P],
                #     ]
                # )
                cp.bmat(
                    [
                        [s_hat, li],
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
        problem = cp.Problem(
            # cp.Minimize(None),
            # cp.Minimize(t),
            cp.Minimize(s_hat),
            [
                F << -EPS * np.eye(nF), 
                *multiplier_constraints, 
                # *size_constraints,
                # *init_constraints
            ],
        )
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception:
            return False  # SDP failed due to solver error
        if not problem.status == "optimal":
            return False  # SDP failed to find feasible solution    
        logger.info(f"SDP analysis problem solved: {problem.status}")

        s = 1/torch.sqrt(torch.tensor(s_hat.value).squeeze())
        logger.info(f"  Initial s from SDP: {s.item():.6f}")
        self.s.data = s
        self.P.data = torch.tensor(P.value)
        self.la.data = torch.tensor(np.diag(M.value))
        # self.M.data = torch.tensor(M.value)
        if self.learn_L:
            self.L.data = torch.tensor(L.value)
        if learn_B:
            self.B.data = torch.tensor(B.value)
        if learn_D21:
            self.D21.data = torch.tensor(D21.value)


        return True  # SDP successfully found feasible solution

    def get_lmis(self):
        lmi_list = []
        """Construct the LMI for stability constraint."""
        alpha = 1/(1+ torch.exp(-self.tau))
        M = torch.diag(self.la)
        def stability_lmi() -> torch.Tensor:
            F = torch_bmat(
                [
                    [
                        -alpha**2 * self.P,
                        torch.zeros((self.nx, self.nd)),
                        self.P @ self.C2.T + self.L.T,
                        self.P @ self.A.T,
                    ],
                    [torch.zeros((self.nd, self.nx)), -torch.eye(self.nd), self.D21.T, self.B.T],
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
            return torch.zeros(size=(B, self.nx, 1))
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

        # Decision variables
        P = cp.Variable((self.nx, self.nx), symmetric=True)
        L = cp.Variable((self.nz, self.nx))
        m = cp.Variable((self.nz, 1))
        M = cp.diag(m)

        # S_hat for optimizing s
        S_hat = cp.Variable((1, 1))

        # Constraints
        constraints = []

        # Main LMI: F <= -eps*I
        F = cp.bmat(
            [
                [-(alpha**2) * P, np.zeros((self.nx, self.nd)), P @ C2.T + L.T, P @ A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P + L, D21, -2 * M, M @ B2.T],
                [A @ P, B, B2 @ M, -P],
            ]
        )
        nF = F.shape[0]
        constraints.append(F << -EPS * np.eye(nF))

        # Locality constraints: [S_hat, li; li', P] >= EPS*I for each row of L
        Gs = []
        for i in range(self.nz):
            li = L[i, :].reshape((1, -1), order="C")
            locality_lmi = cp.bmat([[S_hat, li], [li.T, P]])
            Gs.append(locality_lmi)
            constraints.append(locality_lmi >> EPS * np.eye(self.nx + 1))

        # Objective
        objective = cp.Minimize(S_hat)

        # Solve
        problem = cp.Problem(objective, constraints)
        logger.info(f"Solving SDP with {len(constraints)} constraints using MOSEK...")

        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            logger.error(f"SDP solver failed: {e}")
            return {"success": False, "error": str(e)}

        # Check solution status
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"SDP failed with status: {problem.status}")
            return {"success": False, "status": problem.status}

        logger.info(f"✓ SDP state set solved successfully: {problem.status}")

        S_hat_opt = S_hat.value[0, 0] if hasattr(S_hat.value, "__len__") else S_hat.value
        s_star = np.sqrt(1.0 / S_hat_opt)

        # Verify solution
        max_eig_F = np.max(np.real(np.linalg.eigvals(F.value)))
        logger.info(f"Max eigenvalue of F: {max_eig_F:.6e}")

        for Gi in Gs:
            min_eig_Gi = np.min(np.real(np.linalg.eigvals(Gi.value)))
            if min_eig_Gi < 0:
                logger.warning(f"Locality LMI violated: min eigenvalue = {min_eig_Gi:.6e}")
            else:
                logger.info(f"Locality LMI satisfied: min eigenvalue = {min_eig_Gi:.6e}")

        # update model parameters
        self.P.data = torch.tensor(P.value)
        self.L.data = torch.tensor(L.value)
        self.la.data = torch.tensor(np.diag(M.value))
        # self.M = torch.diag(self.la)
        self.s.data = torch.tensor(s_star)


        # calculate output bound
        C = self.C.cpu().detach().numpy()
        P_star = P.value
        X_star = np.linalg.inv(P_star)
        L_star = L.value

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

    def get_regularization_loss(self, method: Optional[str] = None, return_components: bool = False):
        """
        Compute custom regularization loss on model parameters.

        Args:
            method: Regularization method ('interior_point' or 'dual').
                   If None, uses self.regularization_method.
            return_components: If True, returns a dict with loss breakdown.
                             If False, returns total loss tensor.

        Returns:
            If return_components=False: Regularization loss tensor
            If return_components=True: Dict with keys:
                - 'total': Total regularization loss
                - 'feasibility': Feasibility regularization (barrier or dual)
                - 'parametric': Parametric regularization (L nonzero, etc.)
        """
        if method is None:
            method = self.regularization_method

        if method == "interior_point":
            return self._interior_point_regularization()
        elif method == "dual":
            if return_components:
                return self._dual_regularization(return_components=True)
            else:
                return self._dual_regularization()
        else:
            raise ValueError(f"Unknown regularization method: {method}")

        # add parameter regularization

    def _interior_point_regularization(self):
        """
        Interior point method: uses log-det barrier function.
        Requires strictly feasible parameters (all eigenvalues > 0).
        Gradients explode when constraints are violated.

        Returns:
            Regularization loss (sum of negative log-determinants)
        """
        feasibility_loss = torch.tensor(0.0, device=self.P.device)
        for f_i in self.get_lmis():
            feasibility_loss += -torch.logdet(f_i())
        for s_i in self.get_scalar_inequalities():
            feasibility_loss += -torch.log(s_i()).squeeze()

        return feasibility_loss

    def _dual_regularization(self, return_components: bool = False):
        """
        Dual method: penalizes negative eigenvalues with adaptive penalty coefficient.
        Allows infeasible parameters during training.

        For each LMI F that should be positive definite (F ≻ 0):
        - Compute eigenvalues of F
        - Penalize negative eigenvalues: sum(max(0, -λ)^2)
        - Scale by dual penalty coefficient

        Args:
            return_components: If True, returns dict with breakdown

        Returns:
            If return_components=False: Regularization loss (sum of negative eigenvalue penalties)
            If return_components=True: Dict with 'total', 'feasibility', 'parametric'
        """
        feasibility_loss = torch.tensor(0.0, device=self.P.device)

        for f_i in self.get_lmis():
            F = f_i()
            # Compute eigenvalues (real part, since F should be symmetric)
            eigenvalues = torch.linalg.eigvalsh(F)  # More efficient for symmetric matrices

            # Penalize negative eigenvalues: sum of squared violations
            # max(0, -λ)^2 = (ReLU(-λ))^2
            negative_eigs = torch.relu(-eigenvalues)
            violation = torch.sum(negative_eigs**2)

            feasibility_loss += self.dual_penalty * violation

        # Add parametric regularization to encourage non-zero L
        parametric_loss = self._parametric_regularization()
        total_loss = feasibility_loss + parametric_loss

        if return_components:
            return {
                'total': total_loss,
                'feasibility': feasibility_loss,
                'parametric': parametric_loss
            }
        else:
            return total_loss

    def _parametric_regularization(self) -> torch.Tensor:
        """
        Parametric regularization to encourage specific parameter properties.

        Currently implements:
        - Inverse Frobenius norm penalty on L to encourage non-zero values
          (penalty increases as ||L|| → 0, encouraging larger magnitude)
          Formula: weight / (||L||_F + eps)

        Returns:
            Parametric regularization loss
        """
        param_loss = torch.tensor(0.0, device=self.P.device)

        if self.l_nonzero_weight > 0 and self.learn_L:
            # Inverse norm penalty with numerical stability
            param_loss += self.l_nonzero_weight / (torch.norm(self.L, p="fro") + 1e-6)

        # minimize Frobenius norm of P to encourage smaller values (less conservative)
        # param_loss += self.l_nonzero_weight * torch.norm(self.P, p="fro")

        return param_loss

    def update_dual_penalty(self, constraints_satisfied: bool):
        """
        Update the dual penalty coefficient based on constraint satisfaction.

        Args:
            constraints_satisfied: True if all constraints are satisfied, False otherwise
        """
        if constraints_satisfied:
            # Reduce penalty when constraints are satisfied
            self.dual_penalty *= self.dual_penalty_shrink
        else:
            # Increase penalty when constraints are violated
            self.dual_penalty *= self.dual_penalty_growth

    def get_constraint_violation(self) -> float:
        """
        Compute the total constraint violation (sum of negative eigenvalues).
        Useful for monitoring during training.

        Returns:
            Total violation (0 if all constraints satisfied)
        """
        total_violation = 0.0
        with torch.no_grad():
            for f_i in self.get_lmis():
                F = f_i()
                eigenvalues = torch.linalg.eigvalsh(F)
                negative_eigs = torch.relu(-eigenvalues)
                total_violation += torch.sum(negative_eigs).item()

        return total_violation

    def get_regularization_input(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        return_c: bool = False,
        warmup_steps: int = 0,
    ) -> torch.Tensor:
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

        # Apply ReLU (only penalize violations, i.e., c > 0)
        # and square for smooth penalty
        # Mean over all timesteps and batch samples
        # reg_loss = torch.relu(c).pow(2).mean()
        reg_loss = torch.relu(c).mean()

        if return_c:
            return reg_loss, c

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
