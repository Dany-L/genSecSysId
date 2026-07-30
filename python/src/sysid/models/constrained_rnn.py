import logging
from typing import Dict, List, Literal, Optional, Tuple, Union, overload

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

from sysid.optimization import CertificateSolution, LureCertificateSynthesizer
from sysid.utils import (
    get_volume_of_ellipsoid,
    plot_ellipse_and_parallelogram,
    torch_bmat,
)

from .base import DznActivation, LureSystem, LureSystemClass, LureSystemSafe
from ._lure_initialization import LureInitializationMixin
from ._lure_regularization import LureRegularizationMixin

logger = logging.getLogger(__name__)

EPS = 1e-6


class SimpleLure(LureInitializationMixin, LureRegularizationMixin, nn.Module):
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

        # True once freeze_certificate() has run: (P, L, la, s) are SDP-owned and
        # carry no gradient. Purely informational — the barrier detects the
        # constant terms from requires_grad, not from this flag.
        self.certificate_frozen = False

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

    def _synth(self) -> LureCertificateSynthesizer:
        """Build a certificate synthesizer from the current (fixed) dynamics.

        All the certificate SDPs (MaxS, MaxVol, coverage, feasibility, C2
        calibration) live on :class:`~sysid.optimization.LureCertificateSynthesizer`
        and return typed results from :mod:`sysid.optimization.solutions`. This is
        the single seam between the model and that optimization layer.
        """
        return LureCertificateSynthesizer.from_model(self)

    def feasibility_problem(self, enforce_coverage: bool = False) -> bool:
        """Repair the certificate at the **current** (fixed) ``s`` — the
        within-epoch step 3.1.

        When a gradient update breaks the LMIs, solve the fixed-``s`` feasibility
        SDP (:meth:`LureCertificateSynthesizer.feasibility`) at ``self.s``
        (unchanged) for feasible P, M, L and write them back. Returns ``True`` on
        success; ``False`` when no feasible P, M, L exists at that ``s`` (→ the
        trainer rolls the step back or drops to the second tier). ``s`` is never
        modified here — it is set only by initialization and re-synthesis.

        ``enforce_coverage=True`` adds the **hard coverage floor**
        ``(σ·s)²·C P Cᵀ ⪰ y_max²·I`` to the repair, so it cannot restore
        feasibility by shrinking the certified output image. Requires ``y_max``
        to be set; silently falls back to the floor-free repair otherwise. This
        is the first tier of the trainer's two-tier repair.
        """
        s = float(self.s.cpu().detach().numpy())
        y_max = None
        if enforce_coverage and self.y_max is not None and not bool(torch.isnan(self.y_max)):
            y_max = float(self.y_max)
        sol = self._synth().feasibility(s, y_max=y_max)
        if sol is None:
            return False
        # Write back P, L, Λ only — s stays fixed.
        device, dtype = self.P.device, self.P.dtype
        self.P.data = torch.tensor(sol.P, device=device, dtype=dtype)
        if self.learn_L:
            self.L.data = torch.tensor(sol.L, device=device, dtype=dtype)
        self.la.data = torch.tensor(np.diag(sol.M), device=device, dtype=dtype)
        return True

    # ------------------------------------------------- certificate ownership
    def freeze_certificate(self, freeze_alpha: bool = True) -> List[str]:
        """Take the certificate out of the gradient — the ownership split of the
        re-synthesis scheme: **θ is owned by SGD, (P, L, Λ, s) by the SDPs**.

        Rationale: none of ``P, L, Λ, s`` appears in the prediction loss (the
        rollout uses only A, B, B2, C, C2, D, D12, D21), so their only gradient is
        the interior-point barrier's — which has a preferred direction and no
        counterweight. The locality barrier ``-log det[1/s², l; lᵀ, P]`` rewards
        ``1/s² → ∞``, i.e. ``s → 0``, which is exactly the observed drift.

        ``freeze_alpha`` also freezes ``τ`` (hence ``α``). ``α`` is *not* in the
        rollout either, and the ``-α²P`` block means larger ``α`` slackens the
        stability LMI, so the barrier pushes ``α → 1`` (the weakest contraction
        claim) with nothing pushing back — the same defect as ``s → 0``, bounded
        only by the sigmoid.

        After freezing, the ``nz`` locality LMIs contain **no θ at all**, so their
        barrier terms become additive constants (dropped by
        :meth:`get_regularization_loss`) and a gradient step can no longer violate
        them — only the stability LMI can break.

        Returns the names of the parameters that were frozen.
        """
        frozen: List[str] = []
        names = ["P", "la", "s"] + (["L"] if self.learn_L else [])
        if freeze_alpha:
            names.append("tau")
        for name in names:
            param = getattr(self, name, None)
            if isinstance(param, nn.Parameter) and param.requires_grad:
                param.requires_grad = False
                frozen.append(name)
        self.certificate_frozen = True
        logger.info(
            f"Certificate frozen from autograd: {frozen or 'nothing (already frozen)'} "
            "(P, L, Λ, s are now SDP-owned)"
        )
        return frozen

    def deadzone_activity(
        self,
        inputs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        warmup_steps: int = 0,
    ) -> Dict[str, float]:
        """How much the nonlinearity actually fires on a rollout.

        Rolls the model out on ``inputs`` and reports where the pre-activation
        ``z_k = C2·x_k + D21·d_k`` leaves the dead band ``|z| ≤ 1``:

        * ``firing_rate`` — fraction of (step, unit) pairs with ``|z| > 1``;
        * ``steps_firing`` — fraction of steps where *any* unit fires;
        * ``units_firing`` — fraction of units that fire at least once;
        * ``max_abs_z`` — the largest ``|z|`` reached.

        Why this matters: for the dead-zone activation ``Δ'(z) = 0`` inside the
        band, so when ``firing_rate == 0`` **no gradient from the prediction loss
        reaches B2, C2 or D21** — the model is LTI on that data and the collapse is
        an absorbing state that training cannot escape (only the initialization
        can). It is therefore a first-class init/training diagnostic, not a
        curiosity. Note a *small* rate is normal and can be optimal: the
        nonlinearity's job may be the peaks and the out-of-regime behaviour rather
        than the bulk of the fit.

        Only meaningful for the ``dzn`` activation, where ``|z| ≤ 1`` is exactly
        the linear regime.
        """
        with torch.no_grad():
            if hasattr(self, "forward_unfiltered"):
                _, (x, _), d_applied = self.forward_unfiltered(inputs, x0)
            else:
                _, (x, _), d_applied = self.forward(inputs, x0, warmup_steps=warmup_steps)
            if x.dim() == 4:
                x = x.squeeze(-1)
            n = min(x.shape[1], d_applied.shape[1])
            x, d = x[:, warmup_steps:n, :], d_applied[:, warmup_steps:n, :]
            if x.shape[1] == 0:
                return {"firing_rate": 0.0, "steps_firing": 0.0,
                        "units_firing": 0.0, "max_abs_z": 0.0}
            z = x @ self.C2.T + d @ self.D21.T  # (B, N, nz)
            fired = z.abs() > 1.0
            return {
                "firing_rate": float(fired.double().mean()),
                "steps_firing": float(fired.any(dim=-1).double().mean()),
                "units_firing": float(fired.any(dim=0).any(dim=0).double().mean()),
                "max_abs_z": float(z.abs().max()),
            }

    def coverage_ratio(self) -> Optional[float]:
        """The tightness ratio ``ρ = (ȳ/y_max)ⁿˣ`` of the **current** certificate.

        ``ȳ = σ·s·√(λ_min(C P Cᵀ))`` is the certified physical output half-width in
        the worst output direction, so ``ρ ≥ 1`` ⇔ the certified image covers the
        data level and ``ρ`` is the volume ratio against the minimal covering set.

        This is the **free drift monitor** of the re-synthesis scheme: a few small
        matrix products on parameters already in memory, no SDP. Returns ``None``
        when ``y_max`` is unset (then there is no reference level and re-synthesis
        must run on a cadence instead).
        """
        if self.y_max is None or bool(torch.isnan(self.y_max)):
            return None
        y_max = float(self.y_max)
        if y_max <= 0:
            return None
        with torch.no_grad():
            CPCt = self.C @ self.P @ self.C.T
            lam_min = max(float(torch.linalg.eigvalsh(CPCt).min()), 0.0)
            y_bar = float(self.output_std) * float(self.s) * float(np.sqrt(lam_min))
        return float((y_bar / y_max) ** self.nx)

    def resynthesize_certificate(
        self,
        y_max: Optional[float] = None,
        beta: Optional[float] = 2.0,
        guard_inputs: Optional[torch.Tensor] = None,
        guard_x0: Optional[torch.Tensor] = None,
        warmup_steps: int = 0,
    ) -> dict:
        """Re-solve the certificate from the **current θ** and write it back — the
        per-epoch step of the re-synthesis scheme.

        Calls :meth:`LureCertificateSynthesizer.tight_cert`, which pins
        ``ρ ∈ [1, βⁿˣ]`` by construction (one SDP; see that method for why the
        ``ŝ = 1/s²`` substitution makes it convex). Without ``y_max`` the band
        drops and it degenerates to MaxS.

        ``guard_inputs`` enables the **accept guard**: re-synthesis moves the
        certificate discontinuously, which moves the input-condition landscape
        ``‖u_k‖² ≤ s² − α²x_kᵀP⁻¹x_k`` mid-training, so a new certificate is
        rejected when it would **break a currently clean rollout** — i.e. only when
        the old certificate had *zero* violating trajectories and the new one has
        some.

        The guard deliberately does **not** compare counts. A ``n_new ≤ n_old``
        test locks the certificate up exactly when it must adapt: while the model
        is diverging, every candidate certificate shows violations, so the test
        can never pass and κ stays frozen at a stale value through the whole
        excursion (observed on Duffing — a θ blow-up at epoch 23 vetoed
        re-synthesis for two epochs while ρ ran to 17). Once the rollout is
        already violating, the freshly synthesized certificate is the best
        available claim about the current θ and is always accepted.

        Without ``guard_inputs`` the new certificate is accepted whenever the SDP
        succeeds.

        Returns a flat, log-friendly dict::

            {"success", "applied", "reason", "s", "rho", "y_bar", "beta",
             "band_enforced", "norm_P", "n_violations", "n_violations_before"}
        """
        if y_max is None and self.y_max is not None and not bool(torch.isnan(self.y_max)):
            y_max = float(self.y_max)

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

        n_viol_before = (
            self._count_input_violations(guard_inputs, guard_x0, warmup_steps)
            if guard_inputs is not None else None
        )

        sol = self._synth().tight_cert(y_max=y_max, beta=beta)
        if sol is None:
            return {
                "success": False,
                "applied": False,
                "reason": "sdp_infeasible",
                "n_violations_before": n_viol_before,
            }

        self._apply_certificate_solution(sol)

        n_viol = None
        if guard_inputs is not None:
            n_viol = self._count_input_violations(guard_inputs, guard_x0, warmup_steps)
            # Veto only a clean -> dirty transition; never while the rollout is
            # already violating (see the docstring: a count comparison would
            # freeze the certificate for the whole excursion).
            if n_viol_before == 0 and n_viol > 0:
                _restore()
                logger.info(
                    f"Re-synthesis rejected by the input-condition guard "
                    f"(clean rollout → {n_viol} violating trajectories); keeping the old "
                    "certificate."
                )
                return {
                    "success": True,
                    "applied": False,
                    "reason": "guard_rejected",
                    "s": float(sol.s),
                    "rho": sol.rho,
                    "y_bar": sol.y_bar,
                    "beta": sol.beta,
                    "band_enforced": sol.band_enforced,
                    "norm_P": sol.norm_P,
                    "n_violations": n_viol,
                    "n_violations_before": n_viol_before,
                }

        return {
            "success": True,
            "applied": True,
            "reason": "ok",
            "s": float(sol.s),
            "rho": sol.rho,
            "y_bar": sol.y_bar,
            "beta": sol.beta,
            "band_enforced": sol.band_enforced,
            "norm_P": sol.norm_P,
            "n_violations": n_viol,
            "n_violations_before": n_viol_before,
        }

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

        **Problem 1 — max-feasible-s certificate** (MaxS, :meth:`~sysid.optimization.LureCertificateSynthesizer.max_s`).
        Maximizes the scale ``s`` (minimizes ``1/s²``) subject to the stability +
        locality LMIs only — the largest *regional* certifiable invariant set. It
        is well conditioned (a moderate ``s``, not the tiny-``s``/huge-``P`` corner
        the volume objective falls into) and is the operative certificate written
        back into the model. It does **not** constrain output coverage — the
        coverage floor ``(σ·s)²·C P Cᵀ ⪰ y_max²·I`` is *checked afterwards*
        (``coverage_ok``). Reported: the ellipsoid ``volume`` (``sⁿˣ·√(det P)``),
        the coupling norm ``‖H‖ = ‖L P⁻¹‖``, the scale ``s`` and the certified
        output half-width ``ȳ_c = σ·s·√(C P Cᵀ)`` (physical; ``ne == 1`` only).

        **Problem 2 — tightest coverage** (MinTrProb, :meth:`~sysid.optimization.LureCertificateSynthesizer.coverage_at_s` swept
        over a finite s-grid). The joint problem is bilinear (convex once ``s`` is
        fixed), so it is gridded over ``s ∈ [s_min, s_max]``; reported is the
        smallest feasible certified half-width ``ȳ_f`` — the tightest coverage of
        the demanded ``y_max``. Skipped when ``y_max`` is unset.

        The MaxS solution (ȳ_c) is the one written back into the model; ``ρ`` (the
        volume ratio ``vol(𝒳_MaxS)/vol(𝒳_cov)``) is reported as a tightness
        diagnostic. ``y_max`` is physical; ``None`` falls back to the model's
        stored physical level (which may be unset, in which case Problem 2 and the
        coverage check are skipped).

        Returns a summary dict::

            {
              "success": bool,
              "s_opt": float,                 # operative (MaxS) s == max_s["s"]
              "constraints_satisfied": bool,
              "y_max": Optional[float],       # physical demanded level (or None)
              "max_s": {s, volume, norm_H, y_bar(=ȳ_c), max_eig_F, coverage_ok, rho},
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
            "POST-PROCESSING: (1) max-feasible-s certificate + (2) tightest-coverage sweep"
        )
        logger.info("=" * 80)

        # ------------------------------------------------------------------
        # Problem 1 — MaxS: the largest regional certifiable invariant set
        # (operative, well conditioned — moderate s, not the tiny-s/huge-P corner).
        # ------------------------------------------------------------------
        synth = self._synth()
        max_s_sol = synth.max_s()
        if max_s_sol is None:
            return {"success": False, "status": "max_s_sdp_failed"}

        P_c, L_c, s_c = max_s_sol.P, max_s_sol.L, max_s_sol.s
        vol_c = float(get_volume_of_ellipsoid(P_c, s_c))
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

        logger.info("[Problem 1: MaxS — largest regional invariant set]")
        logger.info(f"  volume   = {vol_c:.3e}")
        logger.info(f"  s        = {_fmt(s_c)}")
        logger.info(f"  ‖H‖      = {_fmt(norm_H_c)}   (H = L P⁻¹)")
        logger.info(f"  ȳ_c      = {_fmt(y_c)}   (σ·s·√(C P Cᵀ))")
        logger.info(f"  max λ(F) = {max_s_sol.max_eig_F:.3e}")
        if coverage_ok is not None:
            logger.info(
                f"  coverage ((σ·s)²·CPCᵀ ≥ y_max²={y_max ** 2:.4g}): "
                f"{'OK' if coverage_ok else 'NOT met'}"
            )

        # ------------------------------------------------------------------
        # Problem 2 — tightest coverage over the s-grid (reported, not applied).
        # ------------------------------------------------------------------
        y_f = s_f = None
        rho = None
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
                # Tightness ratio of the OPERATIVE (MaxS) certificate: how much its
                # own certified image over-covers y_max, as a volume ratio
                # rho = (ȳ_c/y_max)^nx = vol(𝒳_MaxS)/vol(minimal covering set).
                if y_c is not None and y_max > 0:
                    rho = float((y_c / y_max) ** self.nx)
                logger.info("[Problem 2: coverage — tightest ȳ over the s-grid]")
                logger.info(
                    f"  ȳ_f = {_fmt(y_f)}  at s = {_fmt(s_f)}   "
                    f"(target y_max = {_fmt(y_max)}); ρ = (ȳ_c/y_max)^nx = {_fmt(rho)}"
                )
        else:
            logger.info("[Problem 2: coverage] skipped (y_max unset)")

        # ------------------------------------------------------------------
        # Set the model to the largest regional invariant set (MaxS / ȳ_c).
        # ------------------------------------------------------------------
        self._apply_certificate_solution(max_s_sol)
        constraints_ok = self.check_constraints()
        logger.info(
            f"Applied MaxS certificate to the model. Constraints satisfied: {constraints_ok}"
        )
        logger.info("=" * 80)

        return {
            "success": True,
            "s_opt": s_c,
            "constraints_satisfied": constraints_ok,
            "y_max": y_max,
            "max_s": {
                "s": s_c,
                "volume": vol_c,
                "norm_H": norm_H_c,
                "y_bar": y_c,
                "max_eig_F": float(max_s_sol.max_eig_F),
                "coverage_ok": coverage_ok,
                "rho": rho,
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
