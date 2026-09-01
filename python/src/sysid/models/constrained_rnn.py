"""The constrained Lure model itself: parameters, dynamics and its LMIs.

``SimpleLure`` is assembled from three mixins so this module stays about the
model rather than about everything done *to* it:

* :mod:`~sysid.models._lure_initialization` — parameter init + the D21 bootstrap.
* :mod:`~sysid.models._lure_regularization` — the training-time barrier losses
  and feasibility margins.
* :mod:`~sysid.models._lure_post_processing` — the after-training certificate
  synthesis (``post_process``, ``solve_output_coverage_certificate``).

What stays here: the parameters and their structural constraints, the forward
rollout, the LMIs (:meth:`SimpleLure.get_lmis`) every other layer is defined
against, and the two seams to the optimization layer —
:meth:`SimpleLure._synth`, which snapshots θ into a
:class:`~sysid.optimization.LureCertificateSynthesizer`, and
:meth:`SimpleLure._apply_certificate_solution`, which writes a solution back.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sysid.optimization import CertificateSolution, LureCertificateSynthesizer
from sysid.utils import torch_bmat

from .base import DznActivation, LureSystem, LureSystemClass, LureSystemSafe
from ._lure_initialization import LureInitializationMixin
from ._lure_post_processing import LurePostProcessingMixin
from ._lure_regularization import LureRegularizationMixin

logger = logging.getLogger(__name__)

EPS = 1e-6


class SimpleLure(
    LureInitializationMixin,
    LureRegularizationMixin,
    LurePostProcessingMixin,
    nn.Module,
):
    """Discrete-time Lure system ``x⁺ = Ax + Bd + B₂Δ(C₂x + D₂₁d)`` with a
    learnable regional stability certificate.

    The prediction dynamics θ = (A, B, B₂, C, C₂, D₂₁, α) and the certificate
    (P, L, Λ, s) are **both** learnable and move together under one objective:
    prediction loss + log-det barrier. A step that leaves the feasible set is
    repaired by :meth:`feasibility_problem` or rolled back by the trainer.

    See the module docstring for which mixin owns what.
    """

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

        # Input floor u_max = sup_k ||u_k||^2 over the (normalized) training inputs;
        # s >= sqrt(u_max) is necessary for the input condition. Set by
        # set_input_bound(); nan = unset.
        self.register_buffer("u_max", torch.tensor(float("nan")))

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
            
            # Register hook.
            #
            # The mask is a plain tensor captured in the closure, NOT a buffer, so
            # nn.Module.to() does not move it — after model.to("cuda") the parameter
            # and its gradient are on the GPU while the mask is still on the CPU.
            # Re-cast on first use and cache, which also covers a later dtype change.
            def make_hook(mask_tensor):
                def hook(grad):
                    if grad is None:
                        return None
                    nonlocal mask_tensor
                    if (mask_tensor.device != grad.device
                            or mask_tensor.dtype != grad.dtype):
                        mask_tensor = mask_tensor.to(
                            device=grad.device, dtype=grad.dtype
                        )
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

    def feasibility_problem(self) -> bool:
        """Repair the certificate with θ and α held fixed — the within-epoch step.

        When a gradient update breaks the LMIs, solve the feasibility SDP
        (:meth:`LureCertificateSynthesizer.feasibility`) for a certificate that
        satisfies them again and write it back. Returns ``True`` on success;
        ``False`` when none exists, in which case the trainer rolls the step back.

        Two attempts, cheapest first:

        1. **fixed ``s``** — only (P, L, Λ) move. This is the smallest repair and
           leaves the scale where the prediction loss and the barrier put it,
           which matters because ``s`` is a *learned* parameter here.
        2. **free ``s``** — only if (1) is infeasible. The ``ŝ = 1/s²``
           substitution keeps the problem convex, so this is still one solve. It
           overwrites the learned scale, which is why it is the fallback and not
           the default.
        """
        s = float(self.s.cpu().detach().numpy())
        sol = self._synth().feasibility(s)
        if sol is None:
            sol = self._synth().feasibility(None)
            if sol is None:
                return False
            logger.debug(
                f"Fixed-s repair infeasible at s={s:.4g}; repaired with s free -> {sol.s:.4g}"
            )
        self._apply_certificate_solution(sol)
        return True

    def set_input_bound(self, u_max_sq: Optional[float]) -> None:
        """Store the **input floor** ``u_max = sup_{k,i} u_kᵀu_k`` (NORMALIZED units,
        i.e. the inputs as fed to the model), from which ``s ≥ √u_max``.

        The input condition ``‖u_k‖² ≤ s² − α²x_kᵀP⁻¹x_k`` forces
        ``s² ≥ ‖u_k‖² + α²x_kᵀP⁻¹x_k ≥ ‖u_k‖²`` for every sample, because
        ``P ≻ 0`` makes the quadratic form non-negative. So ``s ≥ √u_max`` is
        **necessary for any P and α** — a property of the data, not of the
        certificate — and every solve that is free to choose ``s`` must respect it,
        or it returns a certificate that does not admit its own training inputs.
        Necessary, not sufficient: ``P`` must additionally keep
        ``α²x_kᵀP⁻¹x_k ≤ s² − ‖u_k‖²`` along the rollout.

        ``None``/``nan`` clears the floor (solves then run unconstrained in ``s``).
        """
        device, dtype = self.P.device, self.P.dtype
        self.u_max = torch.tensor(
            float(u_max_sq) if u_max_sq is not None else float("nan"),
            device=device, dtype=dtype,
        )

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

    def analysis_problem_init(self, learn_B: bool = False, learn_D21: bool = False) -> bool:
        """Repair an infeasible identity draw by solving for the input maps too.

        Thin wrapper over :meth:`LureCertificateSynthesizer.bootstrap` — MaxS with
        ``D21`` (and optionally ``B``) as free variables alongside the certificate
        ``(P, L, Lambda, s)``. Writes the solution back and returns ``True``;
        ``False`` when the SDP finds no feasible point.

        Used once, from :meth:`initialize_parameters`, when ``check_constraints()``
        fails after ``_init_identity``.
        """
        sol = self._synth().bootstrap(learn_B=learn_B, learn_D21=learn_D21)
        if sol is None:
            return False

        device, dtype = self.P.device, self.P.dtype
        self._apply_certificate_solution(sol)
        if sol.B is not None:
            self.B.data = torch.tensor(sol.B, device=device, dtype=dtype)
        if sol.D21 is not None:
            self.D21.data = torch.tensor(sol.D21, device=device, dtype=dtype)
        logger.info(f"SDP analysis problem solved: s = {float(self.s):.4g}")
        return True

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
        """Scalar constraints ``g(θ) > 0`` to add to the barrier, alongside the LMIs.

        Currently empty — the certificate is carried entirely by the LMIs in
        :meth:`get_lmis`. The candidate that mattered, ``s > 0``, is deliberately
        NOT registered: its ``-log s`` barrier term pushes ``s`` down with nothing
        in the prediction loss pulling back, so it drove ``s -> 0`` on its own. The
        agreed replacement is a hard coverage-floor LMI whose barrier counter-pushes
        ``s`` up; until that lands, ``s`` is held from below by
        :meth:`Trainer._maybe_maximize_s` instead. ``alpha < 1`` and ``alpha > 0``
        are enforced by the sigmoid parameterization of ``tau``, and the input-size
        condition is handled by the input regularizer.

        Kept as the extension point the three callers
        (:meth:`get_regularization_loss`, :meth:`get_feasibility_margins`,
        :meth:`check_constraints`) already handle — each iterates this list, so a
        new scalar constraint only has to be appended here. See
        ``tests/test_feasibility_margins.py`` for the xfail markers waiting on it.
        """
        return []

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
