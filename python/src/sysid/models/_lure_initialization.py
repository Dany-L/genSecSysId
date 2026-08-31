"""Parameter initialization for SimpleLure.

Split out of ``constrained_rnn.py`` as a mixin. Every method uses ``self`` and
relies on SimpleLure's params/methods (``analysis_problem_init``,
``check_constraints``, ``deadzone_activity``, ``set_output_coverage_level``,
``get_lmis``, and the ``structural_constraints`` dict SimpleLure.__init__ parses)
— provided at runtime via the MRO.

``_is_parameter_fixed`` / ``_should_skip_initialization`` /
``_apply_partial_initialization`` live here rather than next to the rest of the
structural-constraint machinery in ``constrained_rnn.py`` because the
initialization is their only caller: they answer "does this init write reach this
parameter, and which slice of it".
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from sysid.optimization import InitializationReport
from sysid.utils import get_volume_of_ellipsoid, max_abs_output

from ..data import DataNormalizer

logger = logging.getLogger(__name__)


def _zoh_discretize(A_ct: torch.Tensor, ts: float) -> Dict[str, torch.Tensor]:
    r"""Exact zero-order-hold discretisation of ``\dot x = A_ct x + B_ct u``.

    Returns ``{"A": expm(A_ct*ts), "int": int_0^ts expm(A_ct*tau) dtau}`` so the
    caller can form ``B = int @ B_ct``. Both come from one matrix exponential of
    the block matrix ``[[A_ct, I], [0, 0]]``:

        expm([[A_ct, I],[0, 0]] * ts) = [[expm(A_ct*ts), int_0^ts expm(A_ct*tau)dtau],
                                         [0,             I                          ]]

    Forward Euler (``I + A_ct*ts``) is the first-order truncation of the same
    quantity; see running-example/reference-model for what that costs on the
    Duffing (2/60 spurious divergences, 4x the open-loop nrmse).
    """
    n = A_ct.shape[0]
    device, dtype = A_ct.device, A_ct.dtype
    blk = torch.zeros((2 * n, 2 * n), device=device, dtype=dtype)
    blk[:n, :n] = A_ct
    blk[:n, n:] = torch.eye(n, device=device, dtype=dtype)
    E = torch.matrix_exp(blk * float(ts))
    return {"A": E[:n, :n].contiguous(), "int": E[:n, n:].contiguous()}


class LureInitializationMixin:
    def initialize_parameters(
        self,
        train_inputs,
        train_states,
        train_outputs,
        init_config=None,
        normalizer: Optional[DataNormalizer] = None,
    ):
        """Initialize model parameters with the **identity** strategy.

        Two steps, and then training starts from the result:

        1. :meth:`_init_identity` — stable Euler-discretized A, deterministic
           input-scaled B, identity-like C scaled by ``1/output_std``, zero
           D/D12, and random B2/C2/D21 at the configured stds.
        2. If :meth:`check_constraints` fails (it usually does — D21 is drawn
           blind), :meth:`analysis_problem_init` with ``learn_B=False,
           learn_D21=True``: MaxS over ``(D21, P, L, Lambda, s)``. Same LMIs and
           same ``min 1/s^2`` objective as
           :meth:`LureCertificateSynthesizer.max_s`, with D21 additionally free.
           A, B, C, D, D12, B2 and C2 are left exactly as step 1 set them.

        That is the whole initialization. Beyond the two steps only
        ``set_output_coverage_level`` runs, to load the ``output_std`` / ``y_max``
        buffers the output-coverage regularizer reads, plus the dead-zone activity
        probe (diagnostics only). The input floor ``u_max`` is NOT set here; the
        Trainer computes it from the training loader on first use
        (``_ensure_input_bound``). ESN / N4SID inits were removed — ``identity``
        is the only supported method.

        Note that ``s`` therefore ends at the MaxS ceiling for this theta, which
        is typically well below the input floor ``sqrt(u_max)``; the input
        condition is then violated at the input peaks and the regional
        certificate does not cover the training rollout. Whether the initial
        rollout stays bounded depends on the draw — check it per seed.

        Args:
            train_inputs: Training input data (B, N, nd).
            train_states: Training state data (unused by the identity init; kept
                for API symmetry with the loaders).
            train_outputs: Training output data (B, N, ne) — used for y_max.
            init_config: InitializationConfig; ``method`` must be ``'identity'``.
            normalizer: Data normalizer used to scale C/B and derive y_max.

        Returns:
            :class:`~sysid.optimization.solutions.InitializationReport` — the
            established certificate diagnostics; ``to_metrics()`` yields the
            ``initialization/`` mlflow metrics.
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

        # The dead-zone activity probe below rolls the model out, so it needs the
        # inputs in the same normalized units the model sees during training.
        if normalizer is not None:
            train_inputs = normalizer.transform_inputs(train_inputs)

        self._init_identity(normalizer)

        # Common post-initialization
        constraints_ok = self.check_constraints()
        logger.info(f"Initialization complete. Constraints satisfied: {constraints_ok}")
        logger.info("=" * 80)

        # Step 2: repair an infeasible draw with MaxS over (D21, P, L, Lambda, s).
        #
        # _init_identity samples D21 from N(0, std^2) with no reference to the
        # data, so the draw usually lands outside the feasible set. This solve
        # maximizes s (minimizes S_hat = 1/s^2) under the same stability +
        # locality LMIs as LureCertificateSynthesizer.max_s, but with D21 as an
        # additional free variable — A, B, C, D, D12, B2 and C2 keep exactly the
        # values the identity init gave them.
        #
        # D21 has to be in the solve: it is the input->nonlinearity map, and under
        # scale_only normalization the Duffing inputs reach |d_n| ~ 9.7, so a
        # random D21 drives z = C2 x + D21 d far outside the dead band and
        # collapses the certifiable s. Solving it shrinks D21 ~9x and lifts s.
        #
        # learn_B stays False on purpose: with B free too the SDP drives both B
        # and D21 to zero — trivially certifiable, but a dead model (e_hat == 0).
        bootstrap_d21 = (
            bool(getattr(init_config, "bootstrap_d21_on_infeasible", True))
            if init_config is not None else True
        )
        if bootstrap_d21 and not constraints_ok:
            if not self.analysis_problem_init(learn_B=False, learn_D21=True):
                # Nothing downstream repairs this — the bootstrap is the whole
                # initialization — so an infeasible solve here means training
                # would start outside the feasible set.
                raise RuntimeError(
                    "Initialization failed: the identity draw left the stability "
                    "LMI infeasible and the MaxS solve over (D21, P, L, s) found "
                    "no feasible point. Check the identity initialization / "
                    "structural constraints (e.g. A must be stable, alpha < 1), "
                    "or lower the C2/B2 init std."
                )
            constraints_ok = self.check_constraints()
            logger.info(
                f"  MaxS over (D21, P, L, s): ||D21|| = "
                f"{np.linalg.norm(self.D21.detach().cpu().numpy()):.4g}, "
                f"s = {float(self.s):.4g}, constraints satisfied: {constraints_ok}"
            )

        # y_max is PHYSICAL (max |raw training output|); output_std relates the
        # model's normalized C/P/s to physical units.
        sigma = normalizer.output_std if normalizer is not None else 1.0
        sigma_scalar = float(np.asarray(sigma).reshape(-1)[0])
        y_max = max_abs_output(train_outputs) if normalizer is not None else None
        self.set_output_coverage_level(y_max, sigma_scalar)

        # Dead-zone activity of the final initialized model: firing_rate == 0 means
        # the nonlinearity is inert on the training data and, because Δ'(z) = 0 in
        # the dead band, training can never revive it — so this is the number that
        # decides whether the model class is usable at all.
        activity: Dict[str, float] = {}
        if train_inputs is not None:
            try:
                inp = torch.as_tensor(
                    np.asarray(train_inputs), dtype=self.C2.dtype, device=self.C2.device
                )
                if inp.dim() == 2:
                    inp = inp.unsqueeze(-1)
                activity = self.deadzone_activity(inp)
            except Exception as exc:  # diagnostics must never break initialization
                logger.debug(f"Dead-zone activity probe failed: {exc}")

        # Certificate diagnostics read off the model as the bootstrap left it:
        # the bootstrap writes its solution straight into the parameters, so there
        # is no solution object to read here. Coverage (y_bar / coverage_ok) is not
        # measured on this path and stays None.
        P_c = self.P.detach().cpu().numpy()
        L_c = self.L.detach().cpu().numpy()
        s_c = float(self.s)
        report = InitializationReport(
            volume=float(get_volume_of_ellipsoid(P_c, s_c)),
            s=s_c,
            norm_H=float(np.linalg.norm(L_c @ np.linalg.inv(P_c), ord=2)),
            max_eig_F=self._max_eig_stability_lmi(),
            constraints_satisfied=bool(constraints_ok),
            y_max=float(y_max) if y_max is not None else None,
            firing_rate=activity.get("firing_rate"),
            units_firing=activity.get("units_firing"),
            max_abs_z=activity.get("max_abs_z"),
        )
        logger.info(
            "INITIALIZATION certificate (D21 bootstrap): "
            f"volume={report.volume:.3e}, s={report.s:.4f}, ||H||_2={report.norm_H:.4f}, "
            f"y_max={y_max}, constraints_satisfied={constraints_ok}"
        )
        if report.firing_rate is not None:
            logger.info(
                f"INITIALIZATION dead-zone activity: firing_rate="
                f"{100 * report.firing_rate:.3f}% of (step, unit) pairs, "
                f"units_firing={100 * (report.units_firing or 0.0):.0f}%, "
                f"max|z|={report.max_abs_z:.3f} (dead band |z|<=1)"
            )
            if report.firing_rate <= 0.0:
                logger.warning(
                    "INITIALIZATION: the dead zone is INERT on the training data — the "
                    "model is LTI in this regime and, since Δ'(z)=0 inside the band, no "
                    "gradient reaches B2/C2/D21, so training cannot revive it. "
                    "Diagnostic only; the initialization does not optimize for firing."
                )
        self._last_init_report = report
        return report

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
                # Forward-Euler discretisation, matching how the benchmark's
                # reference Lur'e model is built (notebooks/Duffing). The exact ZOH
                # A = expm(A_ct*ts) is available via _zoh_discretize and is more
                # accurate, but it makes the model a different discrete system than
                # the one the dataset's reference model represents; see the wiki note
                # running-example/reference-model.
                A_init = torch.eye(self.nx, device=device, dtype=dtype) + A_ct * self.ts
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
                        input_scale = float(np.asarray(input_std).reshape(-1)[0])
                B_init = input_scale * self.ts * torch.tensor(
                    [[0.0], [1.0]], device=self.B.device, dtype=self.B.dtype
                )
                # B_init = 0.01*self.ts * torch.tensor(
                #     [[0.0], [1.0]], device=self.B.device, dtype=self.B.dtype
                # )
            self._set_param_data('B', B_init)

        # --- B2, C2, D21: random (configurable std), or C2 by breakpoint placement ---
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
                    output_scale = 1.0
                    # raise ValueError(
                    #     "Identity initialization of 'C' requires a normalizer with "
                    #     "'output_std', or an explicit 'identity_init.C.value' / "
                    #     "'identity_init.C.load_from' override in custom_params."
                    # )
                else:
                    output_scale = normalizer.output_std.squeeze()
                C_init = (1.0 / output_scale) * torch.tensor(
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

    def _max_eig_stability_lmi(self) -> float:
        """``max eig F`` of the certificate the model currently carries.

        ``get_lmis()[0]`` is the positive-definite form ``-F``, so the largest
        eigenvalue of ``F`` is minus its smallest. Recomputed from the model
        rather than taken from a solution object, because the certificate that
        ends up applied is not always the one MaxS returned."""
        with torch.no_grad():
            return float(-torch.linalg.eigvalsh(self.get_lmis()[0]()).min())

