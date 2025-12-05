import logging
from typing import Optional

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

from sysid.utils import plot_ellipse_and_parallelogram, torch_bmat

from .base import DznActivation, LureSystem, LureSystemClass

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize the Simple Lure system.

        """
        super().__init__()
        nz = nw
        self.nx = nx
        self.nd = nd
        self.ne = ne
        self.nw = nw
        self.nz = nz

        # Register delta and max_norm_x0 as buffers (saved with model, not trainable)
        self.register_buffer("delta", torch.tensor(delta))
        self.register_buffer("max_norm_x0_buffer", torch.tensor(max_norm_x0))
        self.max_norm_x0 = max_norm_x0  # Keep as attribute for compatibility

        self.P = nn.Parameter(torch.eye(nx))  # Lyapunov matrix
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

        if learn_L:
            self.L = nn.Parameter(torch.zeros((nz, nx)))  # Coupling matrix
            self.alpha = nn.Parameter(torch.tensor(0.99), requires_grad=True)
            self.s = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.L = torch.zeros((nz, nx))  # Coupling matrix, not learnable
            self.alpha = nn.Parameter(torch.tensor(0.99), requires_grad=False)
            self.s = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.la = nn.Parameter(torch.ones(nz))
        self.M = torch.diag(self.la)

        self.A = nn.Parameter(torch.zeros(nx, nx))
        self.B = nn.Parameter(torch.zeros(nx, nd))
        self.B2 = nn.Parameter(torch.zeros(nx, nw))

        self.C = nn.Parameter(torch.zeros(ne, nx))
        self.D = nn.Parameter(torch.zeros(ne, nd))
        self.D12 = nn.Parameter(torch.zeros(ne, nw))

        self.C2 = nn.Parameter(torch.zeros(nz, nx))
        self.D21 = nn.Parameter(torch.zeros(nz, nd))
        self.D22 = nn.Parameter(torch.zeros(nz, nw))

        if activation == "sat":
            Delta = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        elif activation == "dzn":
            Delta = DznActivation()
        elif activation == "tanh":
            Delta = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type: {activation}")

        self.lure = LureSystem(
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

        # self.initialize_parameters()

    def reset_s(self):
        self.s.data = torch.sqrt(self.delta**2 / (1 - self.alpha**2)).squeeze()

    def initialize_parameters(self, train_inputs, train_states, train_outputs):
        """Initialize parameters with small random values."""

        # 1. extract delta from training data delta = max ||d|| and pick s
        # actually this is done in the init already

        # 2. calculate s based on delta and alpha
        # self.s.data = torch.sqrt(self.delta**2 /(1-self.alpha**2)).squeeze()
        self.s.data = torch.sqrt(self.delta**2 / (1 - self.alpha**2)).squeeze()
        # self.s.data = torch.tensor(1.0)

        # 3. choose A to be stable, C2 random but large and get B, D21, P, L and M from solving the analysis problem
        self.A.data = 0.9 * torch.eye(self.nx)
        min_val, max_val = -1, 1
        nn.init.uniform_(self.C2, a=min_val, b=max_val)
        self.analysis_problem_init(learn_B_and_D21=True)

        # 5. set C, D, and D12 based on least squares
        # turns out the fixed choice works just fine, so we switched the order to get some results such that we can check if the condition on the input is satisfied
        self.D12.data = torch.zeros(self.ne, self.nw)
        self.D.data = torch.zeros(self.ne, self.nd)
        self.C.data = torch.zeros(self.ne, self.nx)
        self.C.data[:, 0] = torch.ones(self.ne)

        # 4. simulate the system to get x and w
        with torch.no_grad():
            B, N, _ = train_inputs.shape
            # x0_1 = torch.tensor(train_states[0,0,:].reshape(1,self.nx,1))
            # d_1 = torch.tensor(train_inputs[0,:,:].reshape(1,N,self.nd,1))
            x0s = torch.tensor(train_states[:, 0, :].reshape(B, self.nx, 1))
            ds = torch.tensor(train_inputs.reshape(B, N, self.nd, 1))
            xs, (xs, ws) = self.lure.forward(x0=x0s, d=ds, return_states=True)

        X = np.linalg.inv(self.P.cpu().detach().numpy())
        H = self.L.cpu().detach().numpy() @ X

        fig, ax = plot_ellipse_and_parallelogram(
            X, H, self.s.cpu().detach().numpy(), self.max_norm_x0
        )
        for x0i in x0s:
            ax.plot(
                x0i[0].cpu().detach().numpy(),
                x0i[1].cpu().detach().numpy(),
                "x",
                color="red",
                markersize=2,
            )

        for xi in xs:
            ax.plot(xi[:20, 0].cpu().detach().numpy(), xi[:20, 1].cpu().detach().numpy())

        self.check_constraints()

        return fig

    def analysis_problem(self, learn_B_and_D21: bool = False) -> bool:

        eps = 1e-3

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
        alpha = self.alpha.cpu().detach().numpy()
        s = self.s.cpu().detach().numpy()
        if alpha >= 1:
            self.alpha.data = torch.tensor(0.99)
        # if alpha < 0.9:

        if self.get_scalar_inequalities()[0]() < 0 and self.learn_L:
            logger.info("Condition on inputs is not satisfied")
            self.reset_s()
        #

        multiplier_constraints = []
        if self.learn_L:
            L = cp.Variable((self.nz, self.nx))
            for li in L:
                li = li.reshape((1, -1), "C")
                multiplier_constraints.append(
                    cp.bmat(
                        [
                            [np.array([[1 / s**2]]), li],
                            [li.T, P],
                        ]
                    )
                    >> eps * np.eye(self.nx + 1)
                )
        else:
            L = self.L.cpu().detach().numpy()

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
        problem = cp.Problem(cp.Minimize([None]), [F << -eps * np.eye(nF), *multiplier_constraints])
        try:
            problem.solve(solver=cp.MOSEK)
        except Exception:
            return False  # SDP failed due to solver error
        if not problem.status == "optimal":
            return False  # SDP failed to find feasible solution
        # logger.info(f"SDP analysis problem solved: {problem.status}")

        self.P.data = torch.tensor(P.value)
        self.M.data = torch.tensor(M.value)
        if self.learn_L:
            self.L.data = torch.tensor(L.value)
        if learn_B_and_D21:
            self.B.data = torch.tensor(B.value)
            self.D21.data = torch.tensor(D21.value)

        # self.s.data = torch.tensor(np.sqrt(1/s_hat.value).squeeze())

        return True  # SDP successfully found feasible solution

    def analysis_problem_init(self, learn_B_and_D21: bool = False) -> bool:

        eps = 1e-3

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        la = cp.Variable((self.nz, 1))
        M = cp.diag(la)
        A = self.A.cpu().detach().numpy()
        # s_hat = cp.Variable((1,1))
        # B = self.B.cpu().detach().numpy()
        if learn_B_and_D21:
            B = cp.Variable(self.B.shape)
            D21 = cp.Variable(self.D21.shape)
        else:
            B = self.B.cpu().detach().numpy()
            D21 = self.D21.cpu().detach().numpy()
        B2 = self.B2.cpu().detach().numpy()
        C2 = self.C2.cpu().detach().numpy()

        alpha = self.alpha.cpu().detach().numpy()
        # delta = self.delta.cpu().detach().numpy()  # Currently unused
        s = self.s.cpu().detach().numpy()

        # if delta**2 - (1-alpha**2)*s**2 > 0:
        #     s = np.sqrt(delta**2/(1-alpha**2)).squeeze()

        # D21 = self.D21.cpu().detach().numpy()

        multiplier_constraints = []
        if self.learn_L:
            L = cp.Variable((self.nz, self.nx))
            for li in L:
                li = li.reshape((1, -1), "C")
                multiplier_constraints.append(
                    cp.bmat(
                        [
                            [np.array([[1 / s**2]]), li],
                            [li.T, P],
                        ]
                    )
                    >> eps * np.eye(self.nx + 1)
                )
        else:
            L = self.L.cpu().detach().numpy()

        F = cp.bmat(
            [
                [-(alpha**2) * P, np.zeros((self.nx, self.nd)), P @ C2.T + L.T, P @ A.T],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P + L, D21, -2 * M, M @ B2.T],
                [A @ P, B, B2 @ M, -P],
            ]
        )

        init_constraints = [-P + self.max_norm_x0**2 / s**2 * np.eye(self.nx) << 0]

        nF = F.shape[0]
        problem = cp.Problem(
            cp.Minimize([None]),
            [F << -eps * np.eye(nF), *multiplier_constraints, *init_constraints],
        )
        try:
            problem.solve(solver=cp.MOSEK)
        except Exception:
            return False  # SDP failed due to solver error
        if not problem.status == "optimal":
            return False  # SDP failed to find feasible solution
        logger.info(f"SDP analysis problem solved: {problem.status}")

        self.P.data = torch.tensor(P.value)
        self.M.data = torch.tensor(M.value)
        if self.learn_L:
            self.L.data = torch.tensor(L.value)
        if learn_B_and_D21:
            self.B.data = torch.tensor(B.value)
            self.D21.data = torch.tensor(D21.value)

        self.s.data = torch.tensor(s)

        return True  # SDP successfully found feasible solution

    def get_lmis(self):
        lmi_list = []
        """Construct the LMI for stability constraint."""

        def stability_lmi() -> torch.Tensor:
            F = torch_bmat(
                [
                    [
                        -self.alpha**2 * self.P,
                        torch.zeros((self.nx, self.nd)),
                        self.P @ self.C2.T + self.L.T,
                        self.P @ self.A.T,
                    ],
                    [torch.zeros((self.nd, self.nx)), -torch.eye(self.nd), self.D21.T, self.B.T],
                    [self.C2 @ self.P + self.L, self.D21, -2 * self.M, self.M @ self.B2.T],
                    [self.A @ self.P, self.B, self.B2 @ self.M, -self.P],
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

        def input_size_condition() -> torch.Tensor:
            return -(self.delta**2 - (1 - self.alpha**2) * self.s**2) + 1e-3  # small margin

        inequalities.append(input_size_condition)

        def alpha_smaller_one() -> torch.Tensor:
            return 1.0 - self.alpha

        inequalities.append(alpha_smaller_one)

        def alpha_positive() -> torch.Tensor:
            return self.alpha

        inequalities.append(alpha_positive)

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

    def forward(
        self,
        d: torch.Tensor,  # input
        x0: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            d: Input tensor (batch, seq_len, input_size)
            w: Nonlinearity input tensor (batch, seq_len, nw)
            hidden_state: Hidden state (num_layers, batch, hidden_size)

        Returns:
            e_hat: Predicted output (batch, seq_len, output_size)
        """

        B, N, nd = d.shape  # number of batches, length of sequence, input size
        assert self.lure._nd == nd
        if x0 is None:
            x0 = torch.zeros(size=(B, self.nx))
        ds = d.reshape(shape=(B, N, nd, 1))
        es_hat, x = self.lure.forward(x0=x0, d=ds, return_states=return_state)
        # return (
        #     es_hat.reshape(B, N, self.lure._nd),
        #     (x.reshape(B, self.nx),),
        # )
        if return_state:
            return es_hat.reshape(B, N, self.lure._nd), x
        else:
            return es_hat.reshape(B, N, self.lure._nd)

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
        self.alpha.requires_grad = False
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

    def post_process(self, eps: float = 1e-3) -> dict:
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
        alpha = self.alpha.cpu().detach().numpy()
        s_original = self.s.cpu().detach().numpy()

        P_original = self.P.cpu().detach().numpy()
        # L_original = self.L.cpu().detach().numpy() if self.learn_L else None  # Currently unused

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

        # Multiplier constraints: m(i) >= eps for all i
        for i in range(self.nz):
            constraints.append(m[i, 0] >= eps)

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
        constraints.append(F << -eps * np.eye(nF))

        # Locality constraints: [S_hat, li; li', P] >= eps*I for each row of L
        for i in range(self.nz):
            li = L[i, :].reshape((1, -1), order="C")
            locality_lmi = cp.bmat([[S_hat, li], [li.T, P]])
            constraints.append(locality_lmi >> eps * np.eye(self.nx + 1))

        # P positive definite
        constraints.append(P >> eps * np.eye(self.nx))

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

        logger.info(f"✓ SDP solved successfully: {problem.status}")

        # Extract solution
        P_opt = P.value
        L_opt = L.value
        m_opt = m.value
        M_opt = np.diag(m_opt.flatten())

        S_hat_opt = S_hat.value[0, 0] if hasattr(S_hat.value, "__len__") else S_hat.value
        s_opt = np.sqrt(1.0 / S_hat_opt)

        # Verify solution
        F_value = np.block(
            [
                [
                    -(alpha**2) * P_opt,
                    np.zeros((self.nx, self.nd)),
                    P_opt @ C2.T + L_opt.T,
                    P_opt @ A.T,
                ],
                [np.zeros((self.nd, self.nx)), -np.eye(self.nd), D21.T, B.T],
                [C2 @ P_opt + L_opt, D21, -2 * M_opt, M_opt @ B2.T],
                [A @ P_opt, B, B2 @ M_opt, -P_opt],
            ]
        )
        max_eig_F = np.max(np.real(np.linalg.eigvals(F_value)))

        # Update model parameters
        self.P.data = torch.tensor(P_opt)
        self.L.data = torch.tensor(L_opt)
        self.la.data = torch.tensor(np.diag(M_opt))
        self.M = torch.diag(self.la)
        self.s.data = torch.tensor(s_opt)

        # Verify constraints
        constraints_satisfied = self.check_constraints()

        summary = {
            "original": {
                "s": float(s_original),
                "max_eig_P": float(np.max(np.linalg.eigvals(P_original))),
                "min_eig_P": float(np.min(np.linalg.eigvals(P_original))),
                "cond_P": float(np.linalg.cond(P_original)),
            },
            "optimized": {
                "s": float(s_opt),
                "max_eig_P": float(np.max(np.linalg.eigvals(P_opt))),
                "min_eig_P": float(np.min(np.linalg.eigvals(P_opt))),
                "cond_P": float(np.linalg.cond(P_opt)),
                "max_eig_F": float(max_eig_F),
            },
        }

        # Log results
        logger.info("─" * 80)
        logger.info(f"Original s:      {summary['original']['s']:.6f}")
        logger.info(f"Optimized s:     {summary['optimized']['s']:.6f}")
        logger.info(f"Max eig(F):      {max_eig_F:.6e}")
        logger.info(f"Constraints OK:  {constraints_satisfied}")
        logger.info("=" * 80)

        return {
            "success": True,
            "P_opt": P_opt,
            "L_opt": L_opt,
            "m_opt": m_opt,
            "s_opt": s_opt,
            "S_hat_opt": S_hat_opt,
            "max_eig_F": max_eig_F,
            "constraints_satisfied": constraints_satisfied,
            "summary": summary,
        }

    def get_regularization_loss(self, method: Optional[str] = None) -> torch.Tensor:
        """
        Compute custom regularization loss on model parameters.

        Args:
            method: Regularization method ('interior_point' or 'dual').
                   If None, uses self.regularization_method.

        Returns:
            Regularization loss tensor
        """
        if method is None:
            method = self.regularization_method

        if method == "interior_point":
            return self._interior_point_regularization()
        elif method == "dual":
            return self._dual_regularization()
        else:
            raise ValueError(f"Unknown regularization method: {method}")

        # add parameter regularization

    def _interior_point_regularization(self) -> torch.Tensor:
        """
        Interior point method: uses log-det barrier function.
        Requires strictly feasible parameters (all eigenvalues > 0).
        Gradients explode when constraints are violated.

        Returns:
            Regularization loss (sum of negative log-determinants)
        """
        reg_loss = torch.tensor(0.0, device=self.P.device)
        for f_i in self.get_lmis():
            reg_loss += -torch.logdet(f_i())
        for s_i in self.get_scalar_inequalities():
            reg_loss += -torch.log(s_i()).squeeze()

        # Add parametric regularization to encourage non-zero L
        reg_loss += self._parametric_regularization()

        return reg_loss

    def _dual_regularization(self) -> torch.Tensor:
        """
        Dual method: penalizes negative eigenvalues with adaptive penalty coefficient.
        Allows infeasible parameters during training.

        For each LMI F that should be positive definite (F ≻ 0):
        - Compute eigenvalues of F
        - Penalize negative eigenvalues: sum(max(0, -λ)^2)
        - Scale by dual penalty coefficient

        Returns:
            Regularization loss (sum of negative eigenvalue penalties)
        """
        reg_loss = torch.tensor(0.0, device=self.P.device)

        for f_i in self.get_lmis():
            F = f_i()
            # Compute eigenvalues (real part, since F should be symmetric)
            eigenvalues = torch.linalg.eigvalsh(F)  # More efficient for symmetric matrices

            # Penalize negative eigenvalues: sum of squared violations
            # max(0, -λ)^2 = (ReLU(-λ))^2
            negative_eigs = torch.relu(-eigenvalues)
            violation = torch.sum(negative_eigs**2)

            reg_loss += self.dual_penalty * violation

        # Add parametric regularization to encourage non-zero L
        reg_loss += self._parametric_regularization()

        return reg_loss

    def _parametric_regularization(self) -> torch.Tensor:
        """
        Parametric regularization to encourage specific parameter properties.

        Currently implements:
        - Negative Frobenius norm penalty on L to encourage non-zero values
          (penalizes small ||L||, encourages larger magnitude)

        Returns:
            Parametric regularization loss
        """
        param_loss = torch.tensor(0.0, device=self.P.device)

        if self.l_nonzero_weight > 0 and self.learn_L:
            # param_loss = self.l_nonzero_weight * torch.max(torch.tensor(0.0, device=self.P.device), 1 - torch.linalg.norm(self.L)**2)
            param_loss -= self.l_nonzero_weight * torch.norm(self.L, p="fro")

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
