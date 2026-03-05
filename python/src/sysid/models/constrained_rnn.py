import logging
import os
from pathlib import Path
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
        
        # Check if state padding is enabled (default: True)
        pad_state = custom_params.get("pad_state", True) if custom_params is not None else True
        
        # Store original dataset state dimension
        self.nx_data = nx
        # Optionally pad state dimension to match nz
        self.nx = nz if pad_state else nx
        self.nd = nd
        self.ne = ne
        self.nw = nw
        self.nz = nz
        self.pad_state = pad_state

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

        self.A = nn.Parameter(torch.zeros(self.nx, self.nx))
        self.B = nn.Parameter(torch.zeros(self.nx, nd))
        self.B2 = nn.Parameter(torch.zeros(self.nx, nw))

        self.C = nn.Parameter(torch.zeros(ne, self.nx))
        self.D = nn.Parameter(torch.zeros(ne, nd))
        self.D12 = nn.Parameter(torch.zeros(ne, nw))

        self.C2 = nn.Parameter(torch.zeros(nz, self.nx))
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

    def initialize_parameters(self, train_inputs, train_states, train_outputs, n_restarts: int = 5, data_dir: Optional[str] = None):
        """
        Initialize parameters, preferring N4SID parameters if available.

        If `data_dir` is provided and contains a file ``n4sid_params.mat``, the
        A, B, C, D matrices are loaded from that file (MATLAB idss format).
        If the N4SID state order is smaller than the configured ``nx``, matrices
        are zero-padded in the state dimension (top-left block for A, top rows
        for B, left columns for C). This allows warm-starting larger models
        from a lower-order N4SID initialization. B2 is kept zero so the model
        starts fully linear.
        The SDP is then run (keeping A and B fixed) to obtain feasible P, L.

        If no ``n4sid_params.mat`` is found, the method falls back to the
        random Echo State Network (ESN) reservoir search:
        1. Try `n_restarts` random reservoirs (A, C2) — cheap (no SDP).
           For each, simulate and fit C, D, D12 via least squares. Keep the one
           with the lowest training MSE.
        2. Run the SDP exactly once on the best reservoir to obtain feasible
           B, D21, P, L.
        3. Re-simulate with the SDP-updated B, D21 and refit C, D, D12.
        """
        # ── 0. Compute s ────────────────────────────────────────────────────────
        self.s.data = torch.sqrt(self.delta**2 / (1 - self.alpha**2)).squeeze()

        Bs, N, _ = train_inputs.shape  # batch_size, seq_len, nd

        # ── Optional: load N4SID parameters from mat file ───────────────────────
        n4sid_loaded = False
        if data_dir is not None:
            from scipy.io import loadmat
            mat_path = Path(os.path.expanduser(data_dir)) / "n4sid_params.mat"
            if mat_path.exists():
                logger.info(f"Found n4sid_params.mat at {mat_path} — loading N4SID parameters")
                try:
                    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

                    # Support both direct keys and a nested idss/struct object
                    def _extract(key):
                        if key in mat:
                            return np.atleast_2d(np.array(mat[key], dtype=np.float64))
                        # Search inside any top-level struct
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
                            logger.info(
                                f"Transposing N4SID {name} from shape {mat.shape} to {mat.T.shape}"
                            )
                            return mat.T
                        return mat

                    # Allow common MATLAB orientation differences
                    B_n4 = _maybe_transpose_to_match(B_n4, (A_n4.shape[0], self.nd), "B")
                    C_n4 = _maybe_transpose_to_match(C_n4, (self.ne, A_n4.shape[0]), "C")
                    D_n4 = _maybe_transpose_to_match(D_n4, (self.ne, self.nd), "D")

                    # Validate N4SID state dimension and allow padding to larger configured nx
                    if A_n4.shape[0] != A_n4.shape[1]:
                        raise ValueError(
                            f"N4SID A must be square, got shape {A_n4.shape}."
                        )

                    n4_nx = A_n4.shape[0]
                    if n4_nx > self.nx:
                        raise ValueError(
                            f"N4SID state dimension ({n4_nx}) is larger than configured model nx ({self.nx})."
                        )

                    if B_n4.shape[1] != self.nd:
                        raise ValueError(
                            f"N4SID B has incompatible input dimension: shape {B_n4.shape}, expected second dim {self.nd}."
                        )
                    if B_n4.shape[0] > self.nx:
                        raise ValueError(
                            f"N4SID B has too many state rows: shape {B_n4.shape}, configured nx={self.nx}."
                        )

                    if C_n4.shape[0] != self.ne:
                        raise ValueError(
                            f"N4SID C has incompatible output dimension: shape {C_n4.shape}, expected first dim {self.ne}."
                        )
                    if C_n4.shape[1] > self.nx:
                        raise ValueError(
                            f"N4SID C has too many state columns: shape {C_n4.shape}, configured nx={self.nx}."
                        )

                    if D_n4.shape != (self.ne, self.nd):
                        raise ValueError(
                            f"N4SID D has shape {D_n4.shape}, expected ({self.ne}, {self.nd})."
                        )

                    # Pad state dimensions when N4SID order is smaller than configured nx
                    A_init = np.zeros((self.nx, self.nx), dtype=np.float64)
                    A_init[:n4_nx, :n4_nx] = A_n4

                    B_init = np.zeros((self.nx, self.nd), dtype=np.float64)
                    B_init[:B_n4.shape[0], :] = B_n4

                    C_init = np.zeros((self.ne, self.nx), dtype=np.float64)
                    C_init[:, :C_n4.shape[1]] = C_n4

                    if n4_nx < self.nx:
                        logger.info(
                            f"Padding N4SID initialization from state dim {n4_nx} to model nx {self.nx} "
                            "(A top-left block, B top rows, C left columns)."
                        )

                    # Set parameters from N4SID
                    self.A.data  = torch.tensor(A_init)
                    self.B.data  = torch.tensor(B_init)
                    self.C.data  = torch.tensor(C_init)
                    self.D.data  = torch.tensor(D_n4)
                    self.B2.data = torch.zeros_like(self.B2)   # keep linear
                    # self.B2.data = torch.tensor(np.random.randn(self.nx, self.nw)) * 5*1e-3 # small random B2 to break symmetry

                    # Random C2 scaled to avoid saturation (needed for constraint structure)
                    self.C2.data = torch.tensor(
                        # np.random.randn(self.nz, self.nx) / np.sqrt(self.nx)
                        np.random.randn(self.nz, self.nx)
                    )

                    self.D12.data = torch.tensor(
                        np.random.randn(self.ne, self.nw)
                    )
                    logger.info(
                        f"N4SID matrices loaded. "
                        f"||A||={np.linalg.norm(A_init):.4f}, "
                        f"||B||={np.linalg.norm(B_init):.4f}, "
                        f"||C||={np.linalg.norm(C_init):.4f}, "
                        f"||D||={np.linalg.norm(D_n4):.4f}"
                    )

                    # Run SDP keeping A and B fixed
                    constraints_ok = self.check_constraints()
                    logger.info(f"N4SID initialization complete. Constraints satisfied: {constraints_ok}")
                    n4sid_loaded = True

                    self.analysis_problem_init(learn_B=True, learn_D21=True)

                except (KeyError, ValueError) as e:
                    logger.warning(f"Could not load N4SID parameters: {e} — falling back to ESN init")
            else:
                logger.info(f"No n4sid_params.mat found in {data_dir} — using ESN initialization")

        # Prepare initial states and input tensor (always needed for final simulation / plotting)
        x0s = torch.zeros(Bs, self.nx, 1)
        if train_states is not None:
            x0s_data = torch.tensor(train_states[:, 0, :].reshape(Bs, self.nx_data, 1))
            x0s[:, :self.nx_data, :] = x0s_data
        else:
            x0s = torch.randn(Bs, self.nx, 1) * self.max_norm_x0

        ds = torch.tensor(train_inputs.reshape(Bs, N, self.nd, 1))

        if not n4sid_loaded:
            # ── ESN reservoir search (no SDP) ────────────────────────────────────
            y_target  = torch.tensor(train_outputs.reshape(Bs * N, self.ne))
            alpha_val = float(self.alpha.item())
            delta_val = float(self.delta.item())

            best_mse = float("inf")
            best_reservoir = None  # stores (A, B, B2, C2, D21) as numpy arrays

            for trial in range(n_restarts):
                # Random A with spectral radius = alpha_val (rich, diverse dynamics)
                A_rand = np.random.randn(self.nx, self.nx)
                rho = np.max(np.abs(np.linalg.eigvals(A_rand)))
                A_rand = (alpha_val / max(rho, 1e-8)) * A_rand

                # B2 = 0: keep initialization linear (nonlinearity feedback optimized during training)
                B2_rand = np.zeros((self.nx, self.nw))

                # Random B: scaled by input amplitude / sqrt(nx)
                B_rand = np.random.randn(self.nx, self.nd) * (delta_val / max(np.sqrt(self.nx), 1.0))

                # Random C2 scaled to avoid saturation (pre-activation ~ O(1))
                C2_rand = np.random.randn(self.nz, self.nx) / np.sqrt(self.nx)

                # Small D21
                D21_rand = np.random.randn(self.nz, self.nd) * 0.01

                # Temporarily set params
                self.A.data   = torch.tensor(A_rand)
                self.B.data   = torch.tensor(B_rand)
                self.B2.data  = torch.tensor(B2_rand)
                self.C2.data  = torch.tensor(C2_rand)
                self.D21.data = torch.tensor(D21_rand)

                # Simulate
                with torch.no_grad():
                    _, (xs, ws) = self.lure.forward(x0=x0s, d=ds, return_states=True)

                # Least squares: fit C, D, D12
                x_flat = xs[:, :N, :, :].squeeze(-1).reshape(Bs * N, self.nx)
                w_flat = ws.squeeze(-1).reshape(Bs * N, self.nw)
                u_flat = ds.squeeze(-1).reshape(Bs * N, self.nd)
                regr   = torch.cat([x_flat, u_flat, w_flat], dim=1)
                sol    = torch.linalg.lstsq(regr, y_target).solution
                y_hat  = regr @ sol
                mse    = float(torch.mean((y_hat - y_target) ** 2))

                logger.info(f"  ESN init trial {trial + 1}/{n_restarts}: train MSE = {mse:.6e}")

                if mse < best_mse:
                    best_mse = mse
                    best_reservoir = dict(
                        A=A_rand.copy(), B=B_rand.copy(), B2=B2_rand.copy(),
                        C2=C2_rand.copy(), D21=D21_rand.copy(),
                    )

            logger.info(f"Best reservoir MSE: {best_mse:.6e}. Running SDP for feasibility...")

            # ── SDP: get feasible B, D21, P, L for best reservoir ────────────────
            self.A.data   = torch.tensor(best_reservoir["A"])
            self.B2.data  = torch.tensor(best_reservoir["B2"])
            self.C2.data  = torch.tensor(best_reservoir["C2"])
            self.B.data   = torch.tensor(best_reservoir["B"])
            self.D21.data = torch.tensor(best_reservoir["D21"])

            self.analysis_problem_init(learn_B=True, learn_D21=True)

            # ── Re-simulate and refit C, D, D12 with SDP-updated B, D21 ─────────
            with torch.no_grad():
                _, (xs, ws) = self.lure.forward(x0=x0s, d=ds, return_states=True)

                x_flat = xs[:, :N, :, :].squeeze(-1).reshape(Bs * N, self.nx)
                w_flat = ws.squeeze(-1).reshape(Bs * N, self.nw)
                u_flat = ds.squeeze(-1).reshape(Bs * N, self.nd)
                y_flat = train_outputs.reshape(Bs * N, self.ne)

                regression_matrix = torch.cat([x_flat, u_flat, w_flat], dim=1)
                solution = torch.linalg.lstsq(
                    regression_matrix, torch.tensor(y_flat)
                ).solution

                self.C.data   = solution[: self.nx, :].T
                self.D.data   = solution[self.nx : self.nx + self.nd, :].T
                self.D12.data = solution[self.nx + self.nd :, :].T

                final_mse = float(
                    torch.mean(
                        (regression_matrix @ solution - torch.tensor(y_flat)) ** 2
                    )
                )
                logger.info(f"Initialization complete. Final train MSE (with SDP B,D21): {final_mse:.6e}")
                logger.info(f"  C norm: {torch.norm(self.C).item():.6f}")
                logger.info(f"  D norm: {torch.norm(self.D).item():.6f}")
                logger.info(f"  D12 norm: {torch.norm(self.D12).item():.6f}")
        else:
            # N4SID path: just simulate to get xs for visualization
            with torch.no_grad():
                _, (xs, ws) = self.lure.forward(x0=x0s, d=ds, return_states=True)

        X = np.linalg.inv(self.P.cpu().detach().numpy())
        H = self.L.cpu().detach().numpy() @ X


        if self.nx ==2 :
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
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            

        self.check_constraints()

        return fig

    def analysis_problem(self, learn_B_and_D21: bool = False) -> bool:

        eps = 1e-6

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

    def analysis_problem_init(self, learn_B: bool= False, learn_D21: bool = False) -> bool:

        eps = 1e-6

        P = cp.Variable((self.nx, self.nx), symmetric=True)
        la = cp.Variable((self.nz, 1))
        M = cp.diag(la)
        A = self.A.cpu().detach().numpy()
        # s_hat = cp.Variable((1,1))
        # B = self.B.cpu().detach().numpy()
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

        alpha = self.alpha.cpu().detach().numpy()
        # delta = self.delta.cpu().detach().numpy()  # Currently unused
        s = self.s.cpu().detach().numpy()

        # if delta**2 - (1-alpha**2)*s**2 > 0:
        #     s = np.sqrt(delta**2/(1-alpha**2)).squeeze()

        # D21 = self.D21.cpu().detach().numpy()
        # s_tilde = cp.Variable((1,1))

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
                # multiplier_constraints.append(
                #     cp.bmat(
                #         [
                #             [s_tilde, li],
                #             [li.T, P],
                #         ]
                #     )
                #     >> eps * np.eye(self.nx + 1)
                # )
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
            # cp.Minimize(s_tilde),
            [
                F << -eps * np.eye(nF), 
                *multiplier_constraints, 
                # *init_constraints
            ],
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
        if learn_B:
            self.B.data = torch.tensor(B.value)
        if learn_D21:
            self.D21.data = torch.tensor(D21.value)

        # self.s.data = torch.tensor(s)
        # self.s.data = torch.tensor(np.sqrt(1 / s_tilde.value).squeeze())
        # self.delta = torch.tensor(np.sqrt((1 - alpha**2) * s**2).squeeze())

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
            x0: Initial state (batch, nx_data) or (batch, nx, 1). If nx_data dimension,
                will be padded with zeros to nx dimension.

        Returns:
            e_hat: Predicted output (batch, seq_len, output_size)
        """

        B, N, nd = d.shape  # number of batches, length of sequence, input size
        assert self.lure._nd == nd
        if x0 is None:
            x0 = torch.zeros(size=(B, self.nx, 1))
            # x0 = torch.randn((B, self.nx, 1)) * 2 * self.max_norm_x0 - self.max_norm_x0
            # x0 = torch.random.uniform(-self.max_norm_x0, self.max_norm_x0, size=(B, self.nx, 1))
        else:
            # Handle padding if pad_state is enabled and x0 comes from dataset (nx_data dimension)
            if self.pad_state and x0.shape[1] == self.nx_data:
                x0_padded = torch.zeros(B, self.nx, 1, device=x0.device, dtype=x0.dtype)
                if x0.ndim == 2:
                    x0_padded[:, :self.nx_data, 0] = x0
                else:  # Already has shape (B, nx_data, 1)
                    x0_padded[:, :self.nx_data, :] = x0
                x0 = x0_padded
        ds = d.reshape(shape=(B, N, nd, 1))
        es_hat, x = self.lure.forward(x0=x0, d=ds, return_states=return_state)
        # return (
        #     es_hat.reshape(B, N, self.lure._nd),
        #     (x.reshape(B, self.nx),),
        # )
        if return_state:
            return es_hat.reshape(B, N, self.lure._ne), x
        else:
            return es_hat.reshape(B, N, self.lure._ne)

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


        # norm H
        H = L_opt @ np.linalg.inv(P_opt)
        norm_H = np.linalg.norm(H, ord=2)
        logger.info(f"Norm of H = {norm_H:.6f}")

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
                "norm_P": float(np.linalg.norm(P_original, ord="fro")),
                "norm_L": float(np.linalg.norm(L_original, ord="fro")) if L_original is not None else 0.0,
                "norm_H": float(np.linalg.norm(H_original, ord="fro")) if H_original is not None else 0.0,
            },
            "optimized": {
                "s": float(s_opt),
                "max_eig_P": float(np.max(np.linalg.eigvals(P_opt))),
                "min_eig_P": float(np.min(np.linalg.eigvals(P_opt))),
                "norm_P": float(np.linalg.norm(P_opt, ord="fro")),
                "max_eig_F": float(max_eig_F),
                "norm_H": float(norm_H),
                "norm_L": float(np.linalg.norm(L_opt, ord="fro")),
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
            if return_components:
                return self._interior_point_regularization(return_components=True)
            else:
                return self._interior_point_regularization()
        elif method == "dual":
            if return_components:
                return self._dual_regularization(return_components=True)
            else:
                return self._dual_regularization()
        else:
            raise ValueError(f"Unknown regularization method: {method}")

        # add parameter regularization

    def _interior_point_regularization(self, return_components: bool = False):
        """
        Interior point method: uses log-det barrier function.
        Requires strictly feasible parameters (all eigenvalues > 0).
        Gradients explode when constraints are violated.

        Args:
            return_components: If True, returns dict with breakdown

        Returns:
            If return_components=False: Regularization loss (sum of negative log-determinants)
            If return_components=True: Dict with 'total', 'feasibility', 'parametric'
        """
        feasibility_loss = torch.tensor(0.0, device=self.P.device)
        for f_i in self.get_lmis():
            feasibility_loss += -torch.logdet(f_i())
        for s_i in self.get_scalar_inequalities():
            feasibility_loss += -torch.log(s_i()).squeeze()

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
