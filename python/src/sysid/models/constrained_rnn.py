from typing import Optional
from .base import LureSystem, LureSystemClass, DznActivation
import torch
import torch.nn as nn
from sysid.utils import torch_bmat



class SimpleLure(nn.Module):
    """Simple Lure system model."""
    
    def __init__(
        self,
        nd: int,
        ne: int,
        nx: int,
        nw: int,
        activation: str, # saturation nonlinearity
        custom_params: Optional[dict] = None,
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

        self.P = nn.Parameter(torch.eye(nx)) # Lyapunov matrix
        if custom_params is not None:
            learn_L = custom_params.get("learn_L", True)
            self.regularization_method = custom_params.get("regularization_method", "interior_point")
            self.dual_penalty_init = custom_params.get("dual_penalty_init", 1.0)
            self.dual_penalty_growth = custom_params.get("dual_penalty_growth", 1.1)
            self.dual_penalty_shrink = custom_params.get("dual_penalty_shrink", 0.9)
        else:
            learn_L = True
            self.regularization_method = "interior_point"
            self.dual_penalty_init = 1.0
            self.dual_penalty_growth = 1.1
            self.dual_penalty_shrink = 0.9

        # Dual penalty coefficient (not a parameter, updated manually)
        self.register_buffer('dual_penalty', torch.tensor(self.dual_penalty_init))

        if learn_L:
            self.L = nn.Parameter(torch.zeros((nz, nx)))  # Coupling matrix
        else:
            self.L = torch.zeros((nz, nx))  # Coupling matrix, not learnable
            
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

        self.s = torch.tensor(1.0)
        self.alpha = torch.tensor(.99)
        if activation == "sat":
            Delta = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        elif activation == "dzn":
            Delta = DznActivation()
        elif activation == "tanh":
            Delta = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type: {activation}")

        self.lure = LureSystem(LureSystemClass(
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
        ))

        self.initialize_parameters()
        self.check_constraints()

    def initialize_parameters(self):
        """Initialize parameters with small random values."""
        min_val, max_val = -1e-1, 1e-1
        for name,param in self.named_parameters():
            if not name in ["P", "M", "L"]:
                nn.init.uniform_(param, a=min_val, b=max_val)

    def get_lmis(self):
        lmi_list = []
        """Construct the LMI for stability constraint."""
        def stability_lmi()-> torch.Tensor:
            F = torch_bmat([
                [-self.alpha**2 * self.P, torch.zeros((self.nx, self.nd)), self.P @ self.C2.T + self.L.T, self.P @ self.A.T],
                [torch.zeros((self.nd, self.nx)), -torch.eye(self.nd), self.D21.T, self.B.T],
                [self.C2 @ self.P + self.L, self.D21, -2 * self.M, self.M @ self.B2.T],
                [self.A @ self.P, self.B, self.B2 @ self.M, -self.P]
            ])
            return -0.5*(F + F.T) # to ensure symmetry
        lmi_list.append(stability_lmi)
        
        """Construct the LMIs for locality constraints."""
        for l_i in self.L:
            l_i = l_i.reshape(1,-1)
            def locality_lmi_i(l_i=l_i) -> torch.Tensor:
                return torch_bmat([
                [(1/self.s**2).reshape(1,1), l_i],
                [l_i.T, torch.eye(self.nx)]
                ])
            lmi_list.append(locality_lmi_i)

        return lmi_list

    def check_constraints(self) -> bool:
        """Check if the Lure system constraints are satisfied."""
        with torch.no_grad():
            for lmi in self.get_lmis():
                try:
                    torch.linalg.cholesky(lmi())
                except RuntimeError:
                    return False
        return True
    
    
    def forward(
        self,
        d: torch.Tensor,  # input
        x0: Optional[torch.Tensor] = None,
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
        es_hat, x = self.lure.forward(x0=x0, d=ds)
        # return (
        #     es_hat.reshape(B, N, self.lure._nd),
        #     (x.reshape(B, self.nx),),
        # )
        return es_hat.reshape(B, N, self.lure._nd)


    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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
            violation = torch.sum(negative_eigs ** 2)
            
            reg_loss += self.dual_penalty * violation
        
        return reg_loss
    
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