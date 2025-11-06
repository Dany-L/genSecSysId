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
        self.L = nn.Parameter(torch.zeros(nz, nx)) # Locality condition
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
        self.alpha = torch.tensor(.9)
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

        self.check_constraints()

    def initialize_parameters(self):
        """Initialize parameters with small random values."""
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)

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
            return -F
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
        hidden_state: Optional[torch.Tensor] = None,
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
        if hidden_state is None:
            x0 = torch.zeros(size=(B, self.nx))
        else:
            x0 = x0[0]
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
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute custom regularization loss on model parameters.
        This can be overridden by subclasses for specific constraints.
        
        Returns:
            Regularization loss tensor
        """
        reg_loss = torch.tensor(0.0)
        for f_i in self.get_lmis():
            reg_loss += -torch.logdet(f_i())
        return reg_loss