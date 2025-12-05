"""Base model class for RNN-based system identification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LureSystemClass:
    A: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    D12: torch.Tensor
    C2: torch.Tensor
    D21: torch.Tensor
    D22: torch.Tensor
    Delta: torch.nn.Module


class DznActivation(nn.Module):
    def forward(self, z):
        return z - nn.Hardtanh(min_val=-1.0, max_val=1.0)(z)


class Linear(nn.Module):
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt: torch.Tensor = torch.tensor(0.01),
    ) -> None:
        super().__init__()
        self._nx = A.shape[0]
        self._nd = B.shape[1]
        self._ne = C.shape[0]

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt

    def _init_weights(self) -> None:
        for p in self.parameters():
            torch.nn.init.uniform_(tensor=p, a=-np.sqrt(1 / self._nd), b=np.sqrt(1 / self._nd))

    def state_dynamics(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.A @ x + self.B @ d

    def output_dynamics(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.C @ x + self.D @ d

    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]] = None,
        theta: Optional[np.ndarray] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        n_batch, N, _, _ = d.shape
        x = torch.zeros(size=(n_batch, N + 1, self._nx, 1))
        e_hat = torch.zeros(size=(n_batch, N, self._ne, 1))
        if x0 is not None:
            if isinstance(x0, Tuple):
                x[:, 0, :, :] = x0[0]  # Use first element if tuple
            else:
                x[:, 0, :, :] = x0

        for k in range(N):
            x[:, k + 1, :, :] = self.state_dynamics(x=x[:, k, :, :], d=d[:, k, :, :])
            e_hat[:, k, :, :] = self.output_dynamics(x=x[:, k, :, :], d=d[:, k, :, :])

        return e_hat, (x,)

    def is_stable(self) -> bool:
        """Check if the system is stable by checking the eigenvalues of A."""
        eigenvalues = torch.linalg.eigvals(self.A)
        return bool(torch.all(torch.abs(eigenvalues) < 1.0))


class LureSystem(Linear):
    def __init__(
        self,
        sys: LureSystemClass,
    ) -> None:
        super().__init__(A=sys.A, B=sys.B, C=sys.C, D=sys.D)
        self._nw = sys.B2.shape[1]
        self._nz = sys.C2.shape[0]
        assert self._nw == self._nz
        self.B2 = sys.B2
        self.C2 = sys.C2
        self.D12 = sys.D12
        self.D21 = sys.D21
        self.Delta = sys.Delta  # static nonlinearity

    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]] = None,
        return_states: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]],
    ]:
        n_batch, N, _, _ = d.shape
        e_hat = torch.zeros(size=(n_batch, N, self._ne, 1))

        if return_states:

            w = torch.zeros(size=(n_batch, N, self._nw, 1))
            x = torch.zeros(size=(n_batch, N + 1, self._nx, 1))
            x[:, 0, :, :] = x0.reshape(n_batch, self._nx, 1)

            for k in range(N):
                w[:, k, :, :] = self.Delta(self.C2 @ x[:, k, :, :] + self.D21 @ d[:, k, :, :])
                x[:, k + 1, :, :] = (
                    super().state_dynamics(x=x[:, k, :, :], d=d[:, k, :, :])
                    + self.B2 @ w[:, k, :, :]
                )
                e_hat[:, k, :, :] = (
                    super().output_dynamics(x=x[:, k, :, :], d=d[:, k, :, :])
                    + self.D12 @ w[:, k, :, :]
                )

            return (e_hat, (x, w))

        else:
            x = x0.reshape(n_batch, self._nx, 1)
            for k in range(N):
                w = self.Delta(self.C2 @ x + self.D21 @ d[:, k, :, :])
                x = super().state_dynamics(x=x, d=d[:, k, :, :]) + self.B2 @ w
                e_hat[:, k, :, :] = super().output_dynamics(x=x, d=d[:, k, :, :]) + self.D12 @ w

            return (e_hat, (x, w))


class BaseRNN(nn.Module, ABC):
    """Base class for RNN models."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize the base RNN model.

        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(
        self,
        d: torch.Tensor,  # input
        hidden_state: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Initial hidden state (optional)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        pass

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute custom regularization loss on model parameters.
        This can be overridden by subclasses for specific constraints.

        Returns:
            Regularization loss tensor
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_parameter_dict(self) -> Dict[str, Any]:
        """
        Get dictionary of model parameters for logging/analysis.

        Returns:
            Dictionary of parameter names and values
        """
        param_dict = {}
        for name, param in self.named_parameters():
            param_dict[name] = {
                "shape": list(param.shape),
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "min": param.data.min().item(),
                "max": param.data.max().item(),
            }
        return param_dict

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save model state dict."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "output_size": self.output_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseRNN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        config.update(kwargs)  # Allow override

        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])

        return model
