"""Base model class for RNN-based system identification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

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

    def state_dynamics(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.A @ x + self.B @ d

    def output_dynamics(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.C @ x + self.D @ d


class LureSystem(Linear):
    """Lure dynamics x⁺ = Ax + Bd + B₂Δ(C₂x + D₂₁d), without safety filtering.

    Forward always returns ``(e_hat, (x, w), d)`` with the full state and
    nonlinearity trajectories — no flags, no shape variants.
    """

    def __init__(self, sys: LureSystemClass) -> None:
        super().__init__(A=sys.A, B=sys.B, C=sys.C, D=sys.D)
        self._nw = sys.B2.shape[1]
        self._nz = sys.C2.shape[0]
        assert self._nw == self._nz
        self.B2 = sys.B2
        self.C2 = sys.C2
        self.D12 = sys.D12
        self.D21 = sys.D21
        self.Delta = sys.Delta  # static nonlinearity

    def input_filter(
        self,
        X: torch.Tensor,
        s: torch.Tensor,
        alpha: torch.Tensor,
        x_k: torch.Tensor,
        d_k: torch.Tensor,
    ) -> torch.Tensor:
        # The safety projection is value-only: gradients must not flow through d_max.
        # Otherwise sqrt(...) produces NaN gradients near/outside the safe-set boundary,
        # and the model can't recover. The optimizer still gets the right signal because
        # clamping increases prediction error, which the loss already penalizes.
        eps = 0
        with torch.no_grad():
            X_x_squared = torch.stack([x_k_i.T @ X @ x_k_i for x_k_i in x_k])
            radicand = torch.clamp(s**2 - alpha**2 * X_x_squared, min=0.0)
            d_max = torch.sqrt(radicand) - eps

        return torch.clamp(d_k, min=-d_max, max=d_max)

    def _rollout(
        self,
        d: torch.Tensor,
        x0: torch.Tensor,
        clamp_step=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Step the Lure dynamics over the full sequence.

        ``clamp_step(k, x_k, d_k) -> d_k`` lets subclasses inject a per-step
        input filter; the default is identity (no clamping).
        """
        n_batch, N, _, _ = d.shape
        x_k = x0.reshape(n_batch, self._nx, 1)

        e_hat_list = []
        w_list = []
        x_list = [x_k]
        d_list = []

        for k in range(N):
            d_k = d[:, k, :, :]
            d_k_safe = clamp_step(k, x_k, d_k) if clamp_step is not None else d_k

            w_k = self.Delta(self.C2 @ x_k + self.D21 @ d_k_safe)
            e_hat_k = super().output_dynamics(x=x_k, d=d_k_safe) + self.D12 @ w_k
            x_k_1 = super().state_dynamics(x=x_k, d=d_k_safe) + self.B2 @ w_k

            e_hat_list.append(e_hat_k)
            w_list.append(w_k)
            x_list.append(x_k_1)
            d_list.append(d_k_safe)

            x_k = x_k_1

        e_hat = torch.stack(e_hat_list, dim=1)   # (n_batch, N, ne, 1)
        x_tensor = torch.stack(x_list, dim=1)     # (n_batch, N+1, nx, 1)
        w_tensor = torch.stack(w_list, dim=1)     # (n_batch, N, nw, 1)
        d_tensor = torch.stack(d_list, dim=1)     # (n_batch, N, nd, 1)

        return e_hat, (x_tensor, w_tensor), d_tensor

    def forward(
        self,
        d: torch.Tensor,
        x0: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self._rollout(d, x0, clamp_step=None)


class LureSystemSafe(LureSystem):
    """LureSystem with the safety input filter wired in.

    Forward requires ``X``, ``s``, ``alpha`` (no runtime ``ValueError`` — the
    arguments are just non-optional). Returns the same 3-tuple as ``LureSystem``,
    with ``d`` containing the *filtered* inputs that were actually applied.
    """

    def forward(
        self,
        d: torch.Tensor,
        x0: torch.Tensor,
        X: torch.Tensor,
        s: torch.Tensor,
        alpha: torch.Tensor,
        warmup_steps: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        def clamp_step(k: int, x_k: torch.Tensor, d_k: torch.Tensor) -> torch.Tensor:
            # if k < warmup_steps:
            #     return d_k
            return self.input_filter(X, s, alpha, x_k, d_k)

        return self._rollout(d, x0, clamp_step=clamp_step)


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

    def get_regularization_input(self, inputs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute input constraint regularization loss.
        This can be overridden by subclasses for specific constraints.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_size)
            states: State tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Regularization loss tensor
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)

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
