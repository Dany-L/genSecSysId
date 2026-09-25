"""Standard RNN architectures, used as unconstrained baselines.

They follow the calling convention the trainer and evaluator use for every
model, ``model(d, x0, warmup_steps=...) -> (e_hat, (x, w), d)``, so they train
and evaluate through the same pipeline as the CRNN arms. They carry no
certificate: ``w`` is ``None`` and every constraint check passes trivially.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseRNN


class _TorchRNNBaseline(BaseRNN):
    """Shared forward for the ``torch.nn`` recurrent baselines.

    Subclasses register the batch-first ``nn.RNN``/``nn.LSTM``/``nn.GRU`` under
    the attribute named by ``_core_name`` and set ``self.fc`` (hidden -> output
    readout). The attribute names are the historical ones (``rnn``/``lstm``/
    ``gru``) so existing checkpoints keep their state_dict keys.
    """

    _core_name = "rnn"
    fc: nn.Linear

    def forward(
        self,
        d: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        warmup_steps: int = 0,
        hidden_state=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, None], torch.Tensor]:
        """
        Forward pass.

        Args:
            d: Input tensor (batch, seq_len, input_size)
            x0: Physical initial state. Ignored: the hidden state of a black-box
                RNN has no physical meaning, so it starts at zero (or at
                ``hidden_state``) and the washout ``warmup_steps`` absorbs the
                transient, as for the CRNN.
            warmup_steps: Accepted for interface compatibility. The loss slicing
                happens in the trainer/evaluator, not here.
            hidden_state: Initial torch hidden state, ``(num_layers, batch,
                hidden_size)`` (a ``(h, c)`` tuple for the LSTM).

        Returns:
            e_hat: Predicted output (batch, seq_len, output_size)
            (x, w): Last layer's hidden-state sequence (batch, seq_len,
                hidden_size) and ``None`` (there is no nonlinearity channel).
            d: The input, unchanged (no safety filter).
        """
        x, _ = getattr(self, self._core_name)(d, hidden_state)
        e_hat = self.fc(x)
        return e_hat, (x, None), d

    def check_constraints(self) -> bool:
        """Unconstrained model: always feasible."""
        return True


class SimpleRNN(_TorchRNNBaseline):
    """Simple RNN model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "tanh",
    ):
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)

        self.activation = activation

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity=activation,
        )

        self.fc = nn.Linear(hidden_size, output_size)


class LSTM(_TorchRNNBaseline):
    """LSTM model."""

    _core_name = "lstm"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)


class GRU(_TorchRNNBaseline):
    """GRU model."""

    _core_name = "gru"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)
