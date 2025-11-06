"""Standard RNN architectures."""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseRNN


class SimpleRNN(BaseRNN):
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
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # x: (batch, seq_len, input_size)
        rnn_out, hidden = self.rnn(x, hidden)
        # rnn_out: (batch, seq_len, hidden_size)
        
        output = self.fc(rnn_out)
        # output: (batch, seq_len, output_size)
        
        return output


class LSTM(BaseRNN):
    """LSTM model."""
    
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
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # x: (batch, seq_len, input_size)
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out: (batch, seq_len, hidden_size)
        
        output = self.fc(lstm_out)
        # output: (batch, seq_len, output_size)
        
        return output


class GRU(BaseRNN):
    """GRU model."""
    
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
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # x: (batch, seq_len, input_size)
        gru_out, hidden = self.gru(x, hidden)
        # gru_out: (batch, seq_len, hidden_size)
        
        output = self.fc(gru_out)
        # output: (batch, seq_len, output_size)
        
        return output
