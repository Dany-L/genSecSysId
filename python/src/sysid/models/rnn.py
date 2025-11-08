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
        d: torch.Tensor,  # input
        hidden_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            d: Input tensor (batch, seq_len, input_size)
            hidden_state: Hidden state (num_layers, batch, hidden_size)
            
        Returns:
            e_hat: Predicted output (batch, seq_len, output_size)
        """
        # d: (batch, seq_len, input_size)
        x, hidden_state = self.rnn(d, hidden_state)  # x: hidden state
        # x: (batch, seq_len, hidden_size)
        
        e_hat = self.fc(x)  # e_hat: predicted output
        # e_hat: (batch, seq_len, output_size)
        
        return e_hat


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
        d: torch.Tensor,  # input
        x0: Optional[tuple] = None,
        hidden_state: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            d: Input tensor (batch, seq_len, input_size)
            hidden_state: Hidden state tuple (h, c) where each is (num_layers, batch, hidden_size)
            
        Returns:
            e_hat: Predicted output (batch, seq_len, output_size)
        """
        # d: (batch, seq_len, input_size)
        x, hidden_state = self.lstm(d, hidden_state)  # x: hidden state
        # x: (batch, seq_len, hidden_size)
        
        e_hat = self.fc(x)  # e_hat: predicted output
        # e_hat: (batch, seq_len, output_size)
        
        return e_hat
    
    def check_constraints(self) -> bool:
        """Check if the LSTM constraints are satisfied."""
        return True  # No constraints for standard LSTM


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
        d: torch.Tensor,  # input
        hidden_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            d: Input tensor (batch, seq_len, input_size)
            hidden_state: Hidden state (num_layers, batch, hidden_size)
            
        Returns:
            e_hat: Predicted output (batch, seq_len, output_size)
        """
        # d: (batch, seq_len, input_size)
        x, hidden_state = self.gru(d, hidden_state)  # x: hidden state
        # x: (batch, seq_len, hidden_size)
        
        e_hat = self.fc(x)  # e_hat: predicted output
        # e_hat: (batch, seq_len, output_size)
        
        return e_hat
