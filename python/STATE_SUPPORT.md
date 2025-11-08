# Initial State Support

## Overview

The data loading pipeline now supports optional **initial state** information that can be passed through the entire training and evaluation workflow. Each sequence receives only the **initial state** (state at the first timestep), not the entire state trajectory. This is particularly useful for models that need initial conditions (e.g., LMI-constrained RNNs that use initial states for stability constraints).

## Features

- **Optional by default**: Initial states are not required - the system works with or without them
- **Full pipeline support**: Initial states flow through dataset → dataloader → trainer → evaluator
- **Backward compatible**: Existing code without states continues to work
- **Proper handling**: Custom collate function handles None values correctly
- **Initial state only**: Returns state at first timestep of sequence, not full state trajectory

## Usage

### 1. Data Loading with States

#### From CSV Folders (Recommended)

```python
from sysid.data import load_split_data

# Load data with state columns
train_data, val_data, test_data = load_split_data(
    data_dir="data/prepared",
    input_col=[0, 1],        # Input columns
    output_col=[2],          # Output columns
    state_col=[3, 4, 5],     # State columns (optional)
    normalize=True
)

# Unpack data
train_inputs, train_outputs, train_states, normalizer = train_data
val_inputs, val_outputs, val_states, _ = val_data
test_inputs, test_outputs, test_states, _ = test_data
```

#### Manual Data Loading

```python
from sysid.data import create_dataloaders
import numpy as np

# Your data arrays
train_inputs = np.random.randn(100, 50, 2)   # (samples, time, inputs)
train_outputs = np.random.randn(100, 50, 1)  # (samples, time, outputs)
train_states = np.random.randn(100, 50, 3)   # (samples, time, states)

# Create loaders with states
train_loader, val_loader, test_loader, normalizer = create_dataloaders(
    train_inputs, train_outputs,
    val_inputs, val_outputs,
    test_inputs, test_outputs,
    train_states=train_states,  # Optional
    val_states=val_states,      # Optional
    test_states=test_states,    # Optional
    batch_size=32,
    normalize=True
)
```

### 2. Iterating Through Data Loaders

All data loaders now return 3-tuples: `(inputs, outputs, initial_states)`

The initial state is the state at the **first timestep** of each sequence.

```python
# Training loop
for batch in train_loader:
    # Unpack the batch (always 3 elements)
    if len(batch) == 3:
        inputs, outputs, initial_states = batch
    else:  # Backward compatibility
        inputs, outputs = batch
        initial_states = None
    
    # inputs:  shape (batch_size, seq_length, n_inputs)
    # outputs: shape (batch_size, seq_length, n_outputs)
    # initial_states: shape (batch_size, n_states) <- ONLY initial state!
    
    # Use inputs and outputs as before
    # initial_states will be a tensor if provided, None otherwise
    if initial_states is not None:
        # Use initial state for your model (e.g., as initial condition)
        predictions = model(inputs, initial_state=initial_states)
```

### 3. Without States (Default Behavior)

```python
from sysid.data import create_dataloaders

# Create loaders WITHOUT states (initial_states will be None)
train_loader, val_loader, test_loader, normalizer = create_dataloaders(
    train_inputs, train_outputs,
    val_inputs, val_outputs,
    batch_size=32,
    normalize=True
    # train_states, val_states, test_states are omitted
)

# When iterating, initial_states will be None
for inputs, outputs, initial_states in train_loader:
    assert initial_states is None  # True when no states provided
    # Process inputs and outputs normally
```

## Implementation Details

### Dataset

`TimeSeriesDataset` now accepts an optional `states` parameter and returns only the **initial state**:

```python
from sysid.data import TimeSeriesDataset

dataset = TimeSeriesDataset(
    inputs,   # shape: (n_samples, seq_length, n_inputs)
    outputs,  # shape: (n_samples, seq_length, n_outputs)
    states=None,  # shape: (n_samples, seq_length, n_states) - FULL trajectory
    sequence_length=50
)

# __getitem__ returns (input_seq, output_seq, initial_state)
# input_seq:     shape (seq_length, n_inputs)
# output_seq:    shape (seq_length, n_outputs)  
# initial_state: shape (n_states,) <- ONLY first timestep!
```

### Custom Collate Function

A custom collate function (`collate_with_optional_states`) handles batching when initial states might be None:

```python
from sysid.data import collate_with_optional_states
from torch.utils.data import DataLoader

# Used automatically by create_dataloaders
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_with_optional_states  # Handles None initial states
)

# Returns: (batched_inputs, batched_outputs, batched_initial_states or None)
# batched_initial_states shape: (batch_size, n_states) if provided
```

### Trainer Integration

The `Trainer` class automatically handles batches with or without initial states:

```python
# Training loop (simplified)
def train_epoch(self):
    for batch in self.train_loader:
        # Generic unpacking handles both 2-tuple and 3-tuple
        if len(batch) == 3:
            d, e, x0 = batch  # x0: initial states (batch_size, n_states) or None
        else:
            d, e = batch
            x0 = None
        
        # d:  inputs  (batch_size, seq_length, n_inputs)
        # e:  outputs (batch_size, seq_length, n_outputs)
        # x0: initial states (batch_size, n_states) <- ONLY initial state!
```

### Evaluator Integration

The `Evaluator` saves initial states if they were provided:

```python
# Initial states are collected during evaluation
# Saved to output_dir/initial_states.npy if present
# Results dict includes 'states_shape' if available
# Shape: (n_samples, n_states) - only initial states, not trajectories
```

## Data Format

### Initial State Array Shape

Initial states should be provided as full trajectories, but only the **first timestep** will be extracted and used:

- **Input Shape**: `(n_samples, sequence_length, n_states)` - full state trajectories
- **Output Shape**: `(batch_size, n_states)` - only initial states (first timestep)
- **Type**: NumPy array or PyTorch tensor
- **Example**: 
  - Input: 100 samples, 50 time steps, 3 states → `(100, 50, 3)`
  - Dataset returns: initial state only → `(3,)` per sample
  - Batched: `(batch_size, 3)`

### CSV Format

When using `load_split_data` with state columns:

```
input1, input2, output, state1, state2, state3
0.1,    0.2,    0.5,   1.0,    2.0,    3.0
0.2,    0.3,    0.6,   1.1,    2.1,    3.1
...
```

Configuration:
```python
input_col=[0, 1]      # columns 0, 1
output_col=[2]        # column 2
state_col=[3, 4, 5]   # columns 3, 4, 5
```

## Testing

Run the built-in test to verify initial state support:

```bash
cd python
python3 -c "
from sysid.data import create_dataloaders
import numpy as np

# Test with initial states
train_inputs = np.random.randn(10, 50, 2)
train_outputs = np.random.randn(10, 50, 1)
train_states = np.random.randn(10, 50, 3)  # Full trajectories
val_inputs = np.random.randn(5, 50, 2)
val_outputs = np.random.randn(5, 50, 1)
val_states = np.random.randn(5, 50, 3)

train_loader, val_loader, _, _ = create_dataloaders(
    train_inputs, train_outputs,
    val_inputs, val_outputs,
    train_states=train_states,
    val_states=val_states,
    batch_size=3,
    normalize=False
)

for d, e, x0 in train_loader:
    print(f'Inputs: {d.shape}, Outputs: {e.shape}, Initial States: {x0.shape}')
    # Expected: Inputs: (3, 50, 2), Outputs: (3, 50, 1), Initial States: (3, 3)
    break

# Test without states
train_loader2, _, _, _ = create_dataloaders(
    train_inputs, train_outputs,
    val_inputs, val_outputs,
    batch_size=3,
    normalize=False
)

for d, e, x0 in train_loader2:
    print(f'Inputs: {d.shape}, Outputs: {e.shape}, Initial States: {x0}')
    # Expected: Inputs: (3, 50, 2), Outputs: (3, 50, 1), Initial States: None
    break
"
```

Expected output:
```
Inputs: torch.Size([3, 50, 2]), Outputs: torch.Size([3, 50, 1]), Initial States: torch.Size([3, 3])
Inputs: torch.Size([3, 50, 2]), Outputs: torch.Size([3, 50, 1]), Initial States: None
```

## Migration Guide

### For Existing Code

No changes required! Existing code without states continues to work:

```python
# Old code (still works)
for inputs, outputs in train_loader:
    # ...
```

### To Add Initial State Support

Just add the optional parameters:

```python
# New code with initial states
train_loader, val_loader, test_loader, normalizer = create_dataloaders(
    train_inputs, train_outputs,
    val_inputs, val_outputs,
    train_states=train_states,  # Add this (full trajectories)
    val_states=val_states,      # Add this
    batch_size=32,
    normalize=True
)

# Update iteration to use initial states
for inputs, outputs, initial_states in train_loader:
    # inputs:  (batch_size, seq_length, n_inputs)
    # outputs: (batch_size, seq_length, n_outputs)
    # initial_states: (batch_size, n_states) <- ONLY first timestep!
    
    if initial_states is not None:
        # Use initial states for your model
        predictions = model(inputs, initial_state=initial_states)
```

## Use Cases

### 1. LMI-Constrained RNN Training with Initial Conditions

Initial states can be used as initial conditions for RNN models:

```python
class LMIConstrainedRNN(nn.Module):
    def forward(self, inputs, initial_state=None):
        # Use initial_state as starting hidden state
        if initial_state is not None:
            h = initial_state  # shape: (batch_size, n_states)
        else:
            h = torch.zeros(inputs.shape[0], self.hidden_size)
        
        outputs = []
        for t in range(inputs.shape[1]):
            h = self.cell(inputs[:, t, :], h)  # RNN step
            outputs.append(h)
        return torch.stack(outputs, dim=1)

# Training
for inputs, outputs, initial_states in train_loader:
    predictions = model(inputs, initial_state=initial_states)
    loss = F.mse_loss(predictions, outputs)
```

### 2. Lyapunov Stability with Initial State Constraints

Enforce stability constraints based on initial state:

```python
def train_step(self, inputs, outputs, initial_states):
    predictions = self.model(inputs, initial_state=initial_states)
    
    # Standard prediction loss
    pred_loss = F.mse_loss(predictions, outputs)
    
    # Lyapunov constraint: V(x0) should be within safe region
    if initial_states is not None:
        V0 = self.lyapunov_function(initial_states)  # V at initial state
        # Penalize if initial state is outside safe region
        safety_loss = torch.relu(V0 - self.V_max).mean()
        total_loss = pred_loss + self.lambda_safety * safety_loss
    else:
        total_loss = pred_loss
    
    return total_loss
```

## API Reference

### `create_dataloaders`

```python
def create_dataloaders(
    train_inputs: np.ndarray,
    train_outputs: np.ndarray,
    val_inputs: np.ndarray,
    val_outputs: np.ndarray,
    test_inputs: Optional[np.ndarray] = None,
    test_outputs: Optional[np.ndarray] = None,
    train_states: Optional[np.ndarray] = None,  # NEW
    val_states: Optional[np.ndarray] = None,    # NEW
    test_states: Optional[np.ndarray] = None,   # NEW
    batch_size: int = 32,
    sequence_length: Optional[int] = None,
    normalize: bool = True,
    normalization_method: str = "minmax",
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataNormalizer]]:
    """Create data loaders with optional state support."""
```

### `TimeSeriesDataset`

```python
class TimeSeriesDataset:
    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        states: Optional[np.ndarray] = None,  # NEW
        sequence_length: Optional[int] = None,
    ):
        """Dataset with optional state information."""
    
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Returns (input_seq, output_seq, initial_state)."""
```

### `collate_with_optional_states`

```python
def collate_with_optional_states(
    batch: List[Tuple]
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Custom collate function that handles optional initial state tensors.
    
    Returns:
        (batched_inputs, batched_outputs, batched_initial_states or None)
    """
```

## Notes

- **No normalization of initial states**: Initial states are passed through as-is, without normalization
- **Memory efficient**: Only initial states are extracted and stored in batches
- **Type safety**: All components properly handle `Optional[Tensor]` for initial states
- **Saved with evaluations**: Initial states are automatically saved during model evaluation if present
- **Initial state only**: Full state trajectories can be provided as input, but only the first timestep is used
