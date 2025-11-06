# Naming Conventions - System Identification Literature Compatibility

## Overview

Updated all code to use consistent naming conventions that align with system identification literature, making the code more intuitive for researchers in this field.

## Variable Naming

### Core Variables

| Variable | Meaning | Description |
|----------|---------|-------------|
| **`d`** | Input signal | System input (disturbance/drive signal) |
| **`e`** | Output signal | Actual system output (measured) |
| **`e_hat`** | Predicted output | Model's prediction of the output |
| **`x`** or **`hidden_state`** | Hidden state | RNN internal/hidden state |

### Usage Guidelines

1. **Use explicit names for clarity:**
   - When introducing variables: `d` (input), `e` (output), `e_hat` (predicted output)
   - In documentation/comments: "input" or "d", "output" or "e", "predicted output" or "e_hat"

2. **Hidden state naming:**
   - Variable: `x` or `hidden_state` (both acceptable)
   - Documentation: Always use "hidden state" or "hidden_state"
   - This distinguishes from the input `d` which is NOT called `x`

3. **Avoid generic names:**
   - ❌ `inputs`, `targets`, `outputs`, `predictions`
   - ✅ `d`, `e`, `e_hat`, `hidden_state`

## Examples

### Model Forward Pass

```python
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
```

### Training Loop

```python
for d, e in train_loader:  # d: input, e: output
    d = d.to(device)
    e = e.to(device)
    
    # Forward pass
    e_hat = model(d)  # e_hat: predicted output
    
    # Compute loss
    loss = loss_fn(e_hat, e)
```

### Evaluation

```python
for d, e in test_loader:  # d: input, e: output (target)
    d = d.to(device)
    e = e.to(device)
    
    # Forward pass
    e_hat = model(d)  # e_hat: predicted output
    
    # Compute metrics
    metrics = compute_metrics(e_hat, e)
```

## Files Updated

All major files updated for consistency:

1. **Models** (`src/sysid/models/rnn.py`):
   - `SimpleRNN.forward()`: `d`, `hidden_state` → `e_hat`
   - `LSTM.forward()`: `d`, `hidden_state` → `e_hat`
   - `GRU.forward()`: `d`, `hidden_state` → `e_hat`

2. **Training** (`src/sysid/training/trainer.py`):
   - `train_epoch()`: Uses `d`, `e`, `e_hat`
   - `validate()`: Uses `d`, `e`, `e_hat`

3. **Evaluation** (`src/sysid/evaluation/`):
   - `evaluator.py`: Uses `d`, `e`, `e_hat` throughout
   - `metrics.py`: `compute_metrics(e_hat, e)`

4. **Scripts** (`scripts/evaluate.py`):
   - Uses `d`, `e`, `e_hat` for loading and plotting

## Benefits

1. **Literature Compatibility**: Aligns with standard system identification notation
2. **Clarity**: Immediately clear what each variable represents
3. **Consistency**: Same naming across all modules
4. **Brevity**: Short variable names reduce code clutter
5. **Type Hints**: Combined with type hints for best of both worlds

## Mathematical Notation Correspondence

| Code | Math | Description |
|------|------|-------------|
| `d[k]` | $d_k$ | Input at time $k$ |
| `e[k]` | $e_k$ | Output at time $k$ |
| `e_hat[k]` | $\hat{e}_k$ | Predicted output at time $k$ |
| `x[k]` | $x_k$ | Hidden state at time $k$ |

## System Identification Context

This naming follows the convention where:
- **System**: $e_k = f(d_k, d_{k-1}, ..., x_{k-1})$
- **Model**: $\hat{e}_k = f_{\theta}(d_k, d_{k-1}, ..., x_{k-1})$
- **Error**: $\epsilon_k = e_k - \hat{e}_k$

Where:
- $d$: Input/disturbance signal (what we control or observe)
- $e$: Output/error signal (what we measure)
- $x$: Internal state (hidden dynamics)
- $\theta$: Model parameters (RNN weights)
