# MIMO System Support and State Information

## Overview

The package now supports **Multiple Input Multiple Output (MIMO)** systems and optional **state information** for advanced system identification scenarios.

## Key Features

### 1. MIMO Support

Previously limited to single input/output (SISO), now supports:
- Multiple inputs (e.g., `d1`, `d2`, `d3`)
- Multiple outputs (e.g., `e1`, `e2`)
- Automatic dimension detection from config

### 2. Optional State Information

For special cases where internal state measurements are available:
- Optional `state_col` parameter
- Can include state measurements in data loading
- Empty list `[]` means no state information (default)

## Configuration

### SISO (Single Input Single Output)

```yaml
data:
  train_path: "data/prepared"
  input_col: ["d"]      # Single input
  output_col: ["e"]     # Single output
  state_col: []         # No state (default)
```

Or use defaults:
```yaml
data:
  train_path: "data/prepared"
  # input_col defaults to ["d"]
  # output_col defaults to ["e"]
  # state_col defaults to []
```

### MIMO (Multiple Input Multiple Output)

```yaml
data:
  train_path: "data/prepared"
  input_col: ["d1", "d2", "d3"]  # 3 inputs
  output_col: ["e1", "e2"]       # 2 outputs
  state_col: []                   # No state
```

### MIMO with State Information

```yaml
data:
  train_path: "data/prepared"
  input_col: ["u1", "u2"]     # 2 control inputs
  output_col: ["y1", "y2"]    # 2 measured outputs
  state_col: ["x1", "x2", "x3"]  # 3 state measurements
```

## Data Format

### CSV File Structure

**SISO Example:**
```csv
d,e
0.5,0.3
0.7,0.4
...
```

**MIMO Example:**
```csv
d1,d2,d3,e1,e2
0.5,0.2,0.1,0.3,0.8
0.7,0.3,0.2,0.4,0.9
...
```

**MIMO with States:**
```csv
d1,d2,e1,e2,x1,x2,x3
0.5,0.2,0.3,0.8,1.2,0.5,0.3
0.7,0.3,0.4,0.9,1.3,0.6,0.4
...
```

## Code Changes

### 1. Config (`src/sysid/config.py`)

**DataConfig now supports lists:**
```python
@dataclass
class DataConfig:
    input_col: list = None   # List of input column names
    output_col: list = None  # List of output column names
    state_col: list = None   # List of state column names (optional)
    
    def __post_init__(self):
        if self.input_col is None:
            self.input_col = ["d"]
        if self.output_col is None:
            self.output_col = ["e"]
        if self.state_col is None:
            self.state_col = []  # Empty = no states
```

### 2. Data Loader (`src/sysid/data/direct_loader.py`)

**Updated signatures:**
```python
def load_csv_folder(
    folder_path: str,
    input_col: Union[str, List[str]] = "d",
    output_col: Union[str, List[str]] = "e",
    state_col: Union[str, List[str], None] = None,
    pattern: str = "*.csv",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Returns:
        - inputs: (n_files, seq_len, n_inputs)
        - outputs: (n_files, seq_len, n_outputs)
        - states: (n_files, seq_len, n_states) or None
        - filenames: List of file names
    """
```

**Automatic dimension handling:**
- String input → converts to list: `"d"` → `["d"]`
- List input → uses as-is: `["d1", "d2"]` → `["d1", "d2"]`
- Supports multiple columns per type

### 3. Model Creation (`scripts/train.py`)

**Automatic dimension detection:**
```python
model = SimpleRNN(
    input_size=len(data_config.input_col),    # Number of inputs
    hidden_size=model_config.nw,
    output_size=len(data_config.output_col),  # Number of outputs
    ...
)
```

## Examples

### Example 1: SISO System (Default)

```yaml
# config/siso.yaml
data:
  train_path: "data/prepared"
  # Uses defaults: input_col=["d"], output_col=["e"]

model:
  model_type: "rnn"
  nw: 64
  num_layers: 2
```

Data shape:
- Inputs: (n_samples, seq_len, 1)
- Outputs: (n_samples, seq_len, 1)
- States: None

### Example 2: MIMO System (2 inputs, 3 outputs)

```yaml
# config/mimo.yaml
data:
  train_path: "data/prepared"
  input_col: ["d1", "d2"]
  output_col: ["e1", "e2", "e3"]

model:
  model_type: "rnn"
  nw: 128
  num_layers: 3
```

Data shape:
- Inputs: (n_samples, seq_len, 2)
- Outputs: (n_samples, seq_len, 3)
- States: None

Model automatically created with:
- `input_size=2`
- `output_size=3`

### Example 3: System with State Measurements

```yaml
# config/with_states.yaml
data:
  train_path: "data/prepared"
  input_col: ["u"]
  output_col: ["y"]
  state_col: ["x1", "x2", "x3", "x4"]  # 4 state measurements

model:
  model_type: "rnn"
  nw: 64
  num_layers: 2
```

Data shape:
- Inputs: (n_samples, seq_len, 1)
- Outputs: (n_samples, seq_len, 1)
- States: (n_samples, seq_len, 4)

**Note:** State information is loaded and available but not yet used in standard RNN models. This is for future advanced features or custom models.

## Usage

### Training with MIMO

```bash
# SISO (default)
python scripts/train.py --config configs/siso.yaml

# MIMO
python scripts/train.py --config configs/mimo.yaml

# With states
python scripts/train.py --config configs/with_states.yaml
```

### Evaluation with MIMO

```bash
# Same config determines input/output columns
python scripts/evaluate.py \
    --config configs/mimo.yaml \
    --model models/mimo/best_model.pt \
    --test-data data/prepared/test
```

## Implementation Details

### Backward Compatibility

✅ **Fully backward compatible:**
- Old configs without lists still work (converted to lists internally)
- String column names converted to single-element lists
- Default values: `input_col=["d"]`, `output_col=["e"]`, `state_col=[]`

### Type Handling

```python
# All these are equivalent:
input_col: "d"           → ["d"]
input_col: ["d"]         → ["d"]

# MIMO examples:
input_col: ["d1", "d2"]  → ["d1", "d2"]
```

### Data Loading

1. **Config specifies columns:**
   ```yaml
   input_col: ["d1", "d2"]
   ```

2. **Loader extracts columns:**
   ```python
   inputs = df[["d1", "d2"]].values  # shape: (seq_len, 2)
   ```

3. **Stacks into array:**
   ```python
   all_inputs = np.array([...])  # shape: (n_files, seq_len, 2)
   ```

## Benefits

### 1. Flexibility
- Single config change to switch between SISO and MIMO
- Same codebase handles all cases

### 2. Clarity
- Explicit column naming in config
- Easy to see system structure

### 3. Extensibility
- State information ready for advanced models
- Future support for state-space models
- Custom models can use state measurements

### 4. Type Safety
- List-based approach clearer than string
- IDE autocomplete works better
- Dimension mismatches caught early

## Common Use Cases

### 1. Multi-Actuator Systems
```yaml
input_col: ["motor1_cmd", "motor2_cmd", "motor3_cmd"]
output_col: ["position", "velocity"]
```

### 2. Sensor Fusion
```yaml
input_col: ["control_input"]
output_col: ["sensor1", "sensor2", "sensor3", "sensor4"]
```

### 3. State Estimation
```yaml
input_col: ["control"]
output_col: ["measurement"]
state_col: ["estimated_x1", "estimated_x2"]  # From Kalman filter, etc.
```

### 4. Multi-Variable Process Control
```yaml
input_col: ["valve1", "valve2", "heater_power"]
output_col: ["temperature", "pressure", "flow_rate"]
```

## Limitations

### Current
- State columns loaded but not yet integrated into standard models
- Custom models needed to utilize state information
- All sequences must have same number of inputs/outputs/states

### Future Enhancements
- State-space RNN models
- Physics-informed neural networks using states
- Latent variable models
- State-based regularization

## Migration Guide

### From Old Config (String-based)

**Before:**
```yaml
data:
  input_col: "d"   # String
  output_col: "e"  # String
```

**After (Recommended):**
```yaml
data:
  input_col: ["d"]   # List (explicit)
  output_col: ["e"]  # List (explicit)
```

**Or use defaults:**
```yaml
data:
  # Omit - defaults to ["d"] and ["e"]
```

### Adding MIMO

**Just update config:**
```yaml
data:
  input_col: ["d1", "d2", "d3"]  # Add more columns
  output_col: ["e1", "e2"]       # Add more columns
```

**CSV must have these columns:**
```csv
d1,d2,d3,e1,e2
...
```

## Testing

Tested scenarios:
- ✅ SISO with defaults
- ✅ SISO with explicit lists
- ✅ MIMO (2 in, 3 out)
- ✅ MIMO with states (3 in, 2 out, 4 states)
- ✅ Backward compatibility (string → list conversion)
- ✅ Empty state_col (no states)
- ✅ Model dimension auto-detection

## Summary

The package now fully supports MIMO systems and optional state information:

1. ✅ **Config updated** - List-based column specification
2. ✅ **Data loader updated** - Handles lists and states
3. ✅ **Scripts updated** - Auto-detect dimensions
4. ✅ **Backward compatible** - Old configs still work
5. ✅ **Well documented** - Complete examples
6. ✅ **Type-safe** - Better IDE support

You can now:
- Train MIMO systems by simply updating config
- Include state measurements for advanced use cases
- Mix and match input/output dimensions as needed
- Use same codebase for SISO and MIMO
