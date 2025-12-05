# Generalized Sector-Bounded System Identification (genSecSysId)

Research project on system identification using recurrent neural networks with Lyapunov-based stability certificates.

## Overview

This repository contains the implementation and research materials for identifying nonlinear dynamical systems using constrained RNN architectures with provable stability guarantees. The work focuses on regionally stable **Lur'e models** (Lur'e systems with sector-bounded nonlinearities) and leverages semidefinite programming to certify stability regions via Lyapunov functions.

## Repository Structure

```
genSecSysId/
├── python/                    # Python implementation
│   ├── src/sysid/            # Main package source code
│   ├── scripts/              # Training, evaluation, and analysis scripts
│   ├── configs/              # Configuration files for experiments
│   ├── tests/                # Unit tests
│   └── README.md            # Detailed Python package documentation
│
├── matlab/                    # MATLAB scripts and simulations
│   ├── models/               # MATLAB model implementations
│   ├── analysis/             # Analysis and visualization scripts
│   └── experiments/          # Experimental data and results
│
└── literature/               # Research papers and references
    ├── papers/               # Relevant publications
    └── notes/                # Research notes and summaries
```

## Python Package

The `python/` directory contains a complete PyTorch-based system identification package with:

- **Constrained RNN architectures** (CRNN) with Lyapunov certificates
- **SimpleLure models** with (generalized) sector-bounded nonlinearities
- **Post-processing optimization** via semidefinite programming (SDP)
- **MLflow integration** for experiment tracking
- **Comprehensive evaluation** with stability visualization

**Key Features:**
- Direct CSV data loading
- Adaptive training with interior point method for constraint satisfaction
- Lyapunov ellipse and polytope visualization
- Multi-model comparison tools
- MATLAB export functionality

See [`python/README.md`](python/README.md) for detailed documentation and usage examples.

## MATLAB Scripts

The `matlab/` directory contains complementary MATLAB implementations for:

- System simulation and data generation
- Alternative model implementations
- Specialized analysis tools
- Integration with MATLAB Control System Toolbox

## Quick Start

### Python Package

```bash
# Navigate to Python package
cd python

# Install dependencies
pip install -e .

# Optional: Install SDP solver for post-processing
pip install cvxpy mosek

# Train a constrained RNN model
python scripts/train.py --config configs/constrained_rnn_lmi.yaml

# Evaluate and visualize
python scripts/evaluate.py --run-id <mlflow_run_id> --test-data ../data/test
```

See [`python/README.md`](python/README.md) for comprehensive documentation.

## Development

> **Note**: This project was developed with assistance from GitHub Copilot and Claude (Anthropic) AI coding assistants for architecture design, implementation, and documentation. The constrained RNN models that are based on Lur'e type systems where implemented by the author. This includes the LMIs that are used to verify feasibility of the learned parameters.

### Python Package Development
```bash
cd python
pip install -e ".[dev]"
pytest tests/
```

### Code Structure
- **Python**: PyTorch-based, modular design with dataclasses for configuration
- **MATLAB**: Traditional scripting with specialized control toolbox integration


## License

See LICENSE file.
