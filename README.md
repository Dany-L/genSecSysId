# Generalized Sector-Bounded System Identification (genSecSysId)

Research project on system identification using recurrent neural networks with Lyapunov-based stability certificates.

## Overview

This repository contains the implementation and research materials for identifying nonlinear dynamical systems using constrained RNN architectures with provable stability guarantees. The work focuses on **SimpleLure models** (Lure systems with sector-bounded nonlinearities) and leverages semidefinite programming to certify stability regions via Lyapunov functions.

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
- **SimpleLure models** with sector-bounded nonlinearities
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

## Literature

The `literature/` directory contains:

- Key publications on Lure systems and sector bounds
- Papers on Lyapunov stability theory
- RNN stability analysis research
- System identification literature

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

## Research Context

This work addresses the challenge of learning dynamical system models that are:
1. **Accurate**: High-fidelity predictions from data
2. **Stable**: Certified stability regions via Lyapunov theory
3. **Interpretable**: Structured state-space representations

The approach combines:
- Neural network expressiveness (RNN architectures)
- Control-theoretic guarantees (Lyapunov certificates)
- Convex optimization (SDP post-processing)

## Key Concepts

### SimpleLure Models
State-space models with sector-bounded nonlinearities:
```
x(k+1) = Ax(k) + Bw(k) + Bu(k)
y(k) = Cx(k) + Dw(k) + Du(k)
w(k) = φ(z(k))
z(k) = Lx(k)
```
where φ satisfies sector bounds: ||φ(z)||_∞ ≤ ||z||_∞

### Lyapunov Certificates
Quadratic Lyapunov function V(x) = x^T P x guarantees stability within the region:
- {x: x^T P x ≤ α} ∩ {x: ||Lx||_∞ ≤ 1}

Visualized as the intersection of an ellipse and a polytope.

## Development

> **Note**: This project was developed with assistance from GitHub Copilot and Claude (Anthropic) AI coding assistants for architecture design, implementation, and documentation.

### Python Package Development
```bash
cd python
pip install -e ".[dev]"
pytest tests/
```

### Code Structure
- **Python**: PyTorch-based, modular design with dataclasses for configuration
- **MATLAB**: Traditional scripting with specialized control toolbox integration

## Citation

If you use this code in your research, please cite:

```bibtex
@software{genSecSysId2025,
  title = {Generalized Sector-Bounded System Identification},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/Dany-L/genSecSysId}
}
```

## License

See LICENSE file.

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

## Related Work

- **Lure Systems**: Classical framework for analyzing nonlinear feedback systems
- **Sector Bounds**: Constraints on nonlinearity for stability analysis
- **Lyapunov Theory**: Rigorous stability certification via energy-like functions
- **Semidefinite Programming**: Convex optimization for finding Lyapunov certificates
- **RNN Stability**: Neural network architectures with stability properties

## Acknowledgments

This research builds upon foundational work in:
- Nonlinear control theory
- System identification
- Neural network stability
- Convex optimization

Special thanks to the open-source communities behind PyTorch, CVXPY, MLflow, and MOSEK.
