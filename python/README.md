# System Identification Package

[![Tests](https://github.com/Dany-L/genSecSysId/actions/workflows/tests.yml/badge.svg)](https://github.com/Dany-L/genSecSysId/actions/workflows/tests.yml)

RNN-based identification of nonlinear dynamical systems from data, with
constrained (Lure) models that carry a certified regional-stability
certificate. Built on PyTorch with MLflow experiment tracking.

> Developed with assistance from GitHub Copilot and Claude (Anthropic).

## Installation

```bash
cd python
pip install -e .            # runtime
pip install -e ".[dev]"     # + test/lint tooling
```

Post-processing (SDP for the Lyapunov certificate) additionally needs a
convex solver:

```bash
pip install cvxpy mosek     # MOSEK is free for academia
```

## Basic usage

The three core steps operate on a config file and pass an MLflow **run id**
between each other. `evaluate` and `post_process` resolve the config,
checkpoint, and normalizer from the run id automatically.

### 1. Train

```bash
python scripts/train.py --config ~/genSecSysId-Data/configs/crnn_gen-sec_duffing.yaml
```

Trains the model, tracks the run in MLflow, and saves the best checkpoint.
Add `--run-id-out run_id.txt` to capture the run id for the next steps.

### 2. Evaluate

```bash
python scripts/evaluate.py --run-id <run_id>
```

Computes metrics (RMSE, NRMSE, …) on the test split and — for SimpleLure
models with `nx=2` — plots the Lyapunov ellipse and sector-bound polytope.
Override the test set with `--test-data <path>`.

### 3. Post-process

```bash
python scripts/post_process.py --run-id <run_id> --true-dynamics duffing
```

Solves an SDP for the optimal Lyapunov certificate (`P`, `L`) with the trained
dynamics fixed, then runs regional verification (driving the model past the
learned constraint to check divergence, optionally against registered true
dynamics). Tune the verification with `--rv-violation-factors`,
`--rv-initial-state-scale`, `--rv-num-trajectories`, and `--rv-horizon`.

## Testing

```bash
pytest tests/                              # run the suite
pytest tests/ --cov=sysid --cov-report=html   # with coverage
```

Tests that need the MOSEK solver skip automatically when no license is
present, so the suite runs without one.

## MLflow

Runs are tracked locally under `mlruns/` (no server needed). Browse them with:

```bash
mlflow ui   # then open http://127.0.0.1:5000
```

## More

- Project structure, data loading, and advanced usage: [docs/](docs/)
- Logging layout: [LOGGING.md](LOGGING.md)
- Evaluation metrics: [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md)

## License

See [LICENSE](LICENSE).
