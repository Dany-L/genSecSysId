.PHONY: help install install-dev format lint test clean ui

help:
	@echo "Available commands:"
	@echo "  make install      - Install package in development mode"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run flake8 linter"
	@echo "  make test         - Run pytest test suite"
	@echo "  make ui           - Start the MLflow UI on ./mlruns (http://127.0.0.1:5000)"
	@echo "  make clean        - Remove build artifacts and caches"

# MLflow >= 3.16 refuses a file-backed store (./mlruns) unless this is set:
# the file store is in maintenance mode. Every run this project has produced
# lives there, so opt out rather than migrate. Must run from this directory --
# there is a second, stale ./mlruns at the repo root with only a handful of runs.
ui:
	MLFLOW_ALLOW_FILE_STORE=true mlflow ui

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

format:
	@echo "Running black..."
	python -m black src/sysid scripts/ --line-length 100
	@echo "Running isort..."
	python -m isort src/sysid scripts/ --profile black --line-length 100
	@echo "✓ Code formatted successfully"

lint:
	@echo "Running flake8..."
	python -m flake8 src/sysid scripts/
	@echo "✓ Linting complete"

test:
	@echo "Running pytest..."
	pytest tests/ -v
	@echo "✓ Tests complete"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "✓ Clean complete"
