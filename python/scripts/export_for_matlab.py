#!/usr/bin/env python
"""
Export trained SimpleLure model matrices to a .mat file for MATLAB analysis.

The model, config, and (best) checkpoint are resolved from the standard
training layout via sysid.config.resolve_run_artifacts — no MLflow server
interaction is needed.

Usage:
    python scripts/export_for_matlab.py --run-id <run_id> [--output <path.mat>]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

from sysid.config import resolve_run_artifacts
from sysid.models import SimpleLure, load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "~/genSecSysId-Data"


def export_model_to_matlab(model: SimpleLure, output_path: str):
    """
    Export SimpleLure model matrices to MATLAB .mat file.

    Args:
        model: SimpleLure model instance
        output_path: Path to output .mat file
    """
    logger.info("Extracting model matrices...")

    # Extract all matrices
    matrices = {
        # State dimensions
        "nx": model.nx,
        "nd": model.nd,
        "ne": model.ne,
        "nw": model.nw,
        "nz": model.nz,
        # System matrices
        "A": model.A.detach().cpu().numpy(),
        "B": model.B.detach().cpu().numpy(),
        "C": model.C.detach().cpu().numpy(),
        "D": model.D.detach().cpu().numpy(),
        "B2": model.B2.detach().cpu().numpy(),
        "C2": model.C2.detach().cpu().numpy(),
        "D12": model.D12.detach().cpu().numpy(),
        "D21": model.D21.detach().cpu().numpy(),
        "D22": model.D22.detach().cpu().numpy(),
        # Lyapunov certificate
        "P": model.P.detach().cpu().numpy(),
        "L": model.L.detach().cpu().numpy() if model.learn_L else np.zeros((model.nz, model.nx)),
        # Multipliers
        "la": model.la.detach().cpu().numpy(),
        "M": np.diag(model.la.detach().cpu().numpy()),
        # Stability parameters. Model stores the unconstrained `tau`; the
        # physical α used everywhere downstream is sigmoid(tau).
        "alpha": float(1.0 / (1.0 + np.exp(-model.tau.detach().cpu().numpy()))),
        "s": model.s.item(),
        "delta": model.delta.item(),
        "max_norm_x0": model.max_norm_x0,
    }

    # Compute derived quantities for convenience
    Pinv = np.linalg.inv(matrices["P"])
    matrices["X"] = Pinv  # Ellipsoid shape matrix
    matrices["H"] = matrices["L"] @ Pinv  # Parallelogram matrix
    matrices["Lambda"] = np.diag(1.0 / np.diag(matrices["M"]))  # Inverse multipliers

    # Save to .mat file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    savemat(str(output_path), matrices, oned_as="column")

    logger.info(f"Exported model to {output_path}")
    logger.info(f"Exported {len(matrices)} variables")

    # Print summary
    print("\n" + "=" * 80)
    print("EXPORTED MATRICES")
    print("=" * 80)
    print(
        f"Dimensions: nx={matrices['nx']}, nd={matrices['nd']}, ne={matrices['ne']}, nw={matrices['nw']}, nz={matrices['nz']}"
    )
    print(
        f"Parameters: alpha={matrices['alpha']:.6f}, s={matrices['s']:.6f}, delta={matrices['delta']:.6f}"
    )
    print("\nMatrices exported:")
    for key in sorted(matrices.keys()):
        val = matrices[key]
        if isinstance(val, np.ndarray) and val.ndim > 0:
            print(f"  {key:15s}: {val.shape}")
        else:
            print(f"  {key:15s}: scalar")
    print("=" * 80 + "\n")

    # MATLAB loading instructions
    print("To load in MATLAB:")
    print(f"  data = load('{output_path}');")
    print("  A = data.A;")
    print("  B = data.B;")
    print("  P = data.P;")
    print("  L = data.L;")
    print("  % etc...\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export SimpleLure model to MATLAB format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--run-id", type=str, required=True, help="MLflow run ID of the trained model"
    )

    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Base directory for per-run artefacts (default: {DEFAULT_DATA_ROOT}).",
    )

    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .mat file path. Defaults to <run_dir>/best_model_export.mat "
             "next to the checkpoint.",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve config + best-checkpoint path from the run id (local disk only).
    try:
        config, model_path, _, _ = resolve_run_artifacts(
            args.run_id, data_root=args.data_root
        )
    except Exception as e:
        logger.error(f"Failed to resolve run_id={args.run_id}: {e}")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}")
    try:
        model = load_model(str(model_path), config, device="cpu")
        if not isinstance(model, SimpleLure):
            logger.error("Model is not a SimpleLure model. Export only supports SimpleLure.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    output_path = (
        Path(args.output).expanduser()
        if args.output is not None
        else Path(model_path).with_name("best_model_export.mat")
    )

    export_model_to_matlab(model, str(output_path))

    logger.info("Done!")


if __name__ == "__main__":
    main()
