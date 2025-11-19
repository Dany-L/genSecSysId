#!/usr/bin/env python
"""
Export trained SimpleLure model matrices to .mat file for MATLAB analysis.

This script loads a trained model from MLflow and exports all system matrices
in MATLAB-compatible format.

Usage:
    python scripts/export_for_matlab.py --run-id <run_id> --output <path.mat>
    
Examples:
    python scripts/export_for_matlab.py --run-id abc123def456 --output model.mat
"""

import argparse
import logging
from pathlib import Path
import sys

import mlflow
import numpy as np
from scipy.io import savemat

from sysid.models import SimpleLure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        'nx': model.nx,
        'nd': model.nd,
        'ne': model.ne,
        'nw': model.nw,
        'nz': model.nz,
        
        # System matrices
        'A': model.A.detach().cpu().numpy(),
        'B': model.B.detach().cpu().numpy(),
        'C': model.C.detach().cpu().numpy(),
        'D': model.D.detach().cpu().numpy(),
        
        'B2': model.B2.detach().cpu().numpy(),
        'C2': model.C2.detach().cpu().numpy(),
        'D12': model.D12.detach().cpu().numpy(),
        'D21': model.D21.detach().cpu().numpy(),
        'D22': model.D22.detach().cpu().numpy(),
        
        # Lyapunov certificate
        'P': model.P.detach().cpu().numpy(),
        'L': model.L.detach().cpu().numpy() if model.learn_L else np.zeros((model.nz, model.nx)),
        
        # Multipliers
        'la': model.la.detach().cpu().numpy(),
        'M': np.diag(model.la.detach().cpu().numpy()),
        
        # Stability parameters
        'alpha': model.alpha.item(),
        's': model.s.item(),
        'delta': model.delta.item(),
        'max_norm_x0': model.max_norm_x0,
    }
    
    # Compute derived quantities for convenience
    Pinv = np.linalg.inv(matrices['P'])
    matrices['X'] = Pinv  # Ellipsoid shape matrix
    matrices['H'] = matrices['L'] @ Pinv  # Parallelogram matrix
    matrices['Lambda'] = np.diag(1.0 / np.diag(matrices['M']))  # Inverse multipliers
    
    # Save to .mat file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    savemat(str(output_path), matrices, oned_as='column')
    
    logger.info(f"Exported model to {output_path}")
    logger.info(f"Exported {len(matrices)} variables")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPORTED MATRICES")
    print("="*80)
    print(f"Dimensions: nx={matrices['nx']}, nd={matrices['nd']}, ne={matrices['ne']}, nw={matrices['nw']}, nz={matrices['nz']}")
    print(f"Parameters: alpha={matrices['alpha']:.6f}, s={matrices['s']:.6f}, delta={matrices['delta']:.6f}")
    print("\nMatrices exported:")
    for key in sorted(matrices.keys()):
        val = matrices[key]
        if isinstance(val, np.ndarray) and val.ndim > 0:
            print(f"  {key:15s}: {val.shape}")
        else:
            print(f"  {key:15s}: scalar")
    print("="*80 + "\n")
    
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
        epilog=__doc__
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='MLflow run ID of the trained model'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='model.mat',
        help='Output .mat file path (default: model.mat)'
    )
    
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: uses current tracking URI)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    logger.info(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Load model from MLflow
    logger.info(f"Loading model from run {args.run_id}")
    try:
        model_uri = f"runs:/{args.run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        
        if not isinstance(model, SimpleLure):
            logger.error("Model is not a SimpleLure model. Export only supports SimpleLure.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Export to MATLAB
    export_model_to_matlab(model, args.output)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
