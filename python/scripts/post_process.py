#!/usr/bin/env python
"""
Post-process a trained model by solving an SDP to find optimal Lyapunov certificate (P and L) 
while keeping system matrices (A, B, C, D) fixed.

This script loads a trained SimpleLure model and calls its post_process() method
to solve a semidefinite program (SDP) for optimal P and L matrices.

Usage:
    python scripts/post_process.py --run-id <run_id> [--optimize-s]
    
Examples:
    # Post-process a model, keeping s fixed
    python scripts/post_process.py --run-id abc123def456
    
    # Post-process and optimize for minimum s
    python scripts/post_process.py --run-id abc123 --optimize-s
"""

import argparse
import logging
from pathlib import Path
import sys
import os

import mlflow
import torch
import numpy as np
from scipy.io import loadmat

from sysid.models import SimpleLure
from sysid.config import Config
from sysid.data.direct_loader import load_csv_folder, load_split_data


torch.set_default_dtype(torch.float64)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_DATA_PATH = '~/genSecSysId-Data/data/prepared/test'# set this is hard coded but will later become a parameter
TRAJECTORY_LIST = [
    93,  16, 110, 106,  26,  40, 144,  62,  82, 
    4, 6, 9, 17, 32, 90, 91, 92,
    2, 25, 47, 97, 101, 161, 255
]
CONFIG_FILE_PATH = '~/genSecSysId-Data/configs/crnn_gen-sec.yaml'



X_H_MAT_PATH = '~/genSecSysId-Data/data/X_H-param_from_mat.mat'

X_H_MAT = loadmat(os.path.expanduser(X_H_MAT_PATH))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process a trained model",
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
        '--config',
        type=str,
        default=CONFIG_FILE_PATH,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-3,
        help='Small positive constant for strict inequalities (default: 1e-3)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='post_processing',
        help='Output directory for post-processed results (default: post_processing)'
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

    # Load configuration early so we can derive directories from config.root_dir
    config_path = Path(args.config)
    # config_path = Path(CONFIG_FILE_PATH)
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        config = Config.from_yaml(os.path.expanduser(config_path))
    elif config_path.suffix == ".json":
        config = Config.from_json(os.path.expanduser(config_path))
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Load model from MLflow
    logger.info(f"Loading model from run {args.run_id}")
    try:
        model_uri = f"runs:/{args.run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        
        if not isinstance(model, SimpleLure):
            logger.error("Model is not a SimpleLure model. Post-processing only supports SimpleLure.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Get the original run
    run = mlflow.get_run(args.run_id)
    experiment_id = run.info.experiment_id
    
    # Use the SAME run (not a new one) - just add artifacts and metrics
    logger.info(f"Logging post-processing results to original run: {args.run_id}")
    
    with mlflow.start_run(run_id=args.run_id):
        # Call model's post_process method
        logger.info("Calling model.post_process()...")

        # Get state column if provided
        state_col = getattr(config.data, 'state_col', None)
        if state_col and len(state_col) == 0:  # Empty list means no state
            state_col = None

        test_path = Path(os.path.expanduser(TEST_DATA_PATH))
                
        test_inputs, test_outputs, test_states, filenames = load_csv_folder(
            folder_path=str(test_path),
            input_col=getattr(config.data, 'input_col', ['d']),
            output_col=getattr(config.data, 'output_col', ['e']),
            state_col=state_col,
            pattern=getattr(config.data, 'pattern', '*.csv')
        )

        test_inputs, test_outputs, test_states = np.stack(test_inputs[TRAJECTORY_LIST]), np.stack(test_outputs[TRAJECTORY_LIST]), np.stack(test_states[TRAJECTORY_LIST])
        result = model.post_process(eps=args.eps)
        
        if not result['success']:
            logger.error(f"Post-processing failed: {result.get('error', result.get('status', 'unknown'))}")
            sys.exit(1)
        
        # Extract results
        summary = result['summary']
        
        # Log metrics to MLflow with post_process prefix
        mlflow.log_metric("post_process/constraints_satisfied", int(result['constraints_satisfied']))
        mlflow.log_metric("post_process/s_original", summary['original']['s'])
        mlflow.log_metric("post_process/s_optimized", summary['optimized']['s'])
        mlflow.log_metric("post_process/max_eig_F", summary['optimized']['max_eig_F'])
        mlflow.log_metric("post_process/cond_P_original", summary['original']['cond_P'])
        mlflow.log_metric("post_process/cond_P_optimized", summary['optimized']['cond_P'])
        
        # Log parameter
        mlflow.log_param("post_processing", True)
        mlflow.log_param("post_process_eps", args.eps)
        
        # Save results to file
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"post_processing_{args.run_id[:8]}.npz"
        np.savez(
            results_path,
            P_original=model.P.cpu().detach().numpy(),
            P_opt=result['P_opt'],
            L_original=model.L.cpu().detach().numpy() if model.learn_L else None,
            L_opt=result['L_opt'],
            m_opt=result['m_opt'],
            s_original=summary['original']['s'],
            s_opt=result['s_opt'],
            S_hat_opt=result['S_hat_opt'],
            max_eig_F=result['max_eig_F'],
            alpha=model.alpha.cpu().detach().numpy(),
            A=model.A.cpu().detach().numpy(),
            B=model.B.cpu().detach().numpy(),
            B2=model.B2.cpu().detach().numpy(),
            C2=model.C2.cpu().detach().numpy(),
            D21=model.D21.cpu().detach().numpy(),
        )
        logger.info(f"Saved results to {results_path}")
        
        # Log results file to MLflow
        mlflow.log_artifact(str(results_path), artifact_path="post_processing")


        
        # Save and log updated model with different name
        # model_path = output_dir / f"post_processing_model_{args.run_id[:8]}.pt"
        # torch.save(model.state_dict(), model_path)
        # mlflow.pytorch.log_model(model, name="model_post_processing")
        
        # Generate ellipse/polytope plot if model is 2D
        if model.nx == 2:
            logger.info("Generating ellipse and polytope visualization...")

            # generate trajectories
            with torch.no_grad():
                e_hat, (xs, ws) = model(torch.tensor(test_inputs), torch.tensor(test_states[:,0,:]), return_state = True)

            try:
                from sysid.utils import plot_ellipse_and_parallelogram, plot_ellipse, plot_polytope
                import matplotlib.pyplot as plt
                import tikzplotlib
                
                X = np.linalg.inv(model.P.cpu().detach().numpy())
                H = model.L.cpu().detach().numpy() @ X
                s = model.s.cpu().detach().numpy()

                fig, ax = plt.subplots(figsize=(8, 8))

                for x, x_hat in zip(test_states, xs):
                    if np.linalg.norm(x[-1,:], ord=2) > np.linalg.norm(x[0,:], ord=2) and not (np.linalg.norm(x[0,:], ord=2) == 0.0):
                        # diverging trajectory
                        ax.plot(x[0,0], x[0,1], 'rx')
                    else:
                        ax.plot(x[0,0], x[0,1], 'go')
                    # ax.plot(x[:,0], x[:,1], '--')
                    ax.plot(x_hat[:,0], x_hat[:,1])

                ellipse_plot_path = output_dir / f"ellipse_polytope_post_{args.run_id[:8]}.png"

                if model.learn_L and config.training.use_custom_regularization:
                    
                    fig, ax = plot_ellipse_and_parallelogram(X, H, s, None, ax=ax, show=False, fill_polytope=True)

                    # plot ellipse and polytope from mat file for comparison
                    X_mat = X_H_MAT['X']
                    H_mat = X_H_MAT['H']
                    s_mat = X_H_MAT['s'][0,0]

                    plot_ellipse(ax, X_mat, s_mat, linetype = 'b--')
                    plot_polytope(ax, H_mat, fill=False, linetype = 'r--')
                    
                    # Save to output directory
                    
                    fig.savefig(ellipse_plot_path, dpi=150, bbox_inches='tight')

                    try:
                        tikzplotlib.save(str(ellipse_plot_path.with_suffix('.tex')))
                        mlflow.log_artifact(str(ellipse_plot_path.with_suffix('.tex')), artifact_path="post_processing")
                    except Exception as e:
                        logger.warning(f"Failed to save TikZ plot: {e}")
                    logger.info(f"Ellipse/polytope plot saved to {ellipse_plot_path}")
                    
                    # Log to MLflow
                    mlflow.log_figure(fig, str(ellipse_plot_path.with_suffix('.png')))
                    plt.close(fig)
                    logger.info("Ellipse/polytope plot logged to MLflow")
                else:
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel(r"$x_1$", fontsize=12)
                    ax.set_ylabel(r"$x_2$", fontsize=12)
                    mlflow.log_figure(fig, str(ellipse_plot_path.with_suffix('.png')))
                    tikzplotlib.save(str(ellipse_plot_path.with_suffix('.tex')))
                    mlflow.log_artifact(str(ellipse_plot_path.with_suffix('.tex')), artifact_path="post_processing")
                    plt.close(fig)


            except Exception as e:
                logger.warning(f"Failed to generate ellipse/polytope plot: {e}")
        
        # plot some prediction for handpicked trajectories 
        logger.info(f"Generating prediction plot for trajectory...")
        try:
            from sysid.utils import plot_predictions
            import matplotlib.pyplot as plt
            import tikzplotlib
            
            with torch.no_grad():
                e_hat = model(torch.tensor(test_inputs), torch.tensor(test_states[:,0,:]))

            pred_plot_path = output_dir / f"prediction_trajectory_post_{args.run_id[:8]}.png"

            fig, axes = plot_predictions(
                output_dir=output_dir,
                e_hat = e_hat,
                e = test_outputs,
                num_samples=3,
                sample_indices=UNSTAB_STAB_ZERO,
                save_path=pred_plot_path,
                return_axes=True
            )
            
            try:
                tikzplotlib.save(str(pred_plot_path.with_suffix('.tex')))
                mlflow.log_artifact(str(pred_plot_path.with_suffix('.tex')), artifact_path="post_processing")
            except Exception as e:
                logger.warning(f"Failed to save TikZ plot: {e}")
            plt.close(fig)
            logger.info(f"Prediction plot saved to {pred_plot_path}")
        except Exception as e:
                logger.warning(f"Failed to generate prediction plot for trajectory")

        logger.info(f"✓ Post-processing complete! Results saved to run: {args.run_id}")


if __name__ == "__main__":
    main()
