"""Utility functions."""

import torch
import numpy as np
import random
from typing import List, Optional
import logging


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ("auto", "cuda", "cpu", "mps")
        
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """Print model summary."""
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)


def torch_bmat(mat: List[List[torch.Tensor]]) -> torch.Tensor:
    mat_list = []
    for col in mat:
        mat_list.append(torch.hstack(col))
    return torch.vstack(mat_list)


def plot_ellipse_and_parallelogram(
    X: np.array,
    H: np.array,
    s: float,
    max_norm_x0: Optional[float] = None,
    ax=None,
    show: bool = False,
):
    """Plot ellipse and parallelogram given by X'HX <= s^2.

    Returns:
        (fig, ax) tuple so caller can further modify the figure.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    H = np.asarray(H)

    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    # map unit circle through ellipse transform
    ellipse = np.linalg.inv(np.linalg.cholesky(1 / s**2 * X)).T @ circle

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(ellipse[0, :], ellipse[1, :], label=r"$x^T X x \leq s^2$")

    # Polytope vertices for ||H x||_infty <= 1
    # This works for H ∈ ℝ^(m×2) where m >= 2
    if H.shape[1] == 2:
        try:
            # Each row of H defines constraints: -1 <= H_i @ x <= 1
            # This gives us 2*m linear inequalities defining a polytope
            # We'll compute the vertices of this polytope
            
            m = H.shape[0]
            # Create inequality matrix: A @ x <= b
            # For each row h_i of H: h_i @ x <= 1 and -h_i @ x <= 1
            A_ub = np.vstack([H, -H])  # Shape: (2m, 2)
            b_ub = np.ones(2 * m)
            
            # Find vertices by solving for intersections of constraint boundaries
            # A vertex occurs where 2 constraints are active (since we're in 2D)
            vertices = []
            
            for i in range(2 * m):
                for j in range(i + 1, 2 * m):
                    # Solve system where constraints i and j are active (equality)
                    A_eq = np.array([A_ub[i], A_ub[j]])
                    b_eq = np.array([b_ub[i], b_ub[j]])
                    
                    # Check if system is solvable (constraints not parallel)
                    if np.abs(np.linalg.det(A_eq)) > 1e-10:
                        vertex = np.linalg.solve(A_eq, b_eq)
                        
                        # Check if vertex satisfies ALL other constraints
                        if np.all(A_ub @ vertex <= b_ub + 1e-8):  # Small tolerance for numerical errors
                            vertices.append(vertex)
            
            if len(vertices) > 0:
                vertices = np.array(vertices)
                
                # Sort vertices by angle to get proper polygon order
                center = np.mean(vertices, axis=0)
                angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                vertices = vertices[sorted_indices]
                
                # Close the polygon by adding first vertex at the end
                vertices = np.vstack([vertices, vertices[0]])
                
                ax.plot(vertices[:, 0], vertices[:, 1], 
                       label=r"$\|Hx\|_\infty \leq 1$", linestyle="--")
            else:
                logging.getLogger(__name__).warning("No vertices found for ||Hx||_infty <= 1 polytope.")
                
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to plot ||Hx||_infty polytope: {e}")
    else:
        logging.getLogger(__name__).warning(f"H has {H.shape[1]} columns, cannot plot in 2D (need 2 columns).")

    if max_norm_x0 is not None:
        # allow scalar or vector max_norm_x0; use its 2-norm as radius
        r = max_norm_x0
        circle_x0 = np.array([r * np.cos(theta), r * np.sin(theta)])
        ax.plot(circle_x0[0, :], circle_x0[1, :], label=r"$\|x^0\|_2 \leq x^0_{max}$", linestyle=":")

    # ax.set_aspect("equal", adjustable="box")
    ax.grid()
    ax.legend()
    ax.set_title("Ellipse and Parallelogram")
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])

    if show and created_fig:
        plt.show()

    return fig, ax
