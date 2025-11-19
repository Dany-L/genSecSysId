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
    fill_polytope: bool = True,
):
    """Plot ellipse and parallelogram given by X'HX <= s^2.
    
    Args:
        X: Inverse of Lyapunov matrix P (X = P^-1)
        H: Coupling matrix (H = L @ P^-1)
        s: Sector bound
        max_norm_x0: Optional maximum norm of initial condition
        ax: Matplotlib axis to plot on (creates new if None)
        show: Whether to show the plot
        fill_polytope: Whether to fill the polytope (default: True)

    Returns:
        (fig, ax) tuple so caller can further modify the figure.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    H = np.asarray(H)



    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True
    else:
        fig = ax.figure

    # Polytope vertices for ||H x||_infty <= 1
    # This works for H ∈ ℝ^(m×2) where m >= 2
    plot_polytope(ax, H, fill_polytope, 'r-')
    plot_ellipse(ax, X, s, 'b-', fill=fill_polytope)
    

    # Plot ellipse on top


    # if max_norm_x0 is not None:
    #     # allow scalar or vector max_norm_x0; use its 2-norm as radius
    #     r = max_norm_x0
    #     circle_x0 = np.array([r * np.cos(theta), r * np.sin(theta)])
    #     ax.plot(circle_x0[0, :], circle_x0[1, :], 'k:', linewidth=1.5,
    #            label=r"$\|x^0\|_2 \leq x^0_{max}$")

    # ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_title("Ellipse and Polytope Regions", fontsize=14)

    if show and created_fig:
        plt.show()

    return fig, ax


def plot_polytope(ax, H: np.array, fill: bool = True, linetype: str = 'r--', name:str =r"$\|Hx\|_\infty \leq 1$"):
    if H.shape[1] == 2:
        try:
            # Each row of H defines constraints: -1 <= H_i @ x <= 1
            # This gives us 2*m linear inequalities defining a polytope
            # We'll compute the vertices of this polytope
            
            m = H.shape[0]
            # Create inequality matrix: A @ x <= b
            # For each row h_i of H: h_i @ x <= 1 and -h_i @ x <= 1
            A = np.vstack([H, -H])  # Shape: (2m, 2)
            b = np.ones(2 * m)
            
            # Find vertices by solving for intersections of constraint boundaries
            # A vertex occurs where 2 constraints are active (since we're in 2D)
            vertices = []
            
            for i in range(2 * m):
                for j in range(i + 1, 2 * m):
                    # Solve system where constraints i and j are active (equality)
                    M = np.array([A[i], A[j]])
                    b_eq = np.array([b[i], b[j]])
                    
                    # Check if system is solvable (constraints not parallel)
                    if np.abs(np.linalg.det(M)) > 1e-10:
                        vertex = np.linalg.solve(M, b_eq)
                        
                        # Check if vertex satisfies ALL constraints
                        if np.all(A @ vertex <= b + 1e-9):  # Small tolerance for numerical errors
                            vertices.append(vertex)
            
            if len(vertices) > 0:
                # Remove duplicates
                vertices = np.array(vertices)
                vertices = np.unique(vertices, axis=0)
                
                # Sort vertices by angle to get proper polygon order
                center = np.mean(vertices, axis=0)
                angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                V = vertices[sorted_indices]
                
                # Plot filled polytope or just edges
                if fill:
                    ax.fill(V[:, 0], V[:, 1], color=[0.8, 0.9, 1.0], 
                           edgecolor='r', linewidth=1.5, 
                           label=name, alpha=0.7)
                else:
                    # Close the polygon by adding first vertex at the end
                    V_closed = np.vstack([V, V[0]])
                    ax.plot(V_closed[:, 0], V_closed[:, 1], linetype,
                           label=name, linewidth=1.5)
            else:
                logging.getLogger(__name__).warning("No vertices found for ||Hx||_infty <= 1 polytope.")
                
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to plot ||Hx||_infty polytope: {e}")
    else:
        logging.getLogger(__name__).warning(f"H has {H.shape[1]} columns, cannot plot in 2D (need 2 columns).")


def plot_ellipse(ax, X:np.array,s: float, linetype: str = 'b--', name: str=r"$\frac{1}{s^2} x^T X x \leq 1$" , fill: bool = False):
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    # map unit circle through ellipse transform
    L = np.linalg.cholesky(1 / s**2 * X)
    X_half_inv = np.linalg.inv(L.T)
    ellipse = X_half_inv @ circle

    if fill:
        ax.fill(ellipse[0, :], ellipse[1, :], color=[1.0, 0.8, 0.8], 
               edgecolor='b', linewidth=1.5, 
               label=name, alpha=0.7)
    else:
        ax.plot(ellipse[0, :], ellipse[1, :], linetype, linewidth=2, 
            label=name)