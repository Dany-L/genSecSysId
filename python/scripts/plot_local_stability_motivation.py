#!/usr/bin/env python
"""
Generate motivational phase plot for local stability analysis.

This script simulates a damped pendulum system and visualizes:
1. An ellipsoid representing the basin of attraction for bounded inputs
2. Three trajectories demonstrating local stability:
   - Trajectory A: Starts inside ellipse → converges to origin
   - Trajectory B: Starts outside but enters ellipse → converges
   - Trajectory C: Starts outside and never enters → diverges/doesn't converge

The ellipsoid size depends on the input bound c, representing the region
where trajectories are guaranteed to converge when ||u|| ≤ c.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Ellipse
import argparse
import os

try:
    import tikzplotlib
    TIKZPLOTLIB_AVAILABLE = True
except ImportError:
    TIKZPLOTLIB_AVAILABLE = False


def pendulum_dynamics(t, x, u_func, m=1.0, l=1.0, b=0.5, g=9.81):
    """
    Damped pendulum dynamics with torque input.
    
    State: x = [theta, theta_dot]
    Dynamics: 
        theta_ddot = -g/l * sin(theta) - b/(m*l^2) * theta_dot + u/(m*l^2)
    
    Args:
        t: Time
        x: State [theta, theta_dot]
        u_func: Function u(t, x) returning control input
        m: Mass (kg)
        l: Length (m)
        b: Damping coefficient
        g: Gravity (m/s^2)
    
    Returns:
        State derivative [theta_dot, theta_ddot]
    """
    theta, theta_dot = x
    u = u_func(t, x)
    
    # Nonlinear pendulum dynamics
    theta_ddot = -(g/l) * np.sin(theta) - (b/(m*l**2)) * theta_dot + u/(m*l**2)
    
    return [theta_dot, theta_ddot]


def simulate_trajectory(x0, u_func, t_span, m=1.0, l=1.0, b=0.5, g=9.81, max_step=0.01):
    """
    Simulate pendulum trajectory.
    
    Args:
        x0: Initial condition [theta0, theta_dot0]
        u_func: Control input function
        t_span: Time span [t0, tf]
        m, l, b, g: System parameters
        max_step: Maximum integration step
    
    Returns:
        Solution object with t and y (states)
    """
    sol = solve_ivp(
        lambda t, x: pendulum_dynamics(t, x, u_func, m, l, b, g),
        t_span,
        x0,
        method='RK45',
        max_step=max_step,
        dense_output=True
    )
    return sol


def compute_lyapunov_ellipse(c, m=1.0, l=1.0, b=0.5, g=9.81):
    """
    Compute Lyapunov-based ellipse for the region of attraction.
    
    For a damped pendulum with bounded input ||u|| ≤ c, we can construct
    a Lyapunov function V(x) = x^T P x where trajectories entering the
    ellipse {x: V(x) ≤ α} are guaranteed to converge.
    
    This is a simplified construction for visualization purposes.
    
    Args:
        c: Input bound
        m, l, b, g: System parameters
    
    Returns:
        P: Lyapunov matrix (2x2)
        alpha: Level set value
    """
    # Simplified Lyapunov matrix (energy-based)
    # V(x) ≈ (1/2) * theta_dot^2 + (g/l) * (1 - cos(theta))
    # Approximation near origin: V(x) ≈ (1/2) * theta_dot^2 + (g/2l) * theta^2
    
    # For small angles, use quadratic approximation
    # Choose P such that ellipse represents region where damping dominates
    
    # Scale with input bound c
    # Larger c (more disturbance) → Smaller region of attraction
    scale = (c + 0.1)  # Larger c → larger P → smaller ellipse for fixed α
    
    P = np.array([
        [g/l * scale, 0],
        [0, 1.0 * scale]
    ])
    
    # Keep alpha constant (or could decrease with c for even smaller ellipse)
    alpha = 1.0  # Fixed level set
    
    return P, alpha


def get_ellipse_parameters(P, alpha):
    """
    Convert Lyapunov ellipse to matplotlib Ellipse parameters.
    
    For ellipse x^T P x = alpha, compute center, width, height, and angle.
    
    Args:
        P: Positive definite matrix (2x2)
        alpha: Level set value
    
    Returns:
        center, width, height, angle for matplotlib Ellipse
    """
    # Eigendecomposition of P
    eigvals, eigvecs = np.linalg.eig(P)
    
    # Semi-axes lengths (from x^T P x = alpha)
    a = np.sqrt(alpha / eigvals[0])  # Semi-axis along first eigenvector
    b = np.sqrt(alpha / eigvals[1])  # Semi-axis along second eigenvector
    
    # Angle of rotation (radians to degrees)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) * 180 / np.pi
    
    # Center at origin
    center = (0, 0)
    
    # Width and height (diameters)
    width = 2 * a
    height = 2 * b
    
    return center, width, height, angle


def main():
    parser = argparse.ArgumentParser(description="Generate local stability motivation plot")
    parser.add_argument("--c", type=float, default=1.0, help="Input bound (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default=".", 
                       help="Output directory (default: current directory)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fixed filename
    output_path = os.path.join(args.output_dir, "local_stability_motivation.png")
    tex_output_path = os.path.join(args.output_dir, "local_stability_motivation.tex")
    
    # System parameters
    m = 1.0   # mass (kg)
    l = 1.0   # length (m)
    b = 2.0   # damping
    g = 9.81  # gravity
    
    c = args.c  # Input bound
    
    # Time span for simulation
    t_span = [0, 10]
    
    # Define control inputs for different scenarios
    # Scenario A: No input (u=0)
    u_zero = lambda t, x: 0.0
    
    # Scenario B: Small bounded input
    u_small = lambda t, x: 0.3 * c * np.sin(2*t)
    
    # Scenario C: Large input that prevents convergence
    u_large = lambda t, x: 10 * c * np.sin(t)
    
        
    # Compute Lyapunov ellipse
    P, alpha = compute_lyapunov_ellipse(c, m, l, b, g)
    
    # Initial conditions
    # A: Inside ellipse
    x0_A = np.array([0.1, 0.5])
    
    # B: Outside but enters
    x0_B = np.array([0.1, -1])
    
    # C: Outside and doesn't enter
    x0_C = np.array([0.2, 0.4])
    
    # Simulate trajectories
    print("Simulating trajectories...")
    sol_A = simulate_trajectory(x0_A, u_zero, t_span, m, l, b, g)
    sol_B = simulate_trajectory(x0_B, u_small, t_span, m, l, b, g)
    sol_C = simulate_trajectory(x0_C, u_large, t_span, m, l, b, g)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ellipse (region of attraction)
    center, width, height, angle = get_ellipse_parameters(P, alpha)
    ellipse = Ellipse(center, width, height, angle=angle, 
                     facecolor='lightblue', edgecolor='blue', 
                     linewidth=2, alpha=0.3, label='Region of Attraction')
    ax.add_patch(ellipse)
    
    # Plot trajectories
    ax.plot(sol_A.y[0], sol_A.y[1], 'g-', linewidth=2, label='Trajectory A: Inside → Converges')
    ax.plot(sol_B.y[0], sol_B.y[1], 'orange', linewidth=2, label='Trajectory B: Outside → Enters → Converges')
    ax.plot(sol_C.y[0], sol_C.y[1], 'r-', linewidth=2, label='Trajectory C: Outside → Diverges')
    
    # Mark initial conditions
    ax.plot(x0_A[0], x0_A[1], 'go', markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(x0_B[0], x0_B[1], 'o', color='orange', markersize=10, markeredgecolor='darkorange', markeredgewidth=2)
    ax.plot(x0_C[0], x0_C[1], 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
    
    # Mark origin (equilibrium)
    ax.plot(0, 0, 'k*', markersize=15, label='Equilibrium (0, 0)')
    
    # Add arrows to show direction
    # for sol, color in [(sol_A, 'g'), (sol_B, 'orange'), (sol_C, 'r')]:
    #     # Add arrow at midpoint
    #     mid_idx = len(sol.t) // 2
    #     x_mid = sol.y[0][mid_idx]
    #     y_mid = sol.y[1][mid_idx]
    #     dx = sol.y[0][mid_idx+1] - sol.y[0][mid_idx]
    #     dy = sol.y[1][mid_idx+1] - sol.y[1][mid_idx]
    #     ax.arrow(x_mid, y_mid, dx*5, dy*5, head_width=0.15, head_length=0.1, 
    #             fc=color, ec=color, alpha=0.7)
    
    # Formatting
    ax.set_xlabel(r'$x_1$', fontsize=14)
    ax.set_ylabel(r'$x_2$', fontsize=14)
    # ax.set_title(f'Local Stability: Damped Pendulum with Bounded Input ($\|u\| \leq {c}$)', 
    #             fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=11, loc='upper right')
    # ax.set_aspect('equal', adjustable='box')
    
    # Set axis limits
    # ax.set_xlim(-3.5, 3.5)
    # ax.set_ylim(-2, 2)
    
    # Add text annotation
    # ax.text(0.02, 0.98, 
    #        f'System: Damped pendulum\n'
    #        f'$m={m}$ kg, $l={l}$ m, $b={b}$ Ns/m\n'
    #        f'Input bound: $\|u\| \leq {c}$ Nm',
    #        transform=ax.transAxes, fontsize=10,
    #        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure as PNG
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"✓ Figure saved to {output_path}")
    
    # Save as TikZ/LaTeX if tikzplotlib is available
    if TIKZPLOTLIB_AVAILABLE:
        tikzplotlib.save(tex_output_path, 
                        axis_width='\\figwidth', 
                        axis_height='\\figheight',
                        strict=False)
        print(f"✓ TikZ/LaTeX file saved to {tex_output_path}")
    else:
        print("⚠ tikzplotlib not available. Install with: pip install tikzplotlib")
    
    # Show figure
    plt.show()


if __name__ == "__main__":
    main()
