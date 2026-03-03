"""Generate pendulum dataset with white noise input and state information."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
from tqdm import tqdm


def low_pass_filter(white_noise, cutoff_freq, sampling_rate, order=4):
    """
    Apply low-pass Butterworth filter to white noise signal.
    
    Args:
        white_noise: Input white noise signal
        cutoff_freq: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = signal.filtfilt(b, a, white_noise)
    return filtered_signal


def generate_input_signal(N, dt, input_magnitude, cutoff_freq, transition_point=0.5):
    """
    Generate low-pass filtered white noise input that goes to zero after transition_point.
    
    Args:
        N: Number of time steps
        dt: Time step size
        input_magnitude: Magnitude of input signal
        cutoff_freq: Cutoff frequency for low-pass filter
        transition_point: Fraction of trajectory where input goes to zero (default: 0.5)
        
    Returns:
        Input signal array
    """
    sampling_rate = 1.0 / dt
    
    # Generate white noise
    white_noise = np.random.randn(N) * input_magnitude
    
    # Apply low-pass filter
    filtered_input = low_pass_filter(white_noise, cutoff_freq, sampling_rate)
    
    # Set input to zero after transition point
    transition_idx = int(N * transition_point)
    filtered_input[transition_idx:] = 0.0
    
    return filtered_input


def pendulum_dynamics(t, state, u_func, m=1.0, L=1.0, b=0.1, g=9.81):
    """
    Pendulum dynamics: second-order ODE with friction.
    
    Equation: m*L^2*theta_ddot + b*theta_dot + m*g*L*sin(theta) = u
    
    Args:
        t: Current time
        state: [theta, theta_dot]
        u_func: Function that returns control input at time t
        m: Mass (kg)
        L: Length (m)
        b: Damping coefficient (N⋅m⋅s/rad)
        g: Gravity (m/s^2)
        
    Returns:
        [theta_dot, theta_ddot]
    """
    theta, theta_dot = state
    u = u_func(t)
    
    # Inertia
    I = m * L**2
    
    # Avoid division by zero
    if I == 0:
        raise ValueError(f"Inertia I = m*L^2 = {m}*{L}^2 = 0. Mass and length must be non-zero.")
    
    # theta_ddot = (u - b*theta_dot - m*g*L*sin(theta)) / I
    theta_ddot = (u - b * theta_dot - m * g * L * np.sin(theta)) / I
    
    return np.array([theta_dot, theta_ddot])


def simulate_pendulum(initial_state, input_signal, dt, m=1.0, L=1.0, b=0.1, g=9.81):
    """
    Simulate pendulum dynamics with given input signal using ODE solver.
    
    Args:
        initial_state: [theta, theta_dot] initial state
        input_signal: Control input (torque) sequence
        dt: Time step size
        m: Mass (kg)
        L: Length (m)
        b: Damping coefficient (N⋅m⋅s/rad)
        g: Gravity (m/s^2)
        
    Returns:
        states: Array of shape (N, 2) with [theta, theta_dot]
        observations: Array of shape (N, 3) with [cos(theta), sin(theta), theta_dot]
    """
    N = len(input_signal)
    time_points = np.arange(N) * dt
    
    # Create interpolation function for input signal
    def u_func(t):
        idx = int(t / dt)
        if idx >= N:
            return 0.0
        return input_signal[idx]
    
    # Solve ODE
    sol = solve_ivp(
        fun=lambda t, y: pendulum_dynamics(t, y, u_func, m, L, b, g),
        t_span=(0, time_points[-1]),
        y0=initial_state,
        t_eval=time_points,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # Check if solver was successful
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    
    # Extract states
    states = sol.y.T  # Shape: (N, 2)
    
    # Create observations: [cos(theta), sin(theta), theta_dot]
    observations = np.zeros((N, 3))
    observations[:, 0] = np.cos(states[:, 0])  # cos(theta)
    observations[:, 1] = np.sin(states[:, 0])  # sin(theta)
    observations[:, 2] = states[:, 1]  # theta_dot
    
    return states, observations


def generate_dataset(
    M,
    N,
    dt,
    output_dir,
    theta_range=(-np.pi, np.pi),
    theta_dot_range=(-8.0, 8.0),
    input_magnitude=2.0,
    cutoff_freq=2.0,
    transition_point=0.5,
    m=1.0,
    L=1.0,
    b=0.1,
    g=9.81,
):
    """
    Generate pendulum dataset with M trajectories of length N.
    
    Args:
        M: Number of trajectories
        N: Length of each trajectory (time steps)
        dt: Time step size
        output_dir: Directory to save CSV files
        theta_range: Range for initial angle (radians)
        theta_dot_range: Range for initial angular velocity (rad/s)
        input_magnitude: Magnitude of input signal
        cutoff_freq: Cutoff frequency for low-pass filter (Hz)
        transition_point: Fraction of trajectory where input goes to zero
        m: Mass (kg)
        L: Length (m)
        b: Damping coefficient (N⋅m⋅s/rad)
        g: Gravity (m/s^2)
        
    Returns:
        trajectories: List of dictionaries with trajectory data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectories = []
    
    print(f"Generating {M} trajectories of length {N} with dt={dt}...")
    print(f"Initial state ranges: theta={theta_range}, theta_dot={theta_dot_range}")
    print(f"Input: magnitude={input_magnitude}, cutoff={cutoff_freq} Hz")
    print(f"Input transitions to zero at {transition_point*100:.0f}% of trajectory")
    print(f"Pendulum parameters: m={m} kg, L={L} m, b={b} N⋅m⋅s/rad, g={g} m/s²")
    print()
    
    for traj_idx in tqdm(range(M), desc="Generating trajectories"):
        # Random initial conditions
        theta_0 = np.random.uniform(*theta_range)
        theta_dot_0 = np.random.uniform(*theta_dot_range)
        initial_state = [theta_0, theta_dot_0]
        
        # Generate input signal
        input_signal = generate_input_signal(N, dt, input_magnitude, cutoff_freq, transition_point)
        
        # Simulate pendulum
        states, observations = simulate_pendulum(initial_state, input_signal, dt, m, L, b, g)
        
        # Create DataFrame with all data
        time = np.arange(N) * dt
        df = pd.DataFrame({
            'time': time,
            'u': input_signal,  # Control input (torque)
            'x_theta': states[:, 0],  # True state: angle
            'x_theta_dot': states[:, 1],  # True state: angular velocity
            'y_cos_theta': observations[:, 0],  # Observed output: cos(theta)
            'y_sin_theta': observations[:, 1],  # Observed output: sin(theta)
            'y_theta_dot': observations[:, 2],  # Observed output: angular velocity
        })
        
        # Save to CSV
        filename = f"trajectory_{traj_idx:04d}.csv"
        csv_path = output_dir / filename
        df.to_csv(csv_path, index=False)
        
        # Store trajectory info
        trajectories.append({
            'filename': filename,
            'initial_state': initial_state,
            'states': states,
            'observations': observations,
            'input': input_signal,
            'time': time,
        })
    
    print(f"\n✓ Dataset generated: {M} trajectories saved to {output_dir}")
    return trajectories


def visualize_trajectories(trajectories, output_dir, num_samples=5):
    """
    Visualize sample trajectories in phase space and input signals.
    
    Args:
        trajectories: List of trajectory dictionaries
        output_dir: Directory to save plots
        num_samples: Number of trajectories to visualize in detailed plots
    """
    output_dir = Path(output_dir)
    num_samples = min(num_samples, len(trajectories))
    
    # Select random samples for detailed plots
    sample_indices = np.random.choice(len(trajectories), size=num_samples, replace=False)
    
    # Create figure with 2 subplots: phase space and inputs (ALL trajectories)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Phase space plot - ALL trajectories
    ax_phase = axes[0]
    for idx in range(len(trajectories)):
        traj = trajectories[idx]
        states = traj['states']
        ax_phase.plot(states[:, 0], states[:, 1], alpha=0.3, linewidth=1.0)
        # Mark initial state
        ax_phase.scatter(states[0, 0], states[0, 1], s=50, marker='o', 
                        edgecolors='black', linewidths=0.5, zorder=5, alpha=0.5)
        # Mark final state
        ax_phase.scatter(states[-1, 0], states[-1, 1], s=50, marker='x', 
                        linewidths=1, zorder=5, alpha=0.5)
    
    ax_phase.set_xlabel('Angle θ [rad]', fontsize=12)
    ax_phase.set_ylabel('Angular Velocity dθ/dt [rad/s]', fontsize=12)
    ax_phase.set_title(f'Phase Space Trajectories (All {len(trajectories)} trajectories)', fontsize=14, fontweight='bold')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_phase.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Input signals plot - ALL trajectories
    ax_input = axes[1]
    for idx in range(len(trajectories)):
        traj = trajectories[idx]
        time = traj['time']
        input_signal = traj['input']
        ax_input.plot(time, input_signal, alpha=0.3, linewidth=1.0)
    
    ax_input.set_xlabel('Time [s]', fontsize=12)
    ax_input.set_ylabel('Input (Torque) [N⋅m]', fontsize=12)
    ax_input.set_title(f'Input Signals (All {len(trajectories)} trajectories)', fontsize=14, fontweight='bold')
    ax_input.grid(True, alpha=0.3)
    ax_input.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / 'trajectory_visualization.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {plot_path}")
    plt.show()
    
    # Create individual plots for SAMPLE trajectories only
    fig2, axes2 = plt.subplots(num_samples, 3, figsize=(18, 4*num_samples))
    if num_samples == 1:
        axes2 = axes2.reshape(1, -1)
    
    for plot_idx, traj_idx in enumerate(sample_indices):
        traj = trajectories[traj_idx]
        time = traj['time']
        states = traj['states']
        input_signal = traj['input']
        
        # Phase space
        ax1 = axes2[plot_idx, 0]
        ax1.plot(states[:, 0], states[:, 1], linewidth=2)
        ax1.scatter(states[0, 0], states[0, 1], s=150, marker='o', c='green', 
                   edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax1.scatter(states[-1, 0], states[-1, 1], s=150, marker='x', c='red',
                   linewidths=3, label='End', zorder=5)
        ax1.set_xlabel('θ [rad]')
        ax1.set_ylabel('dθ/dt [rad/s]')
        ax1.set_title(f'Trajectory {traj_idx}: Phase Space')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Angle over time
        ax2 = axes2[plot_idx, 1]
        ax2.plot(time, states[:, 0], linewidth=2, label='θ')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Angle [rad]')
        ax2.set_title(f'Trajectory {traj_idx}: Angle')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Input
        ax3 = axes2[plot_idx, 2]
        ax3.plot(time, input_signal, linewidth=2, color='orange')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Input [N⋅m]')
        ax3.set_title(f'Trajectory {traj_idx}: Input Signal')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Mark transition point
        transition_idx = np.where(input_signal == 0)[0]
        if len(transition_idx) > 0:
            transition_time = time[transition_idx[0]]
            ax3.axvline(x=transition_time, color='red', linestyle='--', alpha=0.5, 
                       label='Input → 0')
            ax3.legend()
    
    plt.tight_layout()
    plot_path_detailed = output_dir / 'trajectory_details.png'
    plt.savefig(plot_path_detailed, dpi=150, bbox_inches='tight')
    print(f"✓ Detailed plots saved to {plot_path_detailed}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate pendulum dataset')
    parser.add_argument('--M', type=int, default=100, help='Number of trajectories')
    parser.add_argument('--N', type=int, default=500, help='Length of each trajectory (time steps)')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step size (seconds)')
    parser.add_argument('--output-dir', type=str, default='data/pendulum/raw', 
                       help='Output directory for CSV files')
    parser.add_argument('--theta-max', type=float, default=np.pi, 
                       help='Maximum initial angle magnitude (radians), range will be [-theta-max, +theta-max]')
    parser.add_argument('--theta-dot-max', type=float, default=8.0, 
                       help='Maximum initial angular velocity magnitude (rad/s), range will be [-theta-dot-max, +theta-dot-max]')
    parser.add_argument('--input-magnitude', type=float, default=2.0, 
                       help='Magnitude of input signal')
    parser.add_argument('--cutoff-freq', type=float, default=2.0, 
                       help='Cutoff frequency for low-pass filter (Hz)')
    parser.add_argument('--transition-point', type=float, default=0.5, 
                       help='Fraction of trajectory where input goes to zero')
    parser.add_argument('--m', type=float, default=1.0, 
                       help='Pendulum mass (kg)')
    parser.add_argument('--L', type=float, default=1.0, 
                       help='Pendulum length (m)')
    parser.add_argument('--b', type=float, default=0.1, 
                       help='Damping coefficient (N⋅m⋅s/rad)')
    parser.add_argument('--g', type=float, default=9.81, 
                       help='Gravity (m/s^2)')
    parser.add_argument('--visualize', type=int, default=5, 
                       help='Number of trajectories to visualize (0 to skip)')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Generate dataset
    trajectories = generate_dataset(
        M=args.M,
        N=args.N,
        dt=args.dt,
        output_dir=args.output_dir,
        theta_range=(-args.theta_max, args.theta_max),
        theta_dot_range=(-args.theta_dot_max, args.theta_dot_max),
        input_magnitude=args.input_magnitude,
        cutoff_freq=args.cutoff_freq,
        transition_point=args.transition_point,
        m=args.m,
        L=args.L,
        b=args.b,
        g=args.g,
    )
    
    # Save metadata
    metadata = {
        'M': args.M,
        'N': args.N,
        'dt': args.dt,
        'total_time': args.N * args.dt,
        'theta_range': [-args.theta_max, args.theta_max],
        'theta_dot_range': [-args.theta_dot_max, args.theta_dot_max],
        'input_magnitude': args.input_magnitude,
        'cutoff_freq': args.cutoff_freq,
        'transition_point': args.transition_point,
        'pendulum_parameters': {
            'm': args.m,
            'L': args.L,
            'b': args.b,
            'g': args.g,
        },
        'dynamics': 'I*theta_ddot + b*theta_dot + m*g*L*sin(theta) = u, where I = m*L^2',
        'columns': ['time', 'u', 'x_theta', 'x_theta_dot', 'y_cos_theta', 'y_sin_theta', 'y_theta_dot'],
        'description': {
            'time': 'Time in seconds',
            'u': 'Control input (torque) in N⋅m',
            'x_theta': 'True state: angle in radians',
            'x_theta_dot': 'True state: angular velocity in rad/s',
            'y_cos_theta': 'Observed output: cos(theta)',
            'y_sin_theta': 'Observed output: sin(theta)',
            'y_theta_dot': 'Observed output: angular velocity in rad/s',
        }
    }
    
    metadata_path = Path(args.output_dir) / 'metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Visualize sample trajectories
    if args.visualize > 0:
        print(f"\nGenerating visualizations for {args.visualize} sample trajectories...")
        visualize_trajectories(trajectories, args.output_dir, num_samples=args.visualize)
    
    print("\n" + "="*70)
    print("Dataset generation complete!")
    print("="*70)
    print(f"Total trajectories: {args.M}")
    print(f"Trajectory length: {args.N} steps ({args.N * args.dt:.1f} seconds)")
    print(f"Output directory: {args.output_dir}")
    print(f"\nCSV columns: {', '.join(metadata['columns'])}")
    print("\nUsage for training:")
    print(f"  - Input column: 'u'")
    print(f"  - Output columns: 'y_cos_theta', 'y_sin_theta', 'y_theta_dot'")
    print(f"  - State columns: 'x_theta', 'x_theta_dot' (true states, for analysis only)")


if __name__ == '__main__':
    main()
