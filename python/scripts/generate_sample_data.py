"""Generate sample data for testing the system identification package."""

import numpy as np
import argparse
from pathlib import Path


def generate_nonlinear_system_data(
    n_samples: int = 1000,
    sequence_length: int = 100,
    noise_std: float = 0.01,
    system_type: str = "pendulum",
) -> tuple:
    """
    Generate synthetic data from a nonlinear dynamical system.
    
    Args:
        n_samples: Number of trajectories
        sequence_length: Length of each trajectory
        noise_std: Standard deviation of measurement noise
        system_type: Type of system ("pendulum", "duffing", "vdp")
    
    Returns:
        Tuple of (inputs, outputs)
    """
    np.random.seed(42)
    
    inputs_list = []
    outputs_list = []
    
    for _ in range(n_samples):
        if system_type == "pendulum":
            # Simple pendulum: x_dot = [x2, -sin(x1) - 0.1*x2 + u]
            x = np.zeros((sequence_length, 2))
            u = np.zeros((sequence_length, 1))
            
            # Random initial condition
            x[0] = np.random.randn(2) * 0.5
            
            # Generate input signal
            u[:, 0] = 0.5 * np.sin(2 * np.pi * np.linspace(0, 10, sequence_length))
            
            # Simulate system
            dt = 0.01
            for t in range(1, sequence_length):
                x1, x2 = x[t-1]
                u_t = u[t-1, 0]
                
                x1_dot = x2
                x2_dot = -np.sin(x1) - 0.1 * x2 + u_t
                
                x[t, 0] = x1 + dt * x1_dot
                x[t, 1] = x2 + dt * x2_dot
            
            # Output is the angle (first state)
            y = x[:, 0:1]
            
        elif system_type == "duffing":
            # Duffing oscillator: x_dot = [x2, -0.5*x2 - x1 - x1^3 + u]
            x = np.zeros((sequence_length, 2))
            u = np.zeros((sequence_length, 1))
            
            x[0] = np.random.randn(2) * 0.5
            u[:, 0] = np.random.randn(sequence_length) * 0.3
            
            dt = 0.01
            for t in range(1, sequence_length):
                x1, x2 = x[t-1]
                u_t = u[t-1, 0]
                
                x1_dot = x2
                x2_dot = -0.5 * x2 - x1 - x1**3 + u_t
                
                x[t, 0] = x1 + dt * x1_dot
                x[t, 1] = x2 + dt * x2_dot
            
            y = x[:, 0:1]
            
        elif system_type == "vdp":
            # Van der Pol oscillator: x_dot = [x2, (1 - x1^2)*x2 - x1 + u]
            x = np.zeros((sequence_length, 2))
            u = np.zeros((sequence_length, 1))
            
            x[0] = np.random.randn(2) * 0.5
            u[:, 0] = 0.5 * np.sin(2 * np.pi * np.linspace(0, 5, sequence_length))
            
            dt = 0.01
            for t in range(1, sequence_length):
                x1, x2 = x[t-1]
                u_t = u[t-1, 0]
                
                x1_dot = x2
                x2_dot = (1 - x1**2) * x2 - x1 + u_t
                
                x[t, 0] = x1 + dt * x1_dot
                x[t, 1] = x2 + dt * x2_dot
            
            y = x[:, 0:1]
        
        # Add measurement noise
        y = y + np.random.randn(*y.shape) * noise_std
        
        inputs_list.append(u)
        outputs_list.append(y)
    
    inputs = np.array(inputs_list)
    outputs = np.array(outputs_list)
    
    return inputs, outputs


def save_data(inputs, outputs, prefix, output_dir):
    """Save data to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = inputs.shape[0]
    seq_len = inputs.shape[1]
    
    # Flatten sequences for CSV format
    all_data = []
    for i in range(n_samples):
        for t in range(seq_len):
            row = list(inputs[i, t]) + list(outputs[i, t])
            all_data.append(row)
    
    # Save to CSV
    csv_path = output_dir / f"{prefix}.csv"
    np.savetxt(csv_path, all_data, delimiter=",")
    print(f"Saved {n_samples} sequences to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for system identification")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--n-train", type=int, default=800, help="Number of training samples")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test samples")
    parser.add_argument("--seq-length", type=int, default=100, help="Sequence length")
    parser.add_argument("--noise", type=float, default=0.01, help="Measurement noise std")
    parser.add_argument("--system", type=str, default="pendulum", 
                       choices=["pendulum", "duffing", "vdp"],
                       help="System type")
    args = parser.parse_args()
    
    print(f"Generating {args.system} system data...")
    print(f"  Train: {args.n_train} samples")
    print(f"  Val: {args.n_val} samples")
    print(f"  Test: {args.n_test} samples")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Noise std: {args.noise}")
    
    # Generate training data
    train_inputs, train_outputs = generate_nonlinear_system_data(
        n_samples=args.n_train,
        sequence_length=args.seq_length,
        noise_std=args.noise,
        system_type=args.system,
    )
    save_data(train_inputs, train_outputs, "train", args.output_dir)
    
    # Generate validation data
    val_inputs, val_outputs = generate_nonlinear_system_data(
        n_samples=args.n_val,
        sequence_length=args.seq_length,
        noise_std=args.noise,
        system_type=args.system,
    )
    save_data(val_inputs, val_outputs, "val", args.output_dir)
    
    # Generate test data
    test_inputs, test_outputs = generate_nonlinear_system_data(
        n_samples=args.n_test,
        sequence_length=args.seq_length,
        noise_std=args.noise,
        system_type=args.system,
    )
    save_data(test_inputs, test_outputs, "test", args.output_dir)
    
    print(f"\nData generation complete!")
    print(f"Data saved to {args.output_dir}/")
    print(f"\nTo train a model, run:")
    print(f"  python scripts/train.py --config configs/example_config.yaml")


if __name__ == "__main__":
    main()
