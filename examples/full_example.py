#!/usr/bin/env python
"""
Example usage of the quantum classification training framework.
This script demonstrates all major features.
"""
import numpy as np
import pandas as pd
import os
import sys

# Generate synthetic data for demonstration
def generate_synthetic_data(n_samples=200, n_features=8, n_classes=3, output_file='example_data.parquet'):
    """Generate synthetic classification data."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some structure
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if X[i, 0] > 0 and X[i, 1] > 0:
            y[i] = 0
        elif X[i, 0] < 0 and X[i, 1] > 0:
            y[i] = 1
        else:
            y[i] = 2
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=n_samples // 10, replace=False)
    y[noise_idx] = np.random.randint(0, n_classes, size=len(noise_idx))
    
    # Create DataFrame
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['class'] = [f'Class_{i}' for i in y]
    df['case_id'] = [f'sample_{i}' for i in range(n_samples)]
    
    # Save to parquet
    df.to_parquet(output_file)
    print(f"Generated synthetic data: {output_file}")
    print(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
    print(f"  Class distribution: {df['class'].value_counts().to_dict()}")
    
    return output_file


def example_basic_training():
    """Example 1: Basic training with default settings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Training")
    print("="*80)
    
    # Generate data
    data_file = generate_synthetic_data(n_samples=100, n_features=6)
    
    # Train with defaults
    cmd = f"""python scripts/train.py \\
        --data_file {data_file} \\
        --output_dir example_output/basic \\
        --n_qubits 6 \\
        --n_layers 2 \\
        --steps 30 \\
        --verbose"""
    
    print(f"\nCommand:\n{cmd}\n")
    os.system(cmd)
    
    print("\nBasic training complete! Check example_output/basic/ for results.")


if __name__ == '__main__':
    print("Quantum Classification Framework - Example Usage")
    print("="*80)
    print("\nThis is a simplified example. Full examples available in the script.")
    print("Run: python examples/full_example.py\n")
    
    example_basic_training()
