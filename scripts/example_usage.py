#!/usr/bin/env python
"""
Example usage of enhanced quantum classifier with classical readout.

This script demonstrates:
- Training with enhanced features
- Checkpointing and resume
- Metrics visualization
- Using trained model for predictions
"""
import numpy as np
import os
import sys

# Ensure utils and models are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_synthetic_data(n_samples=100, n_features=8, n_classes=3, random_seed=42):
    """Generate synthetic classification data for demonstration."""
    np.random.seed(random_seed)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate random labels
    y = np.random.randint(0, n_classes, n_samples)
    
    # Add some class-dependent structure
    for c in range(n_classes):
        mask = y == c
        X[mask] += c * 0.5  # Shift each class
    
    return X, y


def main():
    print("="*70)
    print("Quantum Classifier with Classical Readout - Example Usage")
    print("="*70)
    print()
    
    # Check if dependencies are available
    try:
        from qml_models import MulticlassQuantumClassifierDR
        print("✓ Successfully imported MulticlassQuantumClassifierDR")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=50, n_features=8, n_classes=3)
    print(f"   Data shape: {X.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create checkpoint directory
    checkpoint_dir = './example_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\n2. Checkpoints will be saved to: {checkpoint_dir}")
    
    # Initialize model
    print("\n3. Initializing quantum classifier...")
    model = MulticlassQuantumClassifierDR(
        n_qubits=8,
        n_layers=2,
        n_classes=3,
        learning_rate=0.01,
        steps=20,  # Small number for demo
        hidden_dim=8,  # Classical readout hidden layer
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=5,
        verbose=True
    )
    print("   Model initialized with:")
    print(f"   - {model.n_qubits} qubits")
    print(f"   - {model.n_layers} layers")
    print(f"   - {model.hidden_dim}-dimensional classical readout head")
    
    # Train model
    print("\n4. Training model (first session)...")
    print("-" * 70)
    model.fit(X_train, y_train)
    print("-" * 70)
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Test accuracy: {accuracy:.3f}")
    
    # Show checkpoint files
    print("\n6. Checking saved files...")
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        print(f"   Found {len(files)} files in checkpoint directory:")
        for f in sorted(files):
            print(f"   - {f}")
    
    # Resume training
    print("\n7. Resuming training for more steps...")
    print("-" * 70)
    model_resumed = MulticlassQuantumClassifierDR(
        n_qubits=8,
        n_layers=2,
        n_classes=3,
        learning_rate=0.01,
        steps=30,  # More steps
        hidden_dim=8,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=5,
        verbose=True
    )
    model_resumed.fit(X_train, y_train, resume='auto')
    print("-" * 70)
    
    # Final evaluation
    print("\n8. Final evaluation after resume...")
    y_pred_final = model_resumed.predict(X_test)
    accuracy_final = np.mean(y_pred_final == y_test)
    print(f"   Test accuracy: {accuracy_final:.3f}")
    
    # Show predictions
    print("\n9. Example predictions (first 5 samples):")
    y_proba = model_resumed.predict_proba(X_test[:5])
    for i in range(5):
        print(f"   Sample {i}: True={y_test[i]}, Pred={y_pred_final[i]}, "
              f"Proba={y_proba[i].round(3)}")
    
    # Summary
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    print("\nKey features demonstrated:")
    print("✓ Classical readout head (MLP)")
    print("✓ Automatic checkpoint saving")
    print("✓ Resume training from checkpoints")
    print("✓ Metrics tracking and logging")
    print("✓ Prediction with probability outputs")
    print("\nCheck the following files for details:")
    print(f"- {checkpoint_dir}/metrics.csv - Training metrics")
    print(f"- {checkpoint_dir}/loss_plot.png - Loss visualization")
    print(f"- {checkpoint_dir}/metrics_plot.png - Metrics visualization")
    print(f"- {checkpoint_dir}/checkpoint_best.joblib - Best model checkpoint")
    print(f"- {checkpoint_dir}/checkpoint_latest.joblib - Latest checkpoint")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
