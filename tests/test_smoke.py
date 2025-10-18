"""
Smoke tests to verify basic functionality of quantum classifiers.
Tests construction, training for a few steps, and checkpoint creation.
"""
import os
import sys
from pennylane import numpy as np  # Use pennylane numpy for requires_grad support
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qml_models import MulticlassQuantumClassifierDR
from utils.optim_adam import SerializableAdam
from utils.io_checkpoint import save_checkpoint, load_checkpoint


def test_basic_classifier_construction():
    """Test that we can construct a classifier."""
    print("Test: Basic classifier construction...")
    model = MulticlassQuantumClassifierDR(
        n_qubits=4,
        n_layers=2,
        n_classes=3,
        learning_rate=0.01,
        steps=5,
        verbose=False,
        use_classical_readout=True,
        hidden_size=8
    )
    assert model is not None
    assert model.n_qubits == 4
    assert model.n_classes == 3
    assert model.use_classical_readout == True
    print("✓ Classifier constructed successfully")


def test_classifier_fit():
    """Test that we can fit a classifier on tiny random data."""
    print("\nTest: Classifier fit on random data...")
    
    # Create tiny random dataset
    np.random.seed(42)
    n_samples = 20
    n_features = 4
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Create temp checkpoint directory
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MulticlassQuantumClassifierDR(
            n_qubits=n_features,
            n_layers=2,
            n_classes=n_classes,
            learning_rate=0.05,
            steps=5,
            verbose=False,
            checkpoint_dir=tmpdir,
            checkpoint_frequency=2,
            use_classical_readout=True,
            hidden_size=8
        )
        
        # Fit model
        model.fit(X, y, validation_frac=0.2, resume=None)
        
        # Check that we can predict
        y_pred = model.predict(X)
        assert y_pred.shape[0] == n_samples
        assert all(0 <= p < n_classes for p in y_pred)
        
        # Check that checkpoint files were created
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith('.joblib')]
        assert len(checkpoint_files) > 0, "No checkpoint files created"
        
        print(f"✓ Classifier fitted successfully, {len(checkpoint_files)} checkpoints created")


def test_optimizer_state_serialization():
    """Test that optimizer state can be saved and loaded."""
    print("\nTest: Optimizer state serialization...")
    
    opt = SerializableAdam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    
    # Perform a few optimization steps
    params = np.random.randn(10)
    params = np.array(params, requires_grad=True)
    
    for _ in range(3):
        def cost(p):
            return np.sum(p ** 2)
        params, loss = opt.step_and_cost(cost, params)
    
    # Get state
    state = opt.get_state()
    assert state is not None
    assert 't' in state
    assert state['t'] == 3
    
    # Create new optimizer and restore state
    opt2 = SerializableAdam(lr=0.01)
    opt2.set_state(state)
    assert opt2.t == 3
    assert opt2.lr == 0.01
    
    print("✓ Optimizer state serialization works")


def test_checkpoint_io():
    """Test checkpoint saving and loading."""
    print("\nTest: Checkpoint I/O...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.joblib')
        
        # Create checkpoint data
        data = {
            'quantum_params': np.random.randn(5, 5),
            'classical_params': {
                'W1': np.random.randn(3, 8),
                'b1': np.random.randn(8),
                'W2': np.random.randn(8, 3),
                'b2': np.random.randn(3)
            },
            'optimizer_state': {'t': 10, 'lr': 0.01},
            'step': 10,
            'best_val_metric': 0.85
        }
        
        # Save checkpoint
        save_checkpoint(checkpoint_path, data)
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        loaded_data = load_checkpoint(checkpoint_path)
        assert loaded_data is not None
        assert loaded_data['step'] == 10
        assert loaded_data['best_val_metric'] == 0.85
        
        print("✓ Checkpoint I/O works")


def test_classifier_without_classical_readout():
    """Test classifier with classical readout disabled."""
    print("\nTest: Classifier without classical readout...")
    
    np.random.seed(42)
    n_samples = 15
    n_features = 4
    n_classes = 2
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=n_features,
        n_layers=2,
        n_classes=n_classes,
        learning_rate=0.05,
        steps=3,
        verbose=False,
        use_classical_readout=False  # Disabled
    )
    
    model.fit(X, y, validation_frac=0.0, resume=None)
    y_pred = model.predict(X)
    assert y_pred.shape[0] == n_samples
    
    print("✓ Classifier without classical readout works")


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running Smoke Tests")
    print("=" * 60)
    
    try:
        test_basic_classifier_construction()
        test_optimizer_state_serialization()
        test_checkpoint_io()
        test_classifier_without_classical_readout()
        test_classifier_fit()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
