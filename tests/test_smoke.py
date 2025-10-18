"""
Smoke tests for quantum classifier training improvements.

These tests verify basic functionality without requiring full training:
- Import modules successfully
- Instantiate models
- Run a few training steps on small synthetic data
- Save and load checkpoints
- Resume training
"""
import os
import sys
import tempfile
import shutil
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all new modules can be imported."""
    try:
        from utils.optim_adam import SerializableAdam
        from utils.io_checkpoint import (
            save_checkpoint, load_checkpoint, 
            save_best_and_latest_checkpoints,
            find_latest_checkpoint, find_best_checkpoint
        )
        from qml_models import MulticlassQuantumClassifierDR
        print("✓ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_serializable_adam():
    """Test the SerializableAdam optimizer."""
    from utils.optim_adam import SerializableAdam
    import pennylane.numpy as pnp
    
    # Create optimizer
    opt = SerializableAdam(lr=0.01)
    
    # Create simple cost function
    def cost(x):
        return (x - 3.0) ** 2
    
    # Initialize parameter
    x = pnp.array(0.0, requires_grad=True)
    
    # Perform a few steps
    for i in range(5):
        x, loss = opt.step_and_cost(cost, x)
    
    # x should have moved closer to 3.0
    assert x > 0.0, "Parameter should have increased"
    
    # Test state save/load
    state = opt.get_state()
    assert 'm' in state
    assert 'v' in state
    assert 't' in state
    assert state['t'] == 5
    
    # Create new optimizer and restore state
    opt2 = SerializableAdam(lr=0.01)
    opt2.set_state(state)
    assert opt2.t == 5
    
    print("✓ SerializableAdam works correctly")


def test_checkpoint_io():
    """Test checkpoint save and load functionality."""
    from utils.io_checkpoint import save_checkpoint, load_checkpoint
    import tempfile
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.joblib') as f:
        temp_path = f.name
    
    try:
        # Save checkpoint
        model_params = {'weights': np.random.randn(3, 4)}
        classical_params = {'W1': np.random.randn(2, 2), 'b1': np.zeros(2)}
        optimizer_state = {'m': {}, 'v': {}, 't': 10}
        metadata = {'epoch': 5, 'loss': 0.123}
        
        save_checkpoint(temp_path, model_params, classical_params, 
                       optimizer_state, None, metadata, None)
        
        # Load checkpoint
        loaded = load_checkpoint(temp_path)
        
        assert 'model_params' in loaded
        assert 'classical_params' in loaded
        assert 'optimizer_state' in loaded
        assert 'metadata' in loaded
        assert loaded['metadata']['epoch'] == 5
        
        print("✓ Checkpoint I/O works correctly")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_quantum_classifier_basic():
    """Test basic quantum classifier instantiation and forward pass."""
    # Skip if pennylane not installed
    try:
        import pennylane as qml
    except ImportError:
        pytest.skip("PennyLane not installed")
        return
    
    from qml_models import MulticlassQuantumClassifierDR
    
    # Create small synthetic dataset
    np.random.seed(42)
    n_samples = 10
    n_features = 4
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create model
    model = MulticlassQuantumClassifierDR(
        n_qubits=4,
        n_layers=2,
        n_classes=n_classes,
        learning_rate=0.01,
        steps=3,  # Very few steps for smoke test
        verbose=False,
        hidden_dim=8
    )
    
    # Fit model (just a few steps)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    assert y_pred.shape == (n_samples,)
    assert all(0 <= p < n_classes for p in y_pred)
    
    # Get probabilities
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (n_samples, n_classes)
    assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)
    
    print("✓ Quantum classifier basic functionality works")


def test_checkpoint_save_load_model():
    """Test saving and loading model checkpoints."""
    # Skip if pennylane not installed
    try:
        import pennylane as qml
    except ImportError:
        pytest.skip("PennyLane not installed")
        return
    
    from qml_models import MulticlassQuantumClassifierDR
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create small dataset
        np.random.seed(42)
        n_samples = 10
        n_features = 4
        n_classes = 3
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Create and train model
        model = MulticlassQuantumClassifierDR(
            n_qubits=4,
            n_layers=2,
            n_classes=n_classes,
            learning_rate=0.01,
            steps=5,
            verbose=False,
            checkpoint_dir=temp_dir,
            checkpoint_frequency=2
        )
        
        model.fit(X, y)
        
        # Check that checkpoints were created
        latest_checkpoint = os.path.join(temp_dir, 'checkpoint_latest.joblib')
        best_checkpoint = os.path.join(temp_dir, 'checkpoint_best.joblib')
        
        assert os.path.exists(latest_checkpoint), "Latest checkpoint should exist"
        assert os.path.exists(best_checkpoint), "Best checkpoint should exist"
        
        print("✓ Model checkpoint saving works")
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_resume_training():
    """Test resuming training from checkpoint."""
    # Skip if pennylane not installed
    try:
        import pennylane as qml
    except ImportError:
        pytest.skip("PennyLane not installed")
        return
    
    from qml_models import MulticlassQuantumClassifierDR
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create small dataset
        np.random.seed(42)
        n_samples = 10
        n_features = 4
        n_classes = 3
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        # First training session
        model1 = MulticlassQuantumClassifierDR(
            n_qubits=4,
            n_layers=2,
            n_classes=n_classes,
            learning_rate=0.01,
            steps=3,
            verbose=False,
            checkpoint_dir=temp_dir,
            checkpoint_frequency=1
        )
        
        model1.fit(X, y)
        loss1 = model1.best_loss
        
        # Second training session (resume)
        model2 = MulticlassQuantumClassifierDR(
            n_qubits=4,
            n_layers=2,
            n_classes=n_classes,
            learning_rate=0.01,
            steps=5,  # More steps
            verbose=False,
            checkpoint_dir=temp_dir,
            checkpoint_frequency=1
        )
        
        model2.fit(X, y, resume='auto')
        loss2 = model2.best_loss
        
        # Loss should be same or better (or similar due to randomness in short training)
        # Just check that resume didn't crash
        assert loss2 is not None
        
        print("✓ Resume training works")
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all smoke tests."""
    print("Running smoke tests...")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("SerializableAdam", test_serializable_adam),
        ("Checkpoint I/O", test_checkpoint_io),
        ("Quantum Classifier Basic", test_quantum_classifier_basic),
        ("Checkpoint Save/Load", test_checkpoint_save_load_model),
        ("Resume Training", test_resume_training),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            test_func()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"  ⊘ SKIPPED: {e}")
            skipped += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
