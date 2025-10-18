"""
Basic smoke test to ensure modified classifiers can be imported and run.
"""
import numpy as np
from qml_models import (
    MulticlassQuantumClassifierDR,
    ClassicalReadoutHead
)


def test_classical_readout_head():
    """Test that ClassicalReadoutHead works."""
    readout = ClassicalReadoutHead(input_size=3, output_size=3, hidden_size=8, activation='tanh')
    
    # Test forward pass
    x = np.array([0.1, 0.2, 0.3])
    params = readout.get_params()
    output = readout.forward(x, *params)
    
    assert output.shape == (3,), "Output shape should match output_size"
    
    # Test state dict
    state_dict = readout.get_state_dict()
    assert 'w1' in state_dict
    assert 'b1' in state_dict
    assert 'w2' in state_dict
    assert 'b2' in state_dict


def test_multiclass_quantum_classifier_basic():
    """Test that MulticlassQuantumClassifierDR can be instantiated and fit."""
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.rand(10, 4)  # 10 samples, 4 features
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # 3 classes
    
    # Create classifier
    clf = MulticlassQuantumClassifierDR(
        n_qubits=4,
        n_layers=2,
        n_classes=3,
        learning_rate=0.1,
        steps=5,  # Very few steps for smoke test
        verbose=False,
        hidden_size=8,
        activation='tanh',
        resume_mode='none'
    )
    
    # Fit
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_train)
    assert y_pred.shape == y_train.shape, "Prediction shape should match target shape"
    
    # Predict proba
    y_proba = clf.predict_proba(X_train)
    assert y_proba.shape == (10, 3), "Probability shape should be (n_samples, n_classes)"
    assert np.allclose(y_proba.sum(axis=1), 1.0, atol=0.01), "Probabilities should sum to 1"


def test_checkpoint_resume():
    """Test that checkpoint saving and resuming works."""
    import tempfile
    import shutil
    
    # Create temporary checkpoint directory
    checkpoint_dir = tempfile.mkdtemp()
    
    try:
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.rand(10, 4)
        y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        # Train for a few steps
        clf1 = MulticlassQuantumClassifierDR(
            n_qubits=4,
            n_layers=2,
            n_classes=3,
            learning_rate=0.1,
            steps=3,
            verbose=False,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
            hidden_size=8,
            resume_mode='none'
        )
        clf1.fit(X_train, y_train)
        
        # Resume training
        clf2 = MulticlassQuantumClassifierDR(
            n_qubits=4,
            n_layers=2,
            n_classes=3,
            learning_rate=0.1,
            steps=5,
            verbose=False,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
            hidden_size=8,
            resume_mode='auto'
        )
        clf2.fit(X_train, y_train)
        
        # Check that training completed (we don't check epoch_history since it may be reset on resume)
        y_pred = clf2.predict(X_train)
        assert y_pred.shape == y_train.shape, "Should be able to predict after resume"
        
    finally:
        # Clean up
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == '__main__':
    print("Running smoke tests...")
    test_classical_readout_head()
    print("✓ ClassicalReadoutHead test passed")
    
    test_multiclass_quantum_classifier_basic()
    print("✓ MulticlassQuantumClassifierDR basic test passed")
    
    test_checkpoint_resume()
    print("✓ Checkpoint resume test passed")
    
    print("\nAll smoke tests passed!")
