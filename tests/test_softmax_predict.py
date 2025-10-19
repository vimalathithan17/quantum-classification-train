"""Tests for improved softmax and predict_proba implementations."""
import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qml_models import (
    MulticlassQuantumClassifierDR,
    MulticlassQuantumClassifierDataReuploadingDR,
    ConditionalMulticlassQuantumClassifierFS,
    ConditionalMulticlassQuantumClassifierDataReuploadingFS
)


def test_softmax_1d():
    """Test softmax with 1D input (single sample)."""
    print("Testing softmax with 1D input...")
    model = MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1)
    
    # Test 1D input
    logits_1d = np.array([1.0, 2.0, 3.0])
    probs = model._softmax(logits_1d)
    
    assert probs.shape == (3,), f"Expected shape (3,), got {probs.shape}"
    assert np.allclose(np.sum(probs), 1.0), f"Probabilities should sum to 1, got {np.sum(probs)}"
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities should be in [0, 1]"
    print("✓ Softmax 1D test passed")


def test_softmax_2d():
    """Test softmax with 2D input (batch)."""
    print("Testing softmax with 2D input...")
    model = MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1)
    
    # Test 2D input
    logits_2d = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5], [-1.0, 0.0, 1.0]])
    probs = model._softmax(logits_2d)
    
    assert probs.shape == (3, 3), f"Expected shape (3, 3), got {probs.shape}"
    assert np.allclose(np.sum(probs, axis=1), 1.0), f"Each row should sum to 1, got {np.sum(probs, axis=1)}"
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities should be in [0, 1]"
    print("✓ Softmax 2D test passed")


def test_softmax_numerical_stability():
    """Test softmax numerical stability with large values."""
    print("Testing softmax numerical stability...")
    model = MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1)
    
    # Test with large values that could cause overflow
    logits_large = np.array([[1000.0, 1001.0, 1002.0], [500.0, 501.0, 499.0]])
    probs = model._softmax(logits_large)
    
    assert not np.any(np.isnan(probs)), "Softmax should not produce NaN values"
    assert not np.any(np.isinf(probs)), "Softmax should not produce Inf values"
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Each row should sum to 1"
    print("✓ Softmax numerical stability test passed")


def test_predict_proba_single_sample():
    """Test predict_proba with single sample input (1D)."""
    print("Testing predict_proba with single sample...")
    
    # Create a simple model with validation disabled
    model = MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0)
    
    # Create simple training data
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    # Fit the model (minimal training)
    model.fit(X_train, y_train)
    
    # Test with single sample as 1D array
    X_single_1d = np.array([0.5, 0.5, 0.5])
    probs = model.predict_proba(X_single_1d)
    
    assert probs.shape == (1, 3), f"Expected shape (1, 3) for single sample, got {probs.shape}"
    assert np.allclose(np.sum(probs), 1.0), f"Probabilities should sum to 1, got {np.sum(probs)}"
    print("✓ predict_proba single sample test passed")


def test_predict_proba_batch():
    """Test predict_proba with batch input (2D)."""
    print("Testing predict_proba with batch...")
    
    # Create a simple model with validation disabled
    model = MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0)
    
    # Create simple training data
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    # Fit the model (minimal training)
    model.fit(X_train, y_train)
    
    # Test with batch
    X_batch = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    probs = model.predict_proba(X_batch)
    
    assert probs.shape == (2, 3), f"Expected shape (2, 3) for batch, got {probs.shape}"
    assert np.allclose(np.sum(probs, axis=1), 1.0), f"Each row should sum to 1, got {np.sum(probs, axis=1)}"
    print("✓ predict_proba batch test passed")


def test_predict_consistency():
    """Test that predict returns correct class indices."""
    print("Testing predict consistency...")
    
    model = MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0)
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    model.fit(X_train, y_train)
    
    # Single sample
    X_single = np.array([0.5, 0.5, 0.5])
    pred_single = model.predict(X_single)
    probs_single = model.predict_proba(X_single)
    
    assert pred_single.shape == (1,), f"Expected shape (1,) for single prediction, got {pred_single.shape}"
    assert pred_single[0] == np.argmax(probs_single[0]), "Prediction should match argmax of probabilities"
    
    # Batch
    X_batch = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    pred_batch = model.predict(X_batch)
    probs_batch = model.predict_proba(X_batch)
    
    assert pred_batch.shape == (2,), f"Expected shape (2,) for batch prediction, got {pred_batch.shape}"
    for i in range(2):
        assert pred_batch[i] == np.argmax(probs_batch[i]), f"Prediction {i} should match argmax of probabilities"
    
    print("✓ predict consistency test passed")


def test_all_model_classes():
    """Test that all model classes have updated implementations."""
    print("Testing all model classes...")
    
    # Test each model class
    models = [
        MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0),
        MulticlassQuantumClassifierDataReuploadingDR(n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0),
    ]
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        
        # Test single sample
        X_single = np.array([0.5, 0.5, 0.5])
        probs = model.predict_proba(X_single)
        assert probs.shape == (1, 3), f"Model {i}: Expected shape (1, 3), got {probs.shape}"
        
        # Test batch
        X_batch = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        probs_batch = model.predict_proba(X_batch)
        assert probs_batch.shape == (2, 3), f"Model {i}: Expected shape (2, 3), got {probs_batch.shape}"
    
    print("✓ All model classes test passed")


def test_conditional_models():
    """Test conditional model classes with tuple input."""
    print("Testing conditional models...")
    
    # Test conditional models that expect tuple input (X_scaled, is_missing_mask)
    model = ConditionalMulticlassQuantumClassifierFS(n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0)
    
    X_scaled = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    is_missing = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    y_train = np.array([0, 1, 2])
    
    model.fit((X_scaled, is_missing), y_train)
    
    # Test single sample as 1D arrays
    X_single = np.array([0.5, 0.5, 0.5])
    mask_single = np.array([0, 0, 0])
    probs = model.predict_proba((X_single, mask_single))
    
    assert probs.shape == (1, 3), f"Expected shape (1, 3), got {probs.shape}"
    assert np.allclose(np.sum(probs), 1.0), "Probabilities should sum to 1"
    
    # Test batch
    X_batch = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    mask_batch = np.array([[0, 0, 0], [0, 0, 0]])
    probs_batch = model.predict_proba((X_batch, mask_batch))
    
    assert probs_batch.shape == (2, 3), f"Expected shape (2, 3), got {probs_batch.shape}"
    
    print("✓ Conditional models test passed")


def test_dtype_enforcement():
    """Test that logits are converted to float64."""
    print("Testing dtype enforcement...")
    
    model = MulticlassQuantumClassifierDR(n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0)
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    model.fit(X_train, y_train)
    
    X_test = np.array([[0.5, 0.5, 0.5]])
    probs = model.predict_proba(X_test)
    
    assert probs.dtype == np.float64, f"Expected dtype float64, got {probs.dtype}"
    print("✓ dtype enforcement test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running softmax and predict_proba tests")
    print("=" * 60)
    
    tests = [
        test_softmax_1d,
        test_softmax_2d,
        test_softmax_numerical_stability,
        test_predict_proba_single_sample,
        test_predict_proba_batch,
        test_predict_consistency,
        test_all_model_classes,
        test_conditional_models,
        test_dtype_enforcement,
    ]
    
    results = []
    for test in tests:
        print(f"\n{test.__name__}:")
        try:
            test()
            results.append(True)
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
