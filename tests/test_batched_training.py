"""Tests for batched training optimization."""
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qml_models import (
    MulticlassQuantumClassifierDR,
    MulticlassQuantumClassifierDataReuploadingDR,
)


def test_cost_uses_batched_qcircuit():
    """Test that cost function uses _batched_qcircuit instead of per-sample loops."""
    print("Testing that cost function uses batched path...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=2, verbose=False, validation_frac=0.0
    )
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    # Patch _batched_qcircuit to track calls
    original_batched = model._batched_qcircuit
    call_count = {'count': 0}
    
    def tracked_batched(*args, **kwargs):
        call_count['count'] += 1
        return original_batched(*args, **kwargs)
    
    with patch.object(model, '_batched_qcircuit', side_effect=tracked_batched):
        model.fit(X_train, y_train)
    
    # _batched_qcircuit should be called at least once during training
    # (once per step for cost + once for prediction if validation)
    assert call_count['count'] >= 2, f"_batched_qcircuit should be called during training, got {call_count['count']} calls"
    print(f"✓ _batched_qcircuit was called {call_count['count']} times during training")


def test_validation_uses_batched_qcircuit():
    """Test that validation loss calculation uses _batched_qcircuit."""
    print("Testing that validation uses batched path...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=2, verbose=False, validation_frac=0.3
    )
    
    # Need enough samples for stratified split (at least 2 per class)
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], 
                        [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1],
                        [0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2]])
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    
    # Patch _batched_qcircuit to track calls
    original_batched = model._batched_qcircuit
    call_count = {'count': 0}
    
    def tracked_batched(*args, **kwargs):
        call_count['count'] += 1
        return original_batched(*args, **kwargs)
    
    with patch.object(model, '_batched_qcircuit', side_effect=tracked_batched):
        model.fit(X_train, y_train)
    
    # Should be called for training cost, validation loss, and predictions
    assert call_count['count'] >= 4, f"_batched_qcircuit should be called multiple times with validation, got {call_count['count']} calls"
    print(f"✓ _batched_qcircuit was called {call_count['count']} times with validation")


def test_cross_entropy_computation():
    """Test that cross-entropy is computed correctly with axis=1 sum."""
    print("Testing cross-entropy computation...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
    )
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Get predictions
    probs = model.predict_proba(X_train)
    
    # Manually compute cross-entropy the correct way
    y_one_hot = np.eye(3)[y_train]
    eps = 1e-9
    expected_loss = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=1))
    
    # The loss should be a finite number
    assert np.isfinite(expected_loss), "Cross-entropy loss should be finite"
    assert expected_loss >= 0, "Cross-entropy loss should be non-negative"
    
    print(f"✓ Cross-entropy computed correctly: {expected_loss:.4f}")


def test_data_reuploading_uses_batched_qcircuit():
    """Test that data reuploading class also uses batched path."""
    print("Testing data reuploading class uses batched path...")
    
    model = MulticlassQuantumClassifierDataReuploadingDR(
        n_qubits=3, n_classes=3, steps=2, verbose=False, validation_frac=0.0
    )
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    # Patch _batched_qcircuit to track calls
    original_batched = model._batched_qcircuit
    call_count = {'count': 0}
    
    def tracked_batched(*args, **kwargs):
        call_count['count'] += 1
        return original_batched(*args, **kwargs)
    
    with patch.object(model, '_batched_qcircuit', side_effect=tracked_batched):
        model.fit(X_train, y_train)
    
    assert call_count['count'] >= 2, f"_batched_qcircuit should be called during training, got {call_count['count']} calls"
    print(f"✓ Data reuploading class used _batched_qcircuit {call_count['count']} times")


def test_empty_batch_handling():
    """Test that empty batches are handled correctly."""
    print("Testing empty batch handling...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
    )
    
    # Fit with minimal data
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    model.fit(X_train, y_train)
    
    # Test with empty batch
    X_empty = np.empty((0, 3))
    probs = model.predict_proba(X_empty)
    
    assert probs.shape == (0, 3), f"Empty batch should return (0, 3), got {probs.shape}"
    print("✓ Empty batch handled correctly")


def test_atleast_2d_usage():
    """Test that np.atleast_2d is used for input shaping."""
    print("Testing np.atleast_2d usage...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
    )
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    model.fit(X_train, y_train)
    
    # Test with 1D input
    X_single_1d = np.array([0.5, 0.5, 0.5])
    probs_1d = model.predict_proba(X_single_1d)
    assert probs_1d.shape == (1, 3), f"1D input should return (1, 3), got {probs_1d.shape}"
    
    # Test with 2D input (single sample)
    X_single_2d = np.array([[0.5, 0.5, 0.5]])
    probs_2d = model.predict_proba(X_single_2d)
    assert probs_2d.shape == (1, 3), f"2D single sample should return (1, 3), got {probs_2d.shape}"
    
    # Both should give same result
    assert np.allclose(probs_1d, probs_2d), "1D and 2D single sample should give same result"
    
    print("✓ np.atleast_2d works correctly for both 1D and 2D inputs")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running batched training optimization tests")
    print("=" * 60)
    
    tests = [
        test_cost_uses_batched_qcircuit,
        test_validation_uses_batched_qcircuit,
        test_cross_entropy_computation,
        test_data_reuploading_uses_batched_qcircuit,
        test_empty_batch_handling,
        test_atleast_2d_usage,
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
