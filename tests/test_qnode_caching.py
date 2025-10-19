"""Tests for QNode caching and batch optimization improvements."""
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qml_models import (
    MulticlassQuantumClassifierDR,
    MulticlassQuantumClassifierDataReuploadingDR,
)


def test_qnode_cached_on_init():
    """Test that QNode is created and cached during __init__."""
    print("Testing QNode caching during initialization...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
    )
    
    # Check that _qcircuit attribute exists
    assert hasattr(model, '_qcircuit'), "Model should have _qcircuit attribute"
    assert model._qcircuit is not None, "_qcircuit should not be None"
    
    # Check that it's callable
    assert callable(model._qcircuit), "_qcircuit should be callable"
    
    print("✓ QNode is cached during initialization")


def test_qnode_reused_not_recreated():
    """Test that cached QNode is reused, not recreated on every call."""
    print("Testing that QNode is reused, not recreated...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=2, verbose=False, validation_frac=0.0
    )
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    
    # Store the original cached qnode
    cached_qnode = model._qcircuit
    
    # Mock _get_circuit to track if it's called
    original_get_circuit = model._get_circuit
    call_count = {'count': 0}
    
    def tracked_get_circuit():
        call_count['count'] += 1
        return original_get_circuit()
    
    with patch.object(model, '_get_circuit', side_effect=tracked_get_circuit):
        model.fit(X_train, y_train)
    
    # _get_circuit should not be called during training since qnode is cached
    # (it was called once during __init__)
    assert call_count['count'] == 0, f"_get_circuit should not be called during training, got {call_count['count']} calls"
    
    # The cached qnode should be the same object
    assert model._qcircuit is cached_qnode, "Cached QNode should be reused"
    
    print("✓ Cached QNode is reused, not recreated")


def test_empty_batch_handling_all_classes():
    """Test that empty batches are handled correctly for all model classes."""
    print("Testing empty batch handling for all classes...")
    
    classes_to_test = [
        MulticlassQuantumClassifierDR,
        MulticlassQuantumClassifierDataReuploadingDR,
    ]
    
    for model_class in classes_to_test:
        print(f"  Testing {model_class.__name__}...")
        model = model_class(
            n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
        )
        
        # Fit with minimal data
        X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        y_train = np.array([0, 1, 2])
        model.fit(X_train, y_train)
        
        # Test with empty batch
        X_empty = np.empty((0, 3))
        probs = model.predict_proba(X_empty)
        
        assert probs.shape == (0, 3), f"Empty batch should return (0, 3) for {model_class.__name__}, got {probs.shape}"
        
        # Test predictions on empty batch
        preds = model.predict(X_empty)
        assert preds.shape == (0,), f"Empty batch predictions should have shape (0,) for {model_class.__name__}, got {preds.shape}"
    
    print("✓ Empty batch handling works for all classes")


def test_debug_logging_on_fallback():
    """Test that debug logging is triggered when batched call fails."""
    print("Testing debug logging on fallback...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
    )
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([0, 1, 2])
    model.fit(X_train, y_train)
    
    # Create a mock qcircuit that raises an exception on batched call
    original_qcircuit = model._qcircuit
    
    def mock_qcircuit(X, weights):
        # Single sample works
        if X.ndim == 1 or (X.ndim == 2 and X.shape[0] == 1):
            return original_qcircuit(X.flatten() if X.ndim == 2 else X, weights)
        # Batched call fails
        raise ValueError("Batched call not supported")
    
    # Set up logging to capture debug messages
    with patch('qml_models.log') as mock_log:
        model._qcircuit = mock_qcircuit
        
        # Call _batched_qcircuit which should trigger fallback
        result = model._batched_qcircuit(X_train, model.weights)
        
        # Check that debug logging was called
        debug_calls = [call for call in mock_log.debug.call_args_list]
        assert len(debug_calls) >= 1, "Debug logging should be called during fallback"
        
        # Check that one of the debug messages mentions the fallback
        debug_messages = [str(call) for call in debug_calls]
        fallback_logged = any('fallback' in msg.lower() for msg in debug_messages)
        assert fallback_logged, "Debug logging should mention fallback"
        
        # Result should still be correct shape
        assert result.shape == (3, 3), f"Result shape should be (3, 3), got {result.shape}"
    
    print("✓ Debug logging works correctly on fallback")


def test_atleast_2d_input_validation():
    """Test that input validation with atleast_2d is applied in fit()."""
    print("Testing input validation with atleast_2d...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
    )
    
    # Test with various input shapes
    # 2D input (normal case)
    X_2d = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y = np.array([0, 1, 2])
    
    model.fit(X_2d, y)
    
    # Model should work correctly
    probs = model.predict_proba(X_2d)
    assert probs.shape == (3, 3), f"Should return (3, 3), got {probs.shape}"
    
    print("✓ Input validation with atleast_2d works correctly")


def test_all_classes_have_cached_qnode():
    """Test that all model classes cache the QNode during initialization."""
    print("Testing all classes have cached QNode...")
    
    classes_to_test = [
        MulticlassQuantumClassifierDR,
        MulticlassQuantumClassifierDataReuploadingDR,
    ]
    
    for model_class in classes_to_test:
        print(f"  Testing {model_class.__name__}...")
        model = model_class(
            n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
        )
        
        assert hasattr(model, '_qcircuit'), f"{model_class.__name__} should have _qcircuit attribute"
        assert model._qcircuit is not None, f"{model_class.__name__}._qcircuit should not be None"
        assert callable(model._qcircuit), f"{model_class.__name__}._qcircuit should be callable"
    
    print("✓ All classes have cached QNode")


def test_batched_qcircuit_uses_getattr():
    """Test that _batched_qcircuit uses getattr to access cached QNode."""
    print("Testing _batched_qcircuit uses getattr for cached QNode...")
    
    model = MulticlassQuantumClassifierDR(
        n_qubits=3, n_classes=3, steps=1, verbose=False, validation_frac=0.0
    )
    
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    y_train = np.array([0, 1])
    model.fit(X_train, y_train)
    
    # Remove the cached qnode to test getattr fallback
    cached_qnode = model._qcircuit
    delattr(model, '_qcircuit')
    
    # Mock _get_circuit to verify it's called as fallback
    call_count = {'count': 0}
    original_get_circuit = model._get_circuit
    
    def tracked_get_circuit():
        call_count['count'] += 1
        model._qcircuit = cached_qnode  # Restore it
        return cached_qnode
    
    with patch.object(model, '_get_circuit', side_effect=tracked_get_circuit):
        result = model._batched_qcircuit(X_train, model.weights)
    
    # _get_circuit should have been called once as fallback
    assert call_count['count'] == 1, f"_get_circuit should be called once as fallback, got {call_count['count']}"
    assert result.shape == (2, 3), f"Result shape should be (2, 3), got {result.shape}"
    
    print("✓ _batched_qcircuit correctly uses getattr with fallback")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running QNode caching and optimization tests")
    print("=" * 60)
    
    tests = [
        test_qnode_cached_on_init,
        test_qnode_reused_not_recreated,
        test_empty_batch_handling_all_classes,
        test_debug_logging_on_fallback,
        test_atleast_2d_input_validation,
        test_all_classes_have_cached_qnode,
        test_batched_qcircuit_uses_getattr,
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
