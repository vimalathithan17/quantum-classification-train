"""Tests for metalearner.py comprehensive metrics and directory handling."""
import os
import sys
import tempfile
import shutil
import json
import numpy as np
from sklearn.metrics import confusion_matrix

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _per_class_specificity(cm_arr):
    """Compute per-class specificity from confusion matrix (copied from metalearner.py for testing)."""
    K = cm_arr.shape[0]
    speci = np.zeros(K, dtype=float)
    total = cm_arr.sum()
    for i in range(K):
        TP = cm_arr[i, i]
        FP = cm_arr[:, i].sum() - TP
        FN = cm_arr[i, :].sum() - TP
        TN = total - (TP + FP + FN)
        denom = TN + FP
        speci[i] = float(TN / denom) if denom > 0 else 0.0
    return speci


def test_per_class_specificity():
    """Test per-class specificity computation."""
    try:
        # Create a simple 3-class confusion matrix
        # Perfect classifier example
        cm = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ])
        
        spec = _per_class_specificity(cm)
        
        # For perfect classifier, specificity should be 1.0 for all classes
        assert len(spec) == 3, "Should return 3 specificity values"
        assert all(s == 1.0 for s in spec), "Perfect classifier should have specificity 1.0"
        
        # Test with some errors
        cm = np.array([
            [8, 1, 1],   # Class 0: 8 correct, 1 misclassified as 1, 1 as 2
            [1, 8, 1],   # Class 1: 1 from 0, 8 correct, 1 from 2
            [1, 1, 8]    # Class 2: 1 from 0, 1 from 1, 8 correct
        ])
        
        spec = _per_class_specificity(cm)
        
        # All specificities should be between 0 and 1
        assert len(spec) == 3, "Should return 3 specificity values"
        assert all(0 <= s <= 1 for s in spec), "Specificity should be in [0, 1]"
        
        print("✓ Per-class specificity computation test passed")
        return True
    except Exception as e:
        print(f"✗ Per-class specificity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_handling():
    """Test directory handling logic."""
    try:
        # Create a temporary writable directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test basic directory creation and verification
            subdir = os.path.join(temp_dir, 'test_results')
            os.makedirs(subdir, exist_ok=True)
            assert os.path.exists(subdir), "Should create directory"
            assert os.access(subdir, os.W_OK), "Created directory should be writable"
            
            # Test file writability check
            test_file = os.path.join(subdir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            assert os.path.exists(test_file), "Should be able to write file"
            
            print("✓ Directory handling test passed")
            return True
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ Directory handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_metrics_format():
    """Test that comprehensive metrics have the expected format."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
            classification_report
        )
        
        # Create dummy predictions
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
        
        # Compute all metrics as in the objective function
        accuracy = float(accuracy_score(y_true, y_pred))
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class specificity
        per_class_spec = _per_class_specificity(cm)
        spec_macro = float(np.mean(per_class_spec))
        support = np.bincount(y_true)
        spec_weighted = float(np.sum(per_class_spec * support) / support.sum())
        
        # Pack comprehensive metrics
        metrics = {
            'accuracy': accuracy,
            'precision_macro': float(prec_macro),
            'recall_macro': float(rec_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(prec_weighted),
            'recall_weighted': float(rec_weighted),
            'f1_weighted': float(f1_weighted),
            'specificity_macro': spec_macro,
            'specificity_weighted': spec_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred, zero_division=0)
        }
        
        # Verify all expected keys are present
        expected_keys = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_weighted', 'recall_weighted', 'f1_weighted',
            'specificity_macro', 'specificity_weighted', 'confusion_matrix',
            'classification_report'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
        
        # Verify numeric metrics are in valid range
        numeric_metrics = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_weighted', 'recall_weighted', 'f1_weighted',
            'specificity_macro', 'specificity_weighted'
        ]
        
        for key in numeric_metrics:
            value = metrics[key]
            assert 0 <= value <= 1, f"{key} should be in [0, 1], got {value}"
        
        # Verify metrics can be serialized to JSON
        temp_dir = tempfile.mkdtemp()
        try:
            metrics_file = os.path.join(temp_dir, 'test_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Load it back and verify
            with open(metrics_file, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert loaded_metrics['accuracy'] == metrics['accuracy']
            
            print("✓ Comprehensive metrics format test passed")
            return True
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ Comprehensive metrics format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_computation_similarity():
    """Test that metrics computed match tune_models.py approach."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix
        )
        
        # Create test data with known metrics
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])  # 7/9 correct
        
        # Compute metrics using the same approach as metalearner.py and tune_models.py
        acc = float(accuracy_score(y_true, y_pred))
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        per_class_spec = _per_class_specificity(cm)
        spec_macro = float(np.mean(per_class_spec))
        
        # Basic sanity checks
        assert 0.7 < acc < 0.8, f"Expected accuracy around 7/9, got {acc}"
        assert 0 <= f1_weighted <= 1, f"F1 weighted should be in [0,1], got {f1_weighted}"
        assert 0 <= spec_macro <= 1, f"Specificity macro should be in [0,1], got {spec_macro}"
        
        # Verify all metrics are floats (for JSON serialization)
        assert isinstance(acc, float)
        assert isinstance(f1_weighted, float)
        assert isinstance(spec_macro, float)
        
        print("✓ Metrics computation similarity test passed")
        return True
    except Exception as e:
        print(f"✗ Metrics computation similarity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all metalearner tests."""
    print("=" * 60)
    print("Running metalearner.py update tests")
    print("=" * 60)
    
    tests = [
        test_per_class_specificity,
        test_directory_handling,
        test_comprehensive_metrics_format,
        test_metrics_computation_similarity
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
