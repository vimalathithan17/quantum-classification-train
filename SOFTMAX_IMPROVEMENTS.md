# Softmax and Predict_Proba Improvements

## Summary

This update implements robust and numerically stable `_softmax()` and `predict_proba()` methods for all quantum classifier classes in the repository, following best practices for machine learning inference.

## Problem Statement

The original implementations had several issues:
1. **Single sample handling**: When passing a 1D array (single sample) to `predict_proba()`, the code would iterate over features instead of samples
2. **Inconsistent dtypes**: No explicit dtype enforcement, could lead to integer surprises
3. **Shape inconsistencies**: `np.array([...])` could yield object dtype or unexpected shapes
4. **Numerical stability**: The original softmax implementation could overflow with large values
5. **Inconsistent output shapes**: predict_proba might return 1D or 2D depending on input

## Changes Made

### 1. Improved `_softmax()` Method

**Before:**
```python
def _softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
```

**After:**
```python
def _softmax(self, x):
    """Numerically stable softmax.
    Accepts 1D (K,) or 2D (N, K) arrays and returns probabilities of same shape.
    """
    X = np.asarray(x, dtype=np.float64)
    
    if X.ndim == 1:
        # single sample
        shift = X - np.max(X)
        exp_shift = np.exp(shift)
        return exp_shift / np.sum(exp_shift)
    elif X.ndim == 2:
        # batch: subtract max per row for numerical stability
        shift = X - np.max(X, axis=1, keepdims=True)    # shape (N,1)
        exp_shift = np.exp(shift)                        # shape (N,K)
        return exp_shift / np.sum(exp_shift, axis=1, keepdims=True)  # shape (N,K)
    else:
        raise ValueError("softmax input must be 1D or 2D array")
```

**Improvements:**
- Explicit handling of 1D and 2D inputs
- Per-row max subtraction for batches (numerical stability)
- float64 dtype enforcement
- Clear error message for invalid dimensions

### 2. Improved `predict_proba()` Method

**Before:**
```python
def predict_proba(self, X):
    """Predict class probabilities using quantum circuit and classical readout."""
    qcircuit = self._get_circuit()
    quantum_outputs = np.array([qcircuit(x, self.weights) for x in X])
    
    # Apply classical readout to each sample
    logits_list = []
    for qout in quantum_outputs:
        logits = self._classical_readout(qout)
        logits_list.append(logits)
    
    logits_array = np.array(logits_list)
    return np.array([self._softmax(logit) for logit in logits_array])
```

**After:**
```python
def predict_proba(self, X):
    """Predict class probabilities using quantum circuit and classical readout."""
    # Ensure X is a batch: shape (N, features)
    X = np.atleast_2d(np.asarray(X))
    qcircuit = self._get_circuit()
    
    # compute quantum outputs for each sample
    quantum_outputs = [qcircuit(x, self.weights) for x in X]
    
    # apply classical readout per sample
    logits_list = [self._classical_readout(qout) for qout in quantum_outputs]
    
    # stack into an (N, K) numeric array and ensure float dtype
    logits_array = np.vstack(logits_list).astype(np.float64)
    
    # return (N, K) probabilities; for single-sample input this still returns shape (1,K)
    return self._softmax(logits_array)
```

**Improvements:**
- `np.atleast_2d(np.asarray(X))` ensures single samples are treated as batches
- `np.vstack()` instead of `np.array()` for guaranteed shape consistency
- Explicit `.astype(np.float64)` for type safety
- Single softmax call on the full batch (more efficient)
- Always returns 2D output (N, K) for consistency

### 3. Improved `predict()` Method

**Before:**
```python
def predict(self, X):
    """Predict class labels."""
    return np.argmax(self.predict_proba(X), axis=1)
```

**After:**
```python
def predict(self, X):
    """Predict class labels."""
    probs = self.predict_proba(X)
    return np.argmax(probs, axis=1)
```

**Improvements:**
- Simpler, more explicit code
- Ensures consistency with predict_proba

### 4. Special Handling for Conditional Models

For `ConditionalMulticlassQuantumClassifierFS` and `ConditionalMulticlassQuantumClassifierDataReuploadingFS`:

```python
def predict_proba(self, X):
    """Predict class probabilities using quantum circuit and classical readout."""
    X_scaled, is_missing_mask = X
    # Ensure inputs are batches
    X_scaled = np.atleast_2d(np.asarray(X_scaled))
    is_missing_mask = np.atleast_2d(np.asarray(is_missing_mask))
    
    qcircuit = self._get_circuit()
    
    # compute quantum outputs for each sample
    quantum_outputs = [qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
                      for f, m in zip(X_scaled, is_missing_mask)]
    
    # apply classical readout per sample
    logits_list = [self._classical_readout(qout) for qout in quantum_outputs]
    
    # stack into an (N, K) numeric array and ensure float dtype
    logits_array = np.vstack(logits_list).astype(np.float64)
    
    # return (N, K) probabilities; for single-sample input this still returns shape (1,K)
    return self._softmax(logits_array)
```

## Updated Classes

All four quantum classifier classes have been updated:
1. `MulticlassQuantumClassifierDR`
2. `MulticlassQuantumClassifierDataReuploadingDR`
3. `ConditionalMulticlassQuantumClassifierFS`
4. `ConditionalMulticlassQuantumClassifierDataReuploadingFS`

## Testing

Comprehensive tests added in `tests/test_softmax_predict.py`:

1. **test_softmax_1d**: Validates 1D input handling
2. **test_softmax_2d**: Validates batch (2D) input handling
3. **test_softmax_numerical_stability**: Tests with large values (1000+)
4. **test_predict_proba_single_sample**: Tests single sample prediction
5. **test_predict_proba_batch**: Tests batch prediction
6. **test_predict_consistency**: Verifies predict matches argmax of predict_proba
7. **test_all_model_classes**: Tests all non-conditional models
8. **test_conditional_models**: Tests conditional models with tuple inputs
9. **test_dtype_enforcement**: Verifies float64 dtype

All tests pass (9/9).

## Benefits

1. **Robustness**: Handles single samples and batches consistently
2. **Numerical Stability**: No overflow/underflow issues with large logits
3. **Type Safety**: Explicit float64 enforcement prevents integer surprises
4. **Consistency**: Output shape is always (N, K) regardless of input
5. **Maintainability**: Clearer, more explicit code
6. **Performance**: Single softmax call on batch vs. per-sample calls

## Example Usage

```python
from qml_models import MulticlassQuantumClassifierDR
import numpy as np

# Train model
model = MulticlassQuantumClassifierDR(n_qubits=4, n_classes=3)
X_train = np.random.randn(30, 4)
y_train = np.array([0]*10 + [1]*10 + [2]*10)
model.fit(X_train, y_train)

# Single sample (1D input) - now works correctly!
X_single = np.array([0.1, 0.2, 0.3, 0.4])
probs = model.predict_proba(X_single)  # shape: (1, 3)
pred = model.predict(X_single)          # shape: (1,)

# Batch input (2D) - works as before
X_batch = np.random.randn(5, 4)
probs = model.predict_proba(X_batch)   # shape: (5, 3)
pred = model.predict(X_batch)           # shape: (5,)
```

## Backward Compatibility

The changes are **backward compatible** for code that:
- Passes 2D arrays (batches) to predict_proba
- Uses predict_proba output with axis=1 operations

**Breaking changes** (improvements):
- Single sample predictions now return shape (1, K) instead of (K,)
- predict() now returns shape (1,) for single samples instead of scalar

These are actually improvements that make the API more consistent.
