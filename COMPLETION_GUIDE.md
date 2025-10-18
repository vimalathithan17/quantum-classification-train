# Guide: Completing Remaining Classifier Updates

This guide provides step-by-step instructions for applying the same enhancements to the remaining 3 classifier classes in `qml_models.py`.

## Classes to Update

1. **MulticlassQuantumClassifierDataReuploadingDR** (starts at line ~374)
2. **ConditionalMulticlassQuantumClassifierFS** (starts at line ~476)
3. **ConditionalMulticlassQuantumClassifierDataReuploadingFS** (starts at line ~606)

## Reference Implementation

Use `MulticlassQuantumClassifierDR` (lines 27-373) as the complete reference. All code patterns and logic are implemented there.

## Step-by-Step Process for Each Classifier

### Step 1: Update `__init__` Method

**Add these parameters** to the `__init__` signature:
```python
hidden_size=16, 
readout_activation='tanh', 
selection_metric='weighted_f1',
resume=None, 
validation_frac=0.1, 
patience=None
```

**Store them** as instance variables:
```python
self.hidden_size = hidden_size
self.readout_activation = readout_activation
self.selection_metric = selection_metric
self.resume = resume
self.validation_frac = validation_frac
self.patience = patience
```

**Initialize classical readout weights** (after quantum weights initialization):
```python
# Quantum measurements - determine n_meas based on circuit
self.n_meas = self.n_qubits  # Or adjust based on circuit design

# Classical readout weights
self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
self.b2 = np.array(np.zeros(n_classes), requires_grad=True)
```

**Update tracking variables**:
```python
self.best_weights = None
self.best_weights_classical = None  # ADD THIS
self.best_loss = float('inf')
self.best_metric = -float('inf')   # ADD THIS
self.best_step = 0
```

### Step 2: Add Helper Methods

**Add after `_get_circuit()` method**:
```python
def _activation(self, x):
    """Apply activation function."""
    if self.readout_activation == 'tanh':
        return np.tanh(x)
    elif self.readout_activation == 'relu':
        return np.maximum(0, x)
    else:
        return x  # linear

def _classical_readout(self, quantum_output):
    """Apply classical readout head to quantum measurements."""
    # quantum_output shape: (n_meas,)
    hidden = self._activation(np.dot(quantum_output, self.W1) + self.b1)
    logits = np.dot(hidden, self.W2) + self.b2
    return logits
```

### Step 3: Update `_get_circuit()` Method

**Change the return statement** to measure ALL qubits (not just n_classes):
```python
# Before:
return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]

# After:
return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
```

### Step 4: Replace `fit()` Method

**This is the most complex change.** Copy the entire `fit()` method from `MulticlassQuantumClassifierDR` (lines ~99-354).

**Key sections to verify/adjust**:

1. **Import optimizer**:
   - Change `qml.AdamOptimizer` to `AdamSerializable`

2. **Cost function parameters**:
   - Ensure all quantum weights are passed
   - Add classical weights: `w1, b1, w2, b2`
   - For conditional classifiers with two quantum weight sets, pass both

3. **Cost function implementation**:
   - Use the appropriate circuit call pattern for the classifier
   - For conditional classifiers: `qcircuit(features, mask, w_ansatz, w_missing)`
   - For standard classifiers: `qcircuit(x, w_quantum)`

4. **Optimizer step**:
   ```python
   # Standard classifier:
   (self.weights, self.W1, self.b1, self.W2, self.b2), loss = opt.step_and_cost(
       cost, self.weights, self.W1, self.b1, self.W2, self.b2
   )
   
   # Conditional classifier with two weight sets:
   (self.weights_ansatz, self.weights_missing, self.W1, self.b1, self.W2, self.b2), loss = opt.step_and_cost(
       cost, self.weights_ansatz, self.weights_missing, self.W1, self.b1, self.W2, self.b2
   )
   ```

5. **Resume logic** - checkpoint loading:
   - Standard classifiers: load `self.weights`
   - Conditional classifiers: load `self.weights_ansatz` and `self.weights_missing`

6. **Best model saving**:
   - Standard classifiers: save `self.weights`
   - Conditional classifiers: save both quantum weight sets

### Step 5: Update `predict_proba()` Method

**Replace the method** to use classical readout:
```python
def predict_proba(self, X):
    """Predict class probabilities using quantum circuit and classical readout."""
    qcircuit = self._get_circuit()
    
    # Standard classifier
    quantum_outputs = np.array([qcircuit(x, self.weights) for x in X])
    
    # OR for conditional classifier
    # X_scaled, is_missing_mask = X
    # quantum_outputs = np.array([qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
    #                             for f, m in zip(X_scaled, is_missing_mask)])
    
    # Apply classical readout to each sample
    logits_list = []
    for qout in quantum_outputs:
        logits = self._classical_readout(qout)
        logits_list.append(logits)
    
    logits_array = np.array(logits_list)
    return np.array([self._softmax(logit) for logit in logits_array])
```

### Step 6: Verify `predict()` Method

Should be simple and likely doesn't need changes:
```python
def predict(self, X):
    """Predict class labels."""
    return np.argmax(self.predict_proba(X), axis=1)
```

## Special Considerations for Conditional Classifiers

### For `ConditionalMulticlassQuantumClassifierFS` and `ConditionalMulticlassQuantumClassifierDataReuploadingFS`:

1. **Multiple quantum weight sets**: 
   - `self.weights_ansatz`
   - `self.weights_missing`

2. **Input unpacking**:
   ```python
   X_scaled, is_missing_mask = X
   ```

3. **Cost function signature**:
   ```python
   def cost(w_ansatz, w_missing, w1, b1, w2, b2):
       quantum_outputs = np.array([qcircuit(f, m, w_ansatz, w_missing) 
                                   for f, m in zip(X_scaled, is_missing_mask)])
       # ... rest of cost computation
   ```

4. **Checkpoint data structure**:
   ```python
   'weights_quantum': {
       'weights_ansatz': self.weights_ansatz,
       'weights_missing': self.weights_missing
   }
   ```

## Testing After Each Update

After updating each classifier:

1. **Syntax check**:
   ```bash
   python -c "import qml_models; print('OK')"
   ```

2. **Basic instantiation**:
   ```python
   from qml_models import MulticlassQuantumClassifierDataReuploadingDR
   model = MulticlassQuantumClassifierDataReuploadingDR(
       n_qubits=4, n_classes=3, hidden_size=8
   )
   print("Instantiation successful")
   ```

3. **Run smoke test** (once all dependencies installed):
   ```bash
   python tests/test_smoke.py
   ```

## Common Pitfalls to Avoid

1. **Forgetting to update n_meas**: Make sure measurements match W1 input dimension
2. **Wrong parameter order**: Cost function parameter order must match optimizer call
3. **Missing weight copies**: Use `.copy()` when saving best weights
4. **Checkpoint keys mismatch**: Resume logic must match save logic exactly
5. **Missing imports**: Ensure all utility functions are imported at top

## Verification Checklist

For each updated classifier, verify:

- [ ] New parameters added to `__init__`
- [ ] Classical readout weights initialized
- [ ] `_activation()` method added
- [ ] `_classical_readout()` method added
- [ ] Circuit measures all qubits
- [ ] `fit()` uses `AdamSerializable`
- [ ] Cost function includes classical weights
- [ ] Metrics computed each epoch
- [ ] Checkpoints include all weights
- [ ] Resume logic handles all weight types
- [ ] `predict_proba()` uses classical readout
- [ ] No syntax errors
- [ ] Can instantiate the class

## Time Estimate

- Per classifier: 30-45 minutes
- Total for 3 classifiers: 1.5-2.5 hours
- Testing and debugging: 1-2 hours
- **Total**: 2.5-4.5 hours

## Getting Help

If issues arise:
1. Compare line-by-line with `MulticlassQuantumClassifierDR`
2. Check parameter shapes match between quantum circuit and classical readout
3. Verify all imports are present
4. Test with minimal example (small n_qubits, few steps)

## Success Criteria

All 4 classifiers should:
- Train with classical readout head
- Save/load complete checkpoints
- Generate metrics CSV and plots
- Support resume modes
- Work with existing training scripts
