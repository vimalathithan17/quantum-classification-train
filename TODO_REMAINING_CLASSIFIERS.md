# TODO: Apply Classical Readout and Enhanced Fit to Remaining Classifiers

The following classifiers need to be updated to match the pattern implemented in `MulticlassQuantumClassifierDR`:

## Classes to Update

1. **MulticlassQuantumClassifierDataReuploadingDR** (lines 327-432 in qml_models.py)
2. **ConditionalMulticlassQuantumClassifierFS** (lines 436-599 in qml_models.py)
3. **ConditionalMulticlassQuantumClassifierDataReuploadingFS** (lines 603-766 in qml_models.py)

## Changes to Apply

For each class, apply the following modifications (see `MulticlassQuantumClassifierDR` as reference):

### 1. Update `__init__` method

Add new parameters:
```python
hidden_size=16, 
use_classical_readout=True
```

Initialize classical readout parameters:
```python
if self.use_classical_readout:
    self.W1 = np.random.randn(self.n_classes, self.hidden_size) * 0.01
    self.W1 = np.array(self.W1, requires_grad=True)
    self.b1 = np.zeros(self.hidden_size, requires_grad=True)
    self.W2 = np.random.randn(self.hidden_size, self.n_classes) * 0.01
    self.W2 = np.array(self.W2, requires_grad=True)
    self.b2 = np.zeros(self.n_classes, requires_grad=True)
else:
    self.W1 = self.b1 = self.W2 = self.b2 = None

self.best_classical_params = None
```

### 2. Add classical readout methods

```python
def _classical_readout(self, measurements):
    """Apply classical MLP readout to quantum measurements."""
    if not self.use_classical_readout:
        return measurements
    hidden = np.tanh(np.dot(measurements, self.W1) + self.b1)
    logits = np.dot(hidden, self.W2) + self.b2
    return logits

def _classical_readout_internal(self, measurements, W1, b1, W2, b2):
    """Internal method for classical readout used in cost function."""
    hidden = np.tanh(np.dot(measurements, W1) + b1)
    logits = np.dot(hidden, W2) + b2
    return logits
```

### 3. Update `fit` method

Replace with enhanced fit method that includes:
- Parameters: `resume='auto', selection_metric='weighted_f1', validation_frac=0.1, batch_size=None`
- Train/validation split
- Custom Adam optimizer (replace `qml.AdamOptimizer` with `SerializableAdam`)
- Resume from checkpoint logic
- Cost function that uses classical readout
- Optimizer step that includes classical parameters
- Validation metrics computation per epoch
- Best model tracking based on selection metric
- Checkpoint saving (latest and best)
- Metrics logging to CSV and plots
- Best weights restoration at end

See lines 90-305 in qml_models.py for the complete enhanced fit method.

### 4. Update `predict_proba` method

```python
def predict_proba(self, X):
    qcircuit = self._get_circuit()
    raw_predictions = np.array([qcircuit(x, self.weights) for x in X])
    
    if self.use_classical_readout:
        logits = np.array([self._classical_readout(r) for r in raw_predictions])
    else:
        logits = raw_predictions
    
    return np.array([self._softmax(p) for p in logits])
```

## Special Considerations

### For Conditional Classifiers

The conditional classifiers (ConditionalMulticlassQuantumClassifierFS and ConditionalMulticlassQuantumClassifierDataReuploadingFS) have:
- Multiple weight sets: `weights_ansatz` and `weights_missing`
- Tuple input: `(X_scaled, is_missing_mask)`
- Different circuit structure

When applying changes:
1. Track both weight sets in checkpoints
2. Update cost function to handle tuple inputs
3. Pass both weight sets to optimizer
4. Update best_classical_params to include both weight sets

## Backward Compatibility

All changes maintain backward compatibility:
- `use_classical_readout=True` by default (can be disabled with `use_classical_readout=False`)
- `resume=None` by default (no resume unless explicitly requested)
- `validation_frac=0.1` for validation split (can be set to 0.0 to disable)
- Existing code using old API will continue to work by using defaults

## Testing

After applying changes, update `tests/test_smoke.py` to include tests for:
- Each classifier type with classical readout
- Each classifier type without classical readout
- Resume functionality for each classifier
- Checkpoint creation for each classifier

## Estimated Work

- Each classifier: ~200 lines of changes
- Total: ~600 lines across 3 classifiers
- Testing updates: ~100 lines

The pattern is established in `MulticlassQuantumClassifierDR` - the work is primarily careful copy-paste with adjustments for conditional classifiers' special requirements.
