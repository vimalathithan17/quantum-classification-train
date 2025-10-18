import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import joblib
import time
from datetime import datetime

# Import the centralized logger
from logging_utils import log

# Import new utilities
from utils.optim_adam import AdamSerializable
from utils.io_checkpoint import (
    save_best_checkpoint, save_periodic_checkpoint,
    find_latest_checkpoint, find_best_checkpoint,
    load_checkpoint
)
from utils.metrics_utils import (
    compute_metrics, save_metrics_to_csv, plot_training_curves
)

# --- Models for Approach 1 (Classical Preprocessing + QML) ---

class MulticlassQuantumClassifierDR(BaseEstimator, ClassifierMixin):
    """Multiclass VQC for pre-processed, dimensionally-reduced data with classical readout."""
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False, 
                 checkpoint_dir=None, checkpoint_frequency=10, keep_last_n=3, max_training_time=None,
                 hidden_size=16, readout_activation='tanh', selection_metric='weighted_f1',
                 resume=None, validation_frac=0.1, patience=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        self.hidden_size = hidden_size
        self.readout_activation = readout_activation
        self.selection_metric = selection_metric
        self.resume = resume
        self.validation_frac = validation_frac
        self.patience = patience
        
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Quantum weights - measurements from all qubits
        self.n_meas = self.n_qubits
        self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        
        # Classical readout weights
        self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
        self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
        self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
        self.b2 = np.array(np.zeros(n_classes), requires_grad=True)
        
        self.best_weights = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
        self.best_step = 0
        self.checkpoint_history = []

    def _get_circuit(self):
        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            # Measure all qubits for classical readout
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return qcircuit
    
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

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def fit(self, X, y):
        """
        Fit the quantum classifier with classical readout head.
        
        Supports checkpointing, resume modes, validation split, and comprehensive metrics logging.
        """
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Split into train/validation if requested
        if self.validation_frac > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_frac, stratify=y, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
        
        y_train_one_hot = np.eye(self.n_classes)[y_train]
        if y_val is not None:
            y_val_one_hot = np.eye(self.n_classes)[y_val]
        
        qcircuit = self._get_circuit()
        
        # Use custom Adam optimizer for serializability
        opt = AdamSerializable(lr=self.learning_rate)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_prec_macro': [],
            'val_prec_weighted': [],
            'val_rec_macro': [],
            'val_rec_weighted': [],
            'val_f1_macro': [],
            'val_f1_weighted': [],
            'val_spec_macro': [],
            'val_spec_weighted': []
        }
        
        # Handle resume logic
        start_step = 0
        if self.resume and self.checkpoint_dir:
            checkpoint_path = None
            
            if self.resume == 'best':
                checkpoint_path = find_best_checkpoint(self.checkpoint_dir)
            elif self.resume == 'latest':
                checkpoint_path = find_latest_checkpoint(self.checkpoint_dir)
            elif self.resume == 'auto':
                # Try latest first, fall back to best
                checkpoint_path = find_latest_checkpoint(self.checkpoint_dir)
                if checkpoint_path is None:
                    checkpoint_path = find_best_checkpoint(self.checkpoint_dir)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    checkpoint = load_checkpoint(checkpoint_path)
                    self.weights = checkpoint['weights_quantum']
                    
                    # Load classical weights if available
                    if 'weights_classical' in checkpoint:
                        self.W1 = checkpoint['weights_classical']['W1']
                        self.b1 = checkpoint['weights_classical']['b1']
                        self.W2 = checkpoint['weights_classical']['W2']
                        self.b2 = checkpoint['weights_classical']['b2']
                    
                    # Load optimizer state if available
                    if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                        opt.set_state(checkpoint['optimizer_state'])
                        start_step = checkpoint.get('step', 0) + 1
                        log.info(f"  Resumed from step {start_step} with optimizer state")
                    else:
                        # No optimizer state - reduce LR and use warmup
                        opt.set_lr(self.learning_rate * 0.1)
                        log.info(f"  Loaded weights but no optimizer state - reduced LR to {opt.lr}")
                    
                    # Load history if available
                    if 'history' in checkpoint:
                        history = checkpoint['history']
                    
                    if 'best_val_metric' in checkpoint:
                        self.best_metric = checkpoint['best_val_metric']
                    
                    log.info(f"  Successfully loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    log.warning(f"  Failed to load checkpoint: {e}")
        
        start_time = time.time()
        step = start_step
        patience_counter = 0
        
        # Training loop
        while True:
            # Define cost function with all parameters
            def cost(w_quantum, w1, b1, w2, b2):
                quantum_outputs = np.array([qcircuit(x, w_quantum) for x in X_train])
                
                # Apply classical readout to each sample
                logits_list = []
                for qout in quantum_outputs:
                    hidden = np.tanh(np.dot(qout, w1) + b1) if self.readout_activation == 'tanh' else np.dot(qout, w1) + b1
                    logits = np.dot(hidden, w2) + b2
                    logits_list.append(logits)
                
                logits_array = np.array(logits_list)
                probabilities = np.array([self._softmax(logit) for logit in logits_array])
                
                # Cross-entropy loss
                loss = -np.mean(y_train_one_hot * np.log(probabilities + 1e-9))
                return loss
            
            # Update all parameters jointly
            (self.weights, self.W1, self.b1, self.W2, self.b2), current_loss = opt.step_and_cost(
                cost, self.weights, self.W1, self.b1, self.W2, self.b2
            )
            
            # Compute training metrics periodically
            if step % 10 == 0 or step == 0:
                train_preds = self.predict(X_train)
                train_metrics = compute_metrics(y_train, train_preds, self.n_classes)
                
                history['train_loss'].append(float(current_loss))
                history['train_acc'].append(train_metrics['accuracy'])
                
                # Validation metrics
                if X_val is not None:
                    val_preds = self.predict(X_val)
                    val_metrics = compute_metrics(y_val, val_preds, self.n_classes)
                    
                    # Compute validation loss
                    val_quantum_outputs = np.array([qcircuit(x, self.weights) for x in X_val])
                    val_logits_list = []
                    for qout in val_quantum_outputs:
                        hidden = np.tanh(np.dot(qout, self.W1) + self.b1) if self.readout_activation == 'tanh' else np.dot(qout, self.W1) + self.b1
                        logits = np.dot(hidden, self.W2) + self.b2
                        val_logits_list.append(logits)
                    val_logits_array = np.array(val_logits_list)
                    val_probs = np.array([self._softmax(logit) for logit in val_logits_array])
                    val_loss = -np.mean(y_val_one_hot * np.log(val_probs + 1e-9))
                    
                    history['val_loss'].append(float(val_loss))
                    history['val_acc'].append(val_metrics['accuracy'])
                    history['val_prec_macro'].append(val_metrics['precision_macro'])
                    history['val_prec_weighted'].append(val_metrics['precision_weighted'])
                    history['val_rec_macro'].append(val_metrics['recall_macro'])
                    history['val_rec_weighted'].append(val_metrics['recall_weighted'])
                    history['val_f1_macro'].append(val_metrics['f1_macro'])
                    history['val_f1_weighted'].append(val_metrics['f1_weighted'])
                    history['val_spec_macro'].append(val_metrics['specificity_macro'])
                    history['val_spec_weighted'].append(val_metrics['specificity_weighted'])
                    
                    # Check if this is the best model based on selection metric
                    current_metric = val_metrics[self.selection_metric]
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.best_weights = self.weights.copy()
                        self.best_weights_classical = {
                            'W1': self.W1.copy(),
                            'b1': self.b1.copy(),
                            'W2': self.W2.copy(),
                            'b2': self.b2.copy()
                        }
                        self.best_step = step
                        patience_counter = 0
                        
                        if self.checkpoint_dir:
                            checkpoint_data = {
                                'step': step,
                                'weights_quantum': self.best_weights,
                                'weights_classical': self.best_weights_classical,
                                'optimizer_state': opt.get_state(),
                                'rng_state': np.random.get_state(),
                                'best_val_metric': self.best_metric,
                                'metric_name': self.selection_metric,
                                'history': history,
                                'meta': {
                                    'lr': self.learning_rate,
                                    'hidden_size': self.hidden_size,
                                    'n_layers': self.n_layers,
                                    'date': datetime.now().isoformat()
                                }
                            }
                            save_best_checkpoint(self.checkpoint_dir, checkpoint_data)
                    else:
                        patience_counter += 1
                
            # Periodic checkpoint
            if self.checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                checkpoint_data = {
                    'step': step,
                    'weights_quantum': self.weights,
                    'weights_classical': {
                        'W1': self.W1,
                        'b1': self.b1,
                        'W2': self.W2,
                        'b2': self.b2
                    },
                    'optimizer_state': opt.get_state(),
                    'rng_state': np.random.get_state(),
                    'best_val_metric': self.best_metric,
                    'metric_name': self.selection_metric,
                    'history': history,
                    'meta': {
                        'lr': self.learning_rate,
                        'hidden_size': self.hidden_size,
                        'n_layers': self.n_layers
                    }
                }
                save_periodic_checkpoint(self.checkpoint_dir, step, checkpoint_data, self.keep_last_n)
                
                # Save history CSV and plots
                if len(history['train_loss']) > 0:
                    save_metrics_to_csv(history, self.checkpoint_dir)
                    plot_training_curves(history, self.checkpoint_dir)

            if self.verbose and (step % 10 == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                val_info = ""
                if X_val is not None and len(history['val_loss']) > 0:
                    val_info = f" - Val Loss: {history['val_loss'][-1]:.4f} - Val {self.selection_metric}: {history[f'val_{self.selection_metric}'][-1]:.4f}"
                log.info(f"  [QML Training] Step {step:>{len(str(self.steps))}}/{self.steps} - Loss: {current_loss:.4f}{val_info} - Time: {elapsed:.1f}s")
            
            step += 1
            
            # Check stopping conditions
            if self.patience and patience_counter >= self.patience:
                log.info(f"  Early stopping triggered at step {step} (patience={self.patience})")
                break
            
            if self.max_training_time:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= self.max_training_time:
                    log.info(f"  [QML Training] Reached max training time of {self.max_training_time:.2f} hours at step {step}")
                    break
            else:
                if step >= self.steps:
                    break
        
        # Load best weights if available
        if self.best_weights is not None:
            self.weights = self.best_weights
            if self.best_weights_classical is not None:
                self.W1 = self.best_weights_classical['W1']
                self.b1 = self.best_weights_classical['b1']
                self.W2 = self.best_weights_classical['W2']
                self.b2 = self.best_weights_classical['b2']
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with {self.selection_metric}: {self.best_metric:.4f}")
        
        return self

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

    def predict(self, X):
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

class MulticlassQuantumClassifierDataReuploadingDR(BaseEstimator, ClassifierMixin):
    """Data Re-uploading Multiclass VQC for pre-processed, dense data."""
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_frequency=10, keep_last_n=3, max_training_time=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.best_weights = None
        self.best_loss = float('inf')
        self.best_step = 0
        self.checkpoint_history = []

    def _get_circuit(self):
        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(inputs, weights):
            for layer in range(self.n_layers):
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
                qml.BasicEntanglerLayers(weights[layer:layer+1], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
        return qcircuit

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def fit(self, X, y):
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        qcircuit = self._get_circuit()
        y_one_hot = np.eye(self.n_classes)[y]
        opt = qml.AdamOptimizer(self.learning_rate)
        
        start_time = time.time()
        step = 0
        
        while True:
            def cost(weights):
                raw_predictions = np.array([qcircuit(x, weights) for x in X])
                probabilities = np.array([self._softmax(p) for p in raw_predictions])
                loss = -np.mean(y_one_hot * np.log(probabilities + 1e-9))
                return loss
            
            self.weights, current_loss = opt.step_and_cost(cost, self.weights)
            
            # Track best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights = self.weights.copy()
                self.best_step = step
                if self.checkpoint_dir:
                    best_path = os.path.join(self.checkpoint_dir, 'best_weights.joblib')
                    joblib.dump({'weights': self.best_weights, 'loss': self.best_loss, 'step': step}, best_path)
            
            # Save checkpoint periodically
            if self.checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.joblib')
                joblib.dump({'weights': self.weights, 'loss': current_loss, 'step': step}, checkpoint_path)
                self.checkpoint_history.append(checkpoint_path)
                
                # Keep only last N checkpoints
                if len(self.checkpoint_history) > self.keep_last_n:
                    old_checkpoint = self.checkpoint_history.pop(0)
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)

            if self.verbose and (step % 10 == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                log.info(f"  [QML Training] Step {step:>{len(str(self.steps))}}/{self.steps} - Loss: {current_loss:.4f} - Best Loss: {self.best_loss:.4f} - Time: {elapsed:.1f}s")
            
            step += 1
            
            # Check stopping conditions
            if self.max_training_time:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= self.max_training_time:
                    log.info(f"  [QML Training] Reached max training time of {self.max_training_time:.2f} hours at step {step}")
                    break
            else:
                if step >= self.steps:
                    break
        
        # Load best weights
        if self.best_weights is not None:
            self.weights = self.best_weights
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with loss: {self.best_loss:.4f}")
        
        return self

    def predict_proba(self, X):
        qcircuit = self._get_circuit()
        raw_predictions = np.array([qcircuit(x, self.weights) for x in X])
        return np.array([self._softmax(p) for p in raw_predictions])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# --- Models for Approach 2 (Conditional Encoding on Selected Features) ---

class ConditionalMulticlassQuantumClassifierFS(BaseEstimator, ClassifierMixin):
    """Conditional Multiclass QVC that expects pre-processed tuple input."""
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_frequency=10, keep_last_n=3, max_training_time=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights_ansatz = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.weights_missing = np.random.uniform(0, 2 * np.pi, self.n_qubits, requires_grad=True)
        self.best_weights_ansatz = None
        self.best_weights_missing = None
        self.best_loss = float('inf')
        self.best_step = 0
        self.checkpoint_history = []

    def _get_circuit(self):
        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(features, is_missing_mask, weights_ansatz, weights_missing):
            for i in range(self.n_qubits):
                if is_missing_mask[i] == 1:
                    qml.RY(weights_missing[i], wires=i)
                else:
                    qml.RY(features[i] * np.pi, wires=i)
            qml.BasicEntanglerLayers(weights_ansatz, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
        return qcircuit

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def fit(self, X, y):
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        X_scaled, is_missing_mask = X
        y_one_hot = np.eye(self.n_classes)[y]
        qcircuit = self._get_circuit()
        opt = qml.AdamOptimizer(self.learning_rate)
        
        start_time = time.time()
        step = 0
        
        while True:
            # The cost function now takes the weights directly as arguments
            def cost(w_ansatz, w_missing):
                raw_predictions = np.array([qcircuit(f, m, w_ansatz, w_missing) for f, m in zip(X_scaled, is_missing_mask)])
                probabilities = np.array([self._softmax(p) for p in raw_predictions])
                loss = -np.mean(y_one_hot * np.log(probabilities + 1e-9))
                return loss
            
            # Pass the weights to step_and_cost and receive them back as a tuple
            (self.weights_ansatz, self.weights_missing), current_loss = opt.step_and_cost(
                cost, self.weights_ansatz, self.weights_missing
            )
            
            # Track best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights_ansatz = self.weights_ansatz.copy()
                self.best_weights_missing = self.weights_missing.copy()
                self.best_step = step
                if self.checkpoint_dir:
                    best_path = os.path.join(self.checkpoint_dir, 'best_weights.joblib')
                    joblib.dump({'weights_ansatz': self.best_weights_ansatz, 
                                'weights_missing': self.best_weights_missing,
                                'loss': self.best_loss, 'step': step}, best_path)
            
            # Save checkpoint periodically
            if self.checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.joblib')
                joblib.dump({'weights_ansatz': self.weights_ansatz,
                            'weights_missing': self.weights_missing,
                            'loss': current_loss, 'step': step}, checkpoint_path)
                self.checkpoint_history.append(checkpoint_path)
                
                # Keep only last N checkpoints
                if len(self.checkpoint_history) > self.keep_last_n:
                    old_checkpoint = self.checkpoint_history.pop(0)
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)

            if self.verbose and (step % 10 == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                log.info(f"  [QML Training] Step {step:>{len(str(self.steps))}}/{self.steps} - Loss: {current_loss:.4f} - Best Loss: {self.best_loss:.4f} - Time: {elapsed:.1f}s")

            step += 1
            
            # Check stopping conditions
            if self.max_training_time:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= self.max_training_time:
                    log.info(f"  [QML Training] Reached max training time of {self.max_training_time:.2f} hours at step {step}")
                    break
            else:
                if step >= self.steps:
                    break
        
        # Load best weights
        if self.best_weights_ansatz is not None:
            self.weights_ansatz = self.best_weights_ansatz
            self.weights_missing = self.best_weights_missing
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with loss: {self.best_loss:.4f}")

        return self

    def predict_proba(self, X):
        X_scaled, is_missing_mask = X
        qcircuit = self._get_circuit()
        raw_predictions = np.array([qcircuit(f, m, self.weights_ansatz, self.weights_missing) for f, m in zip(X_scaled, is_missing_mask)])
        return np.array([self._softmax(p) for p in raw_predictions])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class ConditionalMulticlassQuantumClassifierDataReuploadingFS(BaseEstimator, ClassifierMixin):
    """Data Re-uploading Conditional Multiclass QVC."""
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_frequency=10, keep_last_n=3, max_training_time=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights_ansatz = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.weights_missing = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.best_weights_ansatz = None
        self.best_weights_missing = None
        self.best_loss = float('inf')
        self.best_step = 0
        self.checkpoint_history = []

    def _get_circuit(self):
        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(features, is_missing_mask, weights_ansatz, weights_missing):
            for layer in range(self.n_layers):
                # Data encoding
                for i in range(self.n_qubits):
                    if is_missing_mask[i] == 1:
                        qml.RY(weights_missing[layer, i], wires=i)
                    else:
                        qml.RY(features[i] * np.pi, wires=i)
                # Trainable ansatz
                qml.BasicEntanglerLayers(weights_ansatz[layer:layer+1], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
        return qcircuit

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def fit(self, X, y):
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        X_scaled, is_missing_mask = X
        y_one_hot = np.eye(self.n_classes)[y]
        qcircuit = self._get_circuit()
        opt = qml.AdamOptimizer(self.learning_rate)
        
        start_time = time.time()
        step = 0
        
        while True:
            # The cost function now takes the weights directly as arguments
            def cost(w_ansatz, w_missing):
                raw_predictions = np.array([qcircuit(f, m, w_ansatz, w_missing) for f, m in zip(X_scaled, is_missing_mask)])
                probabilities = np.array([self._softmax(p) for p in raw_predictions])
                loss = -np.mean(y_one_hot * np.log(probabilities + 1e-9))
                return loss

            # Pass the weights to step_and_cost and receive them back as a tuple
            (self.weights_ansatz, self.weights_missing), current_loss = opt.step_and_cost(
                cost, self.weights_ansatz, self.weights_missing
            )
            
            # Track best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights_ansatz = self.weights_ansatz.copy()
                self.best_weights_missing = self.weights_missing.copy()
                self.best_step = step
                if self.checkpoint_dir:
                    best_path = os.path.join(self.checkpoint_dir, 'best_weights.joblib')
                    joblib.dump({'weights_ansatz': self.best_weights_ansatz, 
                                'weights_missing': self.best_weights_missing,
                                'loss': self.best_loss, 'step': step}, best_path)
            
            # Save checkpoint periodically
            if self.checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.joblib')
                joblib.dump({'weights_ansatz': self.weights_ansatz,
                            'weights_missing': self.weights_missing,
                            'loss': current_loss, 'step': step}, checkpoint_path)
                self.checkpoint_history.append(checkpoint_path)
                
                # Keep only last N checkpoints
                if len(self.checkpoint_history) > self.keep_last_n:
                    old_checkpoint = self.checkpoint_history.pop(0)
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)

            if self.verbose and (step % 10 == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                log.info(f"  [QML Training] Step {step:>{len(str(self.steps))}}/{self.steps} - Loss: {current_loss:.4f} - Best Loss: {self.best_loss:.4f} - Time: {elapsed:.1f}s")

            step += 1
            
            # Check stopping conditions
            if self.max_training_time:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= self.max_training_time:
                    log.info(f"  [QML Training] Reached max training time of {self.max_training_time:.2f} hours at step {step}")
                    break
            else:
                if step >= self.steps:
                    break
        
        # Load best weights
        if self.best_weights_ansatz is not None:
            self.weights_ansatz = self.best_weights_ansatz
            self.weights_missing = self.best_weights_missing
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with loss: {self.best_loss:.4f}")

        return self

    def predict_proba(self, X):
        X_scaled, is_missing_mask = X
        qcircuit = self._get_circuit()
        raw_predictions = np.array([qcircuit(f, m, self.weights_ansatz, self.weights_missing) for f, m in zip(X_scaled, is_missing_mask)])
        return np.array([self._softmax(p) for p in raw_predictions])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
        
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
