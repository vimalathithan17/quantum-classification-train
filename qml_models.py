import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import joblib
import time

# Import the centralized logger
from logging_utils import log

# Import utilities
from utils.optim_adam import SerializableAdam
from utils.io_checkpoint import (
    save_checkpoint, load_checkpoint, 
    save_best_checkpoint, save_latest_checkpoint,
    get_latest_checkpoint_path, get_best_checkpoint_path
)
from utils.metrics import (
    compute_classification_metrics,
    save_metrics_to_csv,
    plot_metrics,
    plot_confusion_matrix_standalone
)

# --- Models for Approach 1 (Classical Preprocessing + QML) ---

class MulticlassQuantumClassifierDR(BaseEstimator, ClassifierMixin):
    """Multiclass VQC for pre-processed, dimensionally-reduced data with classical readout."""
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False, 
                 checkpoint_dir=None, checkpoint_frequency=10, keep_last_n=3, max_training_time=None,
                 hidden_size=16, use_classical_readout=True):
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
        self.use_classical_readout = use_classical_readout
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Quantum circuit weights
        self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        
        # Classical readout parameters (MLP: measurement -> hidden -> logits)
        if self.use_classical_readout:
            self.W1 = np.random.randn(self.n_classes, self.hidden_size) * 0.01
            self.W1 = np.array(self.W1, requires_grad=True)
            self.b1 = np.zeros(self.hidden_size, requires_grad=True)
            self.W2 = np.random.randn(self.hidden_size, self.n_classes) * 0.01
            self.W2 = np.array(self.W2, requires_grad=True)
            self.b2 = np.zeros(self.n_classes, requires_grad=True)
        else:
            self.W1 = self.b1 = self.W2 = self.b2 = None
        
        self.best_weights = None
        self.best_classical_params = None
        self.best_loss = float('inf')
        self.best_step = 0
        self.checkpoint_history = []

    def _get_circuit(self):
        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
        return qcircuit

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def _classical_readout(self, measurements):
        """Apply classical MLP readout to quantum measurements."""
        if not self.use_classical_readout:
            return measurements
        # measurements: [n_classes] -> hidden -> logits
        hidden = np.tanh(np.dot(measurements, self.W1) + self.b1)  # [hidden_size]
        logits = np.dot(hidden, self.W2) + self.b2  # [n_classes]
        return logits

    def fit(self, X, y, resume='auto', selection_metric='weighted_f1', validation_frac=0.1, batch_size=None):
        """
        Fit the quantum classifier with optional classical readout.
        
        Args:
            X: Training features
            y: Training labels
            resume: Resume mode - 'auto', 'latest', 'best', or None
            selection_metric: Metric to use for best model selection ('weighted_f1', 'accuracy', etc.)
            validation_frac: Fraction of data to use for validation
            batch_size: Batch size for training (None = full batch)
        """
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        # Split into train and validation
        if validation_frac > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_frac, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
        
        # Setup checkpoint directory
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize optimizer
        opt = SerializableAdam(lr=self.learning_rate)
        
        # Handle resume logic
        start_step = 0
        if resume and self.checkpoint_dir:
            checkpoint_path = None
            if resume == 'auto' or resume == 'latest':
                checkpoint_path = get_latest_checkpoint_path(self.checkpoint_dir)
            elif resume == 'best':
                checkpoint_path = get_best_checkpoint_path(self.checkpoint_dir)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                log.info(f"Resuming from checkpoint: {checkpoint_path}")
                ckpt = load_checkpoint(checkpoint_path)
                self.weights = np.array(ckpt['quantum_params'], requires_grad=True)
                if self.use_classical_readout and 'classical_params' in ckpt:
                    cp = ckpt['classical_params']
                    self.W1 = np.array(cp['W1'], requires_grad=True)
                    self.b1 = np.array(cp['b1'], requires_grad=True)
                    self.W2 = np.array(cp['W2'], requires_grad=True)
                    self.b2 = np.array(cp['b2'], requires_grad=True)
                if 'optimizer_state' in ckpt:
                    opt.set_state(ckpt['optimizer_state'])
                start_step = ckpt.get('step', 0)
                self.best_loss = ckpt.get('best_val_metric', float('inf'))
        
        qcircuit = self._get_circuit()
        y_one_hot = np.eye(self.n_classes)[y_train]
        
        start_time = time.time()
        step = start_step
        
        metrics_history = []
        
        # Dynamic steps based on max_training_time or fixed steps
        while True:
            def cost(*params):
                # Unpack parameters
                if self.use_classical_readout:
                    weights, W1, b1, W2, b2 = params
                else:
                    weights = params[0]
                
                # Forward pass with classical readout
                raw_predictions = np.array([qcircuit(x, weights) for x in X_train])
                
                if self.use_classical_readout:
                    # Apply classical readout to each sample
                    logits = np.array([self._classical_readout_internal(r, W1, b1, W2, b2) 
                                      for r in raw_predictions])
                else:
                    logits = raw_predictions
                
                probabilities = np.array([self._softmax(p) for p in logits])
                # Use cross-entropy loss for multiclass
                loss = -np.mean(y_one_hot * np.log(probabilities + 1e-9))
                return loss
            
            # Update parameters
            if self.use_classical_readout:
                (self.weights, self.W1, self.b1, self.W2, self.b2), current_loss = opt.step_and_cost(
                    cost, self.weights, self.W1, self.b1, self.W2, self.b2
                )
            else:
                self.weights, current_loss = opt.step_and_cost(cost, self.weights)
            
            # Compute validation metrics if validation set exists
            val_metrics = {}
            if X_val is not None:
                y_val_pred = self.predict(X_val)
                val_metrics = compute_classification_metrics(y_val, y_val_pred, self.n_classes)
                val_metrics['val_loss'] = current_loss  # Approximate with training loss
            
            # Select best model based on chosen metric
            current_selection_metric = val_metrics.get(f'val_{selection_metric}', 
                                                       val_metrics.get(selection_metric, current_loss))
            
            # Track best model (higher is better for metrics, lower for loss)
            is_best = False
            if 'loss' in selection_metric:
                is_best = current_selection_metric < self.best_loss
            else:
                is_best = current_selection_metric > self.best_loss
            
            if is_best:
                self.best_loss = current_selection_metric
                self.best_weights = self.weights.copy()
                if self.use_classical_readout:
                    self.best_classical_params = {
                        'W1': self.W1.copy(),
                        'b1': self.b1.copy(),
                        'W2': self.W2.copy(),
                        'b2': self.b2.copy()
                    }
                self.best_step = step
                
                if self.checkpoint_dir:
                    ckpt_data = {
                        'quantum_params': np.array(self.weights),
                        'classical_params': self.best_classical_params if self.use_classical_readout else None,
                        'optimizer_state': opt.get_state(),
                        'rng_state': np.random.get_state(),
                        'step': step,
                        'best_val_metric': self.best_loss,
                        'metadata': {
                            'selection_metric': selection_metric,
                            'n_classes': self.n_classes,
                            'n_qubits': self.n_qubits,
                            'n_layers': self.n_layers
                        }
                    }
                    save_best_checkpoint(self.checkpoint_dir, ckpt_data)
            
            # Save checkpoint periodically
            if self.checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                ckpt_data = {
                    'quantum_params': np.array(self.weights),
                    'classical_params': {
                        'W1': np.array(self.W1),
                        'b1': np.array(self.b1),
                        'W2': np.array(self.W2),
                        'b2': np.array(self.b2)
                    } if self.use_classical_readout else None,
                    'optimizer_state': opt.get_state(),
                    'rng_state': np.random.get_state(),
                    'step': step,
                    'best_val_metric': self.best_loss,
                    'metadata': {
                        'selection_metric': selection_metric,
                        'n_classes': self.n_classes
                    }
                }
                save_latest_checkpoint(self.checkpoint_dir, ckpt_data, step)
                self.checkpoint_history.append(os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.joblib'))
                
                # Keep only last N checkpoints
                if len(self.checkpoint_history) > self.keep_last_n:
                    old_checkpoint = self.checkpoint_history.pop(0)
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
            
            # Log metrics periodically
            if step % 10 == 0:
                epoch_metrics = {
                    'epoch': step,
                    'train_loss': float(current_loss),
                    'train_accuracy': accuracy_score(y_train, self.predict(X_train))
                }
                if X_val is not None:
                    for k, v in val_metrics.items():
                        if k != 'confusion_matrix':
                            epoch_metrics[f'val_{k}' if not k.startswith('val_') else k] = float(v)
                        else:
                            epoch_metrics[k] = v
                metrics_history.append(epoch_metrics)

            if self.verbose and (step % 10 == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                log.info(f"  [QML Training] Step {step:>{len(str(self.steps))}}/{self.steps} - Loss: {current_loss:.4f} - Best: {self.best_loss:.4f} - Time: {elapsed:.1f}s")
            
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
        
        # Save metrics
        if self.checkpoint_dir and metrics_history:
            save_metrics_to_csv(metrics_history, os.path.join(self.checkpoint_dir, 'metrics.csv'))
            plot_metrics(metrics_history, self.checkpoint_dir)
        
        # Load best weights
        if self.best_weights is not None:
            self.weights = self.best_weights
            if self.use_classical_readout and self.best_classical_params:
                self.W1 = self.best_classical_params['W1']
                self.b1 = self.best_classical_params['b1']
                self.W2 = self.best_classical_params['W2']
                self.b2 = self.best_classical_params['b2']
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with {selection_metric}: {self.best_loss:.4f}")
        
        return self
    
    def _classical_readout_internal(self, measurements, W1, b1, W2, b2):
        """Internal method for classical readout used in cost function."""
        hidden = np.tanh(np.dot(measurements, W1) + b1)
        logits = np.dot(hidden, W2) + b2
        return logits

    def predict_proba(self, X):
        qcircuit = self._get_circuit()
        raw_predictions = np.array([qcircuit(x, self.weights) for x in X])
        
        if self.use_classical_readout:
            logits = np.array([self._classical_readout(r) for r in raw_predictions])
        else:
            logits = raw_predictions
        
        return np.array([self._softmax(p) for p in logits])

    def predict(self, X):
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
