import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import joblib
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import the centralized logger
from logging_utils import log

# Import checkpoint utilities and custom optimizer
from utils.io_checkpoint import (
    save_best_and_latest_checkpoints, load_checkpoint, 
    find_latest_checkpoint, find_best_checkpoint,
    get_rng_state, set_rng_state
)
from utils.optim_adam import SerializableAdam


# --- Helper functions for metrics and classical readout ---

def compute_per_class_specificity(y_true, y_pred, n_classes):
    """Compute specificity for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specificities = []
    for i in range(n_classes):
        # True negatives: all correctly predicted non-class-i
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        # False positives: predicted as class i but aren't
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    return specificities


def compute_epoch_metrics(y_true, y_pred, n_classes):
    """Compute comprehensive metrics for an epoch."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Add per-class specificity
    specificities = compute_per_class_specificity(y_true, y_pred, n_classes)
    for i, spec in enumerate(specificities):
        metrics[f'specificity_class_{i}'] = spec
    
    # Add confusion matrix as a flattened array
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    metrics['confusion_matrix'] = cm.flatten().tolist()
    
    return metrics


def save_metrics_to_csv(metrics_history, csv_path):
    """Save metrics history to CSV file."""
    df = pd.DataFrame(metrics_history)
    df.to_csv(csv_path, index=False)


def plot_metrics(metrics_history, plot_dir):
    """Generate and save metric plots."""
    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(metrics_history)
    
    if len(df) == 0:
        return
    
    # Plot loss
    if 'loss' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['loss'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'loss_plot.png'), dpi=100, bbox_inches='tight')
        plt.close()
    
    # Plot main metrics
    metric_names = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    available_metrics = [m for m in metric_names if m in df.columns]
    
    if available_metrics:
        plt.figure(figsize=(12, 8))
        for metric in available_metrics:
            plt.plot(df['epoch'], df[metric], marker='o', label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Training Metrics over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'metrics_plot.png'), dpi=100, bbox_inches='tight')
        plt.close()


# --- Models for Approach 1 (Classical Preprocessing + QML) ---

class MulticlassQuantumClassifierDR(BaseEstimator, ClassifierMixin):
    """
    Multiclass VQC for pre-processed, dimensionally-reduced data.
    Now includes classical readout head and enhanced checkpointing.
    """
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, 
                 verbose=False, checkpoint_dir=None, checkpoint_frequency=10, 
                 keep_last_n=3, max_training_time=None, hidden_dim=16):
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
        self.hidden_dim = hidden_dim  # Hidden layer size for classical readout
        
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Quantum parameters
        self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        
        # Classical readout head parameters (MLP: quantum_output -> hidden -> logits)
        self.W1 = np.random.randn(self.n_classes, self.hidden_dim) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.n_classes) * 0.01
        self.b2 = np.zeros(self.n_classes)
        
        # Make classical params trainable
        self.W1 = np.array(self.W1, requires_grad=True)
        self.b1 = np.array(self.b1, requires_grad=True)
        self.W2 = np.array(self.W2, requires_grad=True)
        self.b2 = np.array(self.b2, requires_grad=True)
        
        # Training state
        self.best_weights = None
        self.best_classical_params = None
        self.best_loss = float('inf')
        self.best_step = 0
        self.checkpoint_history = []
        self.metrics_history = []
        self.optimizer = None

    def _get_circuit(self):
        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
        return qcircuit

    def _classical_readout(self, quantum_output, W1, b1, W2, b2):
        """Apply classical MLP readout head."""
        # Hidden layer with ReLU
        hidden = np.maximum(0, np.dot(quantum_output, W1) + b1)
        # Output layer (logits)
        logits = np.dot(hidden, W2) + b2
        return logits

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def fit(self, X, y, resume='auto'):
        """
        Fit the quantum classifier with classical readout.
        
        Args:
            X: Training data
            y: Training labels
            resume: 'auto', 'latest', 'best', or None
                - 'auto': load latest if optimizer state exists, else best
                - 'latest': load latest checkpoint
                - 'best': load best checkpoint
                - None: start fresh
        """
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Handle resume logic
        start_step = 0
        if resume and self.checkpoint_dir:
            checkpoint_path = None
            if resume == 'auto':
                # Try latest first, fall back to best
                checkpoint_path = find_latest_checkpoint(self.checkpoint_dir)
                if checkpoint_path:
                    log.info(f"  [Resume] Found latest checkpoint, loading...")
                else:
                    checkpoint_path = find_best_checkpoint(self.checkpoint_dir)
                    if checkpoint_path:
                        log.info(f"  [Resume] Loading best checkpoint (no latest found)...")
            elif resume == 'latest':
                checkpoint_path = find_latest_checkpoint(self.checkpoint_dir)
            elif resume == 'best':
                checkpoint_path = find_best_checkpoint(self.checkpoint_dir)
            
            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)
                start_step = self.best_step
                log.info(f"  [Resume] Resumed from step {start_step}")
        
        # Initialize or restore optimizer
        if self.optimizer is None:
            self.optimizer = SerializableAdam(lr=self.learning_rate)
        
        qcircuit = self._get_circuit()
        y_one_hot = np.eye(self.n_classes)[y]
        
        start_time = time.time()
        step = start_step
        
        # Dynamic steps based on max_training_time or fixed steps
        while True:
            def cost(weights, W1, b1, W2, b2):
                # Get quantum outputs
                quantum_outputs = np.array([qcircuit(x, weights) for x in X])
                # Apply classical readout
                logits = np.array([self._classical_readout(q_out, W1, b1, W2, b2) 
                                  for q_out in quantum_outputs])
                probabilities = np.array([self._softmax(logit) for logit in logits])
                # Cross-entropy loss
                loss = -np.mean(y_one_hot * np.log(probabilities + 1e-9))
                return loss
            
            # Update all parameters together
            (self.weights, self.W1, self.b1, self.W2, self.b2), current_loss = \
                self.optimizer.step_and_cost(cost, self.weights, self.W1, self.b1, self.W2, self.b2)
            
            # Compute metrics for this step
            y_pred = self.predict(X)
            step_metrics = compute_epoch_metrics(y, y_pred, self.n_classes)
            step_metrics['epoch'] = step
            step_metrics['loss'] = float(current_loss)
            self.metrics_history.append(step_metrics)
            
            # Track best model
            is_best = False
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights = self.weights.copy()
                self.best_classical_params = {
                    'W1': self.W1.copy(), 'b1': self.b1.copy(),
                    'W2': self.W2.copy(), 'b2': self.b2.copy()
                }
                self.best_step = step
                is_best = True
            
            # Save checkpoints
            if self.checkpoint_dir and (step % self.checkpoint_frequency == 0 or is_best):
                self._save_checkpoint(step, current_loss, step_metrics, is_best)

            if self.verbose and (step % 10 == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                log.info(f"  [QML Training] Step {step:>{len(str(self.steps))}}/{self.steps} - "
                        f"Loss: {current_loss:.4f} - Best: {self.best_loss:.4f} - "
                        f"Acc: {step_metrics['accuracy']:.3f} - Time: {elapsed:.1f}s")
            
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
            self.W1 = self.best_classical_params['W1']
            self.b1 = self.best_classical_params['b1']
            self.W2 = self.best_classical_params['W2']
            self.b2 = self.best_classical_params['b2']
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with loss: {self.best_loss:.4f}")
        
        # Save final metrics and plots
        if self.checkpoint_dir and self.metrics_history:
            csv_path = os.path.join(self.checkpoint_dir, 'metrics.csv')
            save_metrics_to_csv(self.metrics_history, csv_path)
            plot_metrics(self.metrics_history, self.checkpoint_dir)
        
        return self

    def _save_checkpoint(self, step, loss, metrics, is_best):
        """Save checkpoint with all state."""
        model_params = {'weights': self.weights}
        classical_params = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        }
        optimizer_state = self.optimizer.get_state() if self.optimizer else None
        rng_state = get_rng_state()
        
        save_best_and_latest_checkpoints(
            self.checkpoint_dir, model_params, classical_params,
            optimizer_state, rng_state, step, step, loss, metrics, is_best
        )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore state."""
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Restore model params
        if 'model_params' in checkpoint and checkpoint['model_params']:
            self.weights = checkpoint['model_params']['weights']
        
        # Restore classical params
        if 'classical_params' in checkpoint and checkpoint['classical_params']:
            cp = checkpoint['classical_params']
            self.W1 = cp['W1']
            self.b1 = cp['b1']
            self.W2 = cp['W2']
            self.b2 = cp['b2']
        
        # Restore optimizer state
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state']:
            if self.optimizer is None:
                self.optimizer = SerializableAdam(lr=self.learning_rate)
            self.optimizer.set_state(checkpoint['optimizer_state'])
        
        # Restore RNG state
        if 'rng_state' in checkpoint:
            set_rng_state(checkpoint['rng_state'])
        
        # Restore metadata
        if 'metadata' in checkpoint:
            self.best_step = checkpoint['metadata'].get('step', 0)
            self.best_loss = checkpoint['metadata'].get('loss', float('inf'))

    def predict_proba(self, X):
        qcircuit = self._get_circuit()
        quantum_outputs = np.array([qcircuit(x, self.weights) for x in X])
        logits = np.array([self._classical_readout(q_out, self.W1, self.b1, self.W2, self.b2) 
                          for q_out in quantum_outputs])
        return np.array([self._softmax(logit) for logit in logits])

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
