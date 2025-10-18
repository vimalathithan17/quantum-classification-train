import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import os
import joblib
import time
from typing import Optional, Dict, Any, Tuple

# Import the centralized logger
from logging_utils import log

# Import custom utilities
from utils.optim_adam import SerializableAdam
from utils.io_checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_epoch_metrics, save_epoch_history, save_metrics_plots


# --- Classical Readout Head ---

class ClassicalReadoutHead:
    """
    Classical neural network readout head for quantum circuit outputs.
    Applies: quantum_output -> hidden_layer -> output_layer
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 hidden_size: int = 16, activation: str = 'tanh'):
        """
        Initialize classical readout head.
        
        Args:
            input_size: Size of quantum circuit output
            output_size: Number of output classes
            hidden_size: Size of hidden layer (default: 16)
            activation: Activation function 'tanh' or 'relu' (default: 'tanh')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Initialize weights
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        
        # Make weights trainable
        self.w1 = np.array(self.w1, requires_grad=True)
        self.b1 = np.array(self.b1, requires_grad=True)
        self.w2 = np.array(self.w2, requires_grad=True)
        self.b2 = np.array(self.b2, requires_grad=True)
    
    def forward(self, x, w1, b1, w2, b2):
        """
        Forward pass through readout head.
        
        Args:
            x: Input from quantum circuit (array)
            w1, b1, w2, b2: Weight parameters
            
        Returns:
            Output logits
        """
        # Hidden layer
        hidden = np.dot(x, w1) + b1
        
        # Apply activation
        if self.activation == 'relu':
            hidden = np.maximum(0, hidden)
        else:  # tanh
            hidden = np.tanh(hidden)
        
        # Output layer
        output = np.dot(hidden, w2) + b2
        return output
    
    def get_params(self) -> Tuple:
        """Get current parameters as tuple."""
        return (self.w1, self.b1, self.w2, self.b2)
    
    def set_params(self, params: Tuple) -> None:
        """Set parameters from tuple."""
        self.w1, self.b1, self.w2, self.b2 = params
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization."""
        return {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'activation': self.activation
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.w1 = state_dict['w1']
        self.b1 = state_dict['b1']
        self.w2 = state_dict['w2']
        self.b2 = state_dict['b2']
        self.input_size = state_dict['input_size']
        self.output_size = state_dict['output_size']
        self.hidden_size = state_dict['hidden_size']
        self.activation = state_dict['activation']


# --- Models for Approach 1 (Classical Preprocessing + QML) ---

class MulticlassQuantumClassifierDR(BaseEstimator, ClassifierMixin):
    """Multiclass VQC for pre-processed, dimensionally-reduced data with classical readout head."""
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False, 
                 checkpoint_dir=None, checkpoint_frequency=10, keep_last_n=3, max_training_time=None,
                 hidden_size=16, activation='tanh', resume_mode='auto'):
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
        self.activation = activation
        self.resume_mode = resume_mode
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.readout_head = ClassicalReadoutHead(self.n_classes, self.n_classes, hidden_size, activation)
        self.optimizer = None
        self.best_weights = None
        self.best_readout_params = None
        self.best_loss = float('inf')
        self.best_step = 0
        self.checkpoint_history = []
        self.epoch_history = []

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

    def fit(self, X, y):
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        # Try to resume from checkpoint
        start_step = 0
        if self.checkpoint_dir and self.resume_mode != 'none':
            checkpoint = load_checkpoint(self.checkpoint_dir, self.resume_mode)
            if checkpoint:
                log.info(f"Resuming from checkpoint (mode: {self.resume_mode})")
                self.weights = checkpoint['quantum_params']
                if checkpoint.get('classical_params'):
                    self.readout_head.load_state_dict(checkpoint['classical_params'])
                start_step = checkpoint.get('step', 0) + 1
                self.best_loss = checkpoint.get('loss', float('inf'))
                
                # Restore optimizer state if available
                if checkpoint.get('optimizer_state'):
                    self.optimizer = SerializableAdam(self.learning_rate)
                    self.optimizer.set_state(checkpoint['optimizer_state'])
                    log.info("Restored optimizer state from checkpoint")
                else:
                    # Reinitialize optimizer with reduced LR and warmup
                    log.warning("Optimizer state not found in checkpoint - reinitializing with 0.1x LR")
                    self.optimizer = SerializableAdam(self.learning_rate * 0.1)
        
        if self.optimizer is None:
            self.optimizer = SerializableAdam(self.learning_rate)
        
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        qcircuit = self._get_circuit()
        y_one_hot = np.eye(self.n_classes)[y]
        
        start_time = time.time()
        step = start_step
        
        # Dynamic steps based on max_training_time or fixed steps
        while True:
            def cost(*params):
                # params = (weights, w1, b1, w2, b2)
                weights = params[0]
                readout_params = params[1:]
                
                # Get quantum outputs
                quantum_out = np.array([qcircuit(x, weights) for x in X])
                
                # Pass through classical readout head
                logits = np.array([self.readout_head.forward(qo, *readout_params) for qo in quantum_out])
                
                # Apply softmax
                probabilities = np.array([self._softmax(l) for l in logits])
                
                # Cross-entropy loss
                loss = -np.mean(y_one_hot * np.log(probabilities + 1e-9))
                return loss
            
            # Combine all parameters for optimization
            all_params = (self.weights,) + self.readout_head.get_params()
            updated_params, current_loss = self.optimizer.step_and_cost(cost, *all_params)
            
            # Unpack updated parameters
            self.weights = updated_params[0]
            self.readout_head.set_params(updated_params[1:])
            
            # Track best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights = self.weights.copy()
                self.best_readout_params = self.readout_head.get_params()
                self.best_step = step
                
                if self.checkpoint_dir:
                    save_checkpoint(
                        self.checkpoint_dir,
                        quantum_params=self.best_weights,
                        classical_params=self.readout_head.get_state_dict(),
                        optimizer_state=self.optimizer.get_state(),
                        step=step,
                        loss=self.best_loss,
                        is_best=True
                    )
            
            # Compute metrics periodically
            if step % 10 == 0:
                y_pred = self.predict(X)
                metrics = compute_epoch_metrics(y, y_pred, loss=current_loss, n_classes=self.n_classes)
                metrics['step'] = step
                metrics['epoch'] = step  # For compatibility
                self.epoch_history.append(metrics)
            
            # Save checkpoint periodically
            if self.checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                save_checkpoint(
                    self.checkpoint_dir,
                    quantum_params=self.weights,
                    classical_params=self.readout_head.get_state_dict(),
                    optimizer_state=self.optimizer.get_state(),
                    step=step,
                    loss=current_loss,
                    is_best=False
                )

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
            if self.best_readout_params is not None:
                self.readout_head.set_params(self.best_readout_params)
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with loss: {self.best_loss:.4f}")
        
        # Save epoch history
        if self.checkpoint_dir and self.epoch_history:
            save_epoch_history(self.checkpoint_dir, self.epoch_history)
            save_metrics_plots(self.epoch_history, self.checkpoint_dir)
        
        return self

    def predict_proba(self, X):
        qcircuit = self._get_circuit()
        quantum_out = np.array([qcircuit(x, self.weights) for x in X])
        readout_params = self.readout_head.get_params()
        logits = np.array([self.readout_head.forward(qo, *readout_params) for qo in quantum_out])
        return np.array([self._softmax(l) for l in logits])

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
