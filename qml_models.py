import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import os
import joblib
import time

# Import the centralized logger
from logging_utils import log

# --- Models for Approach 1 (Classical Preprocessing + QML) ---

class MulticlassQuantumClassifierDR(BaseEstimator, ClassifierMixin):
    """Multiclass VQC for pre-processed, dimensionally-reduced data."""
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
        
        # Dynamic steps based on max_training_time or fixed steps
        while True:
            def cost(weights):
                raw_predictions = np.array([qcircuit(x, weights) for x in X])
                probabilities = np.array([self._softmax(p) for p in raw_predictions])
                # Use cross-entropy loss for multiclass
                loss = -np.mean(y_one_hot * np.log(probabilities + 1e-9))
                return loss
            
            self.weights, current_loss = opt.step_and_cost(cost, self.weights)
            
            # Track best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights = self.weights.copy()
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
            log.info(f"  [QML Training] Loaded best weights with loss: {self.best_loss:.4f}")
        
        return self

    def predict_proba(self, X):
        qcircuit = self._get_circuit()
        raw_predictions = np.array([qcircuit(x, self.weights) for x in X])
        return np.array([self._softmax(p) for p in raw_predictions])

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
            log.info(f"  [QML Training] Loaded best weights with loss: {self.best_loss:.4f}")
        
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
            log.info(f"  [QML Training] Loaded best weights with loss: {self.best_loss:.4f}")

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
            log.info(f"  [QML Training] Loaded best weights with loss: {self.best_loss:.4f}")

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
