import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import joblib
import time
import shutil
from datetime import datetime
import wandb

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


def _ensure_writable_checkpoint_dir(checkpoint_dir, checkpoint_fallback_dir=None):
    """
    Ensure checkpoint directory is writable. If not, try fallback or warn.
    
    Args:
        checkpoint_dir: Primary checkpoint directory
        checkpoint_fallback_dir: Fallback directory if primary is read-only
        
    Returns:
        tuple: (writable_dir, is_fallback) where writable_dir is the path to use
               and is_fallback indicates if we're using the fallback
    """
    if checkpoint_dir is None:
        return None, False
        
    # Try to create directory if it doesn't exist
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except (OSError, PermissionError):
        pass
    
    # Check if writable
    if os.path.exists(checkpoint_dir) and os.access(checkpoint_dir, os.W_OK):
        return checkpoint_dir, False
    
    # Primary is read-only or creation failed
    if checkpoint_fallback_dir:
        log.warning(f"Checkpoint directory '{checkpoint_dir}' is not writable. Trying fallback: '{checkpoint_fallback_dir}'")
        
        try:
            os.makedirs(checkpoint_fallback_dir, exist_ok=True)
            
            if os.access(checkpoint_fallback_dir, os.W_OK):
                # Copy existing checkpoints from primary to fallback
                if os.path.exists(checkpoint_dir):
                    for filename in os.listdir(checkpoint_dir):
                        if filename.endswith('.joblib'):
                            src = os.path.join(checkpoint_dir, filename)
                            dst = os.path.join(checkpoint_fallback_dir, filename)
                            try:
                                shutil.copy2(src, dst)
                                log.info(f"Copied checkpoint: {filename}")
                            except Exception as e:
                                log.warning(f"Could not copy checkpoint {filename}: {e}")
                
                log.info(f"Using fallback checkpoint directory: '{checkpoint_fallback_dir}'")
                return checkpoint_fallback_dir, True
        except Exception as e:
            log.error(f"Could not create fallback directory '{checkpoint_fallback_dir}': {e}")
    
    # No writable path available
    log.warning(f"No writable checkpoint directory available. Checkpointing will be disabled.")
    log.warning(f"Primary: '{checkpoint_dir}' - not writable")
    if checkpoint_fallback_dir:
        log.warning(f"Fallback: '{checkpoint_fallback_dir}' - not available")
    return None, False


def _initialize_wandb(use_wandb, wandb_project, wandb_run_name, config_dict=None):
    """
    Initialize Weights & Biases logging if requested.
    
    Args:
        use_wandb: Whether to use wandb
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        config_dict: Configuration dictionary to log
        
    Returns:
        wandb module if initialized, None otherwise
    """
    if not use_wandb:
        return None
    
    try:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config_dict or {},
            reinit=True
        )
        log.info(f"Initialized W&B logging: project='{wandb_project}', run='{wandb_run_name}'")
        return wandb
    except Exception as e:
        log.warning(f"Failed to initialize wandb: {e}")
        return None

# --- Models for Approach 1 (Classical Preprocessing + QML) ---

class MulticlassQuantumClassifierDR(BaseEstimator, ClassifierMixin):
    """Multiclass VQC for pre-processed, dimensionally-reduced data with classical readout.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of ansatz layers.
        n_classes (int): Number of output classes.
        learning_rate (float): Learning rate for optimizer.
        steps (int): Number of training steps.
        verbose (bool): Enable verbose logging.
        checkpoint_dir (str): Primary directory for saving checkpoints.
        checkpoint_fallback_dir (str): Fallback directory if primary is read-only.
        checkpoint_frequency (int): Save checkpoint every N steps.
        keep_last_n (int): Keep only last N checkpoints.
        max_training_time (float): Maximum training time in hours (overrides steps).
        hidden_size (int): Size of hidden layer in classical readout.
        readout_activation (str): Activation function ('tanh', 'relu', 'linear').
        selection_metric (str): Metric for best model selection.
        resume (str): Resume mode ('auto', 'latest', 'best').
        validation_frac (float): Fraction of data for validation.
        validation_frequency (int): Compute validation metrics every N steps.
        patience (int): Early stopping patience (steps without improvement).
        use_wandb (bool): Enable Weights & Biases logging.
        wandb_project (str): W&B project name.
        wandb_run_name (str): W&B run name.
    """
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False, 
                 checkpoint_dir=None, checkpoint_fallback_dir=None, checkpoint_frequency=10, keep_last_n=3, 
                 max_training_time=None, hidden_size=16, readout_activation='tanh', selection_metric='f1_weighted',
                 resume=None, validation_frac=0.1, validation_frequency=10, patience=None,
                 use_wandb=False, wandb_project=None, wandb_run_name=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_fallback_dir = checkpoint_fallback_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        self.hidden_size = hidden_size
        self.readout_activation = readout_activation
        self.selection_metric = selection_metric
        self.resume = resume
        self.validation_frac = validation_frac
        self.validation_frequency = validation_frequency
        self.patience = patience
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        
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
        Supports checkpointing, resume modes, validation split, and comprehensive metrics logging.
        """
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        # Set default fallback dir based on checkpoint dir name
        if self.checkpoint_dir and not self.checkpoint_fallback_dir:
            checkpoint_name = os.path.basename(self.checkpoint_dir.rstrip('/'))
            self.checkpoint_fallback_dir = checkpoint_name
        
        # Ensure checkpoint directory is writable
        checkpoint_dir, used_fallback = _ensure_writable_checkpoint_dir(
            self.checkpoint_dir, self.checkpoint_fallback_dir
        )
        
        # Initialize W&B logging if requested
        wandb = _initialize_wandb(
            self.use_wandb, self.wandb_project, self.wandb_run_name,
            config_dict={
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'n_classes': self.n_classes,
                'learning_rate': self.learning_rate,
                'steps': self.steps,
                'hidden_size': self.hidden_size,
                'readout_activation': self.readout_activation
            }
        )
        
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
        if self.resume and checkpoint_dir:
            checkpoint_path = None
            
            if self.resume == 'best':
                checkpoint_path = find_best_checkpoint(checkpoint_dir)
            elif self.resume == 'latest':
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            elif self.resume == 'auto':
                # Try latest first, fall back to best
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
                if checkpoint_path is None:
                    checkpoint_path = find_best_checkpoint(checkpoint_dir)
            
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
            if step % self.validation_frequency == 0 or step == 0:
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
                    
                    
                    # Log validation metrics to W&B
                    if wandb:
                        wandb.log({
                            'step': step,
                            'train_loss': history['train_loss'][-1],
                            'train_acc': history['train_acc'][-1],
                            'val_loss': history['val_loss'][-1],
                            'val_acc': history['val_acc'][-1],
                            'val_f1_weighted': history['val_f1_weighted'][-1],
                            'val_prec_weighted': history['val_prec_weighted'][-1],
                            'val_rec_weighted': history['val_rec_weighted'][-1]
                        })
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
                        
                        if checkpoint_dir:
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
                            save_best_checkpoint(checkpoint_dir, checkpoint_data)
                    else:
                        patience_counter += 1
                
            # Periodic checkpoint
            if checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
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
                save_periodic_checkpoint(checkpoint_dir, step, checkpoint_data, self.keep_last_n)
                
                # Save history CSV and plots
                if len(history['train_loss']) > 0:
                    save_metrics_to_csv(history, checkpoint_dir)
                    plot_training_curves(history, checkpoint_dir)

            if self.verbose and (step % self.validation_frequency == 0 or step == self.steps - 1):
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
    """Data Re-uploading Multiclass VQC for pre-processed, dense data with classical readout.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of ansatz layers.
        n_classes (int): Number of output classes.
        learning_rate (float): Learning rate for optimizer.
        steps (int): Number of training steps.
        verbose (bool): Enable verbose logging.
        checkpoint_dir (str): Primary directory for saving checkpoints.
        checkpoint_fallback_dir (str): Fallback directory if primary is read-only.
        checkpoint_frequency (int): Save checkpoint every N steps.
        keep_last_n (int): Keep only last N checkpoints.
        max_training_time (float): Maximum training time in hours (overrides steps).
        hidden_size (int): Size of hidden layer in classical readout.
        readout_activation (str): Activation function ('tanh', 'relu', 'linear').
        selection_metric (str): Metric for best model selection.
        resume (str): Resume mode ('auto', 'latest', 'best').
        validation_frac (float): Fraction of data for validation.
        validation_frequency (int): Compute validation metrics every N steps.
        patience (int): Early stopping patience (steps without improvement).
        use_wandb (bool): Enable Weights & Biases logging.
        wandb_project (str): W&B project name.
        wandb_run_name (str): W&B run name.
    """
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_fallback_dir=None, checkpoint_frequency=10, keep_last_n=3, 
                 max_training_time=None, hidden_size=16, readout_activation='tanh', selection_metric='f1_weighted',
                 resume=None, validation_frac=0.1, validation_frequency=10, patience=None,
                 use_wandb=False, wandb_project=None, wandb_run_name=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_fallback_dir = checkpoint_fallback_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        self.hidden_size = hidden_size
        self.readout_activation = readout_activation
        self.selection_metric = selection_metric
        self.resume = resume
        self.validation_frac = validation_frac
        self.validation_frequency = validation_frequency
        self.patience = patience
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        
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
            for layer in range(self.n_layers):
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
                qml.BasicEntanglerLayers(weights[layer:layer+1], wires=range(self.n_qubits))
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
        Supports checkpointing, resume modes, validation split, and comprehensive metrics logging.
        """
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        # Set default fallback dir based on checkpoint dir name
        if self.checkpoint_dir and not self.checkpoint_fallback_dir:
            checkpoint_name = os.path.basename(self.checkpoint_dir.rstrip('/'))
            self.checkpoint_fallback_dir = checkpoint_name
        
        # Ensure checkpoint directory is writable
        checkpoint_dir, used_fallback = _ensure_writable_checkpoint_dir(
            self.checkpoint_dir, self.checkpoint_fallback_dir
        )
        
        # Initialize W&B logging if requested
        wandb = _initialize_wandb(
            self.use_wandb, self.wandb_project, self.wandb_run_name,
            config_dict={
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'n_classes': self.n_classes,
                'learning_rate': self.learning_rate,
                'steps': self.steps,
                'hidden_size': self.hidden_size,
                'readout_activation': self.readout_activation
            }
        )
        
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
        if self.resume and checkpoint_dir:
            checkpoint_path = None
            
            if self.resume == 'best':
                checkpoint_path = find_best_checkpoint(checkpoint_dir)
            elif self.resume == 'latest':
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            elif self.resume == 'auto':
                # Try latest first, fall back to best
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
                if checkpoint_path is None:
                    checkpoint_path = find_best_checkpoint(checkpoint_dir)
            
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
            if step % self.validation_frequency == 0 or step == 0:
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
                    
                    
                    # Log validation metrics to W&B
                    if wandb:
                        wandb.log({
                            'step': step,
                            'train_loss': history['train_loss'][-1],
                            'train_acc': history['train_acc'][-1],
                            'val_loss': history['val_loss'][-1],
                            'val_acc': history['val_acc'][-1],
                            'val_f1_weighted': history['val_f1_weighted'][-1],
                            'val_prec_weighted': history['val_prec_weighted'][-1],
                            'val_rec_weighted': history['val_rec_weighted'][-1]
                        })
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
                        
                        if checkpoint_dir:
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
                            save_best_checkpoint(checkpoint_dir, checkpoint_data)
                    else:
                        patience_counter += 1
                
            # Periodic checkpoint
            if checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
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
                save_periodic_checkpoint(checkpoint_dir, step, checkpoint_data, self.keep_last_n)
                
                # Save history CSV and plots
                if len(history['train_loss']) > 0:
                    save_metrics_to_csv(history, checkpoint_dir)
                    plot_training_curves(history, checkpoint_dir)

            if self.verbose and (step % self.validation_frequency == 0 or step == self.steps - 1):
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

# --- Models for Approach 2 (Conditional Encoding on Selected Features) ---

class ConditionalMulticlassQuantumClassifierFS(BaseEstimator, ClassifierMixin):
    """Conditional Multiclass QVC that expects pre-processed tuple input with classical readout.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of ansatz layers.
        n_classes (int): Number of output classes.
        learning_rate (float): Learning rate for optimizer.
        steps (int): Number of training steps.
        verbose (bool): Enable verbose logging.
        checkpoint_dir (str): Primary directory for saving checkpoints.
        checkpoint_fallback_dir (str): Fallback directory if primary is read-only.
        checkpoint_frequency (int): Save checkpoint every N steps.
        keep_last_n (int): Keep only last N checkpoints.
        max_training_time (float): Maximum training time in hours (overrides steps).
        hidden_size (int): Size of hidden layer in classical readout.
        readout_activation (str): Activation function ('tanh', 'relu', 'linear').
        selection_metric (str): Metric for best model selection.
        resume (str): Resume mode ('auto', 'latest', 'best').
        validation_frac (float): Fraction of data for validation.
        validation_frequency (int): Compute validation metrics every N steps.
        patience (int): Early stopping patience (steps without improvement).
        use_wandb (bool): Enable Weights & Biases logging.
        wandb_project (str): W&B project name.
        wandb_run_name (str): W&B run name.
    """
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_fallback_dir=None, checkpoint_frequency=10, keep_last_n=3, 
                 max_training_time=None, hidden_size=16, readout_activation='tanh', selection_metric='f1_weighted',
                 resume=None, validation_frac=0.1, validation_frequency=10, patience=None,
                 use_wandb=False, wandb_project=None, wandb_run_name=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_fallback_dir = checkpoint_fallback_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        self.hidden_size = hidden_size
        self.readout_activation = readout_activation
        self.selection_metric = selection_metric
        self.resume = resume
        self.validation_frac = validation_frac
        self.validation_frequency = validation_frequency
        self.patience = patience
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Quantum weights - measurements from all qubits
        self.n_meas = self.n_qubits
        self.weights_ansatz = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.weights_missing = np.random.uniform(0, 2 * np.pi, self.n_qubits, requires_grad=True)
        
        # Classical readout weights
        self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
        self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
        self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
        self.b2 = np.array(np.zeros(n_classes), requires_grad=True)
        
        self.best_weights_ansatz = None
        self.best_weights_missing = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
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
        Supports checkpointing, resume modes, validation split, and comprehensive metrics logging.
        """
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        # Set default fallback dir based on checkpoint dir name
        if self.checkpoint_dir and not self.checkpoint_fallback_dir:
            checkpoint_name = os.path.basename(self.checkpoint_dir.rstrip('/'))
            self.checkpoint_fallback_dir = checkpoint_name
        
        # Ensure checkpoint directory is writable
        checkpoint_dir, used_fallback = _ensure_writable_checkpoint_dir(
            self.checkpoint_dir, self.checkpoint_fallback_dir
        )
        
        # Initialize W&B logging if requested
        wandb = _initialize_wandb(
            self.use_wandb, self.wandb_project, self.wandb_run_name,
            config_dict={
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'n_classes': self.n_classes,
                'learning_rate': self.learning_rate,
                'steps': self.steps,
                'hidden_size': self.hidden_size,
                'readout_activation': self.readout_activation
            }
        )
        
        # Unpack the conditional input
        X_scaled, is_missing_mask = X
        
        # Split into train/validation if requested
        if self.validation_frac > 0:
            X_train_scaled, X_val_scaled, mask_train, mask_val, y_train, y_val = train_test_split(
                X_scaled, is_missing_mask, y, test_size=self.validation_frac, stratify=y, random_state=42
            )
        else:
            X_train_scaled, mask_train, y_train = X_scaled, is_missing_mask, y
            X_val_scaled, mask_val, y_val = None, None, None
        
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
        if self.resume and checkpoint_dir:
            checkpoint_path = None
            
            if self.resume == 'best':
                checkpoint_path = find_best_checkpoint(checkpoint_dir)
            elif self.resume == 'latest':
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            elif self.resume == 'auto':
                # Try latest first, fall back to best
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
                if checkpoint_path is None:
                    checkpoint_path = find_best_checkpoint(checkpoint_dir)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    checkpoint = load_checkpoint(checkpoint_path)
                    weights_quantum = checkpoint['weights_quantum']
                    self.weights_ansatz = weights_quantum['weights_ansatz']
                    self.weights_missing = weights_quantum['weights_missing']
                    
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
            def cost(w_ansatz, w_missing, w1, b1, w2, b2):
                quantum_outputs = np.array([qcircuit(f, m, w_ansatz, w_missing) 
                                          for f, m in zip(X_train_scaled, mask_train)])
                
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
            (self.weights_ansatz, self.weights_missing, self.W1, self.b1, self.W2, self.b2), current_loss = opt.step_and_cost(
                cost, self.weights_ansatz, self.weights_missing, self.W1, self.b1, self.W2, self.b2
            )
            
            # Compute training metrics periodically
            if step % self.validation_frequency == 0 or step == 0:
                train_preds = self.predict((X_train_scaled, mask_train))
                train_metrics = compute_metrics(y_train, train_preds, self.n_classes)
                
                history['train_loss'].append(float(current_loss))
                history['train_acc'].append(train_metrics['accuracy'])
                
                # Validation metrics
                if X_val_scaled is not None:
                    val_preds = self.predict((X_val_scaled, mask_val))
                    val_metrics = compute_metrics(y_val, val_preds, self.n_classes)
                    
                    # Compute validation loss
                    val_quantum_outputs = np.array([qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
                                                   for f, m in zip(X_val_scaled, mask_val)])
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
                    
                    
                    # Log validation metrics to W&B
                    if wandb:
                        wandb.log({
                            'step': step,
                            'train_loss': history['train_loss'][-1],
                            'train_acc': history['train_acc'][-1],
                            'val_loss': history['val_loss'][-1],
                            'val_acc': history['val_acc'][-1],
                            'val_f1_weighted': history['val_f1_weighted'][-1],
                            'val_prec_weighted': history['val_prec_weighted'][-1],
                            'val_rec_weighted': history['val_rec_weighted'][-1]
                        })
                    # Check if this is the best model based on selection metric
                    current_metric = val_metrics[self.selection_metric]
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.best_weights_ansatz = self.weights_ansatz.copy()
                        self.best_weights_missing = self.weights_missing.copy()
                        self.best_weights_classical = {
                            'W1': self.W1.copy(),
                            'b1': self.b1.copy(),
                            'W2': self.W2.copy(),
                            'b2': self.b2.copy()
                        }
                        self.best_step = step
                        patience_counter = 0
                        
                        if checkpoint_dir:
                            checkpoint_data = {
                                'step': step,
                                'weights_quantum': {
                                    'weights_ansatz': self.best_weights_ansatz,
                                    'weights_missing': self.best_weights_missing
                                },
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
                            save_best_checkpoint(checkpoint_dir, checkpoint_data)
                    else:
                        patience_counter += 1
                
            # Periodic checkpoint
            if checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                checkpoint_data = {
                    'step': step,
                    'weights_quantum': {
                        'weights_ansatz': self.weights_ansatz,
                        'weights_missing': self.weights_missing
                    },
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
                save_periodic_checkpoint(checkpoint_dir, step, checkpoint_data, self.keep_last_n)
                
                # Save history CSV and plots
                if len(history['train_loss']) > 0:
                    save_metrics_to_csv(history, checkpoint_dir)
                    plot_training_curves(history, checkpoint_dir)

            if self.verbose and (step % self.validation_frequency == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                val_info = ""
                if X_val_scaled is not None and len(history['val_loss']) > 0:
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
        if self.best_weights_ansatz is not None:
            self.weights_ansatz = self.best_weights_ansatz
            self.weights_missing = self.best_weights_missing
            if self.best_weights_classical is not None:
                self.W1 = self.best_weights_classical['W1']
                self.b1 = self.best_weights_classical['b1']
                self.W2 = self.best_weights_classical['W2']
                self.b2 = self.best_weights_classical['b2']
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with {self.selection_metric}: {self.best_metric:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities using quantum circuit and classical readout."""
        X_scaled, is_missing_mask = X
        qcircuit = self._get_circuit()
        quantum_outputs = np.array([qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
                                   for f, m in zip(X_scaled, is_missing_mask)])
        
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

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class ConditionalMulticlassQuantumClassifierDataReuploadingFS(BaseEstimator, ClassifierMixin):
    """Data Re-uploading Conditional Multiclass QVC with classical readout.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of ansatz layers.
        n_classes (int): Number of output classes.
        learning_rate (float): Learning rate for optimizer.
        steps (int): Number of training steps.
        verbose (bool): Enable verbose logging.
        checkpoint_dir (str): Primary directory for saving checkpoints.
        checkpoint_fallback_dir (str): Fallback directory if primary is read-only.
        checkpoint_frequency (int): Save checkpoint every N steps.
        keep_last_n (int): Keep only last N checkpoints.
        max_training_time (float): Maximum training time in hours (overrides steps).
        hidden_size (int): Size of hidden layer in classical readout.
        readout_activation (str): Activation function ('tanh', 'relu', 'linear').
        selection_metric (str): Metric for best model selection.
        resume (str): Resume mode ('auto', 'latest', 'best').
        validation_frac (float): Fraction of data for validation.
        validation_frequency (int): Compute validation metrics every N steps.
        patience (int): Early stopping patience (steps without improvement).
        use_wandb (bool): Enable Weights & Biases logging.
        wandb_project (str): W&B project name.
        wandb_run_name (str): W&B run name.
    """
    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_fallback_dir=None, checkpoint_frequency=10, keep_last_n=3, 
                 max_training_time=None, hidden_size=16, readout_activation='tanh', selection_metric='f1_weighted',
                 resume=None, validation_frac=0.1, validation_frequency=10, patience=None,
                 use_wandb=False, wandb_project=None, wandb_run_name=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_fallback_dir = checkpoint_fallback_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.max_training_time = max_training_time
        self.hidden_size = hidden_size
        self.readout_activation = readout_activation
        self.selection_metric = selection_metric
        self.resume = resume
        self.validation_frac = validation_frac
        self.validation_frequency = validation_frequency
        self.patience = patience
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        
        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Quantum weights - measurements from all qubits
        self.n_meas = self.n_qubits
        self.weights_ansatz = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.weights_missing = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        
        # Classical readout weights
        self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
        self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
        self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
        self.b2 = np.array(np.zeros(n_classes), requires_grad=True)
        
        self.best_weights_ansatz = None
        self.best_weights_missing = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
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
        Supports checkpointing, resume modes, validation split, and comprehensive metrics logging.
        """
        # Store the classes seen during fit (required by sklearn)
        self.classes_ = np.unique(y)
        
        # Set default fallback dir based on checkpoint dir name
        if self.checkpoint_dir and not self.checkpoint_fallback_dir:
            checkpoint_name = os.path.basename(self.checkpoint_dir.rstrip('/'))
            self.checkpoint_fallback_dir = checkpoint_name
        
        # Ensure checkpoint directory is writable
        checkpoint_dir, used_fallback = _ensure_writable_checkpoint_dir(
            self.checkpoint_dir, self.checkpoint_fallback_dir
        )
        
        # Initialize W&B logging if requested
        wandb = _initialize_wandb(
            self.use_wandb, self.wandb_project, self.wandb_run_name,
            config_dict={
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'n_classes': self.n_classes,
                'learning_rate': self.learning_rate,
                'steps': self.steps,
                'hidden_size': self.hidden_size,
                'readout_activation': self.readout_activation
            }
        )
        
        # Unpack the conditional input
        X_scaled, is_missing_mask = X
        
        # Split into train/validation if requested
        if self.validation_frac > 0:
            X_train_scaled, X_val_scaled, mask_train, mask_val, y_train, y_val = train_test_split(
                X_scaled, is_missing_mask, y, test_size=self.validation_frac, stratify=y, random_state=42
            )
        else:
            X_train_scaled, mask_train, y_train = X_scaled, is_missing_mask, y
            X_val_scaled, mask_val, y_val = None, None, None
        
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
        if self.resume and checkpoint_dir:
            checkpoint_path = None
            
            if self.resume == 'best':
                checkpoint_path = find_best_checkpoint(checkpoint_dir)
            elif self.resume == 'latest':
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            elif self.resume == 'auto':
                # Try latest first, fall back to best
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
                if checkpoint_path is None:
                    checkpoint_path = find_best_checkpoint(checkpoint_dir)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    checkpoint = load_checkpoint(checkpoint_path)
                    weights_quantum = checkpoint['weights_quantum']
                    self.weights_ansatz = weights_quantum['weights_ansatz']
                    self.weights_missing = weights_quantum['weights_missing']
                    
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
            def cost(w_ansatz, w_missing, w1, b1, w2, b2):
                quantum_outputs = np.array([qcircuit(f, m, w_ansatz, w_missing) 
                                          for f, m in zip(X_train_scaled, mask_train)])
                
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
            (self.weights_ansatz, self.weights_missing, self.W1, self.b1, self.W2, self.b2), current_loss = opt.step_and_cost(
                cost, self.weights_ansatz, self.weights_missing, self.W1, self.b1, self.W2, self.b2
            )
            
            # Compute training metrics periodically
            if step % self.validation_frequency == 0 or step == 0:
                train_preds = self.predict((X_train_scaled, mask_train))
                train_metrics = compute_metrics(y_train, train_preds, self.n_classes)
                
                history['train_loss'].append(float(current_loss))
                history['train_acc'].append(train_metrics['accuracy'])
                
                # Validation metrics
                if X_val_scaled is not None:
                    val_preds = self.predict((X_val_scaled, mask_val))
                    val_metrics = compute_metrics(y_val, val_preds, self.n_classes)
                    
                    # Compute validation loss
                    val_quantum_outputs = np.array([qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
                                                   for f, m in zip(X_val_scaled, mask_val)])
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
                    
                    
                    # Log validation metrics to W&B
                    if wandb:
                        wandb.log({
                            'step': step,
                            'train_loss': history['train_loss'][-1],
                            'train_acc': history['train_acc'][-1],
                            'val_loss': history['val_loss'][-1],
                            'val_acc': history['val_acc'][-1],
                            'val_f1_weighted': history['val_f1_weighted'][-1],
                            'val_prec_weighted': history['val_prec_weighted'][-1],
                            'val_rec_weighted': history['val_rec_weighted'][-1]
                        })
                    # Check if this is the best model based on selection metric
                    current_metric = val_metrics[self.selection_metric]
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.best_weights_ansatz = self.weights_ansatz.copy()
                        self.best_weights_missing = self.weights_missing.copy()
                        self.best_weights_classical = {
                            'W1': self.W1.copy(),
                            'b1': self.b1.copy(),
                            'W2': self.W2.copy(),
                            'b2': self.b2.copy()
                        }
                        self.best_step = step
                        patience_counter = 0
                        
                        if checkpoint_dir:
                            checkpoint_data = {
                                'step': step,
                                'weights_quantum': {
                                    'weights_ansatz': self.best_weights_ansatz,
                                    'weights_missing': self.best_weights_missing
                                },
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
                            save_best_checkpoint(checkpoint_dir, checkpoint_data)
                    else:
                        patience_counter += 1
                
            # Periodic checkpoint
            if checkpoint_dir and step > 0 and step % self.checkpoint_frequency == 0:
                checkpoint_data = {
                    'step': step,
                    'weights_quantum': {
                        'weights_ansatz': self.weights_ansatz,
                        'weights_missing': self.weights_missing
                    },
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
                save_periodic_checkpoint(checkpoint_dir, step, checkpoint_data, self.keep_last_n)
                
                # Save history CSV and plots
                if len(history['train_loss']) > 0:
                    save_metrics_to_csv(history, checkpoint_dir)
                    plot_training_curves(history, checkpoint_dir)

            if self.verbose and (step % self.validation_frequency == 0 or step == self.steps - 1):
                elapsed = time.time() - start_time
                val_info = ""
                if X_val_scaled is not None and len(history['val_loss']) > 0:
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
        if self.best_weights_ansatz is not None:
            self.weights_ansatz = self.best_weights_ansatz
            self.weights_missing = self.best_weights_missing
            if self.best_weights_classical is not None:
                self.W1 = self.best_weights_classical['W1']
                self.b1 = self.best_weights_classical['b1']
                self.W2 = self.best_weights_classical['W2']
                self.b2 = self.best_weights_classical['b2']
            log.info(f"  [QML Training] Loaded best weights from step {self.best_step} with {self.selection_metric}: {self.best_metric:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities using quantum circuit and classical readout."""
        X_scaled, is_missing_mask = X
        qcircuit = self._get_circuit()
        quantum_outputs = np.array([qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
                                   for f, m in zip(X_scaled, is_missing_mask)])
        
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
        
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
