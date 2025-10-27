import pennylane as qml
from pennylane import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import os
import joblib
from joblib import Parallel, delayed
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


# Small activation helpers at module level (picklable)
def relu(x):
    return np.maximum(0, x)


def identity(x):
    return x


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
    
    Note: Threading backend is used for fallback parallelism. If device is not thread-safe,
    set n_jobs=1 to force sequential execution.
    """

    def __init__(self, n_qubits=8, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_fallback_dir=None, checkpoint_frequency=10, keep_last_n=3,
                 max_training_time=None, hidden_size=16, readout_activation='tanh', selection_metric='f1_weighted',
                 resume=None, validation_frac=0.1, validation_frequency=10, patience=None,
                 use_wandb=False, wandb_project=None, wandb_run_name=None, n_jobs=-1):
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
        self.n_jobs = n_jobs  # used by joblib parallel fallback

        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Cache the qnode once; reuse to avoid repeated re-creation overhead
        self._qcircuit = self._get_circuit()

        # Quantum weights - measurements from all qubits
        self.n_meas = self.n_qubits
        # Use pennylane.numpy random initializations with requires_grad for qnode compatibility
        self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)

        # Classical readout weights (initialized small)
        self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
        self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
        self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
        self.b2 = np.array(np.zeros(n_classes), requires_grad=True)
        
        # Shape validation flag to track if we've done it once
        self._shape_validated = False

        # Choose activation function once and store callable on instance for low overhead
        if self.readout_activation == 'tanh':
            self._activation_fn = np.tanh
        elif self.readout_activation == 'relu':
            self._activation_fn = relu
        else:
            self._activation_fn = identity

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
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return qcircuit

    def __getstate__(self):
        """Prevent pickling of the cached QNode (nested function)."""
        state = self.__dict__.copy()
        # _qcircuit (a QNode) is not picklable when it's a nested function; remove it
        state.pop('_qcircuit', None)
        # _activation_fn may be a wrapped numpy/pennylane function (non-picklable); remove and recreate on unpickle
        state.pop('_activation_fn', None)
        return state

    def __setstate__(self, state):
        # Restore state and recreate cached QNode lazily
        self.__dict__.update(state)
        try:
            self._qcircuit = self._get_circuit()
        except Exception:
            # If device or pennylane isn't available during unpickle, set to None
            self._qcircuit = None
        # Recreate activation callable from readout_activation
        try:
            if getattr(self, 'readout_activation', 'tanh') == 'tanh':
                self._activation_fn = np.tanh
            elif getattr(self, 'readout_activation', None) == 'relu':
                self._activation_fn = relu
            else:
                self._activation_fn = identity
        except Exception:
            # Fallback
            self._activation_fn = identity

    def _activation(self, x):
        """Compatibility wrapper if other code calls _activation; delegates to stored callable."""
        return self._activation_fn(x)

    def _classical_readout(self, quantum_output):
        """Apply classical readout head to quantum measurements (single-sample path)."""
        # quantum_output shape: (n_meas,)
        hidden = self._activation_fn(np.dot(quantum_output, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        return logits

    def _batched_qcircuit(self, X, weights, n_jobs=None):
        """Batched wrapper: try true batched qnode first, otherwise parallel per-sample.

        Returns:
            np.ndarray shape (N, n_meas)
        """
        qcircuit = getattr(self, '_qcircuit', None)
        if qcircuit is None:
            qcircuit = self._get_circuit()
            self._qcircuit = qcircuit

        X_arr = np.asarray(X, dtype=np.float64)

        # Single sample
        if X_arr.ndim == 1:
            qout = qcircuit(X_arr, weights)
            return np.asarray(qout, dtype=np.float64).reshape(1, -1)

        # Empty batch
        if X_arr.shape[0] == 0:
            return np.empty((0, self.n_meas), dtype=np.float64)

        N = X_arr.shape[0]

        # Fast path: try batched call
        try:
            qouts = qcircuit(X_arr, weights)
            qouts = np.asarray(qouts, dtype=np.float64)
            if qouts.ndim == 2 and qouts.shape[0] == N:
                log.debug("qcircuit batched fast-path used")
                return qouts
            # else fall through to sequential
        except Exception as e:
            log.debug(f"qcircuit batched call failed, triggering fallback to sequential evaluation: {e}")

        # Fallback: sequential per-sample evaluation (safe for PennyLane qnodes)
        log.debug("Falling back to sequential per-sample QNode evaluation")
        results = [qcircuit(X_arr[i], weights) for i in range(N)]
        stacked = np.vstack([np.asarray(r, dtype=np.float64) for r in results])
        return stacked

    def _softmax(self, x):
        """Numerically stable softmax. Accepts 1D (K,) or 2D (N, K)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            z = x - np.max(x)
            e = np.exp(z)
            return e / e.sum()
        z = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

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
        
        # Ensure X is at least 2D to avoid per-iteration type surprises
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        
        # Split into train/validation if requested
        if self.validation_frac > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_frac, stratify=y, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
        # Check for empty training set after split
        if X_train.shape[0] == 0:
            raise ValueError("Empty training set after split; reduce validation_frac or provide more data")
        
        # Ensure validation data is also at least 2D
        if X_val is not None:
            X_val = np.atleast_2d(np.asarray(X_val, dtype=np.float64))
        
        y_train_one_hot = np.eye(self.n_classes)[y_train]
        if y_val is not None:
            y_val_one_hot = np.eye(self.n_classes)[y_val]
        
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
                # get batched quantum outputs: shape (N_train, n_meas)
                quantum_outputs = self._batched_qcircuit(X_train, w_quantum)

                # vectorized classical readout using stored activation callable

                # Shape validation (done once)
                if not self._shape_validated:
                    if w1.shape[0] != self.n_meas:
                        raise ValueError(f"W1 shape mismatch: expected {self.n_meas} rows, got {w1.shape[0]}")
                    hidden_size = w1.shape[1]
                    if w2.shape[0] != hidden_size:
                        raise ValueError(f"W2 shape mismatch: expected {hidden_size} rows, got {w2.shape[0]}")
                    if w2.shape[1] != self.n_classes:
                        raise ValueError(f"W2 output dim ({w2.shape[1]}) != n_classes ({self.n_classes})")
                    self._shape_validated = True

                hidden = self._activation_fn(np.dot(quantum_outputs, w1) + b1)  # (N_train, hidden)
                logits_array = np.dot(hidden, w2) + b2                          # (N_train, n_classes)

                # batch softmax -> (N_train, n_classes)
                probabilities = self._softmax(logits_array)

                # Cross-entropy loss: sum over classes per sample, then mean
                eps = 1e-9
                per_sample_loss = -np.sum(y_train_one_hot * np.log(probabilities + eps), axis=1)
                # Detect samples that are all-zero (these correspond to missing data rows
                # that were included to keep datasets aligned). Ignore them during loss
                # computation so they don't contribute to backprop, but still run forward
                # so their predictions are available for the meta-learner.
                try:
                    sample_has_data = np.any(X_train != 0, axis=1)
                except Exception:
                    # If X_train isn't array-like for some reason, fall back to using all samples
                    sample_has_data = np.ones(per_sample_loss.shape[0], dtype=bool)

                if np.any(sample_has_data):
                    loss = np.mean(per_sample_loss[sample_has_data])
                else:
                    # If no samples have data (degenerate), fall back to mean of per-sample loss
                    loss = np.mean(per_sample_loss)
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
                    val_quantum_outputs = self._batched_qcircuit(X_val, self.weights)  # (N_val, n_meas)
                    val_hidden = self._activation_fn(np.dot(val_quantum_outputs, self.W1) + self.b1)
                    val_logits_array = np.dot(val_hidden, self.W2) + self.b2
                    val_probs = self._softmax(val_logits_array)
                    val_loss = -np.mean(np.sum(y_val_one_hot * np.log(val_probs + 1e-9), axis=1))
                    
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
        """Predict class probabilities using quantum circuit and classical readout.

        Always returns a 2D array: (N, n_classes). For a single sample input it returns shape (1, n_classes).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        X_batch = np.atleast_2d(X_arr)   # 1D -> (1, K), 2D unchanged
        # optional: handle empty batch
        if X_batch.shape[0] == 0:
            return np.empty((0, self.n_classes), dtype=np.float64)

        # get quantum outputs shape (N, n_meas)
        quantum_outputs = self._batched_qcircuit(X_batch, self.weights)

        # vectorized classical readout
        W1 = np.asarray(self.W1, dtype=np.float64)
        W2 = np.asarray(self.W2, dtype=np.float64)
        b1 = np.asarray(self.b1, dtype=np.float64).reshape(1, -1)
        b2 = np.asarray(self.b2, dtype=np.float64).reshape(1, -1)

        hidden = self._activation_fn(np.dot(quantum_outputs, W1) + b1)
        logits_array = np.dot(hidden, W2) + b2  # (N, n_classes)

        if not np.isfinite(logits_array).all():
            raise FloatingPointError("Non-finite values detected in logits (NaN or Inf).")

        probs = self._softmax(logits_array)  # (N, n_classes)
        return probs

    def predict(self, X):
        """Predict class labels. Returns 1D array of length N."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class GatedMulticlassQuantumClassifierDR(BaseEstimator, ClassifierMixin):
    """Gated Multiclass VQC that uses indicator features as classical gates to
    include/exclude base-learner outputs before encoding them into the quantum circuit.

    This class purposefully does NOT embed the indicator features as qubits;
    instead the indicators (binary mask) are applied to the base-learner
    probability vectors (element-wise multiplication) prior to AngleEmbedding.

    The expected input for fit()/predict()/predict_proba() is a tuple:
        (base_preds, indicator_mask)
    where:
        - base_preds: array-like shape (N, n_base_preds) with probabilities [0,1]
        - indicator_mask: array-like shape (N, n_base_preds) with 0/1 values

    By skipping indicator encoding into the quantum circuit we reduce the
    number of qubits required by the count of indicator features.
    """

    def __init__(self, n_qubits=None, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_fallback_dir=None, checkpoint_frequency=10, keep_last_n=3,
                 max_training_time=None, hidden_size=16, readout_activation='tanh', selection_metric='f1_weighted',
                 resume=None, validation_frac=0.1, validation_frequency=10, patience=None,
                 use_wandb=False, wandb_project=None, wandb_run_name=None, n_jobs=-1):
        # n_qubits may be None here; we will infer it from the base_preds shape during fit if so
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
        self.n_jobs = n_jobs

        # device created lazily when we know number of qubits
        self.dev = None
        self._qcircuit = None

        # quantum parameters (initialized later)
        self.weights = None
        self.n_meas = None

        # Classical readout (init later when n_meas known)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self._shape_validated = False
        if self.readout_activation == 'tanh':
            self._activation_fn = np.tanh
        elif self.readout_activation == 'relu':
            self._activation_fn = relu
        else:
            self._activation_fn = identity

        self.best_weights = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
        self.best_step = 0
        self.checkpoint_history = []

    def _get_circuit(self):
        assert self.dev is not None, "Quantum device not initialized"

        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(inputs, weights):
            # inputs expected shape: (n_qubits,) and already masked by indicators
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return qcircuit

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_qcircuit', None)
        state.pop('_activation_fn', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            if getattr(self, 'n_qubits', None) is not None:
                self.dev = qml.device('default.qubit', wires=self.n_qubits)
                self._qcircuit = self._get_circuit()
        except Exception:
            self._qcircuit = None
        try:
            if getattr(self, 'readout_activation', 'tanh') == 'tanh':
                self._activation_fn = np.tanh
            elif getattr(self, 'readout_activation', None) == 'relu':
                self._activation_fn = relu
            else:
                self._activation_fn = identity
        except Exception:
            self._activation_fn = identity

    def _initialize_params_if_needed(self, base_preds_shape):
        # base_preds_shape: (N, n_base)
        n_base = int(base_preds_shape[1])
        if self.n_qubits is None:
            self.n_qubits = n_base
        if self.dev is None:
            self.dev = qml.device('default.qubit', wires=self.n_qubits)
        if self._qcircuit is None:
            self._qcircuit = self._get_circuit()
        if self.weights is None:
            self.n_meas = self.n_qubits
            self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        if self.W1 is None:
            self.W1 = np.array(np.random.randn(self.n_meas, self.hidden_size) * 0.01, requires_grad=True)
            self.b1 = np.array(np.zeros(self.hidden_size), requires_grad=True)
            self.W2 = np.array(np.random.randn(self.hidden_size, self.n_classes) * 0.01, requires_grad=True)
            self.b2 = np.array(np.zeros(self.n_classes), requires_grad=True)

    def _batched_qcircuit(self, X_masked, weights, n_jobs=None):
        """X_masked: array (N, n_qubits) already masked by indicators."""
        qcircuit = getattr(self, '_qcircuit', None)
        if qcircuit is None:
            qcircuit = self._get_circuit()
            self._qcircuit = qcircuit

        X_arr = np.asarray(X_masked, dtype=np.float64)
        # delegate to same logic as DR model
        if X_arr.ndim == 1:
            qout = qcircuit(X_arr, weights)
            return np.asarray(qout, dtype=np.float64).reshape(1, -1)
        if X_arr.shape[0] == 0:
            return np.empty((0, self.n_meas), dtype=np.float64)

        N = X_arr.shape[0]
        try:
            qouts = qcircuit(X_arr, weights)
            qouts = np.asarray(qouts, dtype=np.float64)
            if qouts.ndim == 2 and qouts.shape[0] == N:
                return qouts
        except Exception:
            pass

        results = [qcircuit(X_arr[i], weights) for i in range(N)]
        stacked = np.vstack([np.asarray(r, dtype=np.float64) for r in results])
        return stacked

    def _classical_readout(self, quantum_output):
        hidden = self._activation_fn(np.dot(quantum_output, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        return logits

    def fit(self, X, y):
        """X is expected to be a tuple: (base_preds, indicator_mask)

        This method mirrors the behavior, checkpoints, resume logic and
        validation used in `MulticlassQuantumClassifierDR.fit` but applies
        the classical indicator mask to the base-learner predictions before
        sending them to the quantum circuit.
        """
        base_preds, mask = X
        base_preds = np.atleast_2d(np.asarray(base_preds, dtype=np.float64))
        mask = np.atleast_2d(np.asarray(mask, dtype=np.float64))

        # Ensure shapes align
        if base_preds.shape != mask.shape:
            raise ValueError(f"base_preds shape {base_preds.shape} and mask shape {mask.shape} must match")

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

        # Initialize parameters now we know number of base features
        self._initialize_params_if_needed(base_preds.shape)

        # Split into train/validation if requested
        if self.validation_frac > 0:
            X_train_base, X_val_base, mask_train, mask_val, y_train, y_val = train_test_split(
                base_preds, mask, y, test_size=self.validation_frac, stratify=y, random_state=42
            )
        else:
            X_train_base, mask_train, y_train = base_preds, mask, y
            X_val_base, mask_val, y_val = None, None, None

        # Check for empty training set after split
        if X_train_base.shape[0] == 0:
            raise ValueError("Empty training set after split; reduce validation_frac or provide more data")

        # Ensure validation data is also at least 2D
        if X_val_base is not None:
            X_val_base = np.atleast_2d(np.asarray(X_val_base, dtype=np.float64))
            mask_val = np.atleast_2d(np.asarray(mask_val, dtype=np.float64))

        y_train_one_hot = np.eye(self.n_classes)[y_train]
        if y_val is not None:
            y_val_one_hot = np.eye(self.n_classes)[y_val]

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
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
                if checkpoint_path is None:
                    checkpoint_path = find_best_checkpoint(checkpoint_dir)

            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    checkpoint = load_checkpoint(checkpoint_path)
                    self.weights = checkpoint['weights_quantum']
                    if 'weights_classical' in checkpoint:
                        self.W1 = checkpoint['weights_classical']['W1']
                        self.b1 = checkpoint['weights_classical']['b1']
                        self.W2 = checkpoint['weights_classical']['W2']
                        self.b2 = checkpoint['weights_classical']['b2']
                    if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                        opt.set_state(checkpoint['optimizer_state'])
                        start_step = checkpoint.get('step', 0) + 1
                        log.info(f"  Resumed from step {start_step} with optimizer state")
                    else:
                        opt.set_lr(self.learning_rate * 0.1)
                        log.info(f"  Loaded weights but no optimizer state - reduced LR to {opt.lr}")
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
            def cost(w_quantum, w1, b1, w2, b2):
                # Apply mask to training inputs
                X_train_masked = X_train_base * mask_train
                quantum_outputs = self._batched_qcircuit(X_train_masked, w_quantum)

                # Shape validation (done once)
                if not self._shape_validated:
                    if w1.shape[0] != self.n_meas:
                        raise ValueError(f"W1 shape mismatch: expected {self.n_meas} rows, got {w1.shape[0]}")
                    hidden_size = w1.shape[1]
                    if w2.shape[0] != hidden_size:
                        raise ValueError(f"W2 shape mismatch: expected {hidden_size} rows, got {w2.shape[0]}")
                    if w2.shape[1] != self.n_classes:
                        raise ValueError(f"W2 output dim ({w2.shape[1]}) != n_classes ({self.n_classes})")
                    self._shape_validated = True

                hidden = self._activation_fn(np.dot(quantum_outputs, w1) + b1)
                logits_array = np.dot(hidden, w2) + b2
                probabilities = self._softmax(logits_array)

                eps = 1e-9
                per_sample_loss = -np.sum(y_train_one_hot * np.log(probabilities + eps), axis=1)

                try:
                    sample_has_data = np.any(X_train_masked != 0, axis=1)
                except Exception:
                    sample_has_data = np.ones(per_sample_loss.shape[0], dtype=bool)

                if np.any(sample_has_data):
                    loss = np.mean(per_sample_loss[sample_has_data])
                else:
                    loss = np.mean(per_sample_loss)
                return loss

            (self.weights, self.W1, self.b1, self.W2, self.b2), current_loss = opt.step_and_cost(
                cost, self.weights, self.W1, self.b1, self.W2, self.b2
            )

            # Compute training metrics periodically
            if step % self.validation_frequency == 0 or step == 0:
                train_preds = self.predict((X_train_base, mask_train))
                train_metrics = compute_metrics(y_train, train_preds, self.n_classes)

                history['train_loss'].append(float(current_loss))
                history['train_acc'].append(train_metrics['accuracy'])

                # Validation metrics
                if X_val_base is not None:
                    val_preds = self.predict((X_val_base, mask_val))
                    val_metrics = compute_metrics(y_val, val_preds, self.n_classes)

                    # Compute validation loss
                    X_val_masked = X_val_base * mask_val
                    val_quantum_outputs = self._batched_qcircuit(X_val_masked, self.weights)
                    val_hidden = self._activation_fn(np.dot(val_quantum_outputs, self.W1) + self.b1)
                    val_logits_array = np.dot(val_hidden, self.W2) + self.b2
                    val_probs = self._softmax(val_logits_array)
                    val_loss = -np.mean(np.sum(y_val_one_hot * np.log(val_probs + 1e-9), axis=1))

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
                if X_val_base is not None and len(history['val_loss']) > 0:
                    val_info = f" - Val Loss: {history['val_loss'][-1]:.4f} - Val {self.selection_metric}: {history[f'val_{self.selection_metric}'][-1]:.4f}"
                log.info(f"  [GatedQML Training] Step {step:>{len(str(self.steps))}}/{self.steps} - Loss: {current_loss:.4f}{val_info} - Time: {elapsed:.1f}s")

            step += 1

            # Check stopping conditions
            if self.patience and patience_counter >= self.patience:
                log.info(f"  Early stopping triggered at step {step} (patience={self.patience})")
                break

            if self.max_training_time:
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= self.max_training_time:
                    log.info(f"  [GatedQML Training] Reached max training time of {self.max_training_time:.2f} hours at step {step}")
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
            log.info(f"  [GatedQML Training] Loaded best weights from step {self.best_step} with {self.selection_metric}: {self.best_metric:.4f}")

        return self

    def predict_proba(self, X):
        base_preds, mask = X
        base_preds = np.atleast_2d(np.asarray(base_preds, dtype=np.float64))
        mask = np.atleast_2d(np.asarray(mask, dtype=np.float64))
        if base_preds.shape != mask.shape:
            raise ValueError("base_preds and mask must have same shape")

        X_masked = base_preds * mask
        quantum_outputs = self._batched_qcircuit(X_masked, self.weights)
        hidden = self._activation_fn(np.dot(quantum_outputs, self.W1) + self.b1)
        logits_array = np.dot(hidden, self.W2) + self.b2
        if not np.isfinite(logits_array).all():
            raise FloatingPointError("Non-finite values detected in logits (NaN or Inf).")

        return self._softmax(logits_array)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class GatedMulticlassQuantumClassifierDataReuploadingDR(BaseEstimator, ClassifierMixin):
    """Gated Data-Reuploading Multiclass VQC.

    This mirrors `MulticlassQuantumClassifierDataReuploadingDR` but applies a
    classical indicator mask to the base-learner predictions before each
    re-uploading step. Input for fit()/predict() is a tuple
        (base_preds, indicator_mask)
    with shapes (N, n_base) and (N, n_base) respectively.
    """

    def __init__(self, n_qubits=None, n_layers=3, n_classes=3, learning_rate=0.1, steps=50, verbose=False,
                 checkpoint_dir=None, checkpoint_fallback_dir=None, checkpoint_frequency=10, keep_last_n=3,
                 max_training_time=None, hidden_size=16, readout_activation='tanh', selection_metric='f1_weighted',
                 resume=None, validation_frac=0.1, validation_frequency=10, patience=None,
                 use_wandb=False, wandb_project=None, wandb_run_name=None, n_jobs=-1):
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
        self.n_jobs = n_jobs

        # lazy device and qnode initialization
        self.dev = None
        self._qcircuit = None

        # quantum and classical parameters (initialized later)
        self.weights = None
        self.n_meas = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self._shape_validated = False
        if self.readout_activation == 'tanh':
            self._activation_fn = np.tanh
        elif self.readout_activation == 'relu':
            self._activation_fn = relu
        else:
            self._activation_fn = identity

        self.best_weights = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
        self.best_step = 0
        self.checkpoint_history = []

    def _get_circuit(self):
        assert self.dev is not None, "Quantum device not initialized"

        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(inputs, weights):
            # Data reuploading: encode inputs at each layer
            for layer in range(self.n_layers):
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
                qml.BasicEntanglerLayers(weights[layer:layer+1], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return qcircuit

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_qcircuit', None)
        state.pop('_activation_fn', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            if getattr(self, 'n_qubits', None) is not None:
                self.dev = qml.device('default.qubit', wires=self.n_qubits)
                self._qcircuit = self._get_circuit()
        except Exception:
            self._qcircuit = None
        try:
            if getattr(self, 'readout_activation', 'tanh') == 'tanh':
                self._activation_fn = np.tanh
            elif getattr(self, 'readout_activation', None) == 'relu':
                self._activation_fn = relu
            else:
                self._activation_fn = identity
        except Exception:
            self._activation_fn = identity

    def _initialize_params_if_needed(self, base_preds_shape):
        n_base = int(base_preds_shape[1])
        if self.n_qubits is None:
            self.n_qubits = n_base
        if self.dev is None:
            self.dev = qml.device('default.qubit', wires=self.n_qubits)
        if self._qcircuit is None:
            self._qcircuit = self._get_circuit()
        if self.weights is None:
            self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
            self.n_meas = self.n_qubits
        if self.W1 is None:
            self.W1 = np.array(np.random.randn(self.n_meas, self.hidden_size) * 0.01, requires_grad=True)
            self.b1 = np.array(np.zeros(self.hidden_size), requires_grad=True)
            self.W2 = np.array(np.random.randn(self.hidden_size, self.n_classes) * 0.01, requires_grad=True)
            self.b2 = np.array(np.zeros(self.n_classes), requires_grad=True)

    def _batched_qcircuit(self, X_masked, weights, n_jobs=None):
        qcircuit = getattr(self, '_qcircuit', None)
        if qcircuit is None:
            qcircuit = self._get_circuit()
            self._qcircuit = qcircuit

        X_arr = np.asarray(X_masked, dtype=np.float64)
        if X_arr.ndim == 1:
            qout = qcircuit(X_arr, weights)
            return np.asarray(qout, dtype=np.float64).reshape(1, -1)
        if X_arr.shape[0] == 0:
            return np.empty((0, self.n_meas), dtype=np.float64)

        N = X_arr.shape[0]
        try:
            qouts = qcircuit(X_arr, weights)
            qouts = np.asarray(qouts, dtype=np.float64)
            if qouts.ndim == 2 and qouts.shape[0] == N:
                return qouts
        except Exception:
            pass

        results = [qcircuit(X_arr[i], weights) for i in range(N)]
        stacked = np.vstack([np.asarray(r, dtype=np.float64) for r in results])
        return stacked

    def _softmax(self, x):
        X = np.asarray(x, dtype=np.float64)
        if X.ndim == 1:
            shift = X - np.max(X)
            exp_shift = np.exp(shift)
            return exp_shift / np.sum(exp_shift)
        elif X.ndim == 2:
            shift = X - np.max(X, axis=1, keepdims=True)
            exp_shift = np.exp(shift)
            return exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        else:
            raise ValueError("softmax input must be 1D or 2D array")

    def _classical_readout(self, quantum_output):
        hidden = self._activation_fn(np.dot(quantum_output, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        return logits

    def fit(self, X, y):
        base_preds, mask = X
        base_preds = np.atleast_2d(np.asarray(base_preds, dtype=np.float64))
        mask = np.atleast_2d(np.asarray(mask, dtype=np.float64))
        if base_preds.shape != mask.shape:
            raise ValueError("base_preds and mask must have same shape")

        self.classes_ = np.unique(y)
        self._initialize_params_if_needed(base_preds.shape)

        # Split train/val if requested
        if self.validation_frac > 0:
            X_train, X_val, mask_train, mask_val, y_train, y_val = train_test_split(
                base_preds, mask, y, test_size=self.validation_frac, stratify=y, random_state=42
            )
        else:
            X_train, mask_train, y_train = base_preds, mask, y
            X_val, mask_val, y_val = None, None, None

        y_train_one_hot = np.eye(self.n_classes)[y_train]
        if y_val is not None:
            y_val_one_hot = np.eye(self.n_classes)[y_val]

        opt = AdamSerializable(lr=self.learning_rate)

        history = {'train_loss': [], 'val_loss': []}

        start_step = 0
        step = start_step
        patience_counter = 0

        while True:
            def cost(w_quantum, w1, b1, w2, b2):
                X_masked = X_train * mask_train
                quantum_outputs = self._batched_qcircuit(X_masked, w_quantum)

                if not self._shape_validated:
                    if w1.shape[0] != self.n_meas:
                        raise ValueError(f"W1 shape mismatch: expected {self.n_meas} rows, got {w1.shape[0]}")
                    hidden_size = w1.shape[1]
                    if w2.shape[0] != hidden_size:
                        raise ValueError(f"W2 shape mismatch: expected {hidden_size} rows, got {w2.shape[0]}")
                    if w2.shape[1] != self.n_classes:
                        raise ValueError(f"W2 output dim ({w2.shape[1]}) != n_classes ({self.n_classes})")
                    self._shape_validated = True

                hidden = self._activation_fn(np.dot(quantum_outputs, w1) + b1)
                logits_array = np.dot(hidden, w2) + b2
                probabilities = self._softmax(logits_array)
                per_sample_loss = -np.sum(y_train_one_hot * np.log(probabilities + 1e-9), axis=1)
                loss = np.mean(per_sample_loss)
                return loss

            (self.weights, self.W1, self.b1, self.W2, self.b2), current_loss = opt.step_and_cost(
                cost, self.weights, self.W1, self.b1, self.W2, self.b2
            )

            if step % self.validation_frequency == 0 or step == 0:
                history['train_loss'].append(float(current_loss))
                if X_val is not None:
                    X_masked_val = X_val * mask_val
                    val_quantum_outputs = self._batched_qcircuit(X_masked_val, self.weights)
                    val_hidden = self._activation_fn(np.dot(val_quantum_outputs, self.W1) + self.b1)
                    val_logits = np.dot(val_hidden, self.W2) + self.b2
                    val_probs = self._softmax(val_logits)
                    val_loss = -np.mean(np.sum(y_val_one_hot * np.log(val_probs + 1e-9), axis=1))
                    history['val_loss'].append(float(val_loss))
                if self.verbose:
                    log.info(f"  [GatedReupload] Step {step}/{self.steps} - Loss: {current_loss:.4f}")

            step += 1
            if self.patience and patience_counter >= self.patience:
                break
            if self.max_training_time:
                # simplified time check omitted for brevity
                pass
            if step >= self.steps:
                break

        return self

    def predict_proba(self, X):
        base_preds, mask = X
        base_preds = np.atleast_2d(np.asarray(base_preds, dtype=np.float64))
        mask = np.atleast_2d(np.asarray(mask, dtype=np.float64))
        if base_preds.shape != mask.shape:
            raise ValueError("base_preds and mask must have same shape")
        X_masked = base_preds * mask
        quantum_outputs = self._batched_qcircuit(X_masked, self.weights)
        hidden = self._activation_fn(np.dot(quantum_outputs, self.W1) + self.b1)
        logits_array = np.dot(hidden, self.W2) + self.b2
        if not np.isfinite(logits_array).all():
            raise FloatingPointError("Non-finite values detected in logits (NaN or Inf).")

        return self._softmax(logits_array)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class MulticlassQuantumClassifierDataReuploadingDR(BaseEstimator, ClassifierMixin):
    """Data Re-uploading Multiclass VQC for pre-processed, dense data with classical readout.
    
    Note: Threading backend is used for fallback parallelism. If device is not thread-safe,
    set n_jobs=1 to force sequential execution.
    
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
                 use_wandb=False, wandb_project=None, wandb_run_name=None, n_jobs=-1):
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
        self.n_jobs = n_jobs  # used by joblib parallel fallback

        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Cache the qnode once; reuse to avoid repeated re-creation overhead
        self._qcircuit = self._get_circuit()
        
        # Quantum weights - measurements from all qubits
        self.n_meas = self.n_qubits
        self.weights = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        
        # Classical readout weights
        self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
        self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
        self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
        self.b2 = np.array(np.zeros(n_classes), requires_grad=True)

        # Choose activation function once and store callable on instance for low overhead
        if self.readout_activation == 'tanh':
            self._activation_fn = np.tanh
        elif self.readout_activation == 'relu':
            self._activation_fn = relu
        else:
            self._activation_fn = identity

        self.best_weights = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
        self.best_step = 0
        self.checkpoint_history = []
        
        # Shape validation flag to track if we've done it once
        self._shape_validated = False

    def _get_circuit(self):
        @qml.qnode(self.dev, interface='autograd')
        def qcircuit(inputs, weights):
            for layer in range(self.n_layers):
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
                qml.BasicEntanglerLayers(weights[layer:layer+1], wires=range(self.n_qubits))
            # Measure all qubits for classical readout
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return qcircuit

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_qcircuit', None)
        state.pop('_activation_fn', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self._qcircuit = self._get_circuit()
        except Exception:
            self._qcircuit = None
        try:
            if getattr(self, 'readout_activation', 'tanh') == 'tanh':
                self._activation_fn = np.tanh
            elif getattr(self, 'readout_activation', None) == 'relu':
                self._activation_fn = relu
            else:
                self._activation_fn = identity
        except Exception:
            self._activation_fn = identity
    
    def _activation(self, x):
        """Compatibility wrapper if other code calls _activation; delegates to stored callable."""
        return self._activation_fn(x)
    
    def _classical_readout(self, quantum_output):
        """Apply classical readout head to quantum measurements."""
        # quantum_output shape: (n_meas,)
        hidden = self._activation_fn(np.dot(quantum_output, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        return logits

    def _batched_qcircuit(self, X, weights, n_jobs=None):
        """Batched wrapper: try true batched qnode first, otherwise parallel per-sample.

        Returns:
            np.ndarray shape (N, n_meas)
        """
        qcircuit = getattr(self, '_qcircuit', None)
        if qcircuit is None:
            qcircuit = self._get_circuit()
            self._qcircuit = qcircuit

        X_arr = np.asarray(X, dtype=np.float64)

        # Single sample
        if X_arr.ndim == 1:
            qout = qcircuit(X_arr, weights)
            return np.asarray(qout, dtype=np.float64).reshape(1, -1)

        # Empty batch
        if X_arr.shape[0] == 0:
            return np.empty((0, self.n_meas), dtype=np.float64)

        N = X_arr.shape[0]

        # Fast path: try batched call
        try:
            qouts = qcircuit(X_arr, weights)
            qouts = np.asarray(qouts, dtype=np.float64)
            if qouts.ndim == 2 and qouts.shape[0] == N:
                log.debug("qcircuit batched fast-path used")
                return qouts
            # else fall through to sequential
        except Exception as e:
            log.debug(f"qcircuit batched call failed, triggering fallback to sequential evaluation: {e}")

        # Fallback: sequential per-sample evaluation (safe for PennyLane qnodes)
        log.debug("Falling back to sequential per-sample QNode evaluation")
        results = [qcircuit(X_arr[i], weights) for i in range(N)]
        stacked = np.vstack([np.asarray(r, dtype=np.float64) for r in results])
        return stacked

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
        
        # Ensure X is at least 2D to avoid per-iteration type surprises
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        
        # Split into train/validation if requested
        if self.validation_frac > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_frac, stratify=y, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
        # Check for empty training set after split
        if X_train.shape[0] == 0:
            raise ValueError("Empty training set after split; reduce validation_frac or provide more data")
        
        # Ensure validation data is also at least 2D
        if X_val is not None:
            X_val = np.atleast_2d(np.asarray(X_val, dtype=np.float64))
        
        y_train_one_hot = np.eye(self.n_classes)[y_train]
        if y_val is not None:
            y_val_one_hot = np.eye(self.n_classes)[y_val]
        
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
                # get batched quantum outputs: shape (N_train, n_meas)
                quantum_outputs = self._batched_qcircuit(X_train, w_quantum)

                # vectorized classical readout using stored activation callable

                # Shape validation (done once)
                if not self._shape_validated:
                    if w1.shape[0] != self.n_meas:
                        raise ValueError(f"W1 shape mismatch: expected {self.n_meas} rows, got {w1.shape[0]}")
                    hidden_size = w1.shape[1]
                    if w2.shape[0] != hidden_size:
                        raise ValueError(f"W2 shape mismatch: expected {hidden_size} rows, got {w2.shape[0]}")
                    if w2.shape[1] != self.n_classes:
                        raise ValueError(f"W2 output dim ({w2.shape[1]}) != n_classes ({self.n_classes})")
                    self._shape_validated = True

                hidden = self._activation_fn(np.dot(quantum_outputs, w1) + b1)  # (N_train, hidden)
                logits_array = np.dot(hidden, w2) + b2                          # (N_train, n_classes)

                # batch softmax -> (N_train, n_classes)
                probabilities = self._softmax(logits_array)

                # Cross-entropy loss: sum over classes per sample, then mean
                eps = 1e-9
                per_sample_loss = -np.sum(y_train_one_hot * np.log(probabilities + eps), axis=1)
                try:
                    sample_has_data = np.any(X_train != 0, axis=1)
                except Exception:
                    sample_has_data = np.ones(per_sample_loss.shape[0], dtype=bool)
                if np.any(sample_has_data):
                    loss = np.mean(per_sample_loss[sample_has_data])
                else:
                    loss = np.mean(per_sample_loss)
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
                    val_quantum_outputs = self._batched_qcircuit(X_val, self.weights)  # (N_val, n_meas)
                    val_hidden = self._activation_fn(np.dot(val_quantum_outputs, self.W1) + self.b1)
                    val_logits_array = np.dot(val_hidden, self.W2) + self.b2
                    val_probs = self._softmax(val_logits_array)
                    val_loss = -np.mean(np.sum(y_val_one_hot * np.log(val_probs + 1e-9), axis=1))
                    
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
        """Predict class probabilities using quantum circuit and classical readout.

        Always returns a 2D array: (N, n_classes). For a single sample input it returns shape (1, n_classes).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        X_batch = np.atleast_2d(X_arr)   # 1D -> (1, K), 2D unchanged
        # optional: handle empty batch
        if X_batch.shape[0] == 0:
            return np.empty((0, self.n_classes), dtype=np.float64)

        # get quantum outputs shape (N, n_meas)
        quantum_outputs = self._batched_qcircuit(X_batch, self.weights)

        # vectorized classical readout
        W1 = np.asarray(self.W1, dtype=np.float64)
        W2 = np.asarray(self.W2, dtype=np.float64)
        b1 = np.asarray(self.b1, dtype=np.float64).reshape(1, -1)
        b2 = np.asarray(self.b2, dtype=np.float64).reshape(1, -1)

        hidden = self._activation_fn(np.dot(quantum_outputs, W1) + b1)
        logits_array = np.dot(hidden, W2) + b2  # (N, n_classes)

        if not np.isfinite(logits_array).all():
            raise FloatingPointError("Non-finite values detected in logits (NaN or Inf).")

        probs = self._softmax(logits_array)  # (N, n_classes)
        return probs

    def predict(self, X):
        """Predict class labels. Returns 1D array of length N."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# --- Models for Approach 2 (Conditional Encoding on Selected Features) ---

class ConditionalMulticlassQuantumClassifierFS(BaseEstimator, ClassifierMixin):
    """Conditional Multiclass QVC that expects pre-processed tuple input with classical readout.
    
    Note: Threading backend is used for fallback parallelism. If device is not thread-safe,
    set n_jobs=1 to force sequential execution.
    
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
                 use_wandb=False, wandb_project=None, wandb_run_name=None, n_jobs=-1):
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
        self.n_jobs = n_jobs  # used by joblib parallel fallback

        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Cache the qnode once; reuse to avoid repeated re-creation overhead
        self._qcircuit = self._get_circuit()
        
        # Quantum weights - measurements from all qubits
        self.n_meas = self.n_qubits
        self.weights_ansatz = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.weights_missing = np.random.uniform(0, 2 * np.pi, self.n_qubits, requires_grad=True)
        
        # Classical readout weights
        self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
        self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
        self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
        self.b2 = np.array(np.zeros(n_classes), requires_grad=True)

        # Choose activation function once and store callable on instance for low overhead
        if self.readout_activation == 'tanh':
            self._activation_fn = np.tanh
        elif self.readout_activation == 'relu':
            self._activation_fn = relu
        else:
            self._activation_fn = identity

        self.best_weights_ansatz = None
        self.best_weights_missing = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
        self.best_step = 0
        self.checkpoint_history = []
        
        # Shape validation flag to track if we've done it once
        self._shape_validated = False

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

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_qcircuit', None)
        state.pop('_activation_fn', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self._qcircuit = self._get_circuit()
        except Exception:
            self._qcircuit = None
        try:
            if getattr(self, 'readout_activation', 'tanh') == 'tanh':
                self._activation_fn = np.tanh
            elif getattr(self, 'readout_activation', None) == 'relu':
                self._activation_fn = relu
            else:
                self._activation_fn = identity
        except Exception:
            self._activation_fn = identity
    
    def _activation(self, x):
        """Compatibility wrapper if other code calls _activation; delegates to stored callable."""
        return self._activation_fn(x)
    
    def _classical_readout(self, quantum_output):
        """Apply classical readout head to quantum measurements."""
        # quantum_output shape: (n_meas,)
        hidden = self._activation_fn(np.dot(quantum_output, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        return logits

    def _batched_qcircuit(self, X, weights, n_jobs=None):
        """Batched wrapper: try true batched qnode first, otherwise parallel per-sample.

        Returns:
            np.ndarray shape (N, n_meas)
        """
        qcircuit = getattr(self, '_qcircuit', None)
        if qcircuit is None:
            qcircuit = self._get_circuit()
            self._qcircuit = qcircuit
        X_arr = np.asarray(X, dtype=np.float64)

        # Single sample
        if X_arr.ndim == 1:
            qout = qcircuit(X_arr, weights)
            return np.asarray(qout, dtype=np.float64).reshape(1, -1)

        # Empty batch
        if X_arr.shape[0] == 0:
            return np.empty((0, self.n_meas), dtype=np.float64)

        N = X_arr.shape[0]

        # Fast path: try batched call
        try:
            qouts = qcircuit(X_arr, weights)
            qouts = np.asarray(qouts, dtype=np.float64)
            if qouts.ndim == 2 and qouts.shape[0] == N:
                log.debug("qcircuit batched fast-path used")
                return qouts
            # else fall through
        except Exception as e:
            log.debug(f"qcircuit batched call failed, triggering fallback to sequential evaluation: {e}")

        # Fallback: sequential per-sample evaluation (safe for PennyLane qnodes)
        log.debug("Falling back to sequential per-sample QNode evaluation")
        results = [qcircuit(X_arr[i], weights) for i in range(N)]
        stacked = np.vstack([np.asarray(r, dtype=np.float64) for r in results])
        return stacked

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
        

        # Check for empty training set after split
        if X_train_scaled.shape[0] == 0:
            raise ValueError("Empty training set after split; reduce validation_frac or provide more data")
        
        y_train_one_hot = np.eye(self.n_classes)[y_train]
        if y_val is not None:
            y_val_one_hot = np.eye(self.n_classes)[y_val]
        
        # Use custom Adam optimizer for serializability
        opt = AdamSerializable(lr=self.learning_rate)
        # Use cached qnode for this fit session
        qcircuit = getattr(self, '_qcircuit', None)
        if qcircuit is None:
            qcircuit = self._get_circuit()
            self._qcircuit = qcircuit
        
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
                

                # Shape validation (done once)
                if not self._shape_validated:
                    if w1.shape[0] != self.n_meas:
                        raise ValueError(f"W1 shape mismatch: expected {self.n_meas} rows, got {w1.shape[0]}")
                    hidden_size = w1.shape[1]
                    if w2.shape[0] != hidden_size:
                        raise ValueError(f"W2 shape mismatch: expected {hidden_size} rows, got {w2.shape[0]}")
                    if w2.shape[1] != self.n_classes:
                        raise ValueError(f"W2 output dim ({w2.shape[1]}) != n_classes ({self.n_classes})")
                    self._shape_validated = True

                # Apply classical readout to each sample using unified activation callable
                logits_list = []
                for qout in quantum_outputs:
                    hidden = self._activation_fn(np.dot(qout, w1) + b1)
                    logits = np.dot(hidden, w2) + b2
                    logits_list.append(logits)
                
                logits_array = np.array(logits_list)
                probabilities = np.array([self._softmax(logit) for logit in logits_array])
                
                # Cross-entropy loss computed only over samples that actually have data
                per_sample_loss = -np.sum(y_train_one_hot * np.log(probabilities + 1e-9), axis=1)
                # mask_train is a per-sample/per-feature indicator (1 -> missing). A sample has data
                # if any feature is present (i.e., any mask value == 0).
                # Consider a sample as "having data" if it is not an all-zero row.
                # Some datasets include missing samples as all-zero rows (not NaN),
                # so relying only on the `mask_train` (which flags NaNs) will miss
                # those artificially zeroed samples. Treat any sample that is
                # entirely zero across features as missing for loss/backprop.
                try:
                    sample_all_zero = np.all(X_train_scaled == 0, axis=1)
                except Exception:
                    sample_all_zero = np.zeros(per_sample_loss.shape[0], dtype=bool)
                try:
                    # If a feature-mask is available, combine both checks: a sample
                    # is considered present if it's not all-zero OR mask indicates
                    # at least one present feature. This is robust to different
                    # upstream preprocessing pipelines.
                    mask_present = np.any(mask_train == 0, axis=1)
                except Exception:
                    mask_present = np.ones(per_sample_loss.shape[0], dtype=bool)

                sample_has_data = (~sample_all_zero) & mask_present
                # Fallback: if mask_present is all False but there are non-zero rows,
                # allow those through (this covers odd edge-cases).
                nonzero_any = np.any(~sample_all_zero)
                if not np.any(sample_has_data) and nonzero_any:
                    sample_has_data = ~sample_all_zero

                if np.any(sample_has_data):
                    loss = np.mean(per_sample_loss[sample_has_data])
                else:
                    loss = np.mean(per_sample_loss)
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
                        hidden = self._activation_fn(np.dot(qout, self.W1) + self.b1)
                        logits = np.dot(hidden, self.W2) + self.b2
                        val_logits_list.append(logits)
                    val_logits_array = np.array(val_logits_list)
                    val_probs = np.array([self._softmax(logit) for logit in val_logits_array])
                    val_per_sample_loss = -np.sum(y_val_one_hot * np.log(val_probs + 1e-9), axis=1)
                    try:
                        val_sample_has_data = np.any(X_val != 0, axis=1)
                    except Exception:
                        val_sample_has_data = np.ones(val_per_sample_loss.shape[0], dtype=bool)
                    if np.any(val_sample_has_data):
                        val_loss = np.mean(val_per_sample_loss[val_sample_has_data])
                    else:
                        val_loss = np.mean(val_per_sample_loss)
                    
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
        # Ensure inputs are batches
        X_scaled = np.atleast_2d(np.asarray(X_scaled))
        is_missing_mask = np.atleast_2d(np.asarray(is_missing_mask))
        # Use cached qnode
        qcircuit = self._qcircuit
        
        # compute quantum outputs for each sample
        quantum_outputs = [qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
                          for f, m in zip(X_scaled, is_missing_mask)]
        
        # apply classical readout per sample
        logits_list = [self._classical_readout(qout) for qout in quantum_outputs]
        
        # stack into an (N, K) numeric array and ensure float dtype
        logits_array = np.vstack(logits_list).astype(np.float64)
        
        # return (N, K) probabilities; for single-sample input this still returns shape (1,K)
        return self._softmax(logits_array)

    def predict(self, X):
        """Predict class labels. Returns 1D array of length N."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class ConditionalMulticlassQuantumClassifierDataReuploadingFS(BaseEstimator, ClassifierMixin):
    """Data Re-uploading Conditional Multiclass QVC with classical readout.
    
    Note: Threading backend is used for fallback parallelism. If device is not thread-safe,
    set n_jobs=1 to force sequential execution.
    
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
                 use_wandb=False, wandb_project=None, wandb_run_name=None, n_jobs=-1):
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
        self.n_jobs = n_jobs  # used by joblib parallel fallback

        assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Cache the qnode once; reuse to avoid repeated re-creation overhead
        self._qcircuit = self._get_circuit()
        
        # Quantum weights - measurements from all qubits
        self.n_meas = self.n_qubits
        self.weights_ansatz = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        self.weights_missing = np.random.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits), requires_grad=True)
        
        # Classical readout weights
        self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
        self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
        self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
        self.b2 = np.array(np.zeros(n_classes), requires_grad=True)

        # Choose activation function once and store callable on instance for low overhead
        if self.readout_activation == 'tanh':
            self._activation_fn = np.tanh
        elif self.readout_activation == 'relu':
            self._activation_fn = relu
        else:
            self._activation_fn = identity

        self.best_weights_ansatz = None
        self.best_weights_missing = None
        self.best_weights_classical = None
        self.best_loss = float('inf')
        self.best_metric = -float('inf')
        self.best_step = 0
        self.checkpoint_history = []
        
        # Shape validation flag to track if we've done it once
        self._shape_validated = False

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

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_qcircuit', None)
        state.pop('_activation_fn', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self._qcircuit = self._get_circuit()
        except Exception:
            self._qcircuit = None
        try:
            if getattr(self, 'readout_activation', 'tanh') == 'tanh':
                self._activation_fn = np.tanh
            elif getattr(self, 'readout_activation', None) == 'relu':
                self._activation_fn = relu
            else:
                self._activation_fn = identity
        except Exception:
            self._activation_fn = identity
    
    def _activation(self, x):
        """Compatibility wrapper if other code calls _activation; delegates to stored callable."""
        return self._activation_fn(x)
    
    def _classical_readout(self, quantum_output):
        """Apply classical readout head to quantum measurements."""
        # quantum_output shape: (n_meas,)
        hidden = self._activation_fn(np.dot(quantum_output, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        return logits

    def _batched_qcircuit(self, X, weights, n_jobs=None):
        """Batched wrapper: try true batched qnode first, otherwise parallel per-sample.

        Returns:
            np.ndarray shape (N, n_meas)
        """
        qcircuit = getattr(self, '_qcircuit', None)
        if qcircuit is None:
            qcircuit = self._get_circuit()
            self._qcircuit = qcircuit

        X_arr = np.asarray(X, dtype=np.float64)

        # Single sample
        if X_arr.ndim == 1:
            qout = qcircuit(X_arr, weights)
            return np.asarray(qout, dtype=np.float64).reshape(1, -1)

        # Empty batch
        if X_arr.shape[0] == 0:
            return np.empty((0, self.n_meas), dtype=np.float64)

        N = X_arr.shape[0]

        # Fast path: try batched call
        try:
            qouts = qcircuit(X_arr, weights)
            qouts = np.asarray(qouts, dtype=np.float64)
            if qouts.ndim == 2 and qouts.shape[0] == N:
                log.debug("qcircuit batched fast-path used")
                return qouts
            # else fall through to sequential
        except Exception as e:
            log.debug(f"qcircuit batched call failed, triggering fallback to sequential evaluation: {e}")

        # Fallback: sequential per-sample evaluation (safe for PennyLane qnodes)
        log.debug("Falling back to sequential per-sample QNode evaluation")
        results = [qcircuit(X_arr[i], weights) for i in range(N)]
        stacked = np.vstack([np.asarray(r, dtype=np.float64) for r in results])
        return stacked

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
        

        # Check for empty training set after split
        if X_train_scaled.shape[0] == 0:
            raise ValueError("Empty training set after split; reduce validation_frac or provide more data")
        
        y_train_one_hot = np.eye(self.n_classes)[y_train]
        if y_val is not None:
            y_val_one_hot = np.eye(self.n_classes)[y_val]
        
        # Use custom Adam optimizer for serializability
        opt = AdamSerializable(lr=self.learning_rate)
        # Use cached qnode for this fit session
        qcircuit = self._qcircuit
        
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
                

                # Shape validation (done once)
                if not self._shape_validated:
                    if w1.shape[0] != self.n_meas:
                        raise ValueError(f"W1 shape mismatch: expected {self.n_meas} rows, got {w1.shape[0]}")
                    hidden_size = w1.shape[1]
                    if w2.shape[0] != hidden_size:
                        raise ValueError(f"W2 shape mismatch: expected {hidden_size} rows, got {w2.shape[0]}")
                    if w2.shape[1] != self.n_classes:
                        raise ValueError(f"W2 output dim ({w2.shape[1]}) != n_classes ({self.n_classes})")
                    self._shape_validated = True

                # Apply classical readout to each sample using unified activation callable
                logits_list = []
                for qout in quantum_outputs:
                    hidden = self._activation_fn(np.dot(qout, w1) + b1)
                    logits = np.dot(hidden, w2) + b2
                    logits_list.append(logits)
                
                logits_array = np.array(logits_list)
                probabilities = np.array([self._softmax(logit) for logit in logits_array])
                
                # Cross-entropy loss computed only over samples that have data
                per_sample_loss = -np.sum(y_train_one_hot * np.log(probabilities + 1e-9), axis=1)
                try:
                    sample_all_zero = np.all(X_train_scaled == 0, axis=1)
                except Exception:
                    sample_all_zero = np.zeros(per_sample_loss.shape[0], dtype=bool)
                try:
                    mask_present = np.any(mask_train == 0, axis=1)
                except Exception:
                    mask_present = np.ones(per_sample_loss.shape[0], dtype=bool)

                sample_has_data = (~sample_all_zero) & mask_present
                if not np.any(sample_has_data) and np.any(~sample_all_zero):
                    sample_has_data = ~sample_all_zero
                if np.any(sample_has_data):
                    loss = np.mean(per_sample_loss[sample_has_data])
                else:
                    loss = np.mean(per_sample_loss)
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
                    val_per_sample_loss = -np.sum(y_val_one_hot * np.log(val_probs + 1e-9), axis=1)
                    try:
                        val_all_zero = np.all(X_val_scaled == 0, axis=1)
                    except Exception:
                        val_all_zero = np.zeros(val_per_sample_loss.shape[0], dtype=bool)
                    try:
                        val_mask_present = np.any(mask_val == 0, axis=1)
                    except Exception:
                        val_mask_present = np.ones(val_per_sample_loss.shape[0], dtype=bool)

                    val_sample_has_data = (~val_all_zero) & val_mask_present
                    if not np.any(val_sample_has_data) and np.any(~val_all_zero):
                        val_sample_has_data = ~val_all_zero
                    if np.any(val_sample_has_data):
                        val_loss = np.mean(val_per_sample_loss[val_sample_has_data])
                    else:
                        val_loss = np.mean(val_per_sample_loss)
                    
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
        # Ensure inputs are batches
        X_scaled = np.atleast_2d(np.asarray(X_scaled))
        is_missing_mask = np.atleast_2d(np.asarray(is_missing_mask))
        # Use cached qnode
        qcircuit = self._qcircuit
        
        # compute quantum outputs for each sample
        quantum_outputs = [qcircuit(f, m, self.weights_ansatz, self.weights_missing) 
                          for f, m in zip(X_scaled, is_missing_mask)]
        
        # apply classical readout per sample
        logits_list = [self._classical_readout(qout) for qout in quantum_outputs]
        
        # stack into an (N, K) numeric array and ensure float dtype
        logits_array = np.vstack(logits_list).astype(np.float64)
        
        # return (N, K) probabilities; for single-sample input this still returns shape (1,K)
        return self._softmax(logits_array)

    def predict(self, X):
        """Predict class labels. Returns 1D array of length N."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
