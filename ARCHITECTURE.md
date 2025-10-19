# Project Architecture: A Deep Dive

This document provides a detailed breakdown of the architectural decisions, quantum models, and classical machine learning strategies used in this project.

## Table of Contents
- [End-to-End Pipeline Workflow](#-end-to-end-pipeline-workflow)
- [The Quantum Models](#-the-quantum-models-a-deeper-look)
- [Classical Readout Layer](#-classical-readout-layer-hybrid-quantum-classical-architecture)
- [Nested Cross-Validation](#-nested-cross-validation-for-robust-hyperparameter-tuning)
- [Advanced Training Features](#-advanced-training-features)
- [Exploring Advanced Quantum Gates](#-exploring-advanced-quantum-gates)
- [Classical Design Decisions](#-classical-design-decisions-the-rationale)

---

## 📜 End-to-End Pipeline Workflow

The project is structured as a multi-stage pipeline. Each script performs a distinct role, creating artifacts that are used by subsequent stages.

**Stage 1: Global Setup (`create_master_label_encoder.py`)**
- **Purpose:** To ensure consistent class labels across the entire project.
- **Process:** The script scans all `*.parquet` files in the source data directory, collects every unique class name (e.g., 'BRCA', 'LUAD'), and creates a single, master `LabelEncoder`.
- **Output:** `master_label_encoder/label_encoder.joblib`. This artifact is a critical dependency for all subsequent training and inference scripts.

**Stage 2: Hyperparameter Tuning (`tune_models.py`)**
- **Purpose:** To find the optimal set of hyperparameters for each base-learner configuration.
- **Process:** Using Optuna, this script runs a series of trials for a specified data type (`--datatype`), architectural approach (`--approach`), and QML model (`--qml_model`). It uses `StratifiedKFold` cross-validation to robustly evaluate each parameter set.
- **Output:** A JSON file (e.g., `tuning_results/best_params_..._CNV_app1_...json`) for each tuned configuration, containing the best-performing parameters.

**Stage 3: Base-Learner Training**
- **Approaches:** Dimensionality Reduction Encoding (DRE) and Conditional Feature Encoding (CFE)
- **Scripts:** `dre_standard.py`, `dre_relupload.py`, `cfe_standard.py`, `cfe_relupload.py`
- **Purpose:** To train specialized "expert" models for each data type using the best parameters found in Stage 2.
- **Process:** The four training scripts (`dre_standard.py`, `dre_relupload.py`, `cfe_standard.py`, `cfe_relupload.py`) iterate through all available data types. For each one, they:
    1. Find the corresponding `best_params_*.json` file.
    2. Load the data and build the appropriate classical-quantum pipeline.
    3. Use `cross_val_predict` (for DRE) or a manual loop (for CFE) to generate **out-of-fold (OOF) predictions** for the training set. These predictions are crucial for training the meta-learner without data leakage.
    4. Train a final model on the *entire* training set.
    5. Generate predictions on the hold-out test set.
- **Output:** For each data type, the scripts save the OOF predictions, test predictions, and the final trained model artifacts (`.joblib` files) into a dedicated output directory (e.g., `base_learner_outputs_app1_standard/`).

**Stage 4: Meta-Learner Training (`metalearner.py`)**
- **Purpose:** To train a single, powerful "manager" model that learns from the predictions of the expert base-learners.
- **Process:**
    1. **Assembles Meta-Features:** The script loads the OOF predictions generated in Stage 3. It also loads an `indicator_file`, which contains supplementary classical features (e.g., clinical data like age or tumor stage). These are concatenated to form the feature set for the meta-learner.
    2. **Trains Meta-Learner:** It trains a QML model on this combined feature set, using the true labels from the training data.
- **Output:** The final trained meta-learner model (`metalearner_model.joblib`) and a list of the exact feature columns it was trained on (`best_metalearner_params.json`).

**Stage 5: Inference (`inference.py`)**
- **Purpose:** To predict the cancer type for a new, unseen patient.
- **Process:**
    1. Loads all required artifacts: the final base-learner models, the meta-learner, and the master label encoder.
    2. For each data type, it processes the new patient's data using the corresponding base-learner to get a prediction. It intelligently handles different model types (pipelines vs. individual components).
    3. It assembles the base-learner predictions and the patient's indicator features into a meta-feature vector.
    4. It feeds this vector into the final meta-learner to get the ultimate prediction.
    5. It uses the master label encoder to convert the numeric prediction back into a human-readable class name.

---

## 🧠 The Quantum Models: A Deeper Look

At the core of this project are four different **Variational Quantum Circuits (VQCs)**. The term "variational" (or "hybrid quantum-classical") means that they have classical parameters (weights) that are optimized using a familiar classical algorithm like Adam. The workflow for a single training step is:

1.  **Execute Circuit:** The quantum computer (or simulator) runs the circuit with the current set of classical data and trainable weights.
2.  **Calculate Expectation:** It measures the qubits to get an expectation value for each output.
3.  **Compute Loss:** This expectation value is fed into a classical loss function (like cross-entropy) to see how wrong the prediction was.
4.  **Update Weights:** A classical optimizer (Adam) calculates the gradient of the loss and decides how to update the trainable weights to improve the result in the next iteration.

This loop leverages the quantum processor for its unique computational power while relying on robust, classical methods for optimization.

### **1. The Standard Workhorse (`MulticlassQuantumClassifierDR`)**

This model is the foundation for the **Dimensionality Reduction Encoding (DRE)** approach. It's a hybrid quantum-classical architecture that combines quantum circuit processing with a trainable classical neural network readout layer.

*   **How it Works (Step-by-Step):**
    1.  **Encoding (`AngleEmbedding`):** The process begins by loading the classical data vector (e.g., the 8 principal components from PCA) into the quantum circuit. The `AngleEmbedding` layer takes this vector `[x_0, x_1, ..., x_7]` and maps it to 8 qubits. It does this by applying a rotation gate to each qubit, using the feature's value as the rotation angle (e.g., `RY(x_0)` on the first qubit, `RY(x_1)` on the second). This "encodes" the classical information into the quantum state.
    2.  **Processing (`BasicEntanglerLayers`):** This is the trainable, "neural network" part of the quantum model. It's a repeating block of two types of gates:
        *   **Rotation Gates (`RY`):** Each qubit is rotated by a trainable angle. These angles are the `weights` that the model learns during optimization.
        *   **Entangling Gates (`CNOT`):** Gates like `CNOT` are applied between adjacent qubits. This is the most crucial step. **Entanglement** creates non-local correlations between the qubits, allowing them to process information collectively. These layers allow the circuit to create and explore a massive, high-dimensional computational space (the Hilbert space) to find complex patterns that might be inaccessible to classical models of a similar size.
    3.  **Measurement:** After processing, we measure **all qubits** (not just the first `n_classes`). The measurement is the expectation value of the Pauli-Z operator for each qubit, which gives a real number between -1 and 1. This vector of real numbers represents the quantum circuit's raw output.
    4.  **Classical Readout Layer:** This is the hybrid component that processes quantum measurements through a trainable classical neural network:
        *   **Hidden Layer:** The quantum measurements are fed into a hidden layer with configurable size (default: 16 neurons). The transformation is: `hidden = activation(W1 * quantum_output + b1)`, where `W1` and `b1` are trainable weights and biases.
        *   **Activation Function:** The hidden layer uses a configurable activation function (default: tanh, but can also use relu or linear).
        *   **Output Layer:** The hidden layer output is then transformed to class logits: `logits = W2 * hidden + b2`, where `W2` and `b2` are additional trainable parameters.
    5.  **Softmax Activation:** The classical softmax function is applied as a final post-processing step to convert the logits into probabilities that sum to 1 (e.g., `[0.15, 0.70, 0.15]`), which can then be used to make the final classification.

*   **Architectural Rationale:** This hybrid architecture provides several advantages:
    * **Enhanced Expressivity:** The classical readout layer can learn complex non-linear transformations of the quantum measurements, improving the model's ability to represent complex decision boundaries.
    * **Joint Optimization:** All parameters (quantum circuit weights, classical layer weights and biases) are jointly trained using gradient descent, allowing the quantum and classical components to co-adapt.
    * **Flexibility:** Measuring all qubits (rather than just `n_classes` qubits) provides more information to the classical layer, enabling it to extract richer features from the quantum state.
    * **Improved Performance:** This architecture typically achieves better classification accuracy compared to direct softmax on quantum measurements, especially for complex multi-class problems.

### **2. The High-Capacity Model (`MulticlassQuantumClassifierDataReuploadingDR`)**

This model extends the standard architecture with data re-uploading while maintaining the same hybrid quantum-classical readout layer.

*   **How it Works (The Key Difference):** This model modifies the standard architecture by re-inserting the input data between each processing layer. Instead of one encoding at the beginning, there are multiple "data re-uploading" steps. Each layer consists of: `AngleEmbedding` -> `BasicEntanglerLayers`.
*   **Classical Readout:** Uses the same trainable classical neural network layer as the standard model to process quantum measurements.
*   **Architectural Rationale:** We chose this architecture to test the hypothesis that some data types might have patterns that are too complex for the standard VQC. Data re-uploading dramatically increases the model's **expressivity** (its ability to represent complex functions). It has been shown that this technique effectively turns the circuit into a quantum version of a Fourier series, allowing it to approximate much more complex functions. The trade-off is a higher number of parameters and a greater risk of overfitting, making it suitable for situations where we suspect the decision boundary is highly non-linear.

### **3. The Missing Data Specialist (`ConditionalMulticlassQuantumClassifierFS`)**

This model is the foundation for the **Conditional Feature Encoding (CFE)** approach. Its innovation lies in the encoding step, which treats missing data as a first-class citizen, while using the same hybrid quantum-classical architecture.

*   **How it Works (Step-by-Step):**
    1.  **Dual Input:** The model receives two pieces of information for each sample: the feature vector (where `NaN`s are filled with a placeholder like 0) and a binary mask vector that indicates which features were originally missing.
    2.  **Conditional Encoding:** The encoding layer iterates through each qubit. For each qubit `i`, it checks the `i`-th element of the mask vector.
        *   If `mask[i] == 0` (the feature is present), it applies a standard rotation using the feature's value: `RY(feature[i] * np.pi, wires=i)`.
        *   If `mask[i] == 1` (the feature is missing), it applies a rotation using a separate, **trainable parameter**: `RY(weights_missing[i], wires=i)`.
    3.  **Processing and Measurement:** The rest of the circuit (the entangling layers and measurement) is identical to the standard model, measuring all qubits.
    4.  **Classical Readout:** Uses the trainable classical neural network layer to transform quantum measurements into class predictions.
*   **Architectural Rationale:** The core hypothesis here is that **"missingness" is valuable information, not a problem to be fixed**. Instead of using a classical method like mean imputation (which makes an uninformed guess and can shrink variance), we let the model itself *learn* the best possible representation for a missing value. The optimizer might discover that the most effective way to represent a missing protein feature is a specific angle that places the qubit in a superposition—a state that is difficult to represent classically and might be the key to separating two classes. The classical readout layer then learns how to optimally combine these quantum representations with learned missing-value encodings.

### **4. The Ultimate Complexity Test (`ConditionalMulticlassQuantumClassifierDataReuploadingFS`)**

*   **Architectural Rationale:** This model is the logical synthesis of our two experimental hypotheses. It combines the missingness-aware encoding of the CFE approach with the high-capacity data re-uploading architecture, along with the hybrid quantum-classical readout layer. We included this to test if the combination of these three advanced techniques (conditional encoding, data re-uploading, and classical readout) could provide a performance edge on the most challenging datasets, where we suspect that both missingness and pattern complexity are high. It is the most powerful but also the most computationally expensive and data-hungry model in our arsenal.

---

## 🔗 Classical Readout Layer: Hybrid Quantum-Classical Architecture

All quantum classifiers in this project (`MulticlassQuantumClassifierDR`, `MulticlassQuantumClassifierDataReuploadingDR`, `ConditionalMulticlassQuantumClassifierFS`, and `ConditionalMulticlassQuantumClassifierDataReuploadingFS`) now incorporate a **trainable classical neural network layer** that processes the quantum measurement outputs. This hybrid approach bridges quantum and classical machine learning.

### Architecture Design

The classical readout layer is a two-layer fully-connected neural network that transforms quantum measurements into final class predictions:

```
Quantum Measurements → Hidden Layer → Activation → Output Layer → Softmax → Predictions
     (n_qubits)        (hidden_size)              (n_classes)
```

### Key Components

1. **Input:** Raw quantum measurements from all qubits (expectation values of Pauli-Z operators, each in range [-1, 1])

2. **Hidden Layer:**
   - **Size:** Configurable (default: 16 neurons)
   - **Transformation:** `hidden = activation(W1 * measurements + b1)`
   - **Parameters:** Weight matrix `W1` (n_qubits × hidden_size) and bias vector `b1` (hidden_size)

3. **Activation Function:**
   - **Default:** `tanh` - provides smooth, bounded non-linearity
   - **Options:** `relu` (rectified linear), `linear` (no activation)
   - **Purpose:** Introduces non-linearity to learn complex transformations

4. **Output Layer:**
   - **Transformation:** `logits = W2 * hidden + b2`
   - **Parameters:** Weight matrix `W2` (hidden_size × n_classes) and bias vector `b2` (n_classes)

5. **Softmax:** Converts logits to probability distribution over classes

### Training Procedure

The classical readout parameters are **jointly optimized** with the quantum circuit parameters:

- **Optimizer:** Custom serializable Adam optimizer (see `utils/optim_adam.py`)
- **Gradient Computation:** PennyLane's autograd computes gradients through both quantum and classical components
- **Loss Function:** Cross-entropy loss between predicted probabilities and true labels
- **Co-adaptation:** Quantum and classical parameters adapt together, allowing the quantum circuit to generate features optimized for the classical readout

### Benefits of Hybrid Architecture

1. **Increased Expressivity:**
   - The classical layer can learn complex non-linear mappings from quantum measurements
   - Enables the model to represent more intricate decision boundaries

2. **Better Information Utilization:**
   - By measuring **all qubits** (not just n_classes), we extract maximum information from the quantum state
   - The classical layer learns which quantum measurements are most informative for each class

3. **Improved Performance:**
   - Empirically achieves higher accuracy than direct softmax on quantum measurements
   - Particularly effective for complex multi-class problems with subtle class distinctions

4. **Flexibility:**
   - Hidden layer size can be tuned as a hyperparameter
   - Different activation functions can be tested for different data types

### Implementation Details

The classical readout is implemented in all QML models in `qml_models.py`:

```python
# Initialize classical readout weights during model construction
self.W1 = np.array(np.random.randn(self.n_meas, hidden_size) * 0.01, requires_grad=True)
self.b1 = np.array(np.zeros(hidden_size), requires_grad=True)
self.W2 = np.array(np.random.randn(hidden_size, n_classes) * 0.01, requires_grad=True)
self.b2 = np.array(np.zeros(n_classes), requires_grad=True)

# Apply during forward pass
def _classical_readout(self, quantum_output):
    hidden = self._activation(np.dot(quantum_output, self.W1) + self.b1)
    logits = np.dot(hidden, self.W2) + self.b2
    return logits
```

The classical parameters are included in the optimizer state and checkpointing system, ensuring they are properly saved and restored during training.

---

## ⚡ Batched Evaluation Optimization

All quantum models have been optimized with efficient batched evaluation to significantly improve training and inference performance.

### Key Optimizations

1. **Vectorized Cost Functions:**
   - Training cost functions now use fully vectorized NumPy operations
   - Eliminates per-sample loops in forward pass
   - Example transformation:
     ```python
     # Old: Per-sample loop
     for qout in quantum_outputs:
         hidden = activation(np.dot(qout, w1) + b1)
         logits_list.append(np.dot(hidden, w2) + b2)
     
     # New: Vectorized batch operation
     hidden = activation_fn(np.dot(quantum_outputs, w1) + b1)  # (N, hidden)
     logits_array = np.dot(hidden, w2) + b2  # (N, n_classes)
     ```

2. **Batched Quantum Circuit Execution:**
   - The `_batched_qcircuit` method provides intelligent batching:
     ```python
     def _batched_qcircuit(self, X, weights, n_jobs=None):
         # Try true batched call first (fast path)
         try:
             qouts = qcircuit(X_batch, weights)
             if qouts.ndim == 2:  # Successfully batched
                 return qouts
         except:
             pass
         
         # Fallback to threaded parallel execution
         results = Parallel(n_jobs=n_jobs, backend='threading')(
             delayed(qcircuit)(X[i], weights) for i in range(N)
         )
         return np.vstack(results)
     ```
   - Attempts native batched execution first for maximum speed
   - Falls back to threaded parallel execution when batching not supported
   - Threading backend avoids pickling overhead (important for quantum circuits)

3. **Stored Activation Functions:**
   - Activation functions stored as callables during initialization:
     ```python
     # In __init__:
     if self.readout_activation == 'tanh':
         self._activation_fn = np.tanh
     elif self.readout_activation == 'relu':
         self._activation_fn = relu
     else:
         self._activation_fn = identity
     ```
   - Eliminates repeated conditional checks in training loop
   - Reduces overhead for frequently called functions

4. **Validation Loss Optimization:**
   - Validation loss computation also fully vectorized:
     ```python
     # Batched quantum evaluation
     val_quantum_outputs = self._batched_qcircuit(X_val, self.weights)
     # Vectorized classical readout
     val_hidden = self._activation_fn(np.dot(val_quantum_outputs, self.W1) + self.b1)
     val_logits = np.dot(val_hidden, self.W2) + self.b2
     val_probs = self._softmax(val_logits)
     # Vectorized loss computation
     val_loss = -np.mean(np.sum(y_val_one_hot * np.log(val_probs + 1e-9), axis=1))
     ```

### Performance Benefits

- **Training Speed:** 2-5x faster depending on batch size and model complexity
- **Memory Efficiency:** Better memory access patterns with contiguous arrays
- **Scalability:** Performance improvements increase with larger batch sizes
- **Resource Utilization:** Threaded fallback efficiently uses multiple cores

### Implementation Details

All four quantum model classes incorporate these optimizations:
- `MulticlassQuantumClassifierDR`
- `MulticlassQuantumClassifierDataReuploadingDR`  
- `ConditionalMulticlassQuantumClassifierFS`
- `ConditionalMulticlassQuantumClassifierDataReuploadingFS`

The optimizations are completely internal - all external APIs remain unchanged, ensuring backward compatibility with existing code.

### Configurable Parameters

- **`n_jobs`:** Number of parallel jobs for fallback execution (default: -1 for all cores)
  - Only used when batched execution is not available
  - Threading backend used to avoid pickling overhead

---

## 🔄 Nested Cross-Validation for Robust Hyperparameter Tuning

The project implements a **nested cross-validation strategy** to ensure robust hyperparameter selection and unbiased model evaluation. This approach prevents overfitting to the validation set and provides reliable performance estimates.

### The Nested CV Strategy

The pipeline uses a two-level cross-validation approach:

#### **Outer Level: Model Training and Evaluation**
- **Purpose:** Generate out-of-fold (OOF) predictions for the meta-learner
- **Implementation:** 3-fold stratified cross-validation (configurable)
- **Process:** 
  - Split data into 3 folds
  - For each fold: train on 2 folds, predict on the held-out fold
  - Produces predictions for every sample in the training set without data leakage

#### **Inner Level: Hyperparameter Tuning**
- **Purpose:** Find optimal hyperparameters for each base learner
- **Implementation:** Optuna-based Bayesian optimization with stratified k-fold CV
- **Process:**
  - For each hyperparameter configuration (Optuna trial):
    - Perform 3-fold cross-validation
    - Average validation scores across folds
    - Return mean score to Optuna
  - After all trials, save best hyperparameters

### Workflow Details

**Step 1: Hyperparameter Tuning (`tune_models.py`)**
```
For each Optuna trial:
  Sample hyperparameters (n_qubits, n_layers, scaler, etc.)
  
  Inner CV Loop (3 folds):
    For each fold:
      - Split into train/validation
      - Build pipeline with sampled hyperparameters
      - Train on train split
      - Evaluate on validation split
    
  Return: Average validation score across 3 folds
  
Save: Best hyperparameters based on highest average score
```

**Step 2: Final Model Training (e.g., `dre_standard.py`, `cfe_standard.py`)**
```
Load best hyperparameters from tuning

Outer CV Loop (3 folds for OOF predictions):
  For each fold:
    - Split into train/validation  
    - Build pipeline with best hyperparameters
    - Perform fold-specific preprocessing (e.g., feature selection for CFE)
    - Train QML model on train split
    - Generate predictions on validation split
  
Concatenate: All fold predictions → OOF predictions (no data leakage)

Final Model:
  - Train on entire training set with best hyperparameters
  - Generate predictions on test set
```

### Benefits of Nested CV

1. **Unbiased Performance Estimation:**
   - OOF predictions are generated on data the model has never seen during training
   - Provides realistic estimates of model generalization

2. **Prevents Hyperparameter Overfitting:**
   - Hyperparameters are selected on validation sets separate from the final evaluation
   - Reduces risk of selecting hyperparameters that overfit to a specific data split

3. **Robust to Data Variability:**
   - Multiple folds capture different aspects of data distribution
   - More stable hyperparameter selection compared to single train/val split

4. **Meta-Learner Training:**
   - OOF predictions from base learners enable the meta-learner to be trained on predictions generated without data leakage
   - Critical for stacked ensemble to work properly

### Implementation Notes

- **Stratification:** All cross-validation splits use stratified sampling to preserve class distributions
- **Seed Control:** Random states are set for reproducibility (default: 42, configurable via `RANDOM_STATE` environment variable)
- **Optuna Integration:** Hyperparameter tuning uses SQLite storage for persistence and supports parallel trials
- **Skip Options:** Both `--skip_cross_validation` and `--cv_only` flags allow flexibility in training workflows

### Example: Complete Nested CV Flow for CNV Data

```bash
# Step 1: Tune hyperparameters (inner CV)
python tune_models.py --datatype CNV --approach 1 --qml_model standard --n_trials 50

# Step 2: Train with nested CV (outer CV for OOF, final model on full data)
python dre_standard.py --datatypes CNV --verbose

# Result: 
# - train_oof_preds_CNV.csv (OOF predictions from nested CV)
# - test_preds_CNV.csv (predictions on held-out test set)
# - pipeline_CNV.joblib (final model trained on all training data)
```

This nested approach ensures that the meta-learner receives high-quality, unbiased predictions from the base learners, which is crucial for achieving optimal ensemble performance.

---

## 🚀 Advanced Training Features

The project includes several sophisticated training features designed for production-quality quantum machine learning pipelines. These features enable robust, resumable, and well-monitored training processes.

### 1. Comprehensive Checkpointing System

**Purpose:** Enable training recovery, model selection, and long-running experiments.

**Key Features:**
- **Automatic Checkpointing:** Saves model state at configurable intervals (default: every 50 steps)
- **Best Model Tracking:** Continuously tracks and saves the best model based on training loss or validation metrics
- **State Persistence:** Saves complete training state including:
  - Quantum circuit weights
  - Classical readout layer parameters
  - Optimizer state (momentum, velocity, timestep)
  - RNG state for reproducibility
  - Training history and metrics

**Configuration:**
```bash
python dre_standard.py \
    --max_training_time 11 \           # Enable time-based training
    --checkpoint_frequency 50 \        # Checkpoint every 50 steps
    --keep_last_n 3                    # Keep only last 3 checkpoints
```

**Checkpoint Loading and Resume:**

The system automatically detects existing checkpoints and can resume training from a saved state. When a model is initialized with a checkpoint directory that contains existing checkpoints:

1. **Automatic Detection:** The system scans for `best_weights.joblib` or `checkpoint_step_*.joblib` files
2. **State Restoration:** Loads the complete training state including:
   - Model weights (quantum circuit and classical readout parameters)
   - Optimizer state (momentum, velocity, iteration count)
   - Training history (loss, metrics, best model tracking)
   - Random number generator state
3. **Seamless Continuation:** Training resumes exactly where it left off with no loss of optimization momentum

**Resume Modes:**

The checkpoint utilities support different resume strategies:
- **`auto`**: Automatically detects and loads the most recent checkpoint (best or latest)
- **`latest`**: Loads the most recent checkpoint by step number from periodic saves
- **`best`**: Loads the checkpoint with the best validation metric

**Loading Examples:**
```python
from utils.io_checkpoint import find_best_checkpoint, find_latest_checkpoint, load_checkpoint

# Load best checkpoint (typically used after training completes)
best_path = find_best_checkpoint(checkpoint_dir)
if best_path:
    checkpoint_data = load_checkpoint(best_path)
    model.set_weights(checkpoint_data['weights'])

# Load latest checkpoint (typically used to resume interrupted training)
latest_path = find_latest_checkpoint(checkpoint_dir)
if latest_path:
    checkpoint_data = load_checkpoint(latest_path)
    model.set_weights(checkpoint_data['weights'])
    optimizer.set_state(checkpoint_data['optimizer_state'])
```

**Training Interruption Handling:**

If training is interrupted (e.g., system crash, manual stop, time limit):
1. The most recent checkpoint is preserved
2. Restart training with the same checkpoint directory
3. The system automatically loads the last checkpoint and continues
4. No manual intervention required beyond restarting the script

**Implementation:** See `utils/io_checkpoint.py` for checkpoint save/load logic and `qml_models.py` for model-level integration.

### 2. Serializable Adam Optimizer

**Purpose:** Enable true checkpoint/resume functionality with complete optimizer state.

**Standard Problem:** PennyLane's built-in optimizers don't support state serialization, making it impossible to resume training with momentum.

**Solution:** Custom `AdamSerializable` optimizer that:
- Maintains all Adam state variables (first moment, second moment, timestep)
- Provides `get_state()` and `set_state()` methods for serialization
- Compatible with PennyLane's autograd system
- Enables seamless training resumption without losing optimization momentum

**Implementation:** See `utils/optim_adam.py`

**Benefits:**
- Resume training from exact optimizer state
- No loss of training progress when interrupted
- Critical for long-running experiments (e.g., 11-hour training sessions)

### 3. Time-Based Training

**Purpose:** Train for a specified duration rather than a fixed number of steps.

**Traditional Approach:** Fixed number of steps (e.g., `--steps 100`)
- Problem: Different data types/models may require different amounts of time
- May under-train complex models or waste time on simple ones

**Time-Based Approach:** Specify maximum training time (e.g., `--max_training_time 11`)
- Trains until time limit reached, regardless of step count
- Automatically checkpoints periodically
- Loads best model at end based on validation metric

**Use Cases:**
- **Resource-Limited Training:** Utilize available compute time efficiently
- **Fair Comparison:** Give each model the same computational budget
- **Long-Running Experiments:** Train overnight or over weekend

**Example:**
```bash
# Train for 8 hours instead of fixed 100 steps
python cfe_relupload.py --max_training_time 8 --checkpoint_frequency 25
```

### 4. Comprehensive Metrics Logging

**Purpose:** Full observability into training progress and model performance.

**Metrics Tracked:**
- **Training Metrics:** Loss per epoch
- **Validation Metrics (if validation split used):**
  - Accuracy
  - Precision (macro and weighted)
  - Recall (macro and weighted)
  - F1 score (macro and weighted)
  - Specificity (macro and weighted)
  - Confusion matrix per epoch

**Outputs:**
1. **CSV Export:** `history.csv` containing all metrics per epoch
2. **Automatic Plots:**
   - Loss curves (training and validation)
   - F1 score curves (macro and weighted)
   - Precision/recall curves
   - PNG format, saved to model directory

**Model Selection:**
- Configurable selection metric (default: weighted F1)
- Best model determined by validation performance, not training loss
- Prevents overfitting to training data

**Implementation:** See `utils/metrics_utils.py` for metric computation and plotting.

### 5. Flexible Training Modes

The training scripts support multiple operational modes:

**Cross-Validation Modes:**
- **Full Training (default):** Generates OOF predictions + trains final model
- **`--skip_cross_validation`:** Skip CV, only train final model (faster when OOF not needed)
- **`--cv_only`:** Only generate OOF predictions, skip final training (useful for meta-learner prep)

**Hyperparameter Modes:**
- **Use Tuned Parameters (default):** Load best parameters from `tune_models.py`
- **`--skip_tuning`:** Ignore tuned parameters, use CLI arguments or defaults

**These modes enable flexible workflows:**
```bash
# Generate OOF predictions for meta-learner (no final model)
python dre_standard.py --cv_only

# Quick final model training (skip CV)
python dre_standard.py --skip_cross_validation --steps 150

# Exploratory training (skip tuned params)
python dre_standard.py --skip_tuning --n_qubits 8 --n_layers 4
```

### 6. Validation Split and Early Stopping

**Purpose:** Prevent overfitting during training.

**Features:**
- **Validation Split:** Automatic stratified train/validation split (configurable via `validation_frac`)
- **Best Model Selection:** Track best model by validation metric during training
- **Patience (Optional):** Stop training if validation metric doesn't improve for N epochs

**Configuration:**
```python
model = MulticlassQuantumClassifierDR(
    validation_frac=0.1,      # Hold out 10% for validation
    selection_metric='f1_weighted',  # Use weighted F1 for model selection
    patience=20                # Stop if no improvement for 20 epochs (optional)
)
```

### Integration Example

These features work together seamlessly:

```bash
# Long-running training with all features enabled
python dre_standard.py \
    --datatypes CNV Prot \
    --max_training_time 11 \        # Time-based training
    --checkpoint_frequency 25 \      # Frequent checkpoints
    --keep_last_n 5 \               # More checkpoint retention
    --verbose                       # Detailed logging

# Result:
# - Trains for 11 hours with periodic checkpoints
# - Tracks comprehensive metrics in history.csv
# - Generates automatic plots
# - Saves best model based on validation F1
# - Can be resumed if interrupted

# Complete integration: All advanced features combined
python dre_standard.py \
    --datatypes CNV Prot \
    --max_training_time 11 \              # Time-based training
    --checkpoint_frequency 25 \            # Frequent checkpoints
    --keep_last_n 5 \                     # More checkpoint retention
    --checkpoint_fallback_dir /tmp/ckpt \ # Fallback for read-only storage
    --validation_frequency 5 \            # More frequent validation
    --use_wandb \                         # W&B experiment tracking
    --wandb_project qml_experiments \     # W&B project name
    --wandb_run_name dre_full_run \       # W&B run identifier
    --verbose                             # Detailed logging

# Result:
# - Trains for 11 hours with periodic checkpoints
# - Uses fallback directory if primary checkpoint location is read-only
# - Validates every 5 steps for detailed monitoring
# - Logs all metrics to Weights & Biases for visualization
# - Tracks comprehensive metrics in history.csv
# - Generates automatic plots locally
# - Saves best model based on validation F1
# - Can be resumed if interrupted with full optimizer state
# - Provides complete experiment reproducibility and tracking
```

These advanced features make the quantum machine learning pipeline production-ready, enabling reliable and reproducible experiments at scale.

---

## 🔧 Checkpoint Fallback and Resilience

The training infrastructure includes robust checkpoint management to handle read-only storage and ensure training can proceed even in restricted environments.

### Automatic Read-Only Detection

The system automatically detects when checkpoint directories are read-only:
- **Permission checks:** Before writing checkpoints, the system verifies write access to the checkpoint directory
- **Fallback activation:** If the primary directory is read-only, the system automatically switches to a fallback directory
- **Checkpoint migration:** Existing checkpoints from the read-only directory are copied to the fallback location
- **Clear warnings:** Users receive informative messages about directory status and fallback behavior

### Fallback Directory Configuration

Users can specify a fallback directory via the `checkpoint_fallback_dir` parameter:
```python
model = MulticlassQuantumClassifierDR(
    checkpoint_dir='/readonly/storage/checkpoints',
    checkpoint_fallback_dir='/tmp/checkpoints_fallback',
    # ... other parameters
)
```

### Implementation Details

The checkpoint fallback logic is implemented in the `_ensure_writable_checkpoint_dir()` helper function in `qml_models.py`:

1. **Primary directory check:** Attempts to create and verify write access to the primary directory
2. **Fallback attempt:** If primary is not writable and fallback is specified, creates and verifies the fallback
3. **Checkpoint copy:** If fallback is writable, copies existing `.joblib` checkpoint files from primary to fallback
4. **Return result:** Returns the writable directory path (primary or fallback) and a flag indicating which was used
5. **Graceful degradation:** If no writable path exists, returns None and logs clear warnings (training proceeds without checkpointing)

### Benefits

- **Resilience:** Training can continue even when primary storage becomes read-only
- **Flexibility:** Supports various storage configurations (NFS, read-only mounts, restricted containers)
- **Data preservation:** Existing checkpoints are preserved and migrated automatically
- **User-friendly:** Clear messaging guides users when checkpoint issues occur

---

## 📊 Weights & Biases Integration

All quantum machine learning models now support optional Weights & Biases (W&B) integration for comprehensive experiment tracking and visualization.

### Deferred Import Pattern

W&B integration uses a deferred import pattern to avoid forcing the dependency:
- **Optional dependency:** W&B is only imported if `use_wandb=True` is specified
- **Graceful handling:** If W&B is not installed, a warning is logged and training proceeds normally
- **No code bloat:** Models remain lightweight for users who don't need W&B

### Initialization

W&B is initialized at the start of model training via the `_initialize_wandb()` helper function:
```python
wandb = _initialize_wandb(
    use_wandb=True,
    wandb_project='my_project',
    wandb_run_name='experiment_001',
    config_dict={
        'n_qubits': 8,
        'n_layers': 3,
        'learning_rate': 0.1,
        # ... other hyperparameters
    }
)
```

### Automatic Metric Logging

When W&B is enabled, validation metrics are automatically logged during training:
- **Training metrics:** Loss and accuracy at each validation frequency step
- **Validation metrics:** Loss, accuracy, F1 scores, precision, and recall
- **Step tracking:** Each log entry is associated with the training step number
- **No manual logging:** Developers don't need to add W&B logging calls manually

### Implementation Pattern

Logging is added at validation checkpoints in the training loop:
```python
# After computing validation metrics
if wandb:
    wandb.log({
        'step': step,
        'train_loss': history['train_loss'][-1],
        'train_acc': history['train_acc'][-1],
        'val_loss': history['val_loss'][-1],
        'val_acc': history['val_acc'][-1],
        'val_f1_weighted': history['val_f1_weighted'][-1],
        # ... additional metrics
    })
```

### Configuration in Training Scripts

All training scripts support W&B via command-line arguments:
```bash
# Enable W&B with project and run name
python dre_standard.py --use_wandb --wandb_project qml_experiments --wandb_run_name exp001

# Auto-generated run names (includes script name and data type)
python cfe_standard.py --datatypes CNV --use_wandb --wandb_project qml_project

# Works with all training modes
python tune_models.py --datatype CNV --approach 1 --use_wandb --wandb_project tuning
```

### Benefits

- **Experiment tracking:** Centralized logging of all training runs
- **Visualization:** Automatic plots of training curves and metrics
- **Comparison:** Easy comparison of different hyperparameters and architectures
- **Reproducibility:** Complete record of hyperparameters and training configuration
- **Team collaboration:** Shared workspace for team members to view results

---

## ⚡ Configurable Validation Frequency

The training system now supports configurable validation frequency via the `validation_frequency` parameter (default: 10).

### Motivation

Previously, validation metrics were computed at hard-coded intervals (every 10 steps). This approach:
- **Wasted computation:** For large datasets, frequent validation was costly
- **Limited flexibility:** Users couldn't adjust validation frequency based on their needs
- **Hindered debugging:** For debugging, more frequent validation was sometimes needed

### Implementation

The `validation_frequency` parameter replaces all hard-coded `step % 10 == 0` checks:

**Before:**
```python
if step % 10 == 0 or step == 0:
    # Compute validation metrics
```

**After:**
```python
if step % self.validation_frequency == 0 or step == 0:
    # Compute validation metrics
```

This applies consistently to:
- Validation metric computation
- Verbose logging output
- Best model tracking
- W&B metric logging

### Use Cases

1. **Large datasets:** Increase `validation_frequency` (e.g., 20 or 50) to reduce validation overhead
2. **Debugging:** Decrease `validation_frequency` (e.g., 1 or 5) for fine-grained monitoring
3. **Production training:** Use default (10) for balanced speed and observability
4. **Resource-constrained:** Adjust based on available compute budget

### Configuration

All training scripts and tuning support the `--validation_frequency` CLI argument:
```bash
# Validate every 20 steps (reduce overhead)
python dre_standard.py --validation_frequency 20

# Validate every 5 steps (debugging)
python cfe_relupload.py --validation_frequency 5 --verbose

# Use in tuning
python tune_models.py --datatype CNV --approach 1 --validation_frequency 15
```

Direct model instantiation:
```python
model = MulticlassQuantumClassifierDR(
    n_qubits=8,
    n_layers=3,
    validation_frequency=20,  # Validate every 20 steps
    # ... other parameters
)
```

### Benefits

- **Flexibility:** Users control the trade-off between speed and observability
- **Performance:** Reduce validation overhead for large-scale training
- **Debugging:** Fine-grained monitoring when needed
- **Consistency:** Same parameter across all models and scripts

---

## ⚛️ Exploring Advanced Quantum Gates

The current models primarily use `RY` gates for encoding/processing and `CNOT` gates for entanglement. This is a robust and standard choice, but the world of quantum gates is vast. Here are some alternatives that could be explored to potentially enhance model performance.

### **1. More Expressive Rotation Gates**

*   **Current:** `RY(angle)` - Rotates the qubit state vector around the Y-axis of the Bloch sphere.
*   **Alternative: `U(phi, theta, omega)` (Arbitrary Unitary Gate):** This is the most general single-qubit gate. Instead of a single rotation, it allows for three separate rotations around different axes.
    *   **Potential Benefit:** Using `U` gates for the trainable weights in the `BasicEntanglerLayers` would give the optimizer significantly more freedom to manipulate the qubit's state. It could learn more complex transformations than a simple Y-axis rotation, potentially increasing the model's capacity to find subtle patterns.
    *   **Trade-off:** It triples the number of trainable parameters per qubit in each layer, increasing the risk of overfitting and making the optimization landscape more complex.

### **2. Advanced Entangling Gates**

*   **Current:** `CNOT(control, target)` - Flips the `target` qubit if and only if the `control` qubit is in the `|1⟩` state.
*   **Alternative 1: `CZ(control, target)` (Controlled-Z):** Applies a Z-gate (a phase flip) to the `target` qubit if the `control` is `|1⟩`. It's subtly different from `CNOT` but is "symmetric" and can sometimes lead to more efficient circuit compilation on real hardware.
*   **Alternative 2: `ISWAP(q1, q2)`:** This gate partially swaps the states of two qubits. It's a more "gentle" way of creating correlations compared to the hard flip of a `CNOT`.
    *   **Potential Benefit:** For problems where the relationship between features is not a simple "if-then" condition, `ISWAP` or other partial swap gates might create more nuanced entanglement that better reflects the underlying data structure.
*   **Alternative 3: `Toffoli(c1, c2, target)` (CCNOT):** This is a three-qubit gate that flips the `target` qubit only if *both* control qubits (`c1` and `c2`) are in the `|1⟩` state.
    *   **Potential Benefit:** Using multi-qubit entangling gates allows the model to learn more complex, higher-order correlations directly. A `Toffoli` gate can capture a three-way interaction between features that would require a much deeper circuit of `CNOT` gates to approximate. This could lead to more powerful and compact models.

### **3. Data-Driven Encoding Gates**

*   **Current:** `AngleEmbedding` uses `RY` gates.
*   **Alternative: `IQPEmbedding` (Instantaneous Quantum Polynomial):** This is a more complex embedding that uses a combination of `Hadamard` gates, `CNOT` gates, and controlled phase gates (`RZ`).
    *   **Potential Benefit:** `IQPEmbedding` is known to create feature maps that are hard to simulate classically. By encoding the data in a more "quantum" way from the very beginning, it might unlock computational advantages that `AngleEmbedding` cannot access. It's particularly well-suited for kernel-based quantum machine learning methods.

### **How to Implement These Changes**

These advanced gates can be integrated by creating custom layer functions in PennyLane. For example, to create a processing layer with `U` gates and `CZ` gates, one could write a new function and substitute it for the `qml.BasicEntanglerLayers` call in the existing model definitions. This modularity is a key strength of the PennyLane framework.

---
## 🏛️ Classical Design Decisions: The Rationale

### **Dimensionality Reduction: PCA vs. UMAP (Approach 1 - DRE)**
The DRE approach requires us to distill thousands of features into a small, information-rich vector that can be loaded into a quantum circuit.

* **PCA (Principal Component Analysis):** This was chosen as the **baseline** because it's a fast, linear, and widely understood technique. It works by finding the axes of maximum variance in the data. We use it to answer the question: "How well does a quantum model perform when fed features from a standard, linear classical pipeline?"
* **UMAP (Uniform Manifold Approximation and Projection):** This was chosen as the **experimental alternative**. UMAP is a non-linear technique that assumes high-dimensional data lies on a lower-dimensional, curved manifold. The hypothesis is that the biological states in multi-omics data are not simple linear clusters but complex, intertwined manifolds. UMAP might be better at "unraveling" these structures into a set of features that are more meaningful for the quantum classifier.

### **Feature Selection: LightGBM importance (Approach 2 - CFE)**
The CFE approach requires us to select a small subset of the best original features.

* **LightGBM importance-based selection:**
    * **What it is:** A tree-based method using a LightGBM classifier trained on the (scaled) training fold. Feature importances from the trained model are used to rank features, and the top-k features are selected (k is typically the number of qubits).
    * **Where it's used:** This method is used for Approach 2 (CFE) during both hyperparameter tuning and final training. The selection is performed per fold and a final model selection is computed on the full training set.
    * **Why:** LightGBM's tree-based importances are robust to constant features and don't rely on ANOVA assumptions. They also tend to provide useful multivariate signals that can be more informative than purely univariate scores, while still being fast when configured with fewer trees and feature subsampling.

### **The Ensemble: Why Stack?**
The final architectural choice was to not rely on a single model but to build a **stacked ensemble**.

* **Rationale:** Multi-omics data is heterogeneous. The patterns in gene expression data are fundamentally different from those in methylation data. It is highly unlikely that one model architecture (e.g., DRE with PCA) would be the best for all data types.
    1. **Expert Models:** By training specialist "base learners" for each data type, we allow each model to become an expert on one kind of data.
    2. **Learning from Experts:** The **meta-learner** then acts as a "manager," learning from the *predictions* of these experts.
    3. **Supplementary Data:** Crucially, the meta-learner does not *only* see the predictions. It also receives **indicator features** (provided via the `--indicator_file` argument). These are additional classical features about the patient (e.g., age, tumor stage, or flags indicating if an entire data modality was missing).
* **Final Decision:** The meta-learner's final decision is based on both what the experts are saying (their predictions) and the supplementary context (the indicator features). It might learn, for instance, that for a particular cancer type, the `Prot` and `CNV` models are highly reliable, but if the `is_missing_SNV` flag is true, it should adjust its final prediction accordingly. This ability to learn the strengths and weaknesses of each specialist model in different contexts makes the final ensemble more accurate and robust than any single model could be on its own.

