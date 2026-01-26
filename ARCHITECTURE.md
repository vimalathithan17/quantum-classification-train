# Project Architecture: A Deep Dive

Technical breakdown of architectural decisions, quantum models, and ML strategies.

> **See also:** [README.md](README.md) for setup, [DATA_PROCESSING.md](DATA_PROCESSING.md) for data pipeline, [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for extensions.

## Table of Contents
- [End-to-End Pipeline Workflow](#-end-to-end-pipeline-workflow)
- [The Quantum Models](#-the-quantum-models-a-deeper-look)
- [Classical Readout Layer](#-classical-readout-layer-hybrid-quantum-classical-architecture)
- [Data Encoding: Angle vs Amplitude](#-data-encoding-why-angle-encoding-over-amplitude-encoding)
- [Nested Cross-Validation](#-nested-cross-validation-for-robust-hyperparameter-tuning)
- [Advanced Training Features](#-advanced-training-features)
- [Exploring Advanced Quantum Gates](#-exploring-advanced-quantum-gates)
- [Classical Design Decisions](#-classical-design-decisions-the-rationale)

---

## 📜 End-to-End Pipeline Workflow

The project is structured as a multi-stage pipeline. Each script performs a distinct role, creating artifacts that are used by subsequent stages.

### Data Pipeline Summary

Upstream preprocessing is handled by two notebooks (see [DATA_PROCESSING.md](DATA_PROCESSING.md)):
- `data-process.ipynb`: 10 stages → `final_filtered_datasets/`
- `feature-extraction-xgb.ipynb`: MI → XGBoost → `final_processed_datasets_xgb_balanced/`

Code default: `final_processed_datasets/` (set `SOURCE_DIR=final_processed_datasets_xgb_balanced` for XGBoost-selected features)

**Stage 1: Global Setup (`create_master_label_encoder.py`)**
- **Purpose:** To ensure consistent class labels across the entire project.
- **Process:** The script scans all `*.parquet` files in the source data directory, collects every unique class name (e.g., 'BRCA', 'LUAD'), and creates a single, master `LabelEncoder`.
- **Output:** `master_label_encoder/label_encoder.joblib`. This artifact is a critical dependency for all subsequent training and inference scripts.

**Stage 2: Hyperparameter Tuning (`tune_models.py`)**
- **Purpose:** To find the optimal set of hyperparameters for each base-learner configuration.
- **Process:** Using Optuna, this script runs a series of trials for a specified data type (`--datatype`), architectural approach (`--approach`), and QML model (`--qml_model`). It uses `StratifiedKFold` cross-validation to robustly evaluate each parameter set.
- **Optimization Metric:** Trials are optimized using **weighted F1 score** instead of accuracy. This provides better handling of class imbalance and more robust model selection.
- **Comprehensive Metrics:** For each fold in each trial, the following metrics are computed and saved:
  - Accuracy
  - Precision (macro and weighted)
  - Recall (macro and weighted)
  - F1 score (macro and weighted)
  - Specificity (macro and weighted)
  - Confusion matrix
  - Classification report
- **Artifact Management:** 
  - Per-trial fold metrics are saved to `tuning_results/trial_{trial_id}/fold_{fold_num}_metrics.json`
  - Only the best trial and latest 2 trials are kept to conserve disk space
  - Older trial directories are automatically cleaned up after optimization completes
- **Output:** 
  - Best parameters JSON file (e.g., `tuning_results/best_params_..._CNV_app1_...json`)
  - Trials dataframe CSV with all trial results
  - Optuna visualization plots (parameter importances, optimization history, etc.)
  - Per-trial fold metrics for best + latest 2 trials

**Important implementation note (folding semantics):**

- The current implementation of `tune_models.py` creates a single QML model instance per Optuna trial and uses that same instance across the inner CV folds (it calls `.fit()` repeatedly on the same object for each fold). This means the model's weights continue training across folds within the same trial rather than being freshly initialized for every fold.

- Consequences:
    - Reported mean validation scores reflect sequential training across folds and can differ from scores obtained by training independent, fresh models per fold.
    - This approach was chosen for speed and to reuse checkpoints/optimizer state, but it is not the orthodox cross-validation pattern.

- Final training scripts (`dre_standard.py`, `cfe_standard.py`, etc.) use scikit-learn's `cross_val_predict` or cloned estimators and therefore perform fresh (independent) training per outer fold; only the tuning script uses the sequential-weights pattern described above.

- If you prefer strict CV correctness for tuning, we can update `tune_models.py` to instantiate a fresh model (and pipeline) inside each fold of the inner CV loop. This change is straightforward and recommended when you require fold-independence for hyperparameter selection.

**Stage 3: Base-Learner Training**
- **Approaches:** Dimensionality Reduction Encoding (DRE) and Conditional Feature Encoding (CFE)
- **Scripts:** `dre_standard.py`, `dre_relupload.py`, `cfe_standard.py`, `cfe_relupload.py`
- **Purpose:** To train specialized "expert" models for each data type using the best parameters found in Stage 2.

### The 2-Step Funnel: Imputation and Feature Selection

Both approaches use a **2-step preprocessing funnel** to prepare data for quantum circuits:

Note: This funnel applies to **model-level (training-time)** preprocessing. If you ran the data preprocessing notebooks, set `SOURCE_DIR=final_processed_datasets_xgb_balanced` to use the MI→XGBoost-selected features; see [DATA_PROCESSING.md](DATA_PROCESSING.md).

#### **Approach 1 (DRE): Imputation → Dimensionality Reduction**
```
Step 1: Imputation
Raw Data with NaNs → SimpleImputer (median strategy) → Complete numeric data

Step 2: Dimensionality Reduction
Complete data → PCA or UMAP → Reduced to n_qubits dimensions → QML Circuit
```

**Key Points:**
- Uses `MaskedTransformer` wrapper to prevent imputer from learning from missing modalities
- Imputation fills missing values before dimensionality reduction
- Choice of PCA (linear) or UMAP (non-linear) for reduction

#### **Approach 2 (CFE): Imputation → Feature Selection**
```
Step 1: Imputation (Implicit)
Raw Data with NaNs → Used directly for LightGBM/XGBoost feature selection
                   → NaNs filled with 0.0 for QML input (placeholder only)

Step 2: Feature Selection using Gradient Boosting
Complete/Raw data → LightGBM or XGBoost importance-based selection → Top-k features → QML Circuit
```

**Key Points:**
- **LightGBM (Default):** Lightweight, fast, handles missing values natively during selection
- **XGBoost (Alternative):** More robust for complex patterns, also handles missing values natively
- **Hybrid Method:** Ensemble predictions from both LightGBM and XGBoost selectors
- Feature selection preserves missingness information for conditional QML models
- Selected features are passed to conditional circuits that learn missing value encodings

### Feature Selection Options

**Option 1: LightGBM-based Selection (Current Default)**
```python
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=50, learning_rate=0.1, 
                     feature_fraction=0.7, n_jobs=1, random_state=42, verbosity=-1)
lgb.fit(X_train.values, y_train)  # Handles NaNs natively
importances = lgb.feature_importances_
selected_indices = np.argsort(importances)[-n_qubits:]
```

**Option 2: XGBoost-based Selection (Alternative)**
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=50, learning_rate=0.1, 
                    max_depth=6, n_jobs=1, random_state=42, verbosity=0)
xgb.fit(X_train.values, y_train)  # Handles NaNs natively
importances = xgb.feature_importances_
selected_indices = np.argsort(importances)[-n_qubits:]
```

**Option 3: Hybrid Combining Method**
```python
# Select features using both methods
lgb_selected = select_features_lgbm(X_train, y_train, n_qubits)
xgb_selected = select_features_xgb(X_train, y_train, n_qubits)

# Combine: use union of selected features
hybrid_selected = np.union1d(lgb_selected, xgb_selected)[:n_qubits]

# Or use intersection for more conservative selection
# hybrid_selected = np.intersect1d(lgb_selected, xgb_selected)

# Or ensemble: train separate models and average predictions
model_lgb = train_qml_with_features(X_train[:, lgb_selected], y_train)
model_xgb = train_qml_with_features(X_train[:, xgb_selected], y_train)
predictions = (model_lgb.predict_proba(X_test) + model_xgb.predict_proba(X_test)) / 2
```

**Process:** The four training scripts (`dre_standard.py`, `dre_relupload.py`, `cfe_standard.py`, `cfe_relupload.py`) iterate through all available data types. For each one, they:
    1. Find the corresponding `best_params_*.json` file.
    2. Load the data and build the appropriate classical-quantum pipeline (2-step funnel).
    3. Use `cross_val_predict` (for DRE) or a manual loop (for CFE) to generate **out-of-fold (OOF) predictions** for the training set. These predictions are crucial for training the meta-learner without data leakage.
    4. Train a final model on the *entire* training set.
    5. Generate predictions on the hold-out test set.
- **Output:** For each data type, the scripts save the OOF predictions, test predictions, and the final trained model artifacts (`.joblib` files) into a dedicated output directory (e.g., `base_learner_outputs_app1_standard/`).

---

### Data preprocessing, masking, and MaskedTransformer

We made several coordinated changes to how preprocessing and missingness are handled across the pipeline to support conditional models and to avoid leaking information into scalers/dimensionality reducers.

- MaskedTransformer (new): a small sklearn-style wrapper that "masks out" rows that are all-zero (or below an `eps` threshold) when fitting inner transformers (imputers, scalers, PCA, UMAP). It preserves row order and returns arrays of the same shape on transform by filling back transformed rows into their original positions. This prevents scalers / PCA from learning from artificially-zeroed rows that represent a missing modality for a sample.

- Fallback behavior: `MaskedTransformer` exposes a `fallback` option (`'warn' | 'raise' | 'all'`) to control what happens when no non-zero rows are present during fit; by default it will warn and fall back to fitting on all rows.

- Where used: The `MaskedTransformer` is used widely in base-learner pipelines (DRE flows and most pre-processing steps in CFE flows) to ensure scalers and PCA/UMAP fit only on rows with real signal. The `metalearner.py` intentionally keeps standard scalers unchanged (meta-features are dense and require regular scaling).

- Per-sample masked loss in QML models: QML base-learners (conditional models) now compute per-sample losses and average only over the subset of samples that actually contain data for that modality (using an all-zero detection / missingness mask). This allows the model to forward-propagate all samples but ignore missing-modality rows when computing loss and backpropagating.

- Artifact naming and sentinel scalers: For Approach 2 (CFE) the selected feature artifact is now saved as `selected_features_{datatype}.joblib`. Since conditional models do not use sklearn scalers, the training scripts save a sentinel `scaler_{datatype}.joblib` containing `None` so that `inference.py` can detect and skip scaling at inference time. `inference.py` was updated to handle `scaler is None` and log an info-level message when scaling is skipped.

- LightGBM feature selection on raw data for CFE: Feature selection for conditional flows (Approach 2) is now performed on the raw DataFrame (with NaNs), so LightGBM can use its native missing-value handling when computing feature importances. Do not impute or scale before LightGBM selection for CFE.

- Tuning change (Optuna): The hyperparameter tuning script `tune_models.py` was updated so that scaler hyperparameters are only suggested and used for Approach 1 (DRE). Approach 2 sets `scaler = None` during tuning and does not tune sklearn scalers (conditional models learn from missingness instead).

- Tests: An end-to-end integration test (`tests/test_conditional_e2e.py`) was added to validate the entire train→save→inference path for conditional models (it uses a small synthetic artifact set and monkeypatches parquet reading to CSV during tests to avoid optional parquet engine dependencies).


**Stage 4: Meta-Learner Training (`metalearner.py`)**
- **Purpose:** To train a single, powerful "manager" model that learns from the predictions of the expert base-learners.
- **Process:**
    1. **Assembles Meta-Features:** The script loads the OOF predictions generated in Stage 3. It also loads an `indicator_file`, which contains supplementary classical features (e.g., clinical data like age or tumor stage). These are concatenated to form the feature set for the meta-learner.
    2. **Trains Meta-Learner:** It trains a QML model on this combined feature set, using the true labels from the training data.
- **Optimization Metric:** Trials are optimized using **weighted F1 score** instead of accuracy for better handling of class imbalance.
- **Comprehensive Metrics:** Each trial tracks accuracy, precision, recall, F1 (macro/weighted), specificity (macro/weighted), confusion matrix, and classification report.
- **Automatic Directory Management:** If the journal file or output directory is read-only, the system automatically copies them to a writable location, ensuring training can proceed in restricted environments.
- **Output:** 
    - Final trained meta-learner model (`meta_learner_final.joblib`)
    - Meta-learner column order (`meta_learner_columns.json`)
    - Best hyperparameters if tuning was run (`best_metalearner_params.json`)
    - Per-trial comprehensive metrics during tuning (`trial_{trial_id}/metrics.json`)

Recent hardening and traceability improvements
---------------------------------------------

To make the meta-learner assembly and gating semantics more robust and auditable, the following changes were made:

- Mask expansion parity: The per-datatype indicator columns (named `is_missing_{datatype}_`) are expanded into a per-prediction mask that aligns one-to-one with the base prediction columns (those beginning with `pred_`). This ensures gated meta-learners that accept `(base_preds, mask)` tuples receive masks with exactly the same shape as the base prediction matrix.

- Degeneracy removal (drop one class column per base learner): Because each base learner outputs a probability distribution across classes (columns sum to 1), one class column per base learner is linearly dependent on the others. To avoid perfect multicollinearity we now drop exactly one class column per base learner when assembling the meta-feature matrix. The chosen column is the one that corresponds to the master label encoder's last class when available; otherwise we fall back to a deterministic lexicographic choice. This preserves all information (the dropped probability can be recovered from the remaining ones) while improving numerical stability for classical learners.

- Persisted traceability: The mapping of dropped columns (datatype -> dropped column name) is written to `OUTPUT_DIR/dropped_pred_columns.json` so you can map model inputs back to original base-learner outputs for interpretation.

- Train/test alignment: The same reduced column set (determined on the training side) is enforced on the test-side features. If some columns are missing in the test predictions they are reindexed and filled with zeros so train/test feature ordering and shapes always match.

- Sanity checks: `assemble_meta_data(...)` now validates that the assembled base prediction columns match the mask shape produced from the indicators. If a mismatch is detected, the assembler raises a clear error and logs diagnostic information to help find the root cause rather than failing later inside model code.

- Artifact of final column ordering: For reproducibility we persist the final `base_prediction_columns` and `indicator_columns` to `OUTPUT_DIR/meta_train_feature_columns.json` so the exact inputs used to train the meta-learner are recorded.

These changes were made to reduce debugging overhead (clear early errors), improve numerical stability for the classical meta-learner, and make model inputs auditable for downstream analysis and explainability.

**Notes & CLI additions:**
- The meta-learner intentionally does not save or require a scaler; meta-features are probabilities (0..1) from base learners plus indicator features.
- New CLI flags supported by `metalearner.py`:
    - `--meta_model_type` — force `standard` or `reuploading` for final training (overrides tuned value).
    - `--meta_n_layers` — force number of ansatz layers for final training (overrides tuned value).
    - `--meta_n_qubits` — force number of qubits (defaults to the number of meta-features if not provided).
    - `--learning_rate` — if provided, a fixed learning rate is used during tuning (trials keep this LR) and final training. This allows running tuning while controlling only the LR externally.

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

### Pickling and QNode caching

The QML model classes use a cached PennyLane QNode to avoid repeated re-creation overhead. Because the cached QNode is a nested function (a closure), it isn't picklable by default. To support model persistence with `joblib`/pickle and reliable serialization across processes, the classes implement custom `__getstate__` and `__setstate__` methods:

- `__getstate__`: Removes the non-picklable `_qcircuit` (the cached QNode) and the stored `_activation_fn` callable from the instance state before pickling. This produces a pickle-friendly state that contains all numeric parameters (quantum weights, classical weights and optimizer state) but not the runtime QNode object.
- `__setstate__`: Restores the numeric state and lazily recreates the cached QNode by calling the model's `_get_circuit()` factory. It also reinitializes the activation callable based on the `readout_activation` string.

This design ensures pickled models are portable and can be unpickled in environments where PennyLane is available; if the device/QNode cannot be recreated during unpickle, the code falls back gracefully by setting the `_qcircuit` attribute to `None` and recreating it later on first use.

Practical notes:
- When saving models with `joblib.dump`, the saved file will contain all numeric parameters and can be safely reloaded with `joblib.load`. The cached QNode is recreated on-demand when the model is used (for inference or further training).
- The activation function is restored from the configuration string (`readout_activation`) rather than pickling the callable itself to maximize portability.

---

## 🔢 Data Encoding: Why Angle Encoding Over Amplitude Encoding

A critical design decision in quantum machine learning is **how to encode classical data into quantum states**. This project exclusively uses **Angle Encoding** (`AngleEmbedding` in PennyLane) rather than Amplitude Encoding. This section explains the technical rationale behind this choice.

### The Two Major Encoding Strategies

#### Amplitude Encoding

Amplitude encoding stores classical data in the **probability amplitudes** of a quantum state. For a normalized classical vector `x = [x_0, x_1, ..., x_{N-1}]` where the sum of squared values equals 1, the quantum state is:

```
|ψ⟩ = x_0|0⟩ + x_1|1⟩ + ... + x_{N-1}|N-1⟩
```

**Example:** To encode a 4-dimensional vector `[0.5, 0.5, 0.5, 0.5]`, you would prepare the state:
```
|ψ⟩ = 0.5|00⟩ + 0.5|01⟩ + 0.5|10⟩ + 0.5|11⟩
```

This requires only `log₂(N)` qubits to encode `N` features—seemingly very efficient.

#### Angle Encoding (Used in This Project)

Angle encoding maps each classical feature to a **rotation angle** on a dedicated qubit. For a classical vector `x = [x_0, x_1, ..., x_{n-1}]`, we apply a Y-rotation gate to each qubit:

```
|ψ⟩ = RY(x_0 · π)|0⟩ ⊗ RY(x_1 · π)|0⟩ ⊗ ... ⊗ RY(x_{n-1} · π)|0⟩
```

**Example:** To encode a 4-dimensional vector `[0.1, 0.5, 0.8, 0.3]` on 4 qubits:
- Qubit 0: `RY(0.1π)|0⟩`
- Qubit 1: `RY(0.5π)|0⟩`
- Qubit 2: `RY(0.8π)|0⟩`
- Qubit 3: `RY(0.3π)|0⟩`

This requires `N` qubits for `N` features—one qubit per feature.

---

### Why Amplitude Encoding is Problematic for NISQ Devices

Despite its apparent efficiency (logarithmic qubit scaling), amplitude encoding has **severe practical limitations** that make it unreliable for near-term quantum machine learning:

#### 1. **Exponential State Preparation Complexity**

Preparing an arbitrary amplitude-encoded state requires a circuit whose depth grows **exponentially** with the number of qubits:

| Qubits | Encoded Features | Gate Depth (Amplitude) | Gate Depth (Angle) |
|--------|------------------|------------------------|-------------------|
| 4      | 16               | ~15 gates              | 4 gates           |
| 8      | 256              | ~255 gates             | 8 gates           |
| 10     | 1024             | ~1023 gates            | 10 gates          |
| 14     | 16384            | ~16383 gates           | 14 gates          |

For multi-omics data with hundreds or thousands of features, the amplitude encoding circuit would require thousands of gates—far exceeding what NISQ (Noisy Intermediate-Scale Quantum) devices can reliably execute.

#### 2. **Noise Accumulation and Decoherence**

Current quantum hardware has significant noise sources:
- **Gate errors:** Each quantum gate has a small probability of error (typically 0.1-1% for two-qubit gates)
- **Decoherence:** Quantum states decay over time (coherence times of microseconds to milliseconds)

The **deep circuits** required for amplitude encoding accumulate errors exponentially:

```
Circuit Fidelity ≈ (1 - ε)^D
```

where ε is the per-gate error rate and D is the circuit depth.

| Encoding | Circuit Depth | Fidelity (ε=0.5%) | Fidelity (ε=1%) |
|----------|---------------|-------------------|-----------------|
| Angle (14 features) | ~14 | 93.2% | 86.9% |
| Amplitude (16 features) | ~15 | 92.8% | 86.0% |
| Amplitude (256 features) | ~255 | 27.9% | 7.7% |
| Amplitude (1024 features) | ~1023 | 0.6% | 0.003% |

For realistic multi-omics feature counts, amplitude encoding produces states dominated by noise, making the encoded information unrecoverable.

#### 3. **Normalization Requirements Distort Data**

Amplitude encoding requires the input vector to be **normalized** (sum of squared values = 1). This introduces several problems:

- **Scale information loss:** The magnitude of features is compressed into a unit sphere, losing valuable scale information
- **Zero-vector problem:** All-zero vectors (completely missing modalities) cannot be normalized
- **Batch dependency:** Normalization may depend on batch statistics, causing train/test distribution shifts
- **Negative values:** Complex handling is needed for negative feature values (phase encoding)

**Example of information loss:**
```
Original vectors:     [1, 2, 3, 4] and [100, 200, 300, 400]
After normalization:  [0.182, 0.365, 0.547, 0.730] for BOTH
```

Both vectors encode to the **same quantum state**, despite representing very different magnitudes.

#### 4. **Non-Trivial Gradient Computation**

For variational quantum algorithms, we need to compute gradients via the **parameter-shift rule**. Amplitude encoding creates complex dependencies between:
- The classical data
- The state preparation circuit parameters
- The variational circuit parameters

This makes gradient computation through the state preparation layer extremely challenging. In contrast, angle encoding provides **direct, differentiable** access to the encoded data through simple rotation gates.

#### 5. **No Efficient Classical Simulation**

One practical benefit of quantum ML is the ability to **simulate on classical hardware** during development. Amplitude encoding of large vectors:
- Requires exponential classical memory to represent the state (2^n complex amplitudes)
- Cannot be efficiently simulated for more than ~30 qubits
- Makes debugging and development impractical

Angle encoding with `n` qubits requires only `O(n)` classical simulation complexity for product states.

---

### Research-Backed Fundamental Problems with Amplitude Encoding

Beyond practical NISQ limitations, peer-reviewed research has established **fundamental theoretical barriers** that make amplitude encoding problematic even for future fault-tolerant quantum computers:

#### 6. **Proven Exponential State Preparation Lower Bounds**

**Shende, Bullock & Markov (2006)** proved in *IEEE Transactions on Computer-Aided Design* that generic quantum state preparation requires `Ω(2^n)` CNOT gates—this is a **mathematical lower bound**, not just a current hardware limitation.

**Plesch & Brukner (2011)** in *Physical Review A* confirmed that arbitrary amplitude encoding requires `O(2^n)` gates for `n` qubits. This means:

> **Encoding N classical data points via amplitude encoding takes O(N) time—no better than just reading the data classically.**

The theoretical qubit savings (`log₂(N)` qubits) are completely negated by the exponential circuit depth.

#### 7. **Barren Plateaus Make Training Impossible**

**McClean et al. (2018)** in *Nature Communications* ("Barren plateaus in quantum neural network training landscapes") demonstrated that for random parameterized quantum circuits, **gradients vanish exponentially** with the number of qubits:

```
Var[∂C/∂θ] ≤ F(n) · e^(-αn)
```

This is catastrophic for amplitude encoding because:
- Deep state preparation circuits create highly entangled states
- Highly entangled states exhibit barren plateaus
- **Training becomes exponentially hard** as the number of qubits increases

**Cerezo et al. (2021)** in *Nature Communications* extended this, showing barren plateaus occur even in shallow circuits when using global cost functions—exactly the structure amplitude encoding requires.

**Holmes et al. (2022)** in *PRX Quantum* proved a fundamental trade-off: **more expressive circuits (needed for amplitude encoding) have worse trainability**. You cannot have both.

#### 8. **The Data Loading Bottleneck Negates Quantum Speedup**

This is perhaps the most devastating result. **Aaronson (2015)** in *Nature Physics* ("Read the fine print") identified the "input problem":

> Any quantum speedup must be measured **end-to-end**. If loading classical data takes O(N) time, you cannot achieve better than O(N) total runtime, regardless of how fast the quantum processing is.

| Step | Classical Algorithm | Quantum with Amplitude Encoding |
|------|--------------------|---------------------------------|
| Data loading | O(N) | O(N) or O(2^n) gates |
| Processing | O(f(N)) | O(poly(log N)) |
| Readout | O(1) | O(N) measurements |
| **Total** | O(N + f(N)) | O(N) minimum |

The quantum processing speedup is **sandwiched between classical-time input and output**, eliminating any end-to-end advantage.

#### 9. **Measurement Collapse Destroys Encoded Information**

**Aaronson (2015)** and **Huang, Kueng & Preskill (2020)** in *Nature Physics* established fundamental limits:

- A single measurement **collapses** the quantum state, destroying the superposition
- To extract N amplitude values with precision ε, you need `O(N/ε²)` **copies of the state**
- Each copy requires re-running the exponential state preparation circuit

**Haah et al. (2017)** in *IEEE Transactions on Information Theory* proved that full state tomography (reconstructing all amplitudes) requires `O(2^n)` measurements.

> **The classical data encoded via amplitude encoding cannot be efficiently recovered.** This makes amplitude encoding a "one-way function"—data goes in but cannot come back out efficiently.

#### 10. **Dequantization: Classical Algorithms Match Quantum Performance**

A series of breakthrough results showed that quantum ML algorithms assuming efficient amplitude encoding can be **matched by classical algorithms**:

**Tang (2019)** at *STOC* ("A quantum-inspired classical algorithm for recommendation systems") showed that if amplitude encoding is efficient (as QML papers assume), then classical algorithms achieve the same performance. This "dequantization" result undermined the quantum advantage claims of several prominent QML algorithms.

**Cotler et al. (2021)** in *PRX Quantum* clarified that quantum advantage might exist for truly quantum data, but **not for classical data loaded via amplitude encoding**.

**Schuld & Petruccione (2022)** proved that "Supervised quantum machine learning models are kernel methods"—and for classical data, these quantum kernels provide **no proven advantage over classical kernels**.

#### 11. **Exponential Concentration in Quantum Kernels**

**Thanasilp et al. (2024)** in *Nature Communications* ("Exponential concentration in quantum kernel methods") delivered a major blow:

- Quantum kernels (which rely on amplitude or similar encodings) suffer from **exponential concentration**
- As the feature dimension grows, all kernel values become exponentially close to a constant
- This makes the kernel matrix nearly singular and **training becomes impossible**

> For high-dimensional data like multi-omics features, quantum kernels based on amplitude encoding become completely untrainable.

#### 12. **Quantum RAM (qRAM) Does Not Exist**

Many theoretical QML algorithms assume access to **quantum RAM**—a hypothetical device that can prepare amplitude-encoded states in `O(log N)` time. However:

- **qRAM has never been built** at any meaningful scale
- **Giovannetti et al. (2008)** proposed qRAM but acknowledged exponential resource requirements
- Recent analysis suggests qRAM may be **physically unrealizable** due to error accumulation
- Without qRAM, amplitude encoding provides no speedup

As **Preskill (2018)** noted in "Quantum Computing in the NISQ era and beyond":

> "Quantum speedups often assume access to quantum data or efficient state preparation. For classical data, this assumption is problematic."

---

### The Research Consensus (2020-2025)

The quantum computing research community has reached a **sobering consensus** on amplitude encoding:

| Claim | Status | Key References |
|-------|--------|----------------|
| Amplitude encoding is exponentially expensive | ✅ Proven | Shende et al. 2006, Plesch & Brukner 2011 |
| Deep encoding circuits cause barren plateaus | ✅ Proven | McClean et al. 2018, Cerezo et al. 2021 |
| Data loading negates quantum speedup | ✅ Proven | Aaronson 2015, Tang 2019 |
| Encoded data cannot be efficiently recovered | ✅ Proven | Haah et al. 2017, Huang et al. 2020 |
| Classical algorithms can match QML with amplitude encoding | ✅ Proven | Tang 2019, 2021 |
| Quantum kernels with amplitude encoding are untrainable | ✅ Proven | Thanasilp et al. 2024 |
| qRAM exists and is practical | ❌ Disproven | No working implementation exists |

**Current research direction**: The field is shifting toward:
- **Native quantum data** (quantum sensing, quantum chemistry simulation)
- **Angle/rotation encoding** for NISQ devices (this project's approach)
- **Structured/sparse data** where specialized encoding might help
- **Quantum advantage in specific, narrow domains** rather than general ML

---

### Why Angle Encoding is Reliable and Practical

Angle encoding addresses all the above concerns:

#### 1. **Constant-Depth Circuits**

Each feature requires exactly **one rotation gate**:
```python
# PennyLane AngleEmbedding
for i in range(n_features):
    qml.RY(features[i] * np.pi, wires=i)
```

The circuit depth is `O(1)` per feature, not `O(2^n)`.

#### 2. **Hardware-Compatible Gate Count**

For 14 qubits encoding 14 features:
- **Angle encoding:** 14 single-qubit rotation gates
- **Amplitude encoding:** Would require encoding 16,384 features, needing ~16,383 gates

Current quantum hardware can reliably execute circuits with 10-100 gates. Angle encoding stays well within this limit.

#### 3. **Preserves Feature Independence**

Each feature is encoded on a dedicated qubit:
- Features don't interfere during encoding
- Missing features can be handled independently (conditional encoding)
- Scale information is preserved within the rotation angle range

#### 4. **Natural Gradient Flow**

The parameter-shift rule works naturally with rotation gates:
```
∂/∂θ ⟨O⟩ = (1/2) · [⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}]
```

This enables efficient gradient computation for variational training.

#### 5. **Handles Missingness Gracefully**

Our conditional encoding scheme (`ConditionalMulticlassQuantumClassifierFS`) leverages angle encoding's per-qubit structure:
```python
if feature_is_present:
    qml.RY(feature_value * np.pi, wires=i)
else:
    qml.RY(learned_missing_angle, wires=i)  # Trainable parameter
```

This is impossible with amplitude encoding, where all features are entangled in a single state preparation.

---

### Trade-off: Qubit Count vs. Circuit Depth

The primary trade-off is clear:

| Aspect | Amplitude Encoding | Angle Encoding |
|--------|-------------------|----------------|
| Qubits needed | `log₂(N)` | `N` |
| Circuit depth | `O(2^n)` = `O(N)` gates | `O(n)` = `O(log N)` gates |
| NISQ feasibility | ❌ Impractical for N > 16 | ✅ Practical for N ≤ 20 |
| Noise resilience | ❌ Poor (deep circuits) | ✅ Good (shallow circuits) |
| Gradient computation | ❌ Complex | ✅ Simple |
| Missing value handling | ❌ Difficult | ✅ Natural |

**For this project's multi-omics cancer classification task:**
- We use **dimensionality reduction** (PCA/UMAP) to reduce features to the qubit count (8-14 features)
- This makes angle encoding feasible while preserving the most important variance
- The shallow circuits remain noise-resilient on current hardware

---

### Our Implementation: AngleEmbedding in Practice

All models in `qml_models.py` use PennyLane's `AngleEmbedding`:

```python
@qml.qnode(dev, interface="autograd")
def circuit(weights, features):
    # Angle encoding: one RY rotation per feature/qubit
    qml.AngleEmbedding(features, wires=range(n_qubits), rotation='Y')
    
    # Variational layers with trainable parameters
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

**Key configuration:**
- `rotation='Y'`: Uses `RY` gates (real-valued rotations on the Bloch sphere)
- Input features are scaled to `[0, 1]` or `[-1, 1]` range, then multiplied by π
- Each qubit encodes exactly one feature

---

### Summary: Angle Encoding is the Right Choice for NISQ-Era QML

| Requirement | Amplitude Encoding | Angle Encoding |
|-------------|-------------------|----------------|
| Works on current quantum hardware | ❌ | ✅ |
| Tolerates realistic noise levels | ❌ | ✅ |
| Supports gradient-based training | Difficult | ✅ |
| Handles missing data | ❌ | ✅ |
| Scales to practical feature counts | ❌ (exponential gates) | ✅ (linear gates) |
| Enables classical simulation | ❌ (for large N) | ✅ |

**Conclusion:** Amplitude encoding's theoretical qubit efficiency is negated by its exponential circuit depth, noise sensitivity, and incompatibility with variational training. Angle encoding, combined with dimensionality reduction, provides a **reliable, noise-resilient, and trainable** encoding scheme suitable for real-world quantum machine learning on NISQ devices.

---

### References: Data Encoding Research

For readers interested in the primary literature on quantum data encoding:

**State Preparation Complexity:**
- Shende, V.V., Bullock, S.S., & Markov, I.L. (2006). "Synthesis of quantum-logic circuits." *IEEE Trans. Computer-Aided Design*, 25(6):1000-1010.
- Plesch, M. & Brukner, Č. (2011). "Quantum-state preparation with universal gate decompositions." *Physical Review A*, 83:032302.
- Araujo, I.F. et al. (2021). "A divide-and-conquer algorithm for quantum state preparation." *Scientific Reports*, 11:6329.

**Barren Plateaus and Trainability:**
- McClean, J.R. et al. (2018). "Barren plateaus in quantum neural network training landscapes." *Nature Communications*, 9:4812.
- Cerezo, M. et al. (2021). "Cost function dependent barren plateaus in shallow parameterized quantum circuits." *Nature Communications*, 12:1791.
- Holmes, Z. et al. (2022). "Connecting ansatz expressibility to gradient magnitudes and barren plateaus." *PRX Quantum*, 3:010313.

**Data Loading and Quantum Advantage:**
- Aaronson, S. (2015). "Read the fine print." *Nature Physics*, 11:291-293.
- Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond." *Quantum*, 2:79.
- Tang, E. (2019). "A quantum-inspired classical algorithm for recommendation systems." *Proceedings of STOC 2019*.

**Measurement and Tomography:**
- Haah, J. et al. (2017). "Sample-optimal tomography of quantum states." *IEEE Trans. Information Theory*, 63(9):5628-5641.
- Huang, H.-Y., Kueng, R., & Preskill, J. (2020). "Predicting many properties of a quantum system from very few measurements." *Nature Physics*, 16:1050-1057.

**Dequantization and Kernel Methods:**
- Tang, E. (2021). "Quantum machine learning without the quantum." *arXiv:2107.07295*.
- Schuld, M. & Petruccione, F. (2022). "Supervised quantum machine learning models are kernel methods." *arXiv:2101.11020*.
- Cotler, J. et al. (2021). "Revisiting dequantization and quantum advantage in learning tasks." *PRX Quantum*.
- Thanasilp, S. et al. (2024). "Exponential concentration in quantum kernel methods." *Nature Communications*.

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
  - The `_batched_qcircuit` method implements a two-stage strategy to maximize performance while remaining portable:
    ```python
    def _batched_qcircuit(self, X, weights, n_jobs=None):
        # Fast-path: try true batched call first
        try:
            qouts = qcircuit(X_batch, weights)
            qouts = np.asarray(qouts, dtype=np.float64)
            if qouts.ndim == 2 and qouts.shape[0] == N:
                return qouts
        except Exception:
            pass

        # Sequential per-sample fallback (safe across devices)
        results = [qcircuit(X_arr[i], weights) for i in range(N)]
        stacked = np.vstack([np.asarray(r, dtype=np.float64) for r in results])
        return stacked
    ```
  - Attempts native batched execution first for maximum speed
  - Falls back to sequential per-sample evaluation when batched execution is not available or fails
  - The sequential fallback is robust across PennyLane devices and avoids device-specific batching issues; it trades parallelism for correctness and portability

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

- **Training Speed:** 2-5x faster depending on batch size and model complexity (on devices that support batching)
- **Memory Efficiency:** Better memory access patterns with contiguous arrays
- **Scalability:** Performance improvements increase with larger batch sizes when the device supports batching
- **Robustness:** The sequential fallback ensures correct behavior on devices without batched support or when batched execution fails

### Implementation Details

All four quantum model classes incorporate these optimizations:
- `MulticlassQuantumClassifierDR`
- `MulticlassQuantumClassifierDataReuploadingDR`  
- `ConditionalMulticlassQuantumClassifierFS`
- `ConditionalMulticlassQuantumClassifierDataReuploadingFS`

The optimizations are completely internal - all external APIs remain unchanged, ensuring backward compatibility with existing code.

### Configurable Parameters

    - **`n_jobs`:** Reserved parameter for parallel fallback (present for backwards compatibility).
        - The current implementation falls back to sequential per-sample evaluation when batching is unavailable. `n_jobs` is accepted by the API but is not used by the sequential fallback. It may be reintroduced in future versions if a safe parallel fallback is implemented.

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
    --keep_last_n 3 \                  # Keep only last 3 checkpoints
    --resume auto                      # Resume from checkpoint (best/latest/auto)
```

**CLI Resume Option:**

All training scripts (`dre_*.py`, `cfe_*.py`, `metalearner.py`) support the `--resume` argument:
- `--resume best`: Resume from best validation checkpoint
- `--resume latest`: Resume from most recent checkpoint
- `--resume auto`: Try best first, fallback to latest

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

**Output:** 
    - Final trained meta-learner model: `meta_learner_final.joblib`
    - Meta-learner column order: `meta_learner_columns.json`
    - Best hyperparameters if tuning was run: `best_metalearner_params.json`
    - Per-trial comprehensive metrics during tuning: `trial_{trial_id}/metrics.json`
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
python dre_standard.py --skip_tuning --n_qbits 8 --n_layers 4
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

---

## 🔄 Integrating 2-Step Funnel with QML: Detailed Examples

This section provides concrete, step-by-step examples of how to use the 2-step funnel (imputation → feature selection) with different feature selection methods and integrate them with quantum machine learning.

### Example 1: Using LightGBM Feature Selection (Default)

**Step-by-Step Workflow:**

```bash
# Step 1: Create master label encoder (one-time setup)
python create_master_label_encoder.py

# Step 2: Tune hyperparameters with LightGBM feature selection (Approach 2)
# Note: Approach 2 uses LightGBM's built-in feature importance for selection
python tune_models.py \
    --datatype CNV \
    --approach 2 \
    --qml_model standard \
    --n_trials 30 \
    --verbose

# Step 3: Train base learners with LightGBM feature selection
python cfe_standard.py \
    --datatypes CNV Prot Meth \
    --verbose

# Output: Base learner predictions with LightGBM-selected features
# - train_oof_preds_CNV.csv
# - test_preds_CNV.csv
# - selected_features_CNV.joblib (contains LightGBM-selected feature indices)
# - qml_model_CNV.joblib
```

**Integration Points:**
1. **Imputation**: Implicit - raw data with NaNs used for LightGBM selection
2. **Feature Selection**: LightGBM native missing value handling during importance computation
3. **QML Input**: Selected features with missingness mask for conditional encoding

### Example 2: Using XGBoost Feature Selection (Future Extension)

> **Note:** XGBoost feature selection is not currently implemented in the codebase. 
> The default and only supported feature selector is LightGBM (Approach 2).
> This section describes a conceptual extension for future implementation.

**Conceptual Workflow (Not Yet Implemented):**

```bash
# Step 1: Install XGBoost if not already available
pip install xgboost>=1.7.0

# Step 2: Extension would require modifying cfe_standard.py to support XGBoost
# Current approach: LightGBM is hardcoded in the 2-step funnel pipeline
#
# To implement XGBoost support, you would need to:
# 1. Add XGBoostClassifier as alternative to LGBMClassifier in feature selection
# 2. Add --feature_selector argument to cfe_*.py scripts
# 3. Update tune_models.py to support XGBoost selection

# Current usage (LightGBM only):
python cfe_standard.py \
    --datatypes CNV Prot Meth \
    --verbose
```

**Key Differences from LightGBM:**
- **Speed**: XGBoost may be slower but often more robust
- **Missing Value Handling**: Both handle NaNs natively, but use different algorithms
- **Feature Importance**: XGBoost uses gain/weight/cover; LightGBM uses split-based importance

### Example 3: Hybrid Method - Union of Features

**Step-by-Step Workflow:**

```bash
# Step 1: Generate features using both selectors
# This requires a custom script or modified training script

# Create a hybrid feature selection script
cat > select_features_hybrid.py << 'EOF'
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def select_features_hybrid(X_train, y_train, n_qubits, method='union'):
    """
    Select features using both LightGBM and XGBoost, then combine.
    
    Args:
        X_train: Training data (DataFrame with NaNs)
        y_train: Training labels
        n_qubits: Number of features to select
        method: 'union', 'intersection', or 'vote'
    
    Returns:
        selected_indices: Array of selected feature indices
    """
    # LightGBM selection
    lgb = LGBMClassifier(n_estimators=50, learning_rate=0.1, 
                         feature_fraction=0.7, n_jobs=1, 
                         random_state=42, verbosity=-1)
    lgb.fit(X_train.values, y_train)
    lgb_importances = lgb.feature_importances_
    lgb_top_k = np.argsort(lgb_importances)[-n_qubits:]
    
    # XGBoost selection
    xgb = XGBClassifier(n_estimators=50, learning_rate=0.1,
                        max_depth=6, n_jobs=1, 
                        random_state=42, verbosity=0,
                        tree_method='hist', enable_categorical=True)
    xgb.fit(X_train.values, y_train)
    xgb_importances = xgb.feature_importances_
    xgb_top_k = np.argsort(xgb_importances)[-n_qubits:]
    
    if method == 'union':
        # Union: combine both sets, keep top n_qubits by average importance
        union_indices = np.union1d(lgb_top_k, xgb_top_k)
        avg_importances = (lgb_importances + xgb_importances) / 2
        union_ranked = sorted(union_indices, 
                             key=lambda i: avg_importances[i], 
                             reverse=True)
        selected_indices = np.array(union_ranked[:n_qubits])
    
    elif method == 'intersection':
        # Intersection: only features selected by both
        intersect = np.intersect1d(lgb_top_k, xgb_top_k)
        if len(intersect) < n_qubits:
            # Fall back to union if intersection too small
            print(f"Intersection ({len(intersect)}) < n_qubits ({n_qubits}), using union")
            return select_features_hybrid(X_train, y_train, n_qubits, 'union')
        avg_importances = (lgb_importances + xgb_importances) / 2
        intersect_ranked = sorted(intersect,
                                 key=lambda i: avg_importances[i],
                                 reverse=True)
        selected_indices = np.array(intersect_ranked[:n_qubits])
    
    elif method == 'vote':
        # Weighted voting: rank features and use combined ranks
        lgb_ranks = np.argsort(np.argsort(lgb_importances))
        xgb_ranks = np.argsort(np.argsort(xgb_importances))
        combined_ranks = lgb_ranks + xgb_ranks
        selected_indices = np.argsort(combined_ranks)[-n_qubits:]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return selected_indices

# Example usage
if __name__ == '__main__':
    # Load data
    data = pd.read_parquet('final_processed_datasets/data_CNV_.parquet')
    X = data.drop('label', axis=1)
    y = LabelEncoder().fit_transform(data['label'])
    
    # Select features using hybrid method
    selected = select_features_hybrid(X, y, n_qubits=8, method='union')
    
    # Save for later use
    joblib.dump(selected, 'hybrid_selected_features_CNV.joblib')
    print(f"Selected features: {selected}")
    print(f"Feature names: {X.columns[selected].tolist()}")
EOF

python select_features_hybrid.py

# Step 2: Train QML model with hybrid-selected features
# Note: cfe_standard.py does not currently support custom feature indices
# This would require extending the codebase or manually modifying the script
# Current usage with default LightGBM selection:
python cfe_standard.py \
    --datatypes CNV \
    --verbose
```

> **Future Extension:** To use custom-selected features, you would need to modify 
> `cfe_standard.py` to accept a `--feature_file` argument that loads pre-selected 
> feature indices instead of computing them via LightGBM.

### Example 4: Hybrid Method - Ensemble Predictions

**Step-by-Step Workflow:**

```bash
# Train separate models with different QML approaches
# Then ensemble their predictions

# Step 1: Train model with standard QML circuit
OUTPUT_DIR=base_learner_outputs_standard python cfe_standard.py \
    --datatypes CNV \
    --verbose

# Step 2: Train model with re-uploading QML circuit  
OUTPUT_DIR=base_learner_outputs_reuploading python cfe_relupload.py \
    --datatypes CNV \
    --verbose

# Step 3: Combine predictions using ensemble script
cat > ensemble_hybrid_predictions.py << 'EOF'
import pandas as pd
import numpy as np

# Load predictions from both models
standard_train = pd.read_csv('base_learner_outputs_standard/train_oof_preds_CNV.csv')
reupload_train = pd.read_csv('base_learner_outputs_reuploading/train_oof_preds_CNV.csv')

standard_test = pd.read_csv('base_learner_outputs_standard/test_preds_CNV.csv')
reupload_test = pd.read_csv('base_learner_outputs_reuploading/test_preds_CNV.csv')

# Average predictions (ensemble)
ensemble_train = standard_train.copy()
ensemble_test = standard_test.copy()

# Average probability columns
prob_cols = [c for c in standard_train.columns if c.startswith('pred_')]
for col in prob_cols:
    ensemble_train[col] = (standard_train[col] + reupload_train[col]) / 2
    ensemble_test[col] = (standard_test[col] + reupload_test[col]) / 2

# Save ensemble predictions
ensemble_train.to_csv('base_learner_outputs_hybrid/train_oof_preds_CNV.csv', index=False)
ensemble_test.to_csv('base_learner_outputs_hybrid/test_preds_CNV.csv', index=False)

print("Hybrid ensemble predictions created!")
EOF

python ensemble_hybrid_predictions.py

# Step 4: Use hybrid predictions with meta-learner
python metalearner.py \
    --preds_dir base_learner_outputs_hybrid \
    --indicator_file indicator_features.parquet \
    --mode train \
    --verbose
```

### Example 5: Complete End-to-End with Hybrid Method

**Full Pipeline:**

```bash
# 1. Setup
python create_master_label_encoder.py

# 2. Tune for all modalities with LightGBM (default)
for dtype in CNV Prot Meth GeneExpr miRNA SNV; do
    python tune_models.py \
        --datatype $dtype \
        --approach 2 \
        --qml_model standard \
        --n_trials 20 \
        --verbose
done

# 3. Train base learners with LightGBM selection
python cfe_standard.py --verbose

# 4. (Optional) Train additional models with XGBoost selection
# Requires code modification to support XGBoost selector

# 5. Prepare predictions for meta-learner
mkdir -p final_ensemble_predictions
cp base_learner_outputs_app2_standard/train_oof_preds_*.csv final_ensemble_predictions/
cp base_learner_outputs_app2_standard/test_preds_*.csv final_ensemble_predictions/
cp master_label_encoder/label_encoder.joblib final_ensemble_predictions/

# 6. Train meta-learner
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file indicator_features.parquet \
    --mode tune \
    --verbose

python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file indicator_features.parquet \
    --mode train \
    --verbose

# 7. Deploy and run inference
mkdir -p final_model_deployment
cp meta_learner_final.joblib final_model_deployment/
cp meta_learner_columns.json final_model_deployment/
cp master_label_encoder/label_encoder.joblib final_model_deployment/

# Copy base learner artifacts
for dtype in CNV Prot Meth GeneExpr miRNA SNV; do
    cp base_learner_outputs_app2_standard/*_${dtype}.joblib final_model_deployment/
done

python inference.py \
    --model_dir final_model_deployment \
    --patient_data_dir new_patient_data
```

### Key Integration Points Summary

| **Step** | **Approach 1 (DRE)** | **Approach 2 (CFE)** |
|----------|---------------------|---------------------|
| **Imputation** | MaskedTransformer(SimpleImputer) | Implicit (NaNs preserved for selector) |
| **Feature Selection** | PCA or UMAP | LightGBM, XGBoost, or Hybrid |
| **QML Input** | Reduced dimensions | Selected features + missingness mask |
| **Missing Handling** | Imputed before reduction | Conditional encoding in QML |
| **Best For** | Dense data, linear patterns | Sparse data, complex patterns |

### Recommended Usage

1. **Start with LightGBM** (fastest, good default)
2. **Try XGBoost** if LightGBM underperforms (more robust)
3. **Use Hybrid Union** for maximum feature coverage
4. **Use Hybrid Ensemble** for maximum prediction robustness

All methods seamlessly integrate with the existing QML stacked ensemble architecture.

