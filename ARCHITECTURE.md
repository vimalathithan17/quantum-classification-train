# Project Architecture: A Deep Dive

This document provides a detailed breakdown of the architectural decisions, quantum models, and classical machine learning strategies used in this project.

---

## 📜 End-to-End Pipeline Workflow

The project is structured as a multi-stage pipeline. Each script performs a distinct role, creating artifacts that are used by subsequent stages.

![Pipeline Workflow](https://i.imgur.com/your-diagram-image.png) <!-- Placeholder for a visual diagram -->

**Stage 1: Global Setup (`create_master_label_encoder.py`)**
- **Purpose:** To ensure consistent class labels across the entire project.
- **Process:** The script scans all `*.parquet` files in the source data directory, collects every unique class name (e.g., 'BRCA', 'LUAD'), and creates a single, master `LabelEncoder`.
- **Output:** `master_label_encoder/label_encoder.joblib`. This artifact is a critical dependency for all subsequent training and inference scripts.

**Stage 2: Hyperparameter Tuning (`tune_models.py`)**
- **Purpose:** To find the optimal set of hyperparameters for each base-learner configuration.
- **Process:** Using Optuna, this script runs a series of trials for a specified data type (`--datatype`), architectural approach (`--approach`), and QML model (`--qml_model`). It uses `StratifiedKFold` cross-validation to robustly evaluate each parameter set.
- **Output:** A JSON file (e.g., `tuning_results/best_params_..._CNV_app1_...json`) for each tuned configuration, containing the best-performing parameters.

**Stage 3: Base-Learner Training (`dre_*.py`, `cfe_*.py`)**
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

At the core of this project are four different **Variational Quantum Circuits (VQCs)**. The term "variational" means that they have classical parameters (weights) that are optimized using a classical algorithm. The quantum computer (or simulator) is used to calculate a value (the expectation value), and a classical optimizer (like Adam) uses this value to decide how to update the weights in the next iteration.

### **1. The Standard Workhorse (`MulticlassQuantumClassifierDR`)**
This model is the foundation for the **Dimensionality Reduction Encoding (DRE)** approach. It's a standard and powerful architecture for classification tasks.

* **How it Works (Step-by-Step):**
    1.  **Encoding (`AngleEmbedding`):** The process begins by loading the classical data vector (e.g., the 8 principal components from PCA) into the quantum circuit. The `AngleEmbedding` layer takes this vector `[x_0, x_1, ..., x_7]` and maps it to 8 qubits. It does this by applying a rotation gate to each qubit. For instance, it might apply an `RY(x_0)` gate to the first qubit, an `RY(x_1)` gate to the second, and so on. The value of the classical feature `x_i` is directly used as the rotation angle. This effectively "encodes" the classical information into the quantum state.
    2.  **Processing (`BasicEntanglerLayers`):** This is the trainable part of the model. It's a repeating block of two types of gates:
        * **Rotation Gates:** Each qubit is rotated by a trainable angle. These angles are the `weights` that the model learns.
        * **Entangling Gates:** Gates like `CNOT` are applied between adjacent qubits. This is the most crucial step. **Entanglement** creates correlations between the qubits, allowing them to process information collectively. The entangling layers allow the circuit to create and explore a massive, high-dimensional computational space (the Hilbert space) to find complex patterns.
    3.  **Measurement:** After processing, we only measure the first `n_classes` qubits. The measurement is the expectation value of the Pauli-Z operator, which gives a real number between -1 and 1. This vector of real numbers (e.g., `[-0.2, 0.9, -0.5]`) represents the model's raw output.
    4.  **Softmax Activation:** The raw output is not a probability distribution. The classical softmax function is applied as a final step to convert these raw values into probabilities that sum to 1 (e.g., `[0.15, 0.70, 0.15]`), which can then be used to make the final classification.

* **Architectural Rationale:** This architecture is a well-established standard for VQCs. The use of more qubits than classes is a deliberate choice to create "workspace" qubits. These extra qubits participate in the entanglement and processing, allowing for more complex intermediate calculations before the final result is extracted from the first few qubits. This increases the model's **capacity** to learn.

### **2. The High-Capacity Model (`MulticlassQuantumClassifierDataReuploadingDR`)**
* **How it Works (The Key Difference):** This model modifies the standard architecture by re-inserting the input data between each processing layer. Instead of one encoding at the beginning, there are multiple "data re-uploading" steps.
* **Architectural Rationale:** We chose this architecture to test the hypothesis that some data types might have patterns that are too complex for the standard VQC. Data re-uploading dramatically increases the model's **expressivity** (its ability to represent complex functions), effectively turning the circuit into a quantum version of a Fourier series. The trade-off is a higher risk of overfitting and longer training times.

### **3. The Missing Data Specialist (`ConditionalMulticlassQuantumClassifierFS`)**
This model is the foundation for the **Conditional Feature Encoding (CFE)** approach. Its innovation lies entirely in the encoding step.

* **How it Works (Step-by-Step):**
    1.  **Dual Input:** The model receives two pieces of information for each sample: the feature vector (where `NaN`s are filled with a placeholder like 0) and a binary mask vector that indicates which features were originally missing.
    2.  **Conditional Encoding:** The encoding layer iterates through each qubit. For each qubit `i`, it checks the `i`-th element of the mask vector.
        * If `mask[i] == 0` (the feature is present), it applies a standard rotation using the feature's value: `RY(feature[i] * np.pi, wires=i)`.
        * If `mask[i] == 1` (the feature is missing), it applies a rotation using a separate, **trainable parameter**: `RY(weights_missing[i], wires=i)`.
    3.  **Processing and Measurement:** The rest of the circuit (the entangling layers and measurement) is identical to the standard workhorse model.
* **Architectural Rationale:** The core hypothesis here is that **"missingness" is valuable information, not a problem to be fixed**. Instead of using a classical method like mean imputation (which makes an uninformed guess), we let the model itself *learn* the best possible representation for a missing value. The optimizer might discover that the most effective way to represent a missing protein feature is a specific angle that places the qubit in a superposition, a state that is difficult to represent classically.

### **4. The Ultimate Complexity Test (`ConditionalMulticlassQuantumClassifierDataReuploadingFS`)**
* **Architectural Rationale:** This model is the logical synthesis of our two experimental hypotheses. It combines the missingness-aware encoding of the CFE approach with the high-capacity data re-uploading architecture. We included this to test if the combination of these two advanced techniques could provide a performance edge on the most challenging datasets, where we suspect that both missingness and pattern complexity are high.

---
## 🏛️ Classical Design Decisions: The Rationale

### **Dimensionality Reduction: PCA vs. UMAP (Approach 1 - DRE)**
The DRE approach requires us to distill thousands of features into a small, information-rich vector that can be loaded into a quantum circuit.

* **PCA (Principal Component Analysis):** This was chosen as the **baseline** because it's a fast, linear, and widely understood technique. It works by finding the axes of maximum variance in the data. We use it to answer the question: "How well does a quantum model perform when fed features from a standard, linear classical pipeline?"
* **UMAP (Uniform Manifold Approximation and Projection):** This was chosen as the **experimental alternative**. UMAP is a non-linear technique that assumes high-dimensional data lies on a lower-dimensional, curved manifold. The hypothesis is that the biological states in multi-omics data are not simple linear clusters but complex, intertwined manifolds. UMAP might be better at "unraveling" these structures into a set of features that are more meaningful for the quantum classifier.

### **Feature Selection: `SelectKBest` vs. `SelectFromModel` (Approach 2 - CFE)**
The CFE approach requires us to select a small subset of the best original features. The strategy for this selection changes depending on the context.

* **`SelectKBest` with `f_classif`:**
    * **What it is:** A simple, **univariate filter**. It scores each feature individually based on its statistical ANOVA F-value with the class labels and keeps the top "k".
    * **Where it's used:** During the **hyperparameter tuning** phase in `tune_models.py` and for **all non-`Meth` data types** in the final training scripts (`cfe_*.py`).
    * **Why:** It is extremely **fast**. Since feature selection must be performed inside every trial of the Optuna loop and for every fold of cross-validation, speed is essential to make the process computationally feasible.

* **`SelectFromModel` with LightGBM:**
    * **What it is:** A powerful, **embedded method**. It trains a full, gradient-boosted tree model (LightGBM) and selects features based on their importance to that model's predictive power. It can capture complex **interactions** between features that univariate filters would miss.
    * **Where it's used:** Exclusively for the high-dimensional **`Meth` (Methylation) data type** during the **final training phase** in the `cfe_*.py` scripts.
    * **Why:** At this stage, **accuracy is more important than speed**. Methylation data is notoriously sparse and high-dimensional, and its predictive signals often rely on interactions between features. Using a more intelligent selection method for the final model ensures it is as robust and accurate as possible.

### **The Ensemble: Why Stack?**
The final architectural choice was to not rely on a single model but to build a **stacked ensemble**.

* **Rationale:** Multi-omics data is heterogeneous. The patterns in gene expression data are fundamentally different from those in methylation data. It is highly unlikely that one model architecture (e.g., DRE with PCA) would be the best for all data types.
    1. **Expert Models:** By training specialist "base learners" for each data type, we allow each model to become an expert on one kind of data.
    2. **Learning from Experts:** The **meta-learner** then acts as a "manager," learning from the *predictions* of these experts.
    3. **Supplementary Data:** Crucially, the meta-learner does not *only* see the predictions. It also receives **indicator features** (provided via the `--indicator_file` argument). These are additional classical features about the patient (e.g., age, tumor stage, or flags indicating if an entire data modality was missing).
* **Final Decision:** The meta-learner's final decision is based on both what the experts are saying (their predictions) and the supplementary context (the indicator features). It might learn, for instance, that for a particular cancer type, the `Prot` and `CNV` models are highly reliable, but if the `is_missing_SNV` flag is true, it should adjust its final prediction accordingly. This ability to learn the strengths and weaknesses of each specialist model in different contexts makes the final ensemble more accurate and robust than any single model could be on its own.

