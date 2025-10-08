# Project Architecture: A Deep Dive

This document provides a detailed breakdown of the architectural decisions, quantum models, and classical machine learning strategies used in this project.

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

At the core of this project are four different **Variational Quantum Circuits (VQCs)**. The term "variational" (or "hybrid quantum-classical") means that they have classical parameters (weights) that are optimized using a familiar classical algorithm like Adam. The workflow for a single training step is:

1.  **Execute Circuit:** The quantum computer (or simulator) runs the circuit with the current set of classical data and trainable weights.
2.  **Calculate Expectation:** It measures the qubits to get an expectation value for each output.
3.  **Compute Loss:** This expectation value is fed into a classical loss function (like cross-entropy) to see how wrong the prediction was.
4.  **Update Weights:** A classical optimizer (Adam) calculates the gradient of the loss and decides how to update the trainable weights to improve the result in the next iteration.

This loop leverages the quantum processor for its unique computational power while relying on robust, classical methods for optimization.

### **1. The Standard Workhorse (`MulticlassQuantumClassifierDR`)**

This model is the foundation for the **Dimensionality Reduction Encoding (DRE)** approach. It's a standard and powerful architecture for classification tasks where the input features are already dense and information-rich.

*   **How it Works (Step-by-Step):**
    1.  **Encoding (`AngleEmbedding`):** The process begins by loading the classical data vector (e.g., the 8 principal components from PCA) into the quantum circuit. The `AngleEmbedding` layer takes this vector `[x_0, x_1, ..., x_7]` and maps it to 8 qubits. It does this by applying a rotation gate to each qubit, using the feature's value as the rotation angle (e.g., `RY(x_0)` on the first qubit, `RY(x_1)` on the second). This "encodes" the classical information into the quantum state.
    2.  **Processing (`BasicEntanglerLayers`):** This is the trainable, "neural network" part of the model. It's a repeating block of two types of gates:
        *   **Rotation Gates (`RY`):** Each qubit is rotated by a trainable angle. These angles are the `weights` that the model learns during optimization.
        *   **Entangling Gates (`CNOT`):** Gates like `CNOT` are applied between adjacent qubits. This is the most crucial step. **Entanglement** creates non-local correlations between the qubits, allowing them to process information collectively. These layers allow the circuit to create and explore a massive, high-dimensional computational space (the Hilbert space) to find complex patterns that might be inaccessible to classical models of a similar size.
    3.  **Measurement:** After processing, we only measure the first `n_classes` qubits. The measurement is the expectation value of the Pauli-Z operator, which gives a real number between -1 and 1. This vector of real numbers (e.g., `[-0.2, 0.9, -0.5]`) represents the model's raw, "logit-like" output.
    4.  **Softmax Activation:** The raw output is not a probability distribution. The classical softmax function is applied as a final post-processing step to convert these raw values into probabilities that sum to 1 (e.g., `[0.15, 0.70, 0.15]`), which can then be used to make the final classification.

*   **Architectural Rationale:** This architecture is a well-established standard for VQCs. The use of more qubits than classes is a deliberate choice to create "workspace" qubits. These extra qubits participate in the entanglement and processing, allowing for more complex intermediate calculations before the final result is extracted from the first few qubits. This increases the model's **capacity** (its ability to learn complex functions).

### **2. The High-Capacity Model (`MulticlassQuantumClassifierDataReuploadingDR`)**

*   **How it Works (The Key Difference):** This model modifies the standard architecture by re-inserting the input data between each processing layer. Instead of one encoding at the beginning, there are multiple "data re-uploading" steps. Each layer consists of: `AngleEmbedding` -> `BasicEntanglerLayers`.
*   **Architectural Rationale:** We chose this architecture to test the hypothesis that some data types might have patterns that are too complex for the standard VQC. Data re-uploading dramatically increases the model's **expressivity** (its ability to represent complex functions). It has been shown that this technique effectively turns the circuit into a quantum version of a Fourier series, allowing it to approximate much more complex functions. The trade-off is a higher number of parameters and a greater risk of overfitting, making it suitable for situations where we suspect the decision boundary is highly non-linear.

### **3. The Missing Data Specialist (`ConditionalMulticlassQuantumClassifierFS`)**

This model is the foundation for the **Conditional Feature Encoding (CFE)** approach. Its innovation lies entirely in the encoding step, which is designed to treat missing data as a first-class citizen.

*   **How it Works (Step-by-Step):**
    1.  **Dual Input:** The model receives two pieces of information for each sample: the feature vector (where `NaN`s are filled with a placeholder like 0) and a binary mask vector that indicates which features were originally missing.
    2.  **Conditional Encoding:** The encoding layer iterates through each qubit. For each qubit `i`, it checks the `i`-th element of the mask vector.
        *   If `mask[i] == 0` (the feature is present), it applies a standard rotation using the feature's value: `RY(feature[i] * np.pi, wires=i)`.
        *   If `mask[i] == 1` (the feature is missing), it applies a rotation using a separate, **trainable parameter**: `RY(weights_missing[i], wires=i)`.
    3.  **Processing and Measurement:** The rest of the circuit (the entangling layers and measurement) is identical to the standard workhorse model.
*   **Architectural Rationale:** The core hypothesis here is that **"missingness" is valuable information, not a problem to be fixed**. Instead of using a classical method like mean imputation (which makes an uninformed guess and can shrink variance), we let the model itself *learn* the best possible representation for a missing value. The optimizer might discover that the most effective way to represent a missing protein feature is a specific angle that places the qubit in a superposition—a state that is difficult to represent classically and might be the key to separating two classes.

### **4. The Ultimate Complexity Test (`ConditionalMulticlassQuantumClassifierDataReuploadingFS`)**

*   **Architectural Rationale:** This model is the logical synthesis of our two experimental hypotheses. It combines the missingness-aware encoding of the CFE approach with the high-capacity data re-uploading architecture. We included this to test if the combination of these two advanced techniques could provide a performance edge on the most challenging datasets, where we suspect that both missingness and pattern complexity are high. It is the most powerful but also the most computationally expensive and data-hungry model in our arsenal.

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

