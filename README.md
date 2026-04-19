# Quantum Transfer Learning Ensemble for Multiclass Cancer Classification

A stacked ensemble using Quantum Machine Learning (QML) classifiers for multiclass cancer classification from multi-omics data.

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[README.md](README.md)** | This file: QML pipeline setup and commands |
| **[DOCS_GUIDE.md](DOCS_GUIDE.md)** | Navigation guide with decision trees |
| **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** | Extensions integration, class imbalance, troubleshooting |
| **[DATA_PROCESSING.md](DATA_PROCESSING.md)** | Data pipeline: notebooks, outputs, formats |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Quantum circuits and design decisions |
| **[PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md)** | Transformer fusion and contrastive learning |
| **[RESOURCE_OPTIMIZED_WORKFLOW.md](RESOURCE_OPTIMIZED_WORKFLOW.md)** | Memory-constrained pipeline (30GB RAM, 14 qubits) |
| **[examples/README.md](examples/README.md)** | Running example scripts |

### Quick Decision Guide

| Dataset Size | Recommendation | Guide |
|--------------|----------------|-------|
| < 100 samples | QML Only | This README |
| 100-500 samples | Contrastive + QML | [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) |
| 500+ samples | Transformer Fusion | [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) |
| 1000+ samples | Full Hybrid Pipeline | [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) |

> **⚠️ Data Leakage Prevention:** With the new Global Split Data Architecture, always run `python create_global_split.py` to create `data/global_train` and `data/global_test` before running any pretraining, tuning, or base model scripts. Base learners train and tune strictly on 100% of `data/global_train`. At the end of their training scripts, they automatically calculate final hold-out test metrics on `data/global_test` and log them to W&B. The final meta-learner ensemble evaluation is still performed strictly via `inference.py`, but individual base models provide immediate test-set feedback directly to W&B. See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#-data-leakage-prevention-important) for details.

---

## Recent Updates

- **Regularization Defaults Strengthened:** All QML training scripts now use `weight_decay=1e-3` (was 0.0), `validation_frac=0.2` (was 0.1), and `patience=25` (was 50) by default to reduce overfitting. Override via CLI flags if needed.
- Transformer fusion now includes modern regularization: class weighting, label smoothing, AdamW optimizer, ReduceLROnPlateau scheduler, and configurable dropout. See `train_transformer_fusion.py` for new args: `--dropout`, `--weight_decay`, `--label_smoothing`, `--patience`, `--lr_patience`.
- Added validation-based early stopping to contrastive pretraining (`examples/pretrain_contrastive.py` and `performance_extensions/training_utils.py`). Use `--val_size` and `--patience` to enable/adjust early stopping during encoder pretraining.
- All QML classifiers in `qml_models.py` now accept a `weight_decay` parameter and apply L2 regularization to the loss (controlled via `weight_decay`, **default 1e-3**).


## � Kaggle Production Workflow (Recommended)

Our streamlined production workflow is managed through 5 sequentially numbered Kaggle notebooks located in the `kaggle_notebooks/` directory. This isolates dependencies, intermediate artifacts, and avoids Kaggle session timeouts.

**The 5-Step Kaggle Pipeline:**

1. **`01_Global_Data_Split_Kaggle.ipynb` (Data Prep)**
   - **Action:** Runs global train/test splits to prevent data leakage.
   - **Output:** Zips `global_split_data.zip`. 
   - **Next Step:** Download this zip and upload it as a new **Kaggle Dataset** (e.g., `qml-global-split-data`).

2. **`02_Contrastive_Pretraining_Kaggle.ipynb` (Representation Learning)**
   - **Action:** Mounts the internal dataset + the dataset from Notebook 01. Pretrains contrastive encoders.
   - **Output:** Zips `pretrained_embeddings.zip`.
   - **Next Step:** Download and upload as a new **Kaggle Dataset** (e.g., `qml-pretrained-embeddings`).

3. **`03_QML_Tuning_and_Training_Kaggle.ipynb` (Base Learners - Quantum)**
   - **Action:** Mounts the dataset from Notebook 01. Tunes and trains the 4 QML base models.
   - **Output:** Zips OOF predictions (`.csv`) and trained models (`.joblib`).
   - **Next Step:** Download and upload as a new **Kaggle Dataset** (e.g., `qml-base-learner-outputs`).

4. **`04_Transformer_Fusion_Kaggle.ipynb` (Base Learners - Classical)**
   - **Action:** Mounts datasets from Notebook 01 and 02. Trains the Transformer Fusion model.
   - **Output:** Zips OOF predictions (`.csv`) and trained models (`.joblib`).
   - **Next Step:** Download and upload as a new **Kaggle Dataset** (e.g., `qml-transformer-outputs`).

5. **`05_Meta_Learner_and_Inference_Kaggle.ipynb` (Ensemble & Evaluation)**
   - **Action:** Brings together the original dataset and all Kaggle Datasets created in steps 1, 3, and 4.
   - **Output:** Trains the Level-1 XGBoost Meta-Learner, runs final inference on the global holdout test set, and outputs final metrics/confusion matrices.

For a detailed setup guide on Kaggle, see **[KAGGLE_SETUP.md](KAGGLE_SETUP.md)**.

---

## �🔄 Key Feature: 2-Step Preprocessing Funnel

Training-time preprocessing for quantum circuits. If using the data notebooks, set `SOURCE_DIR=final_processed_datasets_xgb_balanced`. See [DATA_PROCESSING.md](DATA_PROCESSING.md).

The pipeline uses a **2-step funnel** to prepare high-dimensional multi-omics data for quantum circuits:

**Approach 1 (DRE): Imputation → Dimensionality Reduction**
1. **Step 1**: Imputation - Fill missing values using SimpleImputer (median strategy)
2. **Step 2**: Dimensionality Reduction - Reduce to n_qubits dimensions using PCA or UMAP

**Approach 2 (CFE): Imputation → Feature Selection**
1. **Step 1**: Implicit Imputation - Preserve NaNs for native handling by selector
2. **Step 2**: Feature Selection - Select top-k features using:
   - **LightGBM** (default): Fast, handles missing values natively
   - **XGBoost** (alternative): More robust, handles missing values natively
   - **Hybrid**: Combine both methods (union, intersection, or ensemble)

Both approaches integrate seamlessly with quantum circuits for classification.

---

## Directory layout (recommended)

```
project_root/
├── final_processed_datasets/     # Input: processed .parquet files (default SOURCE_DIR)
│   ├── data_CNV_.parquet
│   └── ...
├── master_label_encoder/         # Output: label encoder (label_encoder.joblib)
├── tuning_results/               # Output: tuning JSON files
├── base_learner_outputs_app1_standard/  # Example output directory for base learners
├── base_learner_outputs_app1_reuploading/
├── base_learner_outputs_app2_standard/
├── base_learner_outputs_app2_reuploading/
├── final_ensemble_predictions/   # User-curated best-of predictions for meta-learner
├── final_model_deployment/       # User-created directory containing final models for inference
├── meta_learner_final.joblib
├── meta_learner_columns.json
├── meta_learner_best_params.json
├── create_master_label_encoder.py
├── tune_models.py
├── dre_standard.py                      # Approach 1: Dimensionality Reduction Encoding (standard)
├── dre_relupload.py                      # Approach 1: Dimensionality Reduction Encoding (data reuploading)
├── cfe_standard.py                      # Approach 2: Conditional Feature Encoding (standard)
├── cfe_relupload.py                      # Approach 2: Conditional Feature Encoding (data reuploading)
├── metalearner.py
├── inference.py
└── README.md
```

---

## Important: configure SOURCE_DIR (and other dirs) via environment variables

All scripts now read the input `SOURCE_DIR` from the environment with a default of `final_processed_datasets`.

Set it once in your shell session before running scripts if your data lives elsewhere. Example:

```bash
export SOURCE_DIR=/absolute/path/to/your/processed_datasets
# Optional: override output / config dirs too
export OUTPUT_DIR=/absolute/path/to/outputs
export TUNING_RESULTS_DIR=/absolute/path/to/tuning_results
export ENCODER_DIR=/absolute/path/to/master_label_encoder
```

If you don't set these, the defaults will be used:
- SOURCE_DIR: `final_processed_datasets`
- OUTPUT_DIR: script-specific defaults (e.g. `base_learner_outputs_app1_standard`)
- TUNING_RESULTS_DIR: `tuning_results`
- ENCODER_DIR: `master_label_encoder`

---

## 📊 Weights & Biases (W&B) Setup

All training scripts support optional experiment tracking with [Weights & Biases](https://wandb.ai/). This section explains how to set up W&B and use it with this codebase.

### Step 1: Create a W&B Account

1. Go to [https://wandb.ai/](https://wandb.ai/) and click **Sign Up**
2. Create an account (free tier is sufficient for most use cases)
3. After signing in, you'll see your dashboard

### Step 2: Get Your API Key

1. Click on your profile icon (top right) → **Settings**
2. Scroll to **API keys** section
3. Copy your API key (it looks like a long string of letters and numbers)

### Step 3: Install and Login

```bash
# Install wandb (already in requirements.txt)
pip install wandb

# Login with your API key (one-time setup)
wandb login
# Paste your API key when prompted
```

Alternatively, set the API key as an environment variable:
```bash
export WANDB_API_KEY=your_api_key_here
```

### Step 4: Understanding Project and Run Names

W&B organizes experiments hierarchically:

| Concept | Description | Example |
|---------|-------------|---------|
| **Entity** | Your username or team name | `john_doe` |
| **Project** | A collection of related experiments | `qml_cancer_classification` |
| **Run** | A single training execution | `dre_standard_CNV_20260124` |

**`--wandb_project`**: Groups related experiments together. Use consistent project names across related runs:
- `qml_base_learners` - for base learner training
- `qml_metalearner` - for meta-learner experiments
- `qml_tuning` - for hyperparameter tuning
- `cancer_classification_full` - for complete pipeline runs

**`--wandb_run_name`**: Identifies a specific training run. Good naming conventions:
- Include the script type: `dre_standard_`, `cfe_relupload_`, `metalearner_`
- Include the data type: `CNV`, `GeneExpr`, `miRNA`
- Include key parameters: `q8_l3` (8 qubits, 3 layers)
- Example: `dre_standard_CNV_q8_l3_exp01`

If not specified, W&B auto-generates a random run name like `wandering-sunset-42`.

### Step 5: Using W&B with Training Scripts

```bash
# Basic usage with project name
python dre_standard.py --use_wandb --wandb_project qml_base_learners

# With custom run name
python dre_standard.py \
    --datatypes CNV \
    --use_wandb \
    --wandb_project qml_base_learners \
    --wandb_run_name dre_standard_CNV_exp01

# Hyperparameter tuning with W&B
python tune_models.py \
    --datatype GeneExpr \
    --approach 1 \
    --n_trials 50 \
    --use_wandb \
    --wandb_project qml_tuning

# Meta-learner with W&B
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --use_wandb \
    --wandb_project qml_metalearner \
    --wandb_run_name metalearner_final_v1
```

### W&B Dashboard Features

After running experiments, visit [https://wandb.ai/](https://wandb.ai/) to:
- **View training curves**: Loss, accuracy, F1 scores over time
- **Compare runs**: Side-by-side comparison of different hyperparameters
- **Track hyperparameters**: Automatic logging of all configuration
- **System metrics**: GPU/CPU usage, memory consumption
- **Artifacts**: Model checkpoints (when saved to W&B)

### Offline Mode

For environments without internet access:
```bash
export WANDB_MODE=offline
python dre_standard.py --use_wandb --wandb_project my_project

# Later, sync offline runs when connected:
wandb sync ./wandb/offline-run-*
```

### Disabling W&B

Simply omit the `--use_wandb` flag - training proceeds normally without W&B.

---

## Full workflow (commands)

Below are concrete example commands for each step. Commands assume you are in the repository root and have the environment variables configured as needed.

### 1) Create the master label encoder (one-time)

This scans all parquet files in `SOURCE_DIR` and creates a single `label_encoder.joblib` used by all scripts.

```bash
# Ensure SOURCE_DIR is set (optional if using default)
export SOURCE_DIR=final_processed_datasets

python create_master_label_encoder.py
```

Output: `master_label_encoder/label_encoder.joblib`

### 1.5) Create the Global Data Split (create_global_split.py)

To prevent data leakage during tuning, pretraining, and base model training, perform a global stratified split upfront.

```bash
python create_global_split.py --data_dir final_processed_datasets --out_train data/global_train --out_test data/global_test
```

Output: Creates `data/global_train/` and `data/global_test/` directories containing identically named parquet files.
**Crucial Step:** From this point on, set `SOURCE_DIR=data/global_train` for all tuning, pretraining, and base model training. Base learner scripts train on the `global_train` set and at the very end automatically load `data/global_test` to calculate final hold-out metrics and immediately log them to W&B. The actual metadata ensemble step (`inference.py`) evaluates the overall final system on this held-out data.

```bash
export SOURCE_DIR=data/global_train
```

### 2) Hyperparameter tuning (Optuna)

Use `tune_models.py` to tune base learners. Repeat per data type / approach / qml_model / dim reducer as needed.

**The 2-Step Funnel in Tuning:**
- **Approach 1**: Tunes imputation → dimensionality reduction pipeline
- **Approach 2**: Tunes feature selection method (LightGBM by default)

Examples:

```bash
# Tune Approach 1 (2-step: impute → PCA reduction) for CNV (50 trials)
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer pca --n_trials 50 --verbose

# Tune Approach 1 with UMAP instead of PCA (2-step: impute → UMAP reduction)
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer umap --n_trials 50 --verbose

# Tune Approach 2 (2-step: preserve NaNs → LightGBM selection) for Prot (30 trials)
python tune_models.py --datatype Prot --approach 2 --qml_model reuploading --n_trials 30

# Tune with Weights & Biases logging enabled
python tune_models.py --datatype CNV --approach 1 --qml_model standard --n_trials 50 --use_wandb --wandb_project my_project
```

**Understanding the 2-Step Funnel:**

**Approach 1 (DRE):**
- Step 1: `MaskedTransformer(SimpleImputer(strategy='median'))` - imputes missing values
- Step 2: `MaskedTransformer(PCA(n_components=n_qubits))` - reduces to n_qubits dimensions
- Alternative Step 2: `MaskedTransformer(UMAP(n_components=n_qubits))` - non-linear reduction

**Approach 2 (CFE):**
- Step 1: Raw data with NaNs preserved for selector
- Step 2: `LGBMClassifier` feature importance → selects top n_qubits features
- Alternative Step 2: `XGBClassifier` feature importance (requires code modification)
- Hybrid Step 2: Combine LightGBM + XGBoost selections

Notes:
- The script reads data from `SOURCE_DIR` and runs an Optuna study with `--n_trials` trials.
- Both approaches use a 2-step funnel: imputation/preservation → reduction/selection
- The number of training steps for tuning defaults to 100 (can be changed with `--steps`).
- Optimizes weighted F1 score (handles class imbalance better than accuracy)
- Comprehensive metrics computed per fold: accuracy, precision, recall, F1, specificity (macro/weighted)

Output:
- Best parameters JSON saved to `tuning_results/` (default)
- Per-trial per-fold metrics saved to `tuning_results/trial_N/fold_M_metrics.json`

Quick inspection: there is a small helper script included to view saved parameter files:

```bash
# Print one params file
python inspect_params.py tuning_results/best_params_multiclass_qml_tuning_CNV_app1_pca_standard.json

# Print all best_params_*.json files in the tuning_results directory
python inspect_params.py tuning_results/
```

Or from Python interactively/load a single file:

```python
import json
with open('tuning_results/best_params_xyz.json') as f:
	params = json.load(f)
print(params)
```

### 3) Train base learners (final training using tuned params)

Run the appropriate script for each approach variant. The training scripts read tuned parameters from the `tuning_results` folder and apply the **2-step preprocessing funnel** before training QML models.

**The 2-Step Funnel in Training:**

**Approach 1 (DRE):** Imputation → Dimensionality Reduction
```bash
# Uses: MaskedTransformer(SimpleImputer) → MaskedTransformer(PCA/UMAP)
python dre_standard.py --verbose --steps 100

# Data reuploading variant (same 2-step funnel)
python dre_relupload.py
```

**Approach 2 (CFE):** Preservation → Feature Selection
```bash
# Uses: Raw data with NaNs → LightGBM importance-based selection
python cfe_standard.py --verbose

# Data reuploading variant (same 2-step funnel)
python cfe_relupload.py --steps 100
```

Examples (the repository contains multiple `approach` scripts; run the ones you need):

```bash
# Approach 1: 2-step funnel (impute → reduce) with standard QML
python dre_standard.py --verbose --steps 100

# Approach 1: 2-step funnel (impute → reduce) with data reuploading QML
python dre_relupload.py

# Approach 2: 2-step funnel (preserve NaNs → select features) with standard QML
python cfe_standard.py --verbose

# Approach 2: 2-step funnel (preserve NaNs → select features) with data reuploading QML
python cfe_relupload.py --steps 100

You can override tuned parameters directly from the command line when running the training scripts. Supported overrides are:

- `--n_qbits` (int): override number of qubits / selected features used by the model/pipeline.
- `--n_layers` (int): override number of ansatz layers for QML circuits.
- `--steps` (int): override number of training steps.
- `--scaler` (str): override scaler selection using shorthand: `s` (Standard), `m` (MinMax), `r` (Robust); full names are also accepted.

You can also limit which data types are trained in a run by passing `--datatypes` followed by one or more data type names. This overrides the internal `DATA_TYPES_TO_TRAIN` list.

Example (train only CNV and Prot):

```bash
python dre_standard.py --datatypes CNV Prot --verbose
```

Example (override several params):

```bash
python dre_standard.py --n_qbits 8 --n_layers 4 --steps 150 --scaler m --verbose
```
```

Outputs (per data type):
- `train_oof_preds_<datatype>.csv` (used to train meta-learner)
- model artifacts: `pipeline_<datatype>.joblib` or `selected_features_<datatype>.joblib`, `scaler_<datatype>.joblib`, `qml_model_<datatype>.joblib`

### 4) Curate the "best-of" predictions for the meta-learner

Create a directory (for example, `final_ensemble_predictions`) and copy the `train_oof_preds_*` files you want the meta-learner to use. Also copy `label_encoder.joblib` into this folder (the meta-learner reads the encoder to reconstruct class labels).

Example:

```bash
mkdir -p final_ensemble_predictions

# Copy predictions for CNV from approach 1 standard
cp base_learner_outputs_app1_standard/train_oof_preds_CNV.csv final_ensemble_predictions/

# Copy predictions for Prot from approach 2 reuploading
cp base_learner_outputs_app2_reuploading/train_oof_preds_Prot.csv final_ensemble_predictions/

# Copy the master label encoder (required by meta-learner)
cp master_label_encoder/label_encoder.joblib final_ensemble_predictions/
```

The meta-learner training script (`metalearner.py`) accepts one or more prediction directories via the `--preds_dir` argument — pass the `final_ensemble_predictions` folder (or multiple folders) to it.

### 5) Tune and train the meta-learner

You can choose to run the Quantum Meta-Learner (QML) or the Classical Meta-Learner. The Classical Meta-Learner uses classical tree-based models and logistic regression (LightGBM, Random Forest, Logistic Regression) to stack the base learner probabilities.

**Option A: Quantum Meta-Learner**

Tune (optional):

```bash
# Tune the meta-learner hyperparameters with verbose logging
python metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode tune --verbose
```

Train final meta-learner (uses best parameters from tuning stored in the output directory):

```bash
python metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode train --verbose
```

**Option B: Classical Meta-Learner**

The classical meta-learner acts as an extremely fast stacked-ensemble alternative to the QML meta-learner and fully supports Weights and Biases (W&B) logging.

Tune classical algorithms (LightGBM, Random Forest, Logistic Regression) using Optuna:

```bash
python classical_metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode tune --n_trials 50 --use_wandb --wandb_project my_classical_meta
```

Train classical meta-learner (uses best tuned tree/linear configuration):

```bash
python classical_metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode train --use_wandb --wandb_project my_classical_meta
```

Outputs:
- `classical_meta_learner_final.joblib` (or `meta_learner_final.joblib`)
- `classical_meta_learner_columns.json` (exact column order used for training)

### 6) Prepare deployment directory and run inference on a new patient

Create a deployment directory and copy the meta-learner artifacts and the base models you want to use for inference.

Example:

```bash
mkdir -p final_model_deployment

# Meta learner + metadata
cp meta_learner_final.joblib final_model_deployment/
cp meta_learner_columns.json final_model_deployment/
cp master_label_encoder/label_encoder.joblib final_model_deployment/

# Copy selected base learner artifacts (examples):
cp base_learner_outputs_app1_standard/pipeline_CNV.joblib final_model_deployment/
cp base_learner_outputs_app2_reuploading/selected_features_Prot.joblib final_model_deployment/
cp base_learner_outputs_app2_reuploading/scaler_Prot.joblib final_model_deployment/
cp base_learner_outputs_app2_reuploading/qml_model_Prot.joblib final_model_deployment/
```

Prepare a deployment directory and copy the meta-learner artifacts and base models you want to use. Then run inference on the held-out `global_test` data (or any other new patient data).

Run inference on the global test set:
```bash
python inference.py --model_dir final_model_deployment --patient_data_dir data/global_test
```
The script will generate class predictions and evaluate final test performance if true labels are present.

---

## Notes and recommendations

- Reproducibility: run tuning/training with controlled random seeds (where scripts expose them). Some scripts use fixed seeds by default.
- Disk layout: keep `final_processed_datasets/` immutable and write outputs into separate directories to avoid accidental overwrites.
- Centralized config: for many environments it may be convenient to add a small `export_env.sh` file that sets `SOURCE_DIR`, `OUTPUT_DIR`, etc. Example:

```bash
# export_env.sh
export SOURCE_DIR=/abs/path/to/final_processed_datasets
export TUNING_RESULTS_DIR=/abs/path/to/tuning_results
export ENCODER_DIR=/abs/path/to/master_label_encoder
```

and source it before running scripts:

```bash
source export_env.sh
```
---

## Approach mapping — which script implements each approach

This repository provides two families of base-learner designs. The mapping below shows which scripts implement Approach 1 and Approach 2 (the filenames were renamed for clarity):

- Approach 1 — Dimensionality Reduction Encoding (DRE)
	- `dre_standard.py` — DRE with classical dimensionality reduction (PCA or UMAP) followed by a standard QML classifier.
	- `dre_relupload.py` — DRE using data re-uploading QML circuits for datasets where re-uploading is beneficial.

-- Approach 2 — Conditional Feature Encoding (CFE)
	- `cfe_standard.py` — CFE where the QML model is conditioned on a selected subset of features (standard QML circuit).
	- `cfe_relupload.py` — CFE using data re-uploading QML circuits and fold-wise feature selection (LightGBM importance-based selection).

---

## 🔄 Deep Dive: The 2-Step Preprocessing Funnel

This section provides additional context on how the 2-step funnel works in practice during the base model training loops.
> **Note:** This funnel occurs *after* the Global Split. The `X_train` referenced below is sourced exclusively from 80% `data/global_train`, while the remaining 20% `data/global_test` is fiercely protected until the evaluation block at the very end of the training scripts.

### Understanding the Two Approaches

Both approaches prepare high-dimensional multi-omics data for quantum circuits through a **2-step funnel**, but they differ in implementation:

#### **Approach 1 (DRE): Imputation → Dimensionality Reduction**

**Philosophy:** Fill missing values first, then reduce dimensions linearly or non-linearly.

**Step 1: Imputation**
```python
from sklearn.impute import SimpleImputer
from utils.masked_transformers import MaskedTransformer

# Impute missing values with median (only on non-missing modalities)
imputer = MaskedTransformer(SimpleImputer(strategy='median'))
X_imputed = imputer.fit_transform(X_train)
```

**Step 2: Dimensionality Reduction**
```python
from sklearn.decomposition import PCA
from umap import UMAP

# Option A: Linear reduction with PCA
reducer = MaskedTransformer(PCA(n_components=n_qubits))
X_reduced = reducer.fit_transform(X_imputed)

# Option B: Non-linear reduction with UMAP
reducer = MaskedTransformer(UMAP(n_components=n_qubits))
X_reduced = reducer.fit_transform(X_imputed)
```

**Result:** `X_reduced` is a dense array of shape `(n_samples, n_qubits)` ready for quantum circuit.

#### **Approach 2 (CFE): Preservation → Feature Selection**

**Philosophy:** Preserve missingness information, select important features using gradient boosting.

**Step 1: Preservation (Implicit Imputation)**
```python
# Keep raw data with NaNs for feature selector
X_raw_with_nans = X_train  # DataFrame with NaN values

# For QML input: replace NaNs with 0.0 as placeholder (model learns missing encoding)
X_filled = X_raw_with_nans.fillna(0.0)
is_missing = X_raw_with_nans.isnull().astype(int)
```

**Step 2: Feature Selection**

**Option A: LightGBM Selection (Default)**
```python
from lightgbm import LGBMClassifier

# LightGBM handles NaNs natively during training
lgb = LGBMClassifier(n_estimators=50, learning_rate=0.1, 
                     feature_fraction=0.7, random_state=42, verbosity=-1)
lgb.fit(X_raw_with_nans.values, y_train)  # Passes NaNs directly

# Select top k features by importance
importances = lgb.feature_importances_
selected_indices = np.argsort(importances)[-n_qubits:]
X_selected = X_filled[:, selected_indices]
is_missing_selected = is_missing[:, selected_indices]
```

**Option B: XGBoost Selection (Alternative)**
```python
from xgboost import XGBClassifier

# XGBoost also handles NaNs natively
xgb = XGBClassifier(n_estimators=50, learning_rate=0.1,
                    max_depth=6, random_state=42, verbosity=0,
                    tree_method='hist')
xgb.fit(X_raw_with_nans.values, y_train)  # Passes NaNs directly

# Select top k features by importance
importances = xgb.feature_importances_
selected_indices = np.argsort(importances)[-n_qubits:]
X_selected = X_filled[:, selected_indices]
is_missing_selected = is_missing[:, selected_indices]
```

**Option C: Hybrid Union**
```python
# Combine features selected by both methods
lgb_selected = select_features_lgbm(X_raw_with_nans, y_train, n_qubits)
xgb_selected = select_features_xgb(X_raw_with_nans, y_train, n_qubits)

# Union: take features selected by either method
union = np.union1d(lgb_selected, xgb_selected)

# Rank by combined importance and take top n_qubits
lgb_imp = lgb.feature_importances_
xgb_imp = xgb.feature_importances_
combined_imp = (lgb_imp + xgb_imp) / 2
selected_indices = sorted(union, key=lambda i: combined_imp[i], reverse=True)[:n_qubits]
```

**Option D: Hybrid Ensemble**
```python
# Train separate models and average predictions
model_lgb = train_qml_with_features(X_train[:, lgb_selected], y_train)
model_xgb = train_qml_with_features(X_train[:, xgb_selected], y_train)

# At inference: average probabilities
pred_lgb = model_lgb.predict_proba(X_test[:, lgb_selected])
pred_xgb = model_xgb.predict_proba(X_test[:, xgb_selected])
pred_ensemble = (pred_lgb + pred_xgb) / 2
```

**Result:** `(X_selected, is_missing_selected)` is a tuple ready for conditional quantum circuit.

### Practical Example: Complete Pipeline with 2-Step Funnel

```bash
#!/bin/bash
# Complete example using Approach 2 with LightGBM feature selection

# Step 0: Setup
export SOURCE_DIR=final_processed_datasets
export OUTPUT_DIR=base_learner_outputs_app2_standard

# Step 1: Create master label encoder
python create_master_label_encoder.py

# Step 2: Tune hyperparameters (finds best 2-step funnel config)
# This tunes: n_qubits (how many features to select), n_layers (QML depth)
python tune_models.py \
    --datatype CNV \
    --approach 2 \
    --qml_model standard \
    --n_trials 30 \
    --verbose

echo "Tuning complete. Best params saved to tuning_results/"

# Step 3: Train base learner using 2-step funnel
# Step 1 of funnel: Preserve NaNs
# Step 2 of funnel: LightGBM selects top-k features
python cfe_standard.py \
    --datatypes CNV \
    --verbose

echo "Training complete. OOF and test predictions saved to $OUTPUT_DIR/"

# Step 4: Prepare for meta-learner
mkdir -p final_ensemble_predictions
cp $OUTPUT_DIR/train_oof_preds_CNV.csv final_ensemble_predictions/
cp $OUTPUT_DIR/test_preds_CNV.csv final_ensemble_predictions/
cp master_label_encoder/label_encoder.joblib final_ensemble_predictions/

# Step 5: Train meta-learner
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file indicator_features.parquet \
    --mode train \
    --verbose

echo "Meta-learner training complete!"
```

### When to Use Each Approach

| **Criterion** | **Approach 1 (DRE)** | **Approach 2 (CFE)** |
|---------------|---------------------|---------------------|
| **Data Density** | Dense data (few missing values) | Sparse data (many missing values) |
| **Missing Value Philosophy** | Impute and ignore | Learn from missingness |
| **Feature Relationships** | Linear or smooth manifold | Complex, non-linear |
| **Interpretability** | Low (PCA components) | High (selected original features) |
| **Speed** | Faster (PCA) | Slower (LightGBM/XGBoost fitting) |
| **Best For** | Gene expression, methylation | Copy number, mutations, clinical |

### Feature Selection: LightGBM vs XGBoost vs Hybrid

| **Method** | **Speed** | **Robustness** | **Missing Handling** | **When to Use** |
|-----------|-----------|---------------|---------------------|----------------|
| **LightGBM** | ⭐⭐⭐ Fast | ⭐⭐ Good | Native, efficient | Default choice, most datasets |
| **XGBoost** | ⭐⭐ Moderate | ⭐⭐⭐ Excellent | Native, robust | When LightGBM underperforms |
| **Hybrid Union** | ⭐ Slow | ⭐⭐⭐ Excellent | Best of both | Maximum feature coverage |
| **Hybrid Ensemble** | ⭐ Slowest | ⭐⭐⭐ Best | Best of both | Maximum prediction robustness |

**Recommendation:** Start with LightGBM. If performance is suboptimal, try XGBoost. Use hybrid methods for final production models.

### Integration with Quantum Circuits

After the 2-step funnel, data flows into quantum circuits:

```python
# Approach 1: Reduced features → Standard QML
X_reduced = pipeline.transform(X_test)  # Shape: (n_samples, n_qubits)
predictions = qml_model.predict_proba(X_reduced)

# Approach 2: Selected features + mask → Conditional QML  
X_selected = X_test[:, selected_indices]
X_filled = X_selected.fillna(0.0)
is_missing = X_selected.isnull().astype(int)
predictions = qml_model.predict_proba((X_filled, is_missing))
```

The 2-step funnel ensures quantum circuits receive properly preprocessed, dimensionality-reduced inputs that maximize quantum advantage while handling missing data appropriately.

---

Below are the CLI arguments for each script (if not listed, script uses defaults):

1) `create_master_label_encoder.py`
	- No CLI arguments.
	- Behavior: Scans parquet files in `SOURCE_DIR` (env var or `final_processed_datasets`) and writes `label_encoder.joblib` to `OUTPUT_DIR` (default `master_label_encoder` or `ENCODER_DIR` env var if set).

2) `tune_models.py`
	- `--datatype` (str, required): Data type to tune (e.g., `CNV`, `Meth`, `Prot`).
	- `--approach` (int, required): `1` or `2` selecting Approach 1 (Dimensionality Reduction Encoding) or Approach 2 (Conditional Feature Encoding).
	- `--dim_reducer` (str, default `umap`): `pca` or `umap` (used by Approach 1).
	- `--qml_model` (str, default `standard`): `standard` or `reuploading`.
	- `--scalers` (str, default `smr`): String indicating which scalers to try (s: Standard, m: MinMax, r: Robust). E.g., 'sm' for Standard and MinMax.
	- `--n_trials` (int, default 9): Number of NEW Optuna trials to run (if study exists, these are added to existing trials).
	- `--total_trials` (int, optional): Target TOTAL number of trials. If study exists, computes remaining trials needed to reach this total. Mutually exclusive with `--n_trials` - use one or the other.
	- `--study_name` (str, optional): Override the auto-generated study name.
	- `--min_qbits` (int, optional): Minimum number of qubits for tuning. Defaults to `n_classes`.
	- `--max_qbits` (int, default 12): Maximum number of qubits for tuning.
	- `--min_layers` (int, default 2): Minimum number of layers for tuning.
	- `--max_layers` (int, default 5): Maximum number of layers for tuning.
	- `--steps` (int, default 100): Number of training steps for tuning.
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- `--validation_frequency` (int, default 25): Compute validation metrics every N steps.
	- `--use_wandb` (flag): Enable Weights & Biases logging during tuning.
	- `--wandb_project` (str, optional): W&B project name.
	- `--wandb_run_name` (str, optional): W&B run name (auto-generated if not provided).
	- Behavior: Loads data from `os.path.join(SOURCE_DIR, f'data_{datatype}_.parquet')`, runs an Optuna study using `--n_trials`, and writes best param JSON files to `TUNING_RESULTS_DIR`.
	- Note: For Approach 2 (Conditional Feature Encoding) feature selection is performed using a LightGBM classifier to compute feature importances; the top-k important features (k = number of qubits) are selected per fold and for the final model. `SelectKBest` is no longer used for Approach 2.


3) `dre_standard.py` and `dre_relupload.py`
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- `--n_qbits` (int, optional): Override number of qubits (or selected features) used by the model/pipeline.
	- `--datatypes` (str..., optional): Space-separated list of datatypes to train (overrides default `DATA_TYPES_TO_TRAIN`). Example: `--datatypes CNV Prot`.
	- `--n_layers` (int, optional): Override number of ansatz layers for the QML model.
	- `--steps` (int, optional): Override the number of training steps used for QML training.
	- `--scaler` (str, optional): Override scaler with shorthand: `s` (Standard), `m` (MinMax), `r` (Robust) or full name.
	- `--skip_tuning` (flag): Skip loading tuned parameters and use command-line arguments or defaults instead.
	- `--skip_cross_validation` (flag): Skip cross-validation and only train final model on full training set.
	- `--cv_only` (flag): Perform only cross-validation to generate OOF predictions and skip final training (useful for meta-learner training).
	- `--max_training_time` (float, optional): Maximum training time in hours (overrides fixed steps). Example: `--max_training_time 11`.
	- `--checkpoint_frequency` (int, default 50): Save checkpoint every N steps.
	- `--keep_last_n` (int, default 3): Keep last N checkpoints.
	- `--checkpoint_fallback_dir` (str, optional): Fallback directory for checkpoints if primary is read-only.
	- `--resume` (str, optional): Resume from checkpoint. Choices: `best` (best validation), `latest` (most recent), `auto` (try best, fallback to latest). Example: `--resume auto`.
	- `--validation_frequency` (int, default 25): Compute validation metrics every N steps.
	- `--validation_frac` (float, default 0.2): Fraction of training data for internal validation during QML training. Increased from 0.1 for better overfitting detection.
	- `--patience` (int, default 25): Early stopping patience in steps. Reduced from 50 for faster convergence.
	- `--use_wandb` (flag): Enable Weights & Biases logging.
	- `--wandb_project` (str, optional): W&B project name.
	- `--wandb_run_name` (str, optional): W&B run name.
	- Behavior: Each script iterates over `DATA_TYPES_TO_TRAIN` and for each data type will:
		- Look for tuned params in `TUNING_RESULTS_DIR`.
		- Load `data_{datatype}_.parquet` from `SOURCE_DIR`.
		- Train the pipeline (PCA/UMAP + QML) and save:
			- OOF predictions: `train_oof_preds_{datatype}.csv` in script-specific `OUTPUT_DIR`.
			- Test predictions: `test_preds_{datatype}.csv`.
			- Model artifacts: `pipeline_{datatype}.joblib`.


4) `cfe_standard.py` and `cfe_relupload.py`
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- `--n_qbits` (int, optional): Override number of qubits (or selected features) used by the model/pipeline.
	- `--datatypes` (str..., optional): Space-separated list of datatypes to train (overrides default `DATA_TYPES_TO_TRAIN`). Example: `--datatypes CNV Prot`.
	- `--n_layers` (int, optional): Override the number of ansatz layers for the QML model.
	- `--steps` (int, optional): Override the number of training steps used for QML training.
	- `--scaler` (str, optional): Override scaler with shorthand: `s` (Standard), `m` (MinMax), `r` (Robust) or full name.
	- `--skip_tuning` (flag): Skip loading tuned parameters and use command-line arguments or defaults instead.
	- `--skip_cross_validation` (flag): Skip cross-validation and only train final model on full training set.
	- `--cv_only` (flag): Perform only cross-validation to generate OOF predictions and skip final training (useful for meta-learner training).
	- `--max_training_time` (float, optional): Maximum training time in hours (overrides fixed steps). Example: `--max_training_time 11`.
	- `--checkpoint_frequency` (int, default 50): Save checkpoint every N steps.
	- `--keep_last_n` (int, default 3): Keep last N checkpoints.
	- `--checkpoint_fallback_dir` (str, optional): Fallback directory for checkpoints if primary is read-only.
	- `--resume` (str, optional): Resume from checkpoint. Choices: `best` (best validation), `latest` (most recent), `auto` (try best, fallback to latest). Example: `--resume auto`.
	- `--validation_frequency` (int, default 25): Compute validation metrics every N steps.
	- `--validation_frac` (float, default 0.2): Fraction of training data for internal validation during QML training. Increased from 0.1 for better overfitting detection.
	- `--patience` (int, default 25): Early stopping patience in steps. Reduced from 50 for faster convergence.
	- `--use_wandb` (flag): Enable Weights & Biases logging.
	- `--wandb_project` (str, optional): W&B project name.
	- `--wandb_run_name` (str, optional): W&B run name.
	- Behavior: Each script iterates over `DATA_TYPES_TO_TRAIN` and for each data type will:
		- Look for tuned params in `TUNING_RESULTS_DIR`.
		- Load `data_{datatype}_.parquet` from `SOURCE_DIR`.
		- Run fold-wise feature selection and train QML models. Save:
			- OOF predictions: `train_oof_preds_{datatype}.csv`.
			- Test predictions: `test_preds_{datatype}.csv`.
			- Model artifacts: `selected_features_{datatype}.joblib`, `scaler_{datatype}.joblib`, `qml_model_{datatype}.joblib`.

5) `metalearner.py`
	- `--preds_dir` (one or more, required): One or more directories to search for `train_oof_preds_*` and `test_preds_*` files (use your curated `final_ensemble_predictions` directory).
	- `--indicator_file` (str, required): Path to a parquet file containing indicator features and the true `class` column for combining with meta-features.
	- `--mode` (str, default `train`): Operation mode, `train` or `tune`.
	- `--n_trials` (int, default 50): Number of Optuna trials for tuning.
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- `--skip_cross_validation` (flag): Skip cross-validation during tuning (use simple train/val split).
	- `--max_training_time` (float, optional): Maximum training time in hours (overrides fixed steps). Example: `--max_training_time 11`.
	- `--checkpoint_frequency` (int, default 50): Save checkpoint every N steps.
	- `--keep_last_n` (int, default 3): Keep last N checkpoints.
	- `--checkpoint_fallback_dir` (str, optional): Fallback directory for checkpoints if primary is read-only.
	- `--resume` (str, optional): Resume from checkpoint. Choices: `best` (best validation), `latest` (most recent), `auto` (try best, fallback to latest). Example: `--resume auto`.
	- `--validation_frequency` (int, default 25): Compute validation metrics every N steps.
	- `--use_wandb` (flag): Enable Weights & Biases logging.
	- `--wandb_project` (str, optional): W&B project name.
	- `--wandb_run_name` (str, optional): W&B run name.
	- `--meta_model_type` (str, optional): Force meta-learner model type for final training (choices: `gated_standard`, `gated_reuploading`). Overrides tuned value.
	- `--meta_n_layers` (int, optional): Force number of layers for meta-learner during final training (overrides tuned value).
	- `--meta_n_qubits` (int, optional): Force number of qubits to use for the meta-learner (defaults to n_meta_features).
	- `--minlayers` (int, default 3): Minimum number of layers to consider during tuning.
	- `--maxlayers` (int, default 6): Maximum number of layers to consider during tuning.
	- `--learning_rate` (float, default 0.5): Fixed learning rate to use for both tuning and final training. If passed on the CLI it will override tuned params for final training.

6) `inference.py`
	- `--model_dir` (str, required): Path to curated deployment directory that contains at minimum: `meta_learner_final.joblib`, `meta_learner_columns.json`, and `label_encoder.joblib` plus the selected base learner artifacts (pipelines or selector/scaler/qml_model files).
	- `--patient_data_dir` (str, required): Path to a directory containing per-data-type parquet files named `data_<datatype>_.parquet` for the new patient. Missing files are tolerated (treated as missing data).
	- Behavior: The script will detect whether a base-learner is saved as a `pipeline_{datatype}.joblib` (Approach 1) or as `selected_features_{datatype}.joblib` plus `scaler_...` and `qml_model_...` (Approach 2) and will combine base-learner predictions into meta-features and call the meta-learner to predict the final class.

Environment variables relevant to CLI behavior
- `SOURCE_DIR` — directory where all `data_<datatype>_.parquet` files are read from (default `final_processed_datasets`).
- `TUNING_RESULTS_DIR` — directory where tuning outputs are read/written (default `tuning_results`).
- `ENCODER_DIR` — directory for the master `label_encoder.joblib` (default `master_label_encoder`).
- `OUTPUT_DIR` — per-script output directory; most scripts provide a sensible default but will respect the env var when set.

### Command-line arguments for `tune_models.py`

| Argument | Type | Required | Default | Choices | Description |
|---|---|---|---|---|---|
| `--datatype` | str | Yes | - | - | Data type (e.g., `CNV`, `Meth`). |
| `--approach` | int | Yes | - | `1`, `2` | `1` for Classical+QML, `2` for Conditional QML. |
| `--dim_reducer` | str | No | `umap` | `pca`, `umap` | Dimensionality reducer for Approach 1. |
| `--qml_model` | str | No | `standard` | `standard`, `reuploading` | QML circuit type. |
| `--n_trials` | int | No | `9` | - | Number of NEW Optuna trials to run (mutually exclusive with `--total_trials`). |
| `--total_trials` | int | No | `None` | - | Target TOTAL number of trials (computes remaining if study exists, mutually exclusive with `--n_trials`). |
| `--study_name` | str | No | `None` | - | Override the auto-generated study name. |
| `--min_qbits` | int | No | `None` | - | Minimum number of qubits for tuning. Defaults to `n_classes`. |
| `--max_qbits` | int | No | `12` | - | Maximum number of qubits for tuning. |
| `--min_layers` | int | No | `2` | - | Minimum number of layers for tuning. |
| `--max_layers` | int | No | `5` | - | Maximum number of layers for tuning. |
| `--steps` | int | No | `100` | - | Number of training steps for tuning. |
| `--scalers` | str | No | `smr` | - | String indicating which scalers to try (s: Standard, m: MinMax, r: Robust). |
| `--verbose` | flag | No | `False` | - | Enable verbose logging for QML model training steps. |
| `--validation_frequency` | int | No | `25` | - | Compute validation metrics every N steps. |
| `--use_wandb` | flag | No | `False` | - | Enable Weights & Biases logging during tuning. |
| `--wandb_project` | str | No | `None` | - | W&B project name. |
| `--wandb_run_name` | str | No | `None` | - | W&B run name (auto-generated if not provided). |

### Example commands for `tune_models.py`

```bash
# Tune Approach 1 (standard) for CNV with PCA (50 trials) with verbose logging
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer pca --n_trials 50 --verbose

# Tune Approach 2 (reuploading) for Prot (30 trials) with custom qubit and layer ranges
python tune_models.py --datatype Prot --approach 2 --qml_model reuploading --n_trials 30 --min_qbits 8 --max_qbits 16 --min_layers 4 --max_layers 6

# Tune with Weights & Biases logging and custom study name
python tune_models.py --datatype CNV --approach 1 --qml_model standard --n_trials 50 --use_wandb --wandb_project my_qml_project --study_name custom_cnv_study

# Resume existing study to reach 100 total trials
python tune_models.py --datatype CNV --approach 1 --qml_model standard --total_trials 100
```

### Command-line arguments for `metalearner.py`

| Argument | Type | Required | Default | Choices | Description |
|---|---|---|---|---|---|
| `--preds_dir` | str | Yes | - | - | One or more directories with `train_oof_preds_*` and `test_preds_*` files. |
| `--indicator_file` | str | Yes | - | - | Parquet file with indicator features and true `class` column. |
| `--mode` | str | No | `train` | `train`, `tune` | Operation mode. |
| `--n_trials` | int | No | `50` | - | Number of Optuna trials for tuning. |
| `--verbose` | flag | No | `False` | - | Enable verbose logging for QML model training steps. |
| `--skip_cross_validation` | flag | No | `False` | - | Skip cross-validation during tuning (use simple train/val split). |
| `--max_training_time` | float | No | `None` | - | Maximum training time in hours (overrides fixed steps). |
| `--checkpoint_frequency` | int | No | `50` | - | Save checkpoint every N steps. |
| `--keep_last_n` | int | No | `3` | - | Keep last N checkpoints. |
| `--checkpoint_fallback_dir` | str | No | `None` | - | Fallback directory for checkpoints if primary is read-only. |
| `--resume` | str | No | `None` | `best`, `latest`, `auto` | Resume from checkpoint: `best` (best validation), `latest` (most recent), `auto` (try best, fallback to latest). |
| `--validation_frequency` | int | No | `25` | - | Compute validation metrics every N steps. |
| `--validation_frac` | float | No | `0.2` | - | Fraction of training data for internal validation during QML training. Increased from 0.1 for better overfitting detection. |
| `--patience` | int | No | `25` | - | Early stopping patience in steps. Reduced from 50 for faster convergence. |
| `--use_wandb` | flag | No | `False` | - | Enable Weights & Biases logging. |
| `--wandb_project` | str | No | `None` | - | W&B project name. |
| `--wandb_run_name` | str | No | `None` | - | W&B run name. |
| `--meta_model_type` | str | No | `None` | `gated_standard`, `gated_reuploading` | Force meta-learner model type (overrides tuned value). |
| `--meta_n_layers` | int | No | `None` | - | Force number of layers for meta-learner (overrides tuned value). |
| `--meta_n_qubits` | int | No | `None` | - | Force number of qubits for meta-learner (defaults to n_meta_features). |
| `--minlayers` | int | No | `3` | - | Minimum number of layers to consider during tuning. |
| `--maxlayers` | int | No | `6` | - | Maximum number of layers to consider during tuning. |
| `--learning_rate` | float | No | `0.5` | - | Fixed learning rate (overrides tuned params for final training if passed on CLI). |

### Example commands for `metalearner.py`

```bash
# Tune the meta-learner with 100 trials and verbose logging
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode tune \
    --n_trials 100 \
    --minlayers 4 \
    --maxlayers 8 \
    --verbose

# Train final meta-learner with W&B logging
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --verbose \
    --use_wandb \
    --wandb_project meta_learner_training

# Train with time-based stopping instead of fixed steps
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --max_training_time 2.5 \
    --verbose
```