# Quantum Transfer Learning Ensemble for Multiclass Cancer Classification

This repository implements a stacked ensemble that uses Quantum Machine Learning (QML) classifiers as base learners and a QML meta-learner to combine their predictions for multiclass cancer classification from multi-omics data.

## Table of Contents
- [New Features](#-new-features)
- [Directory Layout](#directory-layout-recommended)
- [Environment Variables](#important-configure-source_dir-and-other-dirs-via-environment-variables)
- [Full Workflow](#full-workflow-commands)
  - [1. Create Master Label Encoder](#1-create-the-master-label-encoder-one-time)
  - [2. Hyperparameter Tuning](#2-hyperparameter-tuning-optuna)
  - [3. Train Base Learners](#3-train-base-learners-final-training-using-tuned-params)
  - [4. Curate Predictions](#4-curate-the-best-of-predictions-for-the-meta-learner)
  - [5. Meta-Learner Training](#5-tune-and-train-the-meta-learner)
  - [6. Inference](#6-prepare-deployment-directory-and-run-inference-on-a-new-patient)
- [Approach Mapping](#approach-mapping--which-script-implements-each-approach)
- [Notes and Recommendations](#notes-and-recommendations)

## ✨ New Features

This repository now includes several advanced features for robust quantum machine learning:

### Classical Readout Head
All quantum classifiers now include a trainable classical neural network layer that processes quantum measurement outputs. This hybrid quantum-classical architecture:
- Uses a hidden layer with configurable size (default: 16 neurons) and activation (default: tanh)
- Jointly trains quantum circuit parameters with classical weights
- Improves model expressivity and performance
- Configurable parameters:
  - `hidden_size`: Number of neurons in hidden layer (default: 16)
  - `readout_activation`: Activation function - 'tanh' (default), 'relu', or 'linear'
  - `selection_metric`: Metric for best model selection (default: 'f1_weighted')

### Serializable Adam Optimizer
A custom Adam optimizer with state persistence:
- Full save/restore of optimizer state (momentum, velocity, timestep)
- Enables true checkpoint/resume functionality
- Compatible with PennyLane's autograd system

### Comprehensive Checkpointing & Resume
Robust training state management:
- **Resume modes**: `auto`, `latest`, or `best`
- Saves quantum weights, classical weights, optimizer state, RNG state
- Automatic learning rate reduction when resuming without optimizer state
- Periodic checkpoints with configurable retention policy

### Metrics Logging & Visualization
Full training observability:
- Per-epoch metrics: accuracy, precision, recall, F1 (macro & weighted), specificity
- Confusion matrices per epoch
- CSV export of all metrics (`history.csv`)
- Automatic PNG plots: loss curves, F1 scores, precision/recall
- Configurable selection metric (default: weighted F1)

### Optuna Integration with SQLite
Enhanced hyperparameter tuning:
- Persistent study storage in SQLite database (`optuna_studies.db`)
- Support for distributed/parallel tuning
- Default training steps increased to 100
- TPE sampler with configurable seed for reproducibility

### Nested Cross-Validation Strategy
Robust hyperparameter tuning and model evaluation:
- **Inner CV (Tuning):** 3-fold stratified cross-validation within Optuna trials to select hyperparameters
- **Outer CV (Training):** 3-fold stratified cross-validation to generate out-of-fold predictions without data leakage
- Prevents hyperparameter overfitting and provides unbiased performance estimates
- OOF predictions from outer CV are used to train the meta-learner
- Ensures proper stacked ensemble with no information leakage

### Stratified 80/20 Split
All training scripts now use:
- 80/20 train/test split (previously 70/30)
- Stratified sampling to preserve class distributions
- Optional validation split from training data

### Time-Based Training
Flexible training duration control:
- Train for a specified time duration instead of fixed steps (e.g., `--max_training_time 11` for 11 hours)
- Automatic periodic checkpointing during training
- Best model selected based on validation metrics
- Enables fair comparison across models with same computational budget
- Particularly useful for long-running experiments and resource-constrained environments

### Checkpoint Fallback Directory

The system now includes intelligent checkpoint directory management:
- **Auto-detection of read-only directories:** If the primary checkpoint directory is not writable (e.g., mounted as read-only), the system automatically detects this condition.
- **Fallback directory support:** When a read-only directory is detected, checkpoints are saved to a configurable fallback directory via `--checkpoint_fallback_dir`.
- **Automatic checkpoint migration:** Existing checkpoints from the read-only directory are automatically copied to the fallback location.
- **Clear warning messages:** The system provides clear warnings when checkpoint directories are not writable and explains fallback behavior.

This feature is particularly useful when working with shared or mounted storage that may have read-only restrictions.

### Configurable Validation Frequency

Training scripts now support `--validation_frequency` (default: 10) to control how often validation metrics are computed during training. This allows you to:
- Reduce validation overhead for large datasets by validating less frequently
- Increase validation frequency for fine-grained monitoring of training progress
- Balance between training speed and observability

### Weights & Biases Integration

All training scripts now support optional Weights & Biases (W&B) integration for experiment tracking:
- **Deferred import:** W&B is only imported if `--use_wandb` is enabled, avoiding unnecessary dependencies
- **Automatic metric logging:** When enabled, validation metrics are automatically logged to W&B during training
- **Organized experiments:** Use `--wandb_project` and `--wandb_run_name` to organize experiments in your W&B workspace
- **No code changes required:** Simply add the CLI flags to enable W&B logging

Example usage:
```bash
# Enable W&B logging with custom project and run name
python dre_standard.py --use_wandb --wandb_project my_qml_project --wandb_run_name experiment_001

# Use default run names (auto-generated based on script and data type)
python cfe_relupload.py --use_wandb --wandb_project my_qml_project

# Combine with other features
python tune_models.py --datatype CNV --approach 1 --qml_model standard \
    --use_wandb --wandb_project qml_tuning --validation_frequency 5
```

Note: To use W&B, install with `pip install wandb` and authenticate with `wandb login`.

### Dependencies
All required packages are now specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Testing
Basic smoke tests validate the new functionality:
```bash
python tests/test_smoke.py
```

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
├── final_model_and_predictions/  # Output: meta-learner models (default OUTPUT_DIR for metalearner.py)
│   ├── metalearner_model.joblib
│   ├── metalearner_scaler.joblib
│   └── best_metalearner_params.json (if tuning was run)
├── final_model_deployment/       # User-created directory containing final models for inference
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

### 2) Hyperparameter tuning (Optuna)

Use `tune_models.py` to tune base learners. Repeat per data type / approach / qml_model / dim reducer as needed.

Examples:

```bash
# Tune Approach 1 (standard) for CNV with PCA (50 trials) with verbose logging (tuning uses default 100 steps)
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer pca --n_trials 50 --verbose

# Tune Approach 2 (reuploading) for Prot (30 trials)
python tune_models.py --datatype Prot --approach 2 --qml_model reuploading --n_trials 30
```

Notes:
- The script reads data from `SOURCE_DIR` and runs an Optuna study with `--n_trials` trials.
- Studies are persisted to SQLite database (default: `./optuna_studies.db`).
- The number of training steps for tuning defaults to 100 (configurable via `--steps`).

Output: one or more JSON files saved to `tuning_results/` (default). These contain best parameters.

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

Run the appropriate script for each approach variant. The training scripts read tuned parameters from the `tuning_results` folder and will save out-of-fold (OOF) and test predictions plus models into their respective output directories.

Examples (the repository contains multiple `approach` scripts; run the ones you need):

```bash
# Approach 1 - Dimensionality Reduction Encoding (standard) with verbose logging and override steps
python dre_standard.py --verbose --override_steps 100

# Approach 1 - Dimensionality Reduction Encoding (data reuploading)
python dre_relupload.py

# Approach 2 - Conditional Feature Encoding (standard) with verbose logging
python cfe_standard.py --verbose

# Approach 2 - Conditional Feature Encoding (data reuploading) with override steps
python cfe_relupload.py --override_steps 100
```

All base learner training scripts (`dre_standard.py`, `dre_relupload.py`, `cfe_standard.py`, `cfe_relupload.py`) support the following command-line arguments:

**Hyperparameter overrides:**
- `--n_qbits` (int): override number of qubits / selected features used by the model/pipeline.
- `--n_layers` (int): override number of ansatz layers for QML circuits.
- `--steps` (int): override number of training steps.
- `--override_steps` (int): override number of training steps (same as --steps, kept for backward compatibility).
- `--scaler` (str): override scaler selection using shorthand: `s` (Standard), `m` (MinMax), `r` (Robust); full names are also accepted.

**Data and workflow control:**
- `--datatypes` (str...): space-separated list of data types to train (overrides the default `DATA_TYPES_TO_TRAIN`). Example: `--datatypes CNV Prot`
- `--skip_tuning` (flag): skip loading tuned parameters entirely and use command-line arguments or defaults instead.
- `--skip_cross_validation` (flag): skip cross-validation and only train final model on full training set (skips OOF prediction generation).
- `--cv_only` (flag): perform only cross-validation to generate OOF predictions and skip final training (useful for meta-learner training). Mutually exclusive with `--skip_cross_validation`.

**Training duration and checkpointing:**
- `--max_training_time` (float): maximum training time in hours. When specified, training continues until this time limit is reached instead of using a fixed number of steps. This enables "smart steps" training where the model trains for a specified duration rather than a fixed number of epochs.
- `--checkpoint_frequency` (int, default 50): save a checkpoint every N training steps for recovery and analysis.
- `--keep_last_n` (int, default 3): keep only the last N checkpoints to save disk space (older checkpoints are automatically deleted).
- `--checkpoint_fallback_dir` (str, optional): fallback directory for checkpoints if primary is read-only. If the primary checkpoint directory is not writable, the system will attempt to use this fallback directory and copy any existing checkpoints.
- `--validation_frequency` (int, default 10): compute validation metrics every N training steps. This parameter controls how often validation is performed during training.

**Weights & Biases integration:**
- `--use_wandb` (flag): enable Weights & Biases logging for experiment tracking. When enabled, training metrics and validation results are automatically logged to W&B.
- `--wandb_project` (str, optional): W&B project name for organizing experiments.
- `--wandb_run_name` (str, optional): W&B run name for identifying specific training runs. If not provided, a default name is generated based on the script and data type.


**Logging:**
- `--verbose` (flag): enable verbose logging for QML model training steps.

**Note on checkpointing and best model selection:** All training scripts now automatically track the best model based on training loss. When `--max_training_time` is specified, checkpoints are saved periodically to `checkpoints_{datatype}/` subdirectories. The best model weights are always saved and automatically loaded at the end of training, ensuring you get the best performing model regardless of whether training ends naturally or due to time limits.

Example (train only CNV and Prot):

```bash
python dre_standard.py --datatypes CNV Prot --verbose
```

Example (override several params):

```bash
python dre_standard.py --n_qbits 8 --n_layers 4 --steps 150 --scaler m --verbose
```

Example (skip tuning and use only CLI arguments):

```bash
python dre_standard.py --skip_tuning --n_qbits 10 --n_layers 3 --steps 100 --scaler s --verbose
```

Example (time-based training for 11 hours with checkpointing):

```bash
# Train for up to 11 hours instead of a fixed number of steps
python dre_standard.py --max_training_time 11 --checkpoint_frequency 50 --keep_last_n 3 --verbose

# Train only specific data types with time-based training
python cfe_relupload.py --datatypes CNV Prot --max_training_time 11 --verbose
```

Example (comprehensive - combining multiple options):

```bash
# Complete example: Train specific data types with custom hyperparameters,
# skip tuning (use CLI args), run cross-validation only (for meta-learner),
# and enable verbose logging
python dre_standard.py \
    --datatypes CNV Prot GeneExpr \
    --skip_tuning \
    --n_qbits 10 \
    --n_layers 4 \
    --steps 200 \
    --scaler s \
    --cv_only \
    --verbose

# Complete example: Train all data types with time-based training,
# custom checkpointing, and parameter overrides (will use tuned params if available)
python dre_standard.py \
    --n_qbits 12 \
    --n_layers 5 \
    --scaler m \
    --max_training_time 8 \
    --checkpoint_frequency 25 \
    --keep_last_n 5 \
    --verbose

# Complete example: Skip cross-validation and train only final model
# with custom parameters (useful when you already have OOF predictions)
python cfe_relupload.py \
    --datatypes CNV \
    --skip_cross_validation \
    --n_qbits 8 \
    --n_layers 3 \
    --steps 150 \
    --scaler r \
    --verbose
```

Outputs (per data type):
- `train_oof_preds_<datatype>.csv` (used to train meta-learner)
- `test_preds_<datatype>.csv`
- model artifacts: `pipeline_<datatype>.joblib` or `selector_<datatype>.joblib`, `scaler_<datatype>.joblib`, `qml_model_<datatype>.joblib`
- checkpoints (when using `--max_training_time`): `checkpoints_<datatype>/best_weights.joblib` and `checkpoints_<datatype>/checkpoint_step_*.joblib`

### 4) Curate the "best-of" predictions for the meta-learner

Create a directory (for example, `final_ensemble_predictions`) and copy the `train_oof_preds_*` and `test_preds_*` files you want the meta-learner to use. Also copy `label_encoder.joblib` into this folder (the meta-learner reads the encoder to reconstruct class labels).

Example:

```bash
mkdir -p final_ensemble_predictions

# Copy predictions for CNV from approach 1 standard
cp base_learner_outputs_app1_standard/train_oof_preds_CNV.csv final_ensemble_predictions/
cp base_learner_outputs_app1_standard/test_preds_CNV.csv final_ensemble_predictions/

# Copy predictions for Prot from approach 2 reuploading
cp base_learner_outputs_app2_reuploading/train_oof_preds_Prot.csv final_ensemble_predictions/
cp base_learner_outputs_app2_reuploading/test_preds_Prot.csv final_ensemble_predictions/

# Copy the master label encoder (required by meta-learner)
cp master_label_encoder/label_encoder.joblib final_ensemble_predictions/
```

The meta-learner training script (`metalearner.py`) accepts one or more prediction directories via the `--preds_dir` argument — pass the `final_ensemble_predictions` folder (or multiple folders) to it.

### 5) Tune and train the meta-learner

Tune (optional):

```bash
# Tune the meta-learner hyperparameters with verbose logging
python metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode tune --n_trials 50 --verbose

# Tune with W&B logging for experiment tracking
python metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode tune --n_trials 50 --use_wandb --wandb_project qml_tuning --verbose
```

Train final meta-learner (uses best parameters from tuning if available):

```bash
python metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode train --verbose

# Or with time-based training for extended optimization
python metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode train --max_training_time 11 --verbose

# With W&B logging and checkpoint fallback for resilience
python metalearner.py --preds_dir final_ensemble_predictions --indicator_file indicator_features.parquet --mode train --max_training_time 11 --use_wandb --wandb_project qml_metalearner --checkpoint_fallback_dir /tmp/metalearner_checkpoints --verbose
```

Notes:
- If tuning was run, the script loads parameters from `final_model_and_predictions/best_metalearner_params.json`
- If no tuned parameters exist, the script uses sensible defaults
- When `--max_training_time` is used, checkpoints are saved to `final_model_and_predictions/checkpoints_metalearner/`
- Use `--checkpoint_fallback_dir` if working with read-only storage (e.g., mounted volumes)
- Enable `--use_wandb` to track experiments and visualize training metrics in Weights & Biases

Outputs (saved to `final_model_and_predictions/` by default):
- `metalearner_model.joblib`
- `metalearner_scaler.joblib`
- `best_metalearner_params.json` (if tuning was run)

Note: The default OUTPUT_DIR for metalearner.py is `final_model_and_predictions`. For inference, you'll need to manually copy/rename files to match what `inference.py` expects (see Step 6).

### 6) Prepare deployment directory and run inference on a new patient

Create a deployment directory and copy the meta-learner artifacts and the base models you want to use for inference.

**Important Note on File Naming:** The `inference.py` script expects specific file names that differ from what `metalearner.py` produces. You need to copy/rename files as shown below:

Example:

```bash
mkdir -p final_model_deployment

# Meta learner + metadata (note the file renaming)
cp final_model_and_predictions/metalearner_model.joblib final_model_deployment/meta_learner_final.joblib
cp master_label_encoder/label_encoder.joblib final_model_deployment/

# You also need to create meta_learner_columns.json manually with the column order
# This file should contain a JSON array of the column names in the exact order used during training
# Example content: ["pred_CNV_BRCA", "pred_CNV_LUAD", ..., "is_missing_CNV", ...]

# Copy selected base learner artifacts (examples):
cp base_learner_outputs_app1_standard/pipeline_CNV.joblib final_model_deployment/
cp base_learner_outputs_app2_reuploading/selector_Prot.joblib final_model_deployment/
cp base_learner_outputs_app2_reuploading/scaler_Prot.joblib final_model_deployment/
cp base_learner_outputs_app2_reuploading/qml_model_Prot.joblib final_model_deployment/
```

Prepare a `new_patient_data` directory that contains per-data-type parquet files following the naming convention used elsewhere (e.g., `data_CNV_.parquet`, `data_Prot_.parquet`, ...). Missing files are tolerated and handled by the inference script.

Run inference:

```bash
python inference.py --model_dir final_model_deployment --patient_data_dir new_patient_data
```

The script prints the final predicted class label.

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

This repository provides two families of base-learner designs. All models use a hybrid quantum-classical architecture with trainable classical readout layers. The mapping below shows which scripts implement Approach 1 and Approach 2:

- Approach 1 — Dimensionality Reduction Encoding (DRE)
	- `dre_standard.py` — DRE with classical dimensionality reduction (PCA or UMAP) followed by a standard QML classifier with classical readout layer.
	- `dre_relupload.py` — DRE using data re-uploading QML circuits with classical readout layer for datasets where re-uploading is beneficial.

- Approach 2 — Conditional Feature Encoding (CFE)
	- `cfe_standard.py` — CFE where the QML model is conditioned on a selected subset of features (standard QML circuit) with classical readout layer.
	- `cfe_relupload.py` — CFE using data re-uploading QML circuits with classical readout layer and fold-wise feature selection (LightGBM importance-based selection).

Below are the CLI arguments for each script (if not listed, script uses defaults):

1) `create_master_label_encoder.py`
	- No CLI arguments.
	- Behavior: Scans parquet files in `SOURCE_DIR` (env var or `final_processed_datasets`) and writes `label_encoder.joblib` to `OUTPUT_DIR` (default `master_label_encoder` or `ENCODER_DIR` env var if set).

2) `tune_models.py`
	- `--datatype` (str, required): Data type to tune (e.g., `CNV`, `Meth`, `Prot`).
	- `--approach` (int, required): `1` or `2` selecting Approach 1 (Dimensionality Reduction Encoding) or Approach 2 (Conditional Feature Encoding).
	- `--dim_reducer` (str, default `pca`): `pca` or `umap` (used by Approach 1).
	- `--qml_model` (str, default `standard`): `standard` or `reuploading`.
	- `--scalers` (str, default `smr`): String indicating which scalers to try (s: Standard, m: MinMax, r: Robust). E.g., 'sm' for Standard and MinMax.
	- `--n_trials` (int, default 9): Number of NEW Optuna trials to run. If study exists, these are added to existing trials.
	- `--total_trials` (int, optional): Target TOTAL number of trials. If study exists, computes remaining trials needed to reach this total.
	- `--study_name` (str, optional): Override the auto-generated study name for custom experiment organization.
	- `--min_qbits` (int, optional): Minimum number of qubits for tuning. Defaults to `n_classes`.
	- `--max_qbits` (int, default 12): Maximum number of qubits for tuning.
	- `--min_layers` (int, default 2): Minimum number of layers for tuning.
	- `--max_layers` (int, default 5): Maximum number of layers for tuning.
	- `--steps` (int, default 100): Number of training steps for tuning.
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- Behavior: Loads data from `os.path.join(SOURCE_DIR, f'data_{datatype}_.parquet')`, runs an Optuna study, and writes best param JSON files to `TUNING_RESULTS_DIR`. Automatically handles read-only databases by copying to a writable location. Gracefully handles interruptions (Ctrl+C) by completing the current trial before stopping.
	- Note: For Approach 2 (Conditional Feature Encoding) feature selection is performed using a LightGBM classifier to compute feature importances; the top-k important features (k = number of qubits) are selected per fold and for the final model. `SelectKBest` is no longer used for Approach 2.


3) `dre_standard.py` and `dre_relupload.py`
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- `--override_steps` (int, optional): Override the number of training steps from the tuned parameters.
	- `--n_qbits` (int, optional): Override number of qubits (or selected features) used by the model/pipeline.
	- `--datatypes` (str..., optional): Space-separated list of datatypes to train (overrides default `DATA_TYPES_TO_TRAIN`). Example: `--datatypes CNV Prot`.
	- `--n_layers` (int, optional): Override number of ansatz layers for the QML model.
	- `--steps` (int, optional): Override the number of training steps used for QML training.
	- `--scaler` (str, optional): Override scaler with shorthand: `s` (Standard), `m` (MinMax), `r` (Robust) or full name.
	- `--skip_tuning` (flag, optional): Skip loading tuned parameters and use command-line arguments or defaults instead.
	- `--skip_cross_validation` (flag, optional): Skip cross-validation and only train final model on full training set (skips OOF prediction generation).
	- `--cv_only` (flag, optional): Perform only cross-validation to generate OOF predictions and skip final training (useful for meta-learner training). Mutually exclusive with `--skip_cross_validation`.
	- `--max_training_time` (float, optional): Maximum training time in hours. If specified, training continues until this time limit is reached instead of using fixed steps. Example: `--max_training_time 11` for 11 hours.
	- `--checkpoint_frequency` (int, default 50): Save a checkpoint every N training steps.
	- `--keep_last_n` (int, default 3): Keep only the last N checkpoints to save disk space.
	- Behavior: Each script iterates over `DATA_TYPES_TO_TRAIN` and for each data type will:
		- Look for tuned params in `TUNING_RESULTS_DIR` (unless `--skip_tuning` is used).
		- Load `data_{datatype}_.parquet` from `SOURCE_DIR`.
		- Train the pipeline (PCA/UMAP + QML) with automatic best model selection and optional checkpointing.
		- Save:
			- OOF predictions: `train_oof_preds_{datatype}.csv` in script-specific `OUTPUT_DIR` (unless `--skip_cross_validation` is used).
			- Test predictions: `test_preds_{datatype}.csv` (unless `--cv_only` is used).
			- Model artifacts: `pipeline_{datatype}.joblib` (unless `--cv_only` is used).
			- Checkpoints (if `--max_training_time` is used): `checkpoints_{datatype}/best_weights.joblib` and `checkpoints_{datatype}/checkpoint_step_*.joblib`.


4) `cfe_standard.py` and `cfe_relupload.py`
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- `--override_steps` (int, optional): Override the number of training steps from the tuned parameters.
	- `--n_qbits` (int, optional): Override number of qubits (or selected features) used by the model/pipeline.
	- `--datatypes` (str..., optional): Space-separated list of datatypes to train (overrides default `DATA_TYPES_TO_TRAIN`). Example: `--datatypes CNV Prot`.
	- `--n_layers` (int, optional): Override the number of ansatz layers for the QML model.
	- `--steps` (int, optional): Override the number of training steps used for QML training.
	- `--scaler` (str, optional): Override scaler with shorthand: `s` (Standard), `m` (MinMax), `r` (Robust) or full name.
	- `--skip_tuning` (flag, optional): Skip loading tuned parameters and use command-line arguments or defaults instead.
	- `--skip_cross_validation` (flag, optional): Skip cross-validation and only train final model on full training set (skips OOF prediction generation).
	- `--cv_only` (flag, optional): Perform only cross-validation to generate OOF predictions and skip final training (useful for meta-learner training). Mutually exclusive with `--skip_cross_validation`.
	- `--max_training_time` (float, optional): Maximum training time in hours. If specified, training continues until this time limit is reached instead of using fixed steps. Example: `--max_training_time 11` for 11 hours.
	- `--checkpoint_frequency` (int, default 50): Save a checkpoint every N training steps.
	- `--keep_last_n` (int, default 3): Keep only the last N checkpoints to save disk space.
	- Behavior: Each script iterates over `DATA_TYPES_TO_TRAIN` and for each data type will:
		- Look for tuned params in `TUNING_RESULTS_DIR` (unless `--skip_tuning` is used).
		- Load `data_{datatype}_.parquet` from `SOURCE_DIR`.
		- Run fold-wise feature selection and train QML models with automatic best model selection and optional checkpointing.
		- Save:
			- OOF predictions: `train_oof_preds_{datatype}.csv` (unless `--skip_cross_validation` is used).
			- Test predictions: `test_preds_{datatype}.csv` (unless `--cv_only` is used).
			- Model artifacts: `selector_{datatype}.joblib`, `scaler_{datatype}.joblib`, `qml_model_{datatype}.joblib` (unless `--cv_only` is used).
			- Checkpoints (if `--max_training_time` is used): `checkpoints_{datatype}/best_weights.joblib` and `checkpoints_{datatype}/checkpoint_step_*.joblib`.

5) `metalearner.py`
	- `--preds_dir` (one or more, required): One or more directories to search for `train_oof_preds_*` and `test_preds_*` files (use your curated `final_ensemble_predictions` directory).
	- `--indicator_file` (str, required): Path to a parquet file containing indicator features and the true `class` column for combining with meta-features.
	- `--mode` (str, default `train`): Operation mode, `train` or `tune`.
	- `--n_trials` (int, default 50): Number of Optuna trials for tuning.
	- `--override_steps` (int, optional): Override the number of training steps from the tuned parameters.
	- `--scalers` (str, default 'smr'): String indicating which scalers to try during tuning (s: Standard, m: MinMax, r: Robust). E.g., 'sm' for Standard and MinMax.
	- `--verbose` (flag): Enable verbose logging for QML model training steps.
	- `--max_training_time` (float, optional): Maximum training time in hours. If specified, training continues until this time limit is reached instead of using fixed steps. Example: `--max_training_time 11` for 11 hours.
	- `--checkpoint_frequency` (int, default 50): Save a checkpoint every N training steps.
	- `--keep_last_n` (int, default 3): Keep only the last N checkpoints to save disk space.
	- `--checkpoint_fallback_dir` (str, optional): Fallback directory for checkpoints if primary is read-only. If the primary checkpoint directory is not writable, the system will attempt to use this fallback directory and copy any existing checkpoints.
	- `--validation_frequency` (int, default 10): Compute validation metrics every N training steps. This parameter controls how often validation is performed during training.
	- `--use_wandb` (flag): Enable Weights & Biases logging for experiment tracking. When enabled, training metrics and validation results are automatically logged to W&B.
	- `--wandb_project` (str, optional): W&B project name for organizing experiments.
	- `--wandb_run_name` (str, optional): W&B run name for identifying specific training runs. If not provided, a default name is generated based on the training mode.
	- Behavior: In `tune` mode, runs Optuna to find best hyperparameters and saves to `final_model_and_predictions/best_metalearner_params.json`. In `train` mode, loads tuned params (if available) or uses defaults, trains final meta-learner on combined meta-features and indicator features with automatic best model selection and optional checkpointing, and saves the model and metadata to `final_model_and_predictions/`.

6) `inference.py`
	- `--model_dir` (str, required): Path to curated deployment directory that contains at minimum: `meta_learner_final.joblib`, `meta_learner_columns.json`, and `label_encoder.joblib` plus the selected base learner artifacts (pipelines or selector/scaler/qml_model files).
	- `--patient_data_dir` (str, required): Path to a directory containing per-data-type parquet files named `data_<datatype>_.parquet` for the new patient. Missing files are tolerated (treated as missing data).
	- Behavior: The script will detect whether a base-learner is saved as a `pipeline_{datatype}.joblib` (Approach 1) or as `selector_{datatype}.joblib` plus `scaler_...` and `qml_model_...` (Approach 2) and will combine base-learner predictions into meta-features and call the meta-learner to predict the final class.
	- Note: Due to a naming mismatch between what metalearner.py saves and what inference.py expects, you need to rename `metalearner_model.joblib` to `meta_learner_final.joblib` and manually create `meta_learner_columns.json` (see Step 6 above).

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
| `--dim_reducer` | str | No | `pca` | `pca`, `umap` | Dimensionality reducer for Approach 1. |
| `--qml_model` | str | No | `standard` | `standard`, `reuploading` | QML circuit type. |
| `--n_trials` | int | No | `9` | - | Number of NEW Optuna trials to run (if study exists, these are added to existing trials). |
| `--total_trials` | int | No | `None` | - | Target TOTAL number of trials. If study exists, computes remaining trials needed to reach this total. Mutually exclusive with using `--n_trials` as absolute count. |
| `--study_name` | str | No | `None` | - | Override the auto-generated study name. Useful for organizing multiple tuning experiments. |
| `--min_qbits` | int | No | `None` | - | Minimum number of qubits for tuning. Defaults to `n_classes`. |
| `--max_qbits` | int | No | `12` | - | Maximum number of qubits for tuning. |
| `--min_layers` | int | No | `2` | - | Minimum number of layers for tuning. |
| `--max_layers` | int | No | `5` | - | Maximum number of layers for tuning. |
| `--steps` | int | No | `100` | - | Number of training steps for tuning. |
| `--scalers` | str | No | `smr` | - | String indicating which scalers to try (s: Standard, m: MinMax, r: Robust). E.g., 'sm' for Standard and MinMax. |
| `--verbose` | flag | No | `False` | - | Enable verbose logging for QML model training steps. |

### Example commands for `tune_models.py`

```bash
# Tune Approach 1 (standard) for CNV with PCA (50 trials) with verbose logging
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer pca --n_trials 50 --verbose

# Tune Approach 2 (reuploading) for Prot (30 trials) with custom qubit and layer ranges
python tune_models.py --datatype Prot --approach 2 --qml_model reuploading --n_trials 30 --min_qbits 8 --max_qbits 16 --min_layers 4 --max_layers 6 --steps 75

# Use custom study name and target a total of 100 trials (will run remaining trials to reach 100)
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer pca --total_trials 100 --study_name my_custom_study

# Resume tuning by adding 20 more trials to an existing study
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer pca --n_trials 20 --verbose

# Use with read-only database (automatically copies to writable location)
export OPTUNA_DB_PATH=/path/to/readonly/optuna_studies.db
python tune_models.py --datatype CNV --approach 1 --qml_model standard --dim_reducer pca --n_trials 50
```

**Notes on new features:**
- **Read-only Database Handling**: If the Optuna database is read-only (e.g., mounted from a read-only volume), `tune_models.py` will automatically detect this and copy it to a writable location (current directory or temp directory). The script will inform you where the working copy is stored.
- **Interruption Handling**: Press Ctrl+C during tuning to gracefully stop after the current trial completes. Press Ctrl+C again to force exit (may lose current trial). The script will save all completed trials even if interrupted.
- **Trial Counting**: Use `--total_trials` to specify a target number of trials. If the study already has some trials, it will calculate and run only the remaining trials needed. Use `--n_trials` to add a specific number of new trials to an existing study.
- **Custom Study Names**: Use `--study_name` to organize multiple tuning experiments or to resume a specific study by name.

### Command-line arguments for `metalearner.py`

| Argument | Type | Required | Default | Choices | Description |
|---|---|---|---|---|---|
| `--preds_dir` | str (multiple) | Yes | - | - | One or more directories with `train_oof_preds_*` and `test_preds_*` files. |
| `--indicator_file` | str | Yes | - | - | Parquet file with indicator features and true `class` column. |
| `--mode` | str | No | `train` | `train`, `tune` | Operation mode. |
| `--n_trials` | int | No | `50` | - | Number of Optuna trials for tuning. |
| `--override_steps` | int | No | `None` | - | Override the number of training steps from the tuned parameters. |
| `--scalers` | str | No | `smr` | - | String indicating which scalers to try during tuning (s: Standard, m: MinMax, r: Robust). E.g., 'sm' for Standard and MinMax. |
| `--verbose` | flag | No | `False` | - | Enable verbose logging for QML model training steps. |
| `--max_training_time` | float | No | `None` | - | Maximum training time in hours. Trains until time limit instead of fixed steps. |
| `--checkpoint_frequency` | int | No | `50` | - | Save a checkpoint every N training steps. |
| `--keep_last_n` | int | No | `3` | - | Keep only the last N checkpoints to save disk space. |
| `--checkpoint_fallback_dir` | str | No | `None` | - | Fallback directory for checkpoints if primary is read-only. |
| `--validation_frequency` | int | No | `10` | - | Compute validation metrics every N training steps. |
| `--use_wandb` | flag | No | `False` | - | Enable Weights & Biases logging for experiment tracking. |
| `--wandb_project` | str | No | `None` | - | W&B project name for organizing experiments. |
| `--wandb_run_name` | str | No | `None` | - | W&B run name for identifying specific training runs. |

### Example commands for `metalearner.py`

```bash
# Tune the meta-learner with 100 trials and verbose logging
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode tune \
    --n_trials 100 \
    --verbose

# Train the meta-learner using tuned parameters (or defaults if tuning wasn't run)
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --verbose

# Train the meta-learner with override steps
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --override_steps 150 \
    --verbose

# Train with time-based training and checkpointing
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --max_training_time 11 \
    --checkpoint_frequency 25 \
    --verbose

# Train with W&B logging and custom validation frequency
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --use_wandb \
    --wandb_project qml_metalearner \
    --wandb_run_name final_ensemble_v1 \
    --validation_frequency 5 \
    --verbose

# Complete example with all advanced features
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --max_training_time 8 \
    --checkpoint_frequency 20 \
    --keep_last_n 5 \
    --checkpoint_fallback_dir /tmp/metalearner_checkpoints \
    --validation_frequency 5 \
    --use_wandb \
    --wandb_project qml_experiments \
    --wandb_run_name metalearner_long_run \
    --verbose
```