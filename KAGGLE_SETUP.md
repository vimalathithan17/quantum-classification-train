# Kaggle Environment & Production Workflow Guide

This guide covers setting up the `quantum-classification-train` project in a Kaggle notebook environment and using our **5-Step Production Workflow**.

---

## The 5-Step Notebook Production Workflow

The pipeline is split into 5 modular, sequentially numbered Jupyter Notebooks located in the `kaggle_notebooks/` directory. By splitting the work, we avoid Kaggle 12-hour VM timeouts and isolate our dependencies.

### Step 1: Global Data Split (Notebook 01)
* **File:** `kaggle_notebooks/01_Global_Data_Split_Kaggle.ipynb`
* **Purpose:** Performs stratified splitting of the raw multi-omics data. This creates a rigorous training and holdout test set to prevent ANY data leakage across the entire pipeline.
* **Output:** Generates and zips `global_split_data.zip`.
* **Action Required:** Download `global_split_data.zip` to your local machine, go to Kaggle Datasets → New Dataset, and upload it (name it e.g. `qml-global-split-data`).

### Step 2: Contrastive Pretraining (Notebook 02)
* **File:** `kaggle_notebooks/02_Contrastive_Pretraining_Kaggle.ipynb`
* **Prerequisites:** Attach the primary dataset and your newly created `qml-global-split-data` dataset to this notebook.
* **Purpose:** Trains self-supervised contrastive encoders on the `global_train` dataset to learn rich feature representations before QML or Transformer modeling.
* **Output:** Generates and zips `pretrained_embeddings.zip` and the encoder models.
* **Action Required:** Download the outputs and upload them as another new Kaggle Dataset (e.g., `qml-pretrained-embeddings`).

### Steps 3 & 4: Base Learner Training (Parallelizable)
These two notebooks can be run in parallel on different Kaggle sessions.

#### Notebook 03: QML Tuning and Training
* **File:** `kaggle_notebooks/03_QML_Tuning_and_Training_Kaggle.ipynb`
* **Prerequisites:** Attach the primary dataset and your `qml-global-split-data` dataset.
* **Purpose:** Tunes hyperparameters (via Optuna) and trains our 4 Quantum Base models on the global training set using cross-validation.
* **Output:** Generates base learner OOF (Out-Of-Fold) predictions (`.csv`) and trained models (`.joblib`).
* **Action Required:** Download and upload as a Kaggle Dataset (e.g., `qml-base-learner-outputs`).

#### Notebook 04: Transformer Fusion Training
* **File:** `kaggle_notebooks/04_Transformer_Fusion_Kaggle.ipynb`
* **Prerequisites:** Attach the primary dataset, `qml-global-split-data`, and `qml-pretrained-embeddings` to access the extracted features.
* **Purpose:** Trains the classical multi-modal Transformer on the contrastively-learned embeddings.
* **Output:** Generates Transformer OOF predictions (`.csv`) and model state dictionaries (`.pt`).
* **Action Required:** Download and upload as a Kaggle Dataset (e.g., `qml-transformer-outputs`).

### Step 5: Meta-Learner & Final Inference (Notebook 05)
* **File:** `kaggle_notebooks/05_Meta_Learner_and_Inference_Kaggle.ipynb`
* **Prerequisites:** Attach all previous Kaggle Datasets: `qml-global-split-data`, `qml-base-learner-outputs`, and `qml-transformer-outputs`.
* **Purpose:** Stacks all base model OOF predictions and trains the Level-1 XGBoost Meta-Learner. Then, evaluates the full ensemble (Base + Meta) on the unseen `global_test` set.
* **Output:** Final evaluation metrics, W&B logs, and confusion matrices.

---

## Setup: First-Time Kaggle Configuration

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Set the following notebook settings:
   - **Accelerator:** None (CPU) or GPU T4 x2 (if available)
   - **Persistence:** Files (to save outputs between sessions)
   - **Internet:** On (required for cloning repo and installing packages)

**Each team member creates their own notebook** for parallel training.

---

## Step 2: Add the Dataset

1. Click **"+ Add Input"** in the right sidebar
2. Search for: `gbm_lgg_balanced_xgb_reduced`
3. Select the dataset by `vimalathithan22i272`
4. The dataset will be mounted at: `/kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/`

**Dataset Structure:**
```
/kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/
└── final_processed_datasets_xgb_balanced/
    ├── data_CNV_.parquet
    ├── data_GeneExpr_.parquet
    ├── data_Meth_.parquet
    ├── data_Prot_.parquet
    ├── data_SNV_.parquet      # (will be skipped initially)
    ├── data_miRNA_.parquet
    └── indicator_features.parquet
```

---

## Step 3: Clone the Repository

Run this in the first cell of your notebook:

```python
# Clone the repository
!git clone https://github.com/vimalathithan17/quantum-classification-train.git

# Change to the repo directory
%cd /kaggle/working/quantum-classification-train
```

---

## Step 4: Install Dependencies

```python
# Install all required packages from requirements.txt
!pip install -q -r requirements.txt
```

---

## Step 5: Set Global Environment Variables

```python
import os

# Set the data directory to the Kaggle input path
os.environ['SOURCE_DIR'] = '/kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced'

# Set output directories (must be in /kaggle/working for persistence)
os.environ['OUTPUT_DIR'] = '/kaggle/working/outputs'
os.environ['TUNING_RESULTS_DIR'] = '/kaggle/working/tuning_results'

# Random seed for reproducibility
os.environ['RANDOM_STATE'] = '42'

# Create output directories
!mkdir -p /kaggle/working/outputs
!mkdir -p /kaggle/working/tuning_results
```

---

## Step 6: Create Master Label Encoder (One-time Setup)

Before running any training, create the master label encoder:

```python
# Create label encoder directory
!mkdir -p /kaggle/working/master_label_encoder

# Set encoder directory
os.environ['ENCODER_DIR'] = '/kaggle/working/master_label_encoder'

# Run the encoder creation script
!python create_master_label_encoder.py
```

---

## Step 7: Verify Setup

```python
# Verify dataset is accessible
!ls -la /kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced/

# Verify repo is cloned
!ls -la /kaggle/working/quantum-classification-train/

# Test imports
!python -c "import pennylane; import torch; import optuna; print('All imports successful!')"
```

---

## Quick Reference: Kaggle Paths

| Purpose | Path |
|---------|------|
| **Input Dataset** | `/kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced` |
| **Working Directory** | `/kaggle/working/quantum-classification-train` |
| **Outputs** | `/kaggle/working/outputs` |
| **Pretrained Encoders** | `/kaggle/working/pretrained_encoders_*` |
| **Extracted Features** | `/kaggle/working/pretrained_features_*` |
| **Label Encoder** | `/kaggle/working/master_label_encoder` |

---

## Complete Setup Cell (Copy-Paste Ready)

```python
# === KAGGLE SETUP - RUN THIS FIRST ===

import os

# Clone repo (skip if already cloned)
if not os.path.exists('/kaggle/working/quantum-classification-train'):
    !git clone https://github.com/vimalathithan17/quantum-classification-train.git

# Change to repo directory
%cd /kaggle/working/quantum-classification-train

# Install dependencies from requirements.txt
!pip install -q -r requirements.txt

# Set environment variables
os.environ['SOURCE_DIR'] = '/kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced'
os.environ['OUTPUT_DIR'] = '/kaggle/working/outputs'
os.environ['TUNING_RESULTS_DIR'] = '/kaggle/working/tuning_results'
os.environ['ENCODER_DIR'] = '/kaggle/working/master_label_encoder'
os.environ['RANDOM_STATE'] = '42'

# Create directories
!mkdir -p /kaggle/working/outputs
!mkdir -p /kaggle/working/tuning_results
!mkdir -p /kaggle/working/master_label_encoder

# Create label encoder (if not exists)
if not os.path.exists('/kaggle/working/master_label_encoder/label_encoder.joblib'):
    !python create_master_label_encoder.py

# Verify
!ls /kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced/
print("\n✅ Setup complete!")
```

---

## W&B Login (Recommended for Experiment Tracking)

To enable Weights & Biases logging for the **contrastive-team** project:

```python
import os

# Team W&B API key (stored in .wandb_api_key file in repo root)
# You can also read it from the file: open('.wandb_api_key').read().strip()
os.environ['WANDB_API_KEY'] = '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'

# Login (will use the API key from environment)
import wandb
wandb.login()
```

**Note:** Each team member should use their own W&B API key.

---

## Notes

1. **Separate Notebooks:** Each team member creates their own notebook for parallel training
2. **Persistence:** Only files in `/kaggle/working/` persist between sessions
3. **GPU:** Use `--device cuda` if GPU accelerator is enabled
4. **Time Limits:** Kaggle notebooks have a 12-hour runtime limit
5. **Memory:** Be mindful of RAM limits (~30GB for CPU, ~16GB for GPU)

---

## Data Flow: Where Labels Come From

### Parquet File Structure

Each modality parquet file contains:
- **Feature columns:** Numerical data (gene expression levels, etc.)
- **`class` column:** Cancer type label (GBM or LGG)
- **`case_id` column:** Sample identifier for consistent ordering

### Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  RAW DATA (parquet files)                                            │
│  - Features + class + case_id                                        │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Contrastive Pretraining (pretrain_contrastive.py)          │
│  - Uses: features (excludes class, case_id)                          │
│  - Self-supervised: no labels needed                                 │
│  - Output: pretrained_encoders/encoders/*.pt                         │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Feature Extraction (extract_pretrained_features.py)        │
│  - Uses: pretrained encoders + features                              │
│  - Output:                                                           │
│    - {modality}_embeddings.npy (encoded features)                    │
│    - labels.npy (class labels preserved from parquet)                │
│    - case_ids.npy (sample identifiers)                               │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3A: QML Training (dre_standard.py --use_pretrained_features)  │
│  - Loads: {modality}_embeddings.npy + labels.npy                     │
│  - Supervised: uses labels for classification loss                   │
│                                                                      │
│  STEP 3B: Transformer Fusion (train_transformer_fusion.py)          │
│  - Uses: raw parquet files directly                                  │
│  - Loads pretrained encoders into model                              │
│  - Supervised: uses 'class' column from parquet                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Points

- **Labels are ALWAYS available** - either from parquet files or saved labels.npy
- **case_id is preserved** for tracking samples across pipeline
- **Metadata columns** (class, case_id) are automatically excluded from features

---

## Next Steps

After setup is complete, proceed to [TEAM_TASKS.md](TEAM_TASKS.md) for your assigned training variant.
