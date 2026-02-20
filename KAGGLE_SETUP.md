# Kaggle Environment Setup Guide

## Overview

This guide covers setting up the quantum-classification-train project in a Kaggle notebook environment. Each team member should create their own notebook.

---

## Step 1: Create a New Kaggle Notebook

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
