# Data Processing Pipeline

Complete guide to the data preprocessing pipeline for quantum classification.

---

## Overview

The pipeline transforms raw multi-omics Kaggle datasets into training-ready feature sets:

```
Raw Kaggle Data (4 tumor types, ~400 patients)
    ↓
data-process.ipynb (10 stages)
    ↓
final_filtered_datasets/ (312 cases, full features)
    ↓
feature-extraction-xgb.ipynb (2 stages)
    ↓
final_processed_datasets_xgb_balanced/ (312 cases, 500 features/modality)
```

---

## Notebook 1: data-process.ipynb

**Purpose:** Collect, align, merge, and organize multi-omics data.  
**Runtime:** 3-4 hours  
**Output:** `final_filtered_datasets/`

### 10-Stage Pipeline

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Clone/update repository | Working environment |
| 2 | Create tumor-type TSVs | `outputs/{tumor}.tsv` |
| 3 | Analyze data completeness | Quality metrics |
| 4 | Align columns across types | Common feature set |
| 5 | Merge all tumor types | `outputs_common/merged.tsv` |
| 6 | Analyze feature composition | Modality counts |
| 7 | Split by modality | `feature_subsets/data_{type}_.parquet` |
| 8 | Analyze NaN patterns | Quality report |
| 9 | Create missing indicators | `is_missing_{type}_` columns |
| 10 | Select balanced cases | 78 per class (312 total) |

### Configuration

All paths and parameters are centralized in the `CONFIG` dictionary:

```python
CONFIG = {
    'KAGGLE_INPUT_BASE': '/kaggle/input/organized-gbm-lgg',
    'WORKING_DIR': '/kaggle/working/quantum-classification',
    'TUMOR_TYPES': ['astrocytoma', 'glioblastoma', 'mixed_glioma', 'oligodendroglioma'],
    'FEATURE_PREFIXES': ['Meth_', 'CNV_', 'GeneExpr_', 'miRNA_', 'SNV_', 'Prot_'],
    'TOP_N_CASES_PER_CLASS': 78,
    # ... more settings
}
```

---

## Notebook 2: feature-extraction-xgb.ipynb

**Purpose:** Reduce features using two-stage AI selection.  
**Runtime:** 1-2 hours  
**Input:** `final_filtered_datasets/`  
**Output:** `final_processed_datasets_xgb_balanced/`

### Two-Stage Funnel

```
Original: ~10,000-20,000 features per modality
    ↓
Stage 1: Mutual Information Filter
    → Coarse sieve: keep top 50,000 features
    → Fast, univariate scoring
    ↓
Stage 2: XGBoost Importance
    → Fine filter: keep top 500 features
    → Captures feature interactions
    ↓
Final: 500 features per modality (3,000 total)
```

### Why Two Stages?

| Stage | Speed | What It Detects |
|-------|-------|-----------------|
| MI | ⚡ Fast | Individual feature importance |
| XGBoost | 🐌 Slower | Feature combinations & interactions |
| **Combined** | ✅ Balanced | Complete predictive signal |

---

## Output Directory Structure

```
final_processed_datasets_xgb_balanced/
├── data_CNV_.parquet        # Copy number variation (500 features)
├── data_GeneExpr_.parquet   # Gene expression (500 features)
├── data_miRNA_.parquet      # microRNA (500 features)
├── data_Meth_.parquet       # Methylation (500 features)
├── data_Prot_.parquet       # Protein (500 features)
├── data_SNV_.parquet        # Mutations (500 features)
└── indicator_features.parquet
    └── Columns: case_id, class, is_missing_CNV_, is_missing_GeneExpr_, ...
```

### File Format

Each modality parquet contains:
- `case_id`: Patient identifier (sorted alphabetically)
- `class`: Tumor type (astrocytoma, glioblastoma, mixed_glioma, oligodendroglioma)
- `is_missing_{type}_`: Binary indicator (0=has data, 1=missing)
- 500 feature columns ranked by importance

---

## Using the Output

### Set SOURCE_DIR for Training Scripts

```bash
# If using XGBoost-selected features (recommended):
export SOURCE_DIR=final_processed_datasets_xgb_balanced

# If using full features:
export SOURCE_DIR=final_processed_datasets
```

### Load Data in Python

```python
import pandas as pd

# Load all modalities
modalities = ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']
data = {m: pd.read_parquet(f'{SOURCE_DIR}/data_{m}_.parquet') for m in modalities}

# Load indicators for missing modality handling
indicators = pd.read_parquet(f'{SOURCE_DIR}/indicator_features.parquet')

# Get features and labels
X = np.hstack([data[m].iloc[:, 2:].values for m in modalities])  # 312 × 3000
y = data['CNV']['class'].values  # 312 labels
```

### Use with Conditional Feature Encoding

```python
# Check if modality is missing for a case
case_id = 'TCGA-AB-1234'
if indicators.loc[case_id, 'is_missing_Meth_'] == 1:
    # Handle missing methylation data
    pass
```

---

## Key Concepts

### Missing Data Indicators

Each case has binary flags indicating data availability:
```
Patient TCGA-XY-1234:
  is_missing_CNV_ = 0      (has CNV data)
  is_missing_Meth_ = 1     (missing methylation)
  is_missing_GeneExpr_ = 0 (has gene expression)
```

These indicators are used for:
- **Conditional QML:** Learn encodings for missing vs. present
- **Transformer Fusion:** Attention masking for missing modalities
- **Sample selection:** Prefer cases with complete data

### Selective Imputation

Only completely empty rows (entire modality missing) are imputed with placeholder values. Sporadic NaNs within rows are preserved because:
- LightGBM/XGBoost can learn from missingness patterns
- Imputing sporadic NaNs may introduce bias

---

## Computational Requirements

| Notebook | CPU | GPU | RAM | Time |
|----------|-----|-----|-----|------|
| data-process.ipynb | 4-core | Optional | 8-16 GB | 2-4 hrs |
| feature-extraction-xgb.ipynb | 4-core | Beneficial | 8-16 GB | 1-2 hrs |

GPU accelerates XGBoost training but is not required.

---

## Validation Checklist

After running both notebooks, verify:

- [ ] **312 cases** total across 4 classes
- [ ] **78 cases per class** (balanced)
- [ ] **6 modality files** + indicator file
- [ ] **500 features** per modality
- [ ] **Consistent case_id ordering** across files
- [ ] **No NaN values** in final output (verified by notebook)

---

## Troubleshooting

### "Parquet file too large to load"
Increase thrift limits (already set in `safe_load_parquet`):
```python
pd.read_parquet(file, thrift_string_size_limit=1*1024**3)
```

### "Insufficient memory during XGBoost"
Reduce `n_features_stage1` in CONFIG or use `tree_method='hist'` (already default).

### "Case counts don't match across modalities"
Re-run case selection stage with consistent `master_sort_order`.
