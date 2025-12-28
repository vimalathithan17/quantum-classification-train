# ⚠️ DEPRECATED - This file has been consolidated

**This document has been merged into [DATA_PROCESSING.md](DATA_PROCESSING.md).**

Please use the new consolidated documentation:
- **[DATA_PROCESSING.md](DATA_PROCESSING.md)** - Complete data pipeline guide
- **[README.md](README.md)** - Main entry point

**This file can be safely deleted.**

#### **STAGE 1: Repository & Data Source Setup (Cells 1-4)**
- **Purpose**: Initialize working environment
- **Operations**:
  1. Remove old quantum-classification directory
  2. Clone fresh repository from GitHub
  3. Navigate to cloned directory
  4. Pull latest changes

#### **STAGE 2: Create Tumor-Type Datasets (Cells 5-10)**
- **Purpose**: Extract multi-omics data for each glioma subtype
- **Process**:
  - Runs `py/create_multiomics.py` script for 4 tumor types
  - **Tumor Types Processed**:
    1. Astrocytoma
    2. Glioblastoma
    3. Oligodendroglioma
    4. Mixed Glioma
  - **Input Source**: Kaggle `organized-gbm-lgg` datasets
  - **Output Format**: TSV files with columns [case_id, class, feature_1, ..., feature_N]
  - **Output Location**: `/kaggle/working/quantum-classification/outputs/{tumor_type}.tsv`

- **Command Pattern**:
  ```bash
  python3 py/create_multiomics.py \
    --root /kaggle/input/organized-gbm-lgg/organizedTop10_{tumor_type} \
    --out outputs/{tumor_type}.tsv \
    --label {tumor_type}
  ```

#### **STAGE 3: Quality Check & Column Analysis (Cells 11-20)**
- **Purpose**: Verify data completeness and identify column overlap
- **Operations**:
  - **Cell 17**: Load TSV files and analyze missing data patterns
    - Checks for missing files by data type per case
    - Outputs: `{tumor_type}_missing_files.tsv` indicator files
    - Purpose: Understand which cases have complete multi-omics data
  
  - **Cell 21**: Main column alignment function `process_files_and_log_columns()`
    - **Two-Pass Algorithm**:
      1. **First Pass**: Read headers only (efficient)
         - Identify column set for each tumor type
         - Count total columns per file
      2. **Compute Intersection**: Find common columns across all 4 files
      3. **Log Results**: Save to JSON
         - Total common columns count
         - List of common columns (sorted)
         - Uncommon columns per tumor type
      4. **Second Pass**: Read full files, filter to common columns
         - Load each TSV completely
         - Select only intersecting columns
         - Write filtered TSV to `outputs_common/`
    
    - **Example JSON Output**:
      ```json
      {
        "common_columns_count": 26000,
        "common_columns": ["case_id", "class", "CNV_feature1", ...],
        "uncommon_columns_by_file": {
          "astrocytoma": ["rare_feature_1", "rare_feature_2"],
          "glioblastoma": [...],
          "mixed_glioma": [...],
          "oligodendroglioma": [...]
        }
      }
      ```

#### **STAGE 4: Merge All Tumor Types (Cell 22)**
- **Purpose**: Combine all tumor types into single unified dataset
- **Operation**: Bash command
  ```bash
  { head -n1 outputs_common/astrocytoma.tsv; 
    tail -n+2 outputs_common/astrocytoma.tsv; 
    tail -n+2 outputs_common/glioblastoma.tsv; 
    tail -n+2 outputs_common/mixed_glioma.tsv;
    tail -n+2 outputs_common/oligodendroglioma.tsv; 
  } > outputs_common/merged.tsv
  ```
- **Logic**: 
  - Header from first file (once)
  - All data rows from all 4 tumor type files (no duplicates)
- **Output**: Single TSV with all cases from all tumor types
- **Validation**: Line count verification

#### **STAGE 5: Feature Composition Analysis (Cell 23)**
- **Purpose**: Characterize feature distribution by modality
- **Operation**:
  ```python
  # Count features by prefix (first underscore-separated component)
  count = defaultdict(int)
  for feature in header_columns:
    prefix = feature.split('_')[0]
    count[prefix] += 1
  ```
- **Output Example**: `{'CNV': 5000, 'GeneExpr': 8000, 'Meth': 12000, ...}`
- **Purpose**: Understand relative contribution of each modality

#### **STAGE 6: Split by Feature Modality into Parquets (Cells 25-26)**
- **Purpose**: Separate single merged TSV into modality-specific parquet files
- **Algorithm**: Single-pass streaming split
  - **Configuration**:
    - `CHUNK_SIZE = 100` rows per read
    - `FEATURE_PREFIXES = ['Meth_', 'CNV_', 'GeneExpr_', 'miRNA_', 'SNV_', 'Prot_']`
  - **Process**:
    1. Open iterator on large merged TSV (avoids loading entire file)
    2. For **first chunk**:
       - Identify all columns matching each feature prefix
       - Create ParquetWriter for each modality
       - Write first chunk to each file
    3. For **subsequent chunks**:
       - Write data to pre-initialized ParquetWriter objects
       - Maintains schema consistency
    4. **Final Step**: Close all writers properly
  
  - **Output Structure**: `feature_subsets/` directory
    ```
    feature_subsets/
    ├── data_Meth_.parquet      # Methylation features
    ├── data_CNV_.parquet       # Copy number variation
    ├── data_GeneExpr_.parquet  # Gene expression
    ├── data_miRNA_.parquet     # microRNA
    ├── data_SNV_.parquet       # Single nucleotide variants
    └── data_Prot_.parquet      # Protein levels
    ```
  
  - **Columns per file**: [case_id, class, feature_1, feature_2, ..., feature_N]
  - **Memory efficiency**: Only 100 rows in RAM at any time

#### **STAGE 7: Comprehensive Feature Analysis (Cells 27-30)**
- **Purpose**: Characterize data quality, missingness patterns, and decide preprocessing strategy
- **Analysis Components**:
  
  **7a. Basic Feature Statistics** (Cell 27):
  - Shape (row count, column count)
  - Total NaN cell count and percentage
  - Count of completely empty rows
  - Top 10 features by missingness
  - Sporadic missingness (after removing completely empty rows)
  - Descriptive statistics sample
  
  **7b. Missing Value Distribution** (Cell 28):
  - Bins columns by NaN percentage: 0%, 0-10%, 10-30%, 30-50%, ..., 90-95%, 95-97%, 97-100%
  - Count columns in each bin
  - Identifies which modalities have sparse vs. dense data
  - **Example Output**:
    ```
    Meth data:
      0%      : 1000 columns (8%)
      0-10%   : 500 columns (4%)
      10-30%  : 2000 columns (16%)
      90-95%  : 5000 columns (40%)
      95-100% : 3500 columns (28%)
    ```

#### **STAGE 8: Modality-Specific Preprocessing (Cell 31)**
- **Purpose**: Apply data-type-appropriate preprocessing strategies
- **Three Strategies**:
  
  **Strategy A: Dense-Data Cleaning** (miRNA, GeneExpr, CNV, Prot)
  - **Rationale**: High-quality data with rare outlier columns
  - **Operation**: Drop columns with >90% NaNs
  - **Reasoning**: Rare empty columns provide no signal
  - **Output**: Filtered parquets with indicator columns added
  
  **Strategy B: Preserve Missingness** (Methylation)
  - **Rationale**: Widespread NaNs are informative for specialized models
  - **Operation**: Preserve exact NaN patterns, no imputation
  - **Target Models**: LightGBM, XGBoost (can learn from missingness patterns)
  - **Output**: Original data + indicator column
  
  **Strategy C: Pathway Aggregation** (SNV - currently disabled)
  - **Rationale**: Very sparse SNV features (~95% NaN) need special handling
  - **Original Design**: Group rare mutations by KEGG biological pathways
  - **Status**: Code present but disabled in current notebook
  - **Alternative Path**: Use raw SNV features with XGBoost in next stage
  
- **Common Output**: `feature_subsets_processed_selectively/` with indicator columns
  - Each file gains column: `is_missing_{data_type}` (binary: 0=available, 1=completely empty)
  - Imputes only completely empty rows with value 0
  - Preserves sporadic NaNs

#### **STAGE 9: Create Indicator Summary (Cell 32)**
- **Purpose**: Generate master indicator file for all cases × modalities
- **Process**:
  1. Load base [case_id, class] from first file
  2. Extract `is_missing_*` columns from all modality files
  3. Merge all indicators into single file
  4. Fill any missing indicators with 0
  5. Convert all indicators to integer type
  
 - **Output**: `final_processed_datasets_xgb_balanced/indicator_features.parquet`
  ```
  | case_id | class | is_missing_Meth_ | is_missing_CNV_ | is_missing_GeneExpr_ | ...
  |---------|-------|------------------|-----------------|----------------------|-----
  | Case_1  | astro | 0                | 0               | 1                    | ...
  | Case_2  | gliob | 1                | 0               | 0                    | ...
  ```
- **Purpose**: Enables downstream models to condition on data availability

#### **STAGE 10: Final Case Selection & Filtering (Cell 33)**
- **Purpose**: Create balanced dataset with complete/near-complete data
- **Process**:
  1. Calculate `missing_score` = sum of all `is_missing_*` columns per case
  2. Group by class (tumor type)
  3. Sort by missing_score (ascending) within each class
  4. Select top 78 cases per class (or less if class smaller)
  5. Filter all modality files to keep only selected cases
  6. Save to `final_filtered_datasets/`
  
- **Rationale**:
  - `missing_score = 0`: Complete multi-omics data
  - Selects cases with least missing modalities
  - Balanced sampling (78 per class) = ~312 total cases
  - High-quality dataset for downstream training
  
- **Output**: `final_filtered_datasets/` with:
  - 6 modality parquets (filtered)
  - `indicator_features.parquet` (filtered and indexed)

### Key Data Structures

**Input Data Format (per tumor type):**
```
case_id | class | CNV_feat_1 | CNV_feat_2 | ... | Meth_feat_1 | ...
--------|-------|------------|------------|-----|-------------|-----
Case_1  | tumor | 0.5        | 1.2        | ... | 0.8         | ...
Case_2  | tumor | 0.6        | 1.1        | ... | 0.7         | ...
```

**Output: Feature-Type Specific Parquets**
```
Parquet File (data_CNV_.parquet)
├── Columns: [case_id, class, CNV_feat_1, CNV_feat_2, ...]
├── Row Count: 1000+ cases
└── Memory Efficient: Columnar storage
```

---

## 🔍 Notebook 2: `feature-extraction-xgb.ipynb`

### Purpose
Two-stage feature selection pipeline reducing dimensionality from thousands to 500 features per modality while preserving predictive signal using mutual information and XGBoost importance scores.

### Architecture & Workflow

#### **STAGE 1: Initialize Feature Selection Pipeline (Cell 1)**
- **Purpose**: Load indicator file, establish case ordering, and prepare label encoding
- **Operations**:
  1. **Load Indicator File**:
     - Source: `/kaggle/input/data-process/quantum-classification/final_filtered_datasets/indicator_features.parquet`
     - Contains: case_id, class, and all is_missing_* columns
  
  2. **Define Master Sort Order**:
     - Creates consistent case_id ordering: `sorted(all_case_ids)`
     - Purpose: Ensures all modalities output identically sorted case_ids
  
  3. **Create Label Encoder**:
     - Fits LabelEncoder on class labels
     - Maps: astrocytoma→0, glioblastoma→1, mixed_glioma→2, oligodendroglioma→3
     - Used for: Both MI and XGBoost training
  
  - **Validation Output**: 
    ```
    LabelEncoder created. Classes: ['astrocytoma', 'glioblastoma', 'mixed_glioma', 'oligodendroglioma']
    Master sort order defined with 312 case_ids.
    ```

#### **STAGE 2: Two-Stage Feature Selection Per Modality**
- **Purpose**: Apply coarse and fine feature filtering for each data type
- **Loop**: For each data_type in ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']

##### **STAGE 2a: Load & Prepare Data**
- **Operation**:
  1. Load full parquet: `final_filtered_datasets/data_{data_type}_.parquet`
  2. If features < 500: Skip feature selection, just sort and save
  3. Otherwise: Proceed to two-stage selection
  
- **Data Preparation for Selection**:
  - **Key Insight**: Only use cases that have complete data for THIS modality
  - **Logic**: 
    ```python
    is_missing_col_indicator = f'is_missing_{data_type}_'
    case_ids_to_use = indicators_df[indicators_df[is_missing_col_indicator] == 0].index
    # Only these cases used for feature importance training
    ```
  - **Purpose**: Avoid training feature importance on imputed/missing data
  - **Result**: X_select_all = subset of cases with actual measurements
  - **Memory Optimization**: Delete full dataframe after filtering

##### **STAGE 2b: STAGE 1 - Mutual Information Filter (Coarse Sieve)**
- **Configuration**: `N_FEATURES_STAGE1 = 50000`
- **Method**: SelectKBest with mutual_info_classif
- **Algorithm**:
  ```python
  imputer = SimpleImputer(strategy='median')
  X_imputed = imputer.fit_transform(X_select_all)
  # Median imputation for NaN values in sporadic locations
  
  mi_selector = SelectKBest(mutual_info_classif, k=min(50000, X_select_all.shape[1]))
  mi_selector.fit(X_imputed, y_encoded)
  
  selected_indices = mi_selector.get_support()
  stage1_features = X_select_all.columns[selected_indices].tolist()
  ```
  
- **Why Mutual Information**:
  - Fast computation (O(n*m) vs O(n*m*log(m)) for other methods)
  - Detects non-linear relationships with target
  - Univariate metric: each feature evaluated independently
  - Coarse filtering: identifies clearly irrelevant features
  - Memory efficient: doesn't require loading all features simultaneously
  
- **Output Example**:
  ```
  Original shape for selection: (78, 25000)
  Selected 47832 features after Stage 1.
  # ~91% of features survive (only clearly irrelevant ~9% removed)
  ```

##### **STAGE 2c: STAGE 2 - XGBoost Fine Selection (Fine-Toothed Comb)**
- **Configuration**: `N_FEATURES_TO_KEEP = 500`
- **XGBoost Model Setup**:
  ```python
  XGBClassifier(
    random_state=42,           # Reproducibility
    n_jobs=-1,                 # Multi-core processing
    tree_method='hist',        # GPU/histogram acceleration (memory efficient)
    subsample=0.8,             # 80% row sampling (reduces overfitting)
    colsample_bytree=0.1       # 10% column sampling (strong regularization)
  )
  ```
  
- **Feature Importance Extraction**:
  ```python
  xgb_final_selector = SelectFromModel(
    xgb_selector_model,
    max_features=500,
    prefit=False
  ).fit(X_for_xgb.values, y_encoded)
  
  kept_features = X_for_xgb.columns[xgb_final_selector.get_support()].tolist()
  ```
  
- **Why XGBoost**:
  - Captures multivariate feature interactions
  - Tree-based importance: accounts for feature combinations
  - Naturally handles missing values (no imputation needed)
  - Strong regularization (colsample_bytree=0.1): avoids noise features
  - Fast: histogram-based algorithm
  
- **Output Example**:
  ```
  Fitting XGBoost on 47832 features...
  Selected final 500 features after Stage 2.
  # Aggressive reduction: 47832 → 500 (1% survival)
  ```

##### **STAGE 2d: Reload & Sort Final Features**
- **Purpose**: Memory-efficient loading of only selected features
- **Process**:
  ```python
  columns_to_load = ['case_id', 'class'] + kept_features  # Just 502 columns
  df_final_filtered = safe_load_parquet(input_file, columns=columns_to_load)
  # Only load these 502 columns, not original 25000+
  
  df_final_filtered.set_index('case_id', inplace=True)
  df_sorted = df_final_filtered.reindex(master_sort_order)
  # Ensure consistent ordering across all modality files
  
  df_final = df_sorted.reset_index()
  df_final.to_parquet(output_file, index=False)
  ```
  
- **Why This Approach**:
  - **Pandas Column Selection**: Only reads specified columns from parquet (pushdown)
  - **Memory Savings**: 25000 columns → 502 columns = ~95% memory reduction
  - **Consistent Ordering**: All modalities sorted identically by case_id
  - **Preserves Types**: class labels and case_ids maintain integrity

#### **STAGE 3: Quality Validation (Cell 2)**
- **Purpose**: Verify output dataset quality
- **Validation Steps**:
  1. **Scan Two Directories**:
     - Input: `/kaggle/input/data-process/quantum-classification/final_filtered_datasets`
     - Output: `final_processed_datasets_xgb_balanced`
  
  2. **For Each Parquet File**:
     - Load with safe_load_parquet()
     - Calculate total NaN count: `df.isnull().sum().sum()`
     - Report: 0 NaNs or specific count
  
  3. **Final Summary**:
     ```
     Scanning Directory: final_processed_datasets_xgb_balanced
       - File: data_CNV_.parquet -> No NaN values found.
       - File: data_GeneExpr_.parquet -> No NaN values found.
       - File: data_miRNA_.parquet -> No NaN values found.
       - File: data_Meth_.parquet -> Found 2,456 NaN values.
       ...
     Total NaN values found: [X]
     ```

#### **STAGE 4: Copy Indicators & Package (Cells 3-4)**
- **Cell 3**: Copy indicator file to output directory
  ```bash
  cp /kaggle/input/.../indicator_features.parquet final_processed_datasets_xgb_balanced/
  ```
  - **Purpose**: Include indicators in final output
  - **Used by**: Downstream models for conditional encoding
  
- **Cell 4**: Create deliverable zip file
  ```bash
  zip -r xgb_reduced.zip final_processed_datasets_xgb_balanced
  ```
  - **Output**: Compressed dataset ready for download/transfer

### Two-Stage Selection Rationale

**Why Two Stages Instead of One?**

| Aspect | Stage 1 (MI) | Stage 2 (XGB) | Combined Benefit |
|--------|--------------|---------------|-----------------|
| Speed | Very fast | Slower | Avoids training XGB on 25K features |
| Memory | Low | Medium | Keeps ~50K features to manageable RAM |
| Relationships | Univariate only | Multivariate | Captures both independent + interaction effects |
| Missing Data | Requires imputation | Handles natively | MI imputes, XGB preserves patterns |
| Output Size | 50K features | 500 features | ~1-2% of original → model-ready size |

**Why Not Machine Learning-Based Selection (e.g., Recursive Feature Elimination)?**
- RFE would require training 25 models (one per elimination round)
- Computational cost: 25 × XGB training = prohibitive
- Two-stage approach: 1 MI (fast) + 1 XGB (medium) = practical

---

## 🔗 Integration with Main Pipeline

### Data Output Locations

**From data-process.ipynb:**
- `feature_subsets/` - Raw feature-type parquets (thousands of features)
- `feature_subsets_processed_selectively/` - With indicator columns
- `final_processed_datasets/` - After modality-specific preprocessing
- `final_filtered_datasets/` - Quality-filtered dataset (78 cases per class)
  - **Contains**: 6 modality parquets + indicator_features.parquet
  - **Case Count**: ~312 cases (78 per class × 4 classes)
  - **Feature Count**: Thousands per modality

**From feature-extraction-xgb.ipynb:**
- `final_processed_datasets_xgb_balanced/` - XGBoost-selected features
  - **Contains**: 6 modality parquets (500 features each) + indicator_features.parquet
  - **Feature Count**: 500 per modality (from thousands)
  - **Case Count**: Same 312 cases
  - **Ready For**: Training any downstream model

### Where This Data Feeds Into

1. **Training Scripts** (`dre_standard.py`, `cfe_standard.py`, etc.)
   - **Input Source**: `final_processed_datasets_xgb_balanced/`
   - **Features Used**: 500 selected per modality
   - **Workflow**: Load 6 modality parquets, combine with indicators
   
2. **Contrastive Learning** (`performance_extensions/contrastive_learning.py`)
   - **Input Source**: `final_processed_datasets_xgb_balanced/` or `final_filtered_datasets/`
   - **Feature Count**: Can use 500 (reduced) or thousands (detailed)
   - **Indicators**: Required for conditional encoding
   
3. **Transformer Fusion** (`performance_extensions/transformer_fusion.py`)
   - **Input Source**: `final_processed_datasets_xgb_balanced/`
   - **Benefit**: 500 features reduces memory for transformer layers
   - **Indicators**: Enable masking of missing modalities

### Example Usage in Training Script

```python
import os
import pandas as pd

# After running both notebooks, load processed data:
DATA_DIR = 'final_processed_datasets_xgb_balanced'

# Load indicator file (describes data availability)
indicators = pd.read_parquet(os.path.join(DATA_DIR, 'indicator_features.parquet'))

# Load modality-specific feature sets
data_dict = {}
for modality in ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']:
    data_dict[modality] = pd.read_parquet(
        os.path.join(DATA_DIR, f'data_{modality}_.parquet')
    )

# Data is already:
# ✅ Dimensionally reduced (500 features per modality)
# ✅ Aligned by case_id and class across all modalities
# ✅ With missing data indicators per modality
# ✅ Sorted consistently by case_id
# ✅ Quality filtered (78 cases per class, complete data preference)

# Example: Build feature matrix
X = []
for case_id in data_dict['CNV']['case_id']:
    row = []
    for modality, df in data_dict.items():
        features = df[df['case_id'] == case_id].iloc[0, 2:].values  # Skip case_id, class
        row.extend(features)
    X.append(row)
X = np.array(X)  # Shape: (312, 3000) = 312 cases × (6 modalities × 500 features)
```

---

## 📊 Complete Data Flow Diagram

```
STAGE 1: Raw Collection
┌─────────────────────────────────────────────────────┐
│  Kaggle Datasets: organized-gbm-lgg                │
│  4 Tumor Types:                                     │
│  - Astrocytoma (~100 samples)                       │
│  - Glioblastoma (~100 samples)                      │
│  - Oligodendroglioma (~100 samples)                │
│  - Mixed Glioma (~100 samples)                      │
└─────────────────────────────────────────────────────┘
              ↓ py/create_multiomics.py
STAGE 2: Tumor-Type TSVs
┌─────────────────────────────────────────────────────┐
│  outputs/ (TSV files)                              │
│  - astrocytoma.tsv (~30K features)                 │
│  - glioblastoma.tsv (~30K features)                │
│  - oligodendroglioma.tsv (~25K features)           │
│  - mixed_glioma.tsv (~28K features)                │
│  ⚠️  Different feature counts per tumor type       │
└─────────────────────────────────────────────────────┘
              ↓ process_files_and_log_columns()
STAGE 3: Common Column Alignment
┌─────────────────────────────────────────────────────┐
│  outputs_common/ (TSV files)                       │
│  - astrocytoma.tsv (~20K common features)          │
│  - glioblastoma.tsv                                │
│  - oligodendroglioma.tsv                           │
│  - mixed_glioma.tsv                                │
│  + column_analysis.json (alignment report)         │
│  ✅ Identical feature columns across all files     │
└─────────────────────────────────────────────────────┘
              ↓ bash: head + tail merge
STAGE 4: Single Merged Dataset
┌─────────────────────────────────────────────────────┐
│  outputs_common/merged.tsv                         │
│  - All 400 cases combined                          │
│  - 20K common features across all cases            │
│  - Columns: case_id | class | feature_1 | ...     │
└─────────────────────────────────────────────────────┘
              ↓ Single-pass feature type split
STAGE 5: Feature Type Separation
┌─────────────────────────────────────────────────────┐
│  feature_subsets/ (Parquet files)                  │
│  - data_Meth_.parquet (12K features, ~80% NaN)    │
│  - data_CNV_.parquet (5K features, ~10% NaN)      │
│  - data_GeneExpr_.parquet (8K features, ~5% NaN)  │
│  - data_miRNA_.parquet (2K features, ~2% NaN)     │
│  - data_SNV_.parquet (3K features, ~95% NaN)      │
│  - data_Prot_.parquet (1K features, ~1% NaN)      │
│  Each: case_id | class | feature_1 | ...          │
└─────────────────────────────────────────────────────┘
              ↓ Modality-specific preprocessing
STAGE 6: Quality Processing
┌─────────────────────────────────────────────────────┐
│  feature_subsets_processed_selectively/             │
│  + Selective imputation (empty rows only)          │
│  + Indicator columns: is_missing_{type}            │
│  - Data preserved as-is for model-native handling  │
│  ✅ Ready for model training                       │
└─────────────────────────────────────────────────────┘
              ↓ Generate indicator summary
STAGE 7: Indicator File Creation
┌─────────────────────────────────────────────────────┐
│  final_processed_datasets/                         │
│  - indicator_features.parquet                      │
│    Columns: case_id | class | is_missing_Meth_ |  │
│             is_missing_CNV_ | ... (6 total)        │
│  - (All 6 modality files also in this dir)         │
│  ✅ Tracks data availability per case             │
└─────────────────────────────────────────────────────┘
              ↓ Quality filter: select 78 best per class
STAGE 8: Final Case Selection
┌─────────────────────────────────────────────────────┐
│  final_filtered_datasets/ (DATA-PROCESS OUTPUT)    │
│  - 312 total cases (78 per class)                  │
│  - Cases ranked by data completeness               │
│  - 6 modality parquets (full features)             │
│  - indicator_features.parquet (filtered)           │
│  ✅ High-quality balanced dataset                  │
└─────────────────────────────────────────────────────┘
              ↓ Two-stage feature selection
STAGE 9: XGBoost Feature Reduction
┌─────────────────────────────────────────────────────┐
│  STAGE 9a: Mutual Information Filter               │
│  - Input: 12K features (Meth), 5K (CNV), etc.     │
│  - Output: 50K features (if available)             │
│  - Method: SelectKBest(mutual_info_classif)        │
│  - Purpose: Quick removal of irrelevant features   │
│                                                     │
│  STAGE 9b: XGBoost Fine Selection                  │
│  - Input: 50K features from Stage 1                │
│  - Output: 500 features per modality               │
│  - Method: SelectFromModel(XGBClassifier)          │
│  - Purpose: Capture multivariate interactions      │
└─────────────────────────────────────────────────────┘
              ↓ Final feature-reduced dataset
STAGE 10: Final Output (XGB-EXTRACTION OUTPUT)
┌─────────────────────────────────────────────────────┐
│  final_processed_datasets_xgb_balanced/             │
│  - data_CNV_.parquet (500 features)                │
│  - data_GeneExpr_.parquet (500 features)           │
│  - data_miRNA_.parquet (500 features)              │
│  - data_Meth_.parquet (500 features)               │
│  - data_Prot_.parquet (500 features)               │
│  - data_SNV_.parquet (500 features)                │
│  - indicator_features.parquet (copied)             │
│  ✅ READY FOR TRAINING                             │
│  Total Features: 3000 (6 × 500)                    │
│  Cases: 312                                        │
│  Size: ~50-100 MB (vs ~1-2 GB input)              │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  Training Scripts                                  │
│  ├─ dre_standard.py, dre_relupload.py             │
│  ├─ cfe_standard.py, cfe_relupload.py             │
│  ├─ metalearner.py                                │
│  ├─ contrastive_learning.py                       │
│  └─ transformer_fusion.py                         │
│                                                     │
│  Models trained on final_processed_datasets_xgb    │
│  Results: Optimized F1 with reduced computational │
│  cost and reduced overfitting risk                 │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ How to Run These Notebooks

### Prerequisites

```bash
pip install pandas pyarrow xgboost scikit-learn
```

### Execution Steps

**Step 1: Run `data-process.ipynb`**
- Requires Kaggle API credentials (for dataset access)
- Takes 2-4 hours depending on dataset size
- Outputs: Feature-type-specific parquets in `feature_subsets/`

**Step 2: Run `feature-extraction-xgb.ipynb`**
- Requires GPU for XGBoost (recommended but not required)
- Takes 1-2 hours
- Outputs: Reduced feature sets in `final_processed_datasets_xgb_balanced/`

**Step 3: Use processed data in main training pipeline**
- Update dataset paths in training scripts
- Features are ready for QML, DRE, CFE, or fusion models

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│  Kaggle Datasets (Organized GBM-LGG)               │
│  - Astrocytoma                                      │
│  - Glioblastoma                                     │
│  - Oligodendroglioma                               │
│  - Mixed Glioma                                     │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  data-process.ipynb                                │
│  ├─ Create tumor-type-specific TSVs               │
│  ├─ Find common columns across tumor types        │
│  ├─ Merge into single dataset                     │
│  ├─ Split by feature modality (CNV, Meth, etc.)  │
│  └─ Create indicator features for missing data    │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  feature_subsets/ (parquet files)                  │
│  - 6 data type files: CNV, GeneExpr, miRNA, etc.  │
│  - Each with 10,000-50,000 features               │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  feature-extraction-xgb.ipynb                      │
│  ├─ Stage 1: Mutual Information (→50K features)   │
│  ├─ Stage 2: XGBoost Selection (→500 features)    │
│  ├─ Memory optimization & reloading               │
│  └─ Data quality validation (NaN checks)          │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  final_processed_datasets_xgb_balanced/ (parquet)  │
│  - Ready for training with QML/DRE/CFE/Fusion    │
│  - 500 features per modality                      │
│  - Aligned case IDs and indicators                │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  Training Scripts (dre_standard.py, etc.)          │
│  - Load processed parquets                        │
│  - Train QML/DRE/CFE or Fusion models            │
│  - Evaluate on test set                           │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Customization Points

### If You Want to:

**Use different tumors:**
- Modify `file_list` in `data-process.ipynb`
- Point to different Kaggle input directories

**Change feature reduction aggressiveness:**
- Adjust `N_FEATURES_STAGE1` (e.g., 25000 for more aggressive)
- Adjust `N_FEATURES_TO_KEEP` (e.g., 250 for fewer features)

**Include/exclude modalities:**
- Edit `DATA_TYPES_TO_PROCESS` list
- Add/remove feature prefixes in feature splitting logic

**Use different feature selection methods:**
- Replace Mutual Information with correlation-based selection
- Replace XGBoost with Random Forest feature importance
- Add SHAP values for explainability

---

## 💾 Storage & Memory Considerations

**Input Size (per notebook run):**
- Kaggle datasets: ~50-100 GB (from Kaggle)
- Memory needed: 8-16 GB RAM

**Intermediate Size:**
- Merged TSV: ~10-20 GB
- Parquets in feature_subsets: ~15-25 GB

**Output Size:**
- final_processed_datasets_xgb_balanced: ~2-3 GB (500 features × 6 modalities)

**Typical Execution Time:**
- data-process.ipynb: 2-4 hours
- feature-extraction-xgb.ipynb: 1-2 hours
- **Total: 3-6 hours**

---

## 🐛 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Kaggle dataset not found" | No API credentials | Set up `~/.kaggle/kaggle.json` |
| Out of memory during merging | File too large | Use chunked reading (already implemented) |
| NaN values in output | Imputation failure | Check median imputer strategy |
| Column mismatch in merge | Tumor types have different features | Notebook handles by finding common columns |
| Parquet read error | Thrift buffer too small | Increase buffer limits (already done in notebook) |

---

## 📝 Notes & Future Improvements

### Current Design
- ✅ Handles multi-omics data with different feature counts
- ✅ Manages missing modalities with indicator features
- ✅ Optimized for large files (chunked reading/writing)
- ✅ Two-stage filtering balances signal preservation with dimensionality

### Potential Enhancements
- [ ] Parallelize feature extraction across modalities
- [ ] Add cross-validation for feature selection stability
- [ ] Implement SHAP values for interpretability
- [ ] Add feature correlation analysis before selection
- [ ] Cache intermediate results for faster re-runs
- [ ] Add data quality metrics (feature variance, distributions)

---

**Last Updated:** December 28, 2024  
**Status:** Complete documentation of data processing pipeline
