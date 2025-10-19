# Tuning Metrics Update Summary

This document summarizes the comprehensive updates made to `tune_models.py` to implement F1-based optimization and comprehensive metrics tracking.

## Key Changes

### 1. Imports and Dependencies

Added comprehensive metrics imports:
```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import numpy as _np  # local alias to avoid shadowing pennylane.numpy
```

### 2. Optimization Metric Changed

**Previous:** Trials were optimized using accuracy score
**New:** Trials are now optimized using **weighted F1 score**

This provides:
- Better handling of class imbalance
- More robust model selection
- Alignment with best practices for multiclass classification

### 3. Comprehensive Metrics per Fold

For each fold in each trial, the following metrics are now computed and saved:

- **Accuracy:** Overall classification accuracy
- **Precision:** Macro and weighted averages
- **Recall:** Macro and weighted averages  
- **F1 Score:** Macro and weighted averages (used as optimization metric)
- **Specificity:** Macro and weighted averages (custom implementation)
- **Confusion Matrix:** Full confusion matrix
- **Classification Report:** Detailed per-class metrics

### 4. Helper Functions Added

#### `_per_class_specificity(cm_arr)`
Computes per-class specificity from a confusion matrix:
```python
def _per_class_specificity(cm_arr):
    """Compute per-class specificity from confusion matrix."""
    K = cm_arr.shape[0]
    speci = _np.zeros(K, dtype=float)
    total = cm_arr.sum()
    for i in range(K):
        TP = cm_arr[i, i]
        FP = cm_arr[:, i].sum() - TP
        FN = cm_arr[i, :].sum() - TP
        TN = total - (TP + FP + FN)
        denom = TN + FP
        speci[i] = float(TN / denom) if denom > 0 else 0.0
    return speci
```

#### `ensure_writable_results_dir(results_dir)`
Ensures the tuning results directory is writable, with automatic fallback to current working directory if the primary location is read-only.

#### `cleanup_old_trials(results_dir, study, keep_best=True, keep_latest_n=2)`
Automatically cleans up old trial directories, keeping only:
- The best trial (based on F1 score)
- The latest 2 trials

This prevents disk space issues from accumulating trial artifacts.

### 5. Objective Function Updates

Both Approach 1 and Approach 2 objective functions were updated with:

**Approach 1 (Pipeline-based):**
- Compute predictions using `pipeline.predict(X_val)`
- Extract probabilities if available using `pipeline.predict_proba(X_val)`
- Calculate all metrics from predictions
- Save fold metrics to JSON file
- Report F1 weighted to Optuna for pruning
- Return mean F1 weighted across folds

**Approach 2 (QML-based with feature selection):**
- Compute predictions using `qml_model.predict((X_val_scaled, is_missing_val))`
- Extract probabilities if available
- Calculate all metrics from predictions
- Save fold metrics to JSON file
- Report F1 weighted to Optuna for pruning
- Return mean F1 weighted across folds

### 6. Per-Trial Artifact Saving

Each trial now saves comprehensive artifacts:

```
tuning_results/
├── trial_{trial_id}/
│   ├── fold_1_metrics.json
│   ├── fold_2_metrics.json
│   └── fold_3_metrics.json
```

Each `fold_X_metrics.json` contains:
```json
{
  "accuracy": 0.85,
  "precision_macro": 0.83,
  "recall_macro": 0.82,
  "f1_macro": 0.825,
  "precision_weighted": 0.84,
  "recall_weighted": 0.85,
  "f1_weighted": 0.843,
  "specificity_macro": 0.91,
  "specificity_weighted": 0.915,
  "confusion_matrix": [[10, 2], [3, 15]],
  "classification_report": "..."
}
```

### 7. Trial-Level Aggregation

After all folds complete, trial-level statistics are computed and attached:
- `mean_f1_weighted`: Mean F1 score across folds
- `std_f1_weighted`: Standard deviation of F1 scores
- `fold_{N}_metrics`: Full metrics dictionary for each fold

### 8. Study-Level Artifacts

After optimization completes, the following artifacts are saved:

1. **Best Parameters** (`best_params_{study_name}.json`):
   - Best hyperparameters found
   - Includes training steps

2. **Trials Dataframe** (`trials_{study_name}.csv`):
   - Flat CSV with all trial results
   - Easy to load and analyze offline

3. **Optuna Visualization Plots** (`optuna_plots/`):
   - `param_importances.png`: Parameter importance analysis
   - `optimization_history.png`: Optimization progress over time
   - `slice.png`: Slice plots showing parameter effects
   - `contour.png`: Contour plots showing parameter interactions
   - Saved as PNG if kaleido is available, otherwise as HTML

### 9. Weights & Biases Integration

Each trial now gets a unique W&B run name:
- Format: `tune_{approach}_{datatype}_trial{trial_number}`
- Example: `tune_DR_CNV_trial0`, `tune_DR_CNV_trial1`, etc.
- Custom prefix can be specified via `--wandb_run_name` argument

### 10. Directory Structure

The recommended tuning results directory structure is now:

```
tuning_results/
├── best_params_{study_name}.json
├── trials_{study_name}.csv
├── optuna_plots/
│   ├── param_importances.png
│   ├── optimization_history.png
│   ├── slice.png
│   └── contour.png
└── trial_{trial_id}/          # Only best + latest 2 kept
    ├── fold_1_metrics.json
    ├── fold_2_metrics.json
    └── fold_3_metrics.json
```

## Usage Examples

### Basic Tuning with New Metrics
```bash
python tune_models.py --datatype CNV --approach 1 --qml_model standard --n_trials 50 --verbose
```

### With W&B Tracking
```bash
python tune_models.py --datatype CNV --approach 1 --qml_model standard \
    --n_trials 50 --use_wandb --wandb_project qml_tuning --verbose
```

### Custom Run Name Prefix
```bash
python tune_models.py --datatype CNV --approach 1 --qml_model standard \
    --n_trials 50 --use_wandb --wandb_project qml_tuning \
    --wandb_run_name my_experiment --verbose
```

## Benefits

1. **Better Model Selection:** F1 score is more appropriate for imbalanced classes than accuracy
2. **Comprehensive Insights:** Full metrics provide deep understanding of model performance
3. **Disk Management:** Automatic cleanup prevents disk space issues
4. **Reproducibility:** Complete metrics history enables detailed analysis
5. **Visualization:** Optuna plots provide insights into hyperparameter importance and optimization progress
6. **Experiment Tracking:** Unique W&B run names enable easy comparison of trials

## Compatibility

All changes are backward compatible:
- Existing command-line arguments work unchanged
- Study databases can be reused
- New features are additive, not breaking

## Testing

The implementation was tested with:
- Specificity calculation validation
- Directory handling verification
- Metrics integration testing
- Syntax validation

All tests passed successfully.

## Migration Notes

For existing studies:
1. Old studies will continue to work with new code
2. New trials will use F1 score optimization
3. Old trials (optimized for accuracy) can coexist with new trials
4. Trial cleanup only affects trial directories, not the study database
5. Optuna plots and trials CSV will be generated for existing studies when running additional trials

## Future Enhancements

Possible future improvements:
1. Configurable optimization metric (allow choosing between F1, accuracy, etc.)
2. Per-trial model checkpointing (currently only fold metrics are saved)
3. OOF predictions export for stacking/meta-learning
4. Aggregate results manifest with git SHA, timestamps, etc.
