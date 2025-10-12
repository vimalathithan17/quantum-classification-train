# New Features: Cross-Validation Control and Best Weight Step Tracking

## Overview
This update adds key features to improve training flexibility and transparency:
1. Option to skip cross-validation during training
2. Option to perform only cross-validation and skip final training (for meta-learner OOF prediction generation)
3. Logging of the training step at which best weights were obtained

## Feature 1: Skip Cross-Validation

### Purpose
Allow users to skip the computationally expensive cross-validation step when only final model training is needed. This is useful for:
- Quick iterations during development
- Final production runs where OOF predictions aren't needed
- Time-constrained training scenarios

### Usage
Add the `--skip_cross_validation` flag to any training script:

```bash
# Skip cross-validation in DRE standard training
python dre_standard.py --datatypes CNV --skip_cross_validation --verbose

# Skip cross-validation in DRE reuploading training
python dre_relupload.py --datatypes CNV --skip_cross_validation --verbose

# Skip cross-validation in CFE standard training
python cfe_standard.py --datatypes CNV --skip_cross_validation --verbose

# Skip cross-validation in CFE reuploading training
python cfe_relupload.py --datatypes CNV --skip_cross_validation --verbose

# Skip cross-validation in metalearner training (affects tuning mode)
python metalearner.py --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train --skip_cross_validation --verbose
```

### Affected Scripts
- `dre_standard.py`
- `dre_relupload.py`
- `cfe_standard.py`
- `cfe_relupload.py`
- `metalearner.py`

### Behavior
When `--skip_cross_validation` is used:
- The OOF (out-of-fold) prediction generation is skipped
- Only the final model training on the full training set is performed
- Training time is significantly reduced (approximately 3x faster as 3-fold CV is skipped)
- Test predictions and final model are still generated and saved

## Feature 1b: CV Only Mode

### Purpose
Allow users to perform only cross-validation to generate out-of-fold (OOF) predictions without the expensive final model training step. This is useful for:
- Generating OOF predictions specifically for meta-learner training
- Quick iteration when you only need training set predictions
- Reducing training time when final models aren't needed

### Usage
Add the `--cv_only` flag to any training script:

```bash
# Generate only OOF predictions in DRE standard training
python dre_standard.py --datatypes CNV --cv_only --verbose

# Generate only OOF predictions in DRE reuploading training
python dre_relupload.py --datatypes CNV --cv_only --verbose

# Generate only OOF predictions in CFE standard training
python cfe_standard.py --datatypes CNV --cv_only --verbose

# Generate only OOF predictions in CFE reuploading training
python cfe_relupload.py --datatypes CNV --cv_only --verbose
```

### Affected Scripts
- `dre_standard.py`
- `dre_relupload.py`
- `cfe_standard.py`
- `cfe_relupload.py`

### Behavior
When `--cv_only` is used:
- Cross-validation is performed and OOF predictions are generated
- Final model training on the full training set is skipped
- Test predictions are not generated
- Final model artifacts (pipeline, scaler, etc.) are not saved
- Training time is significantly reduced (approximately 3x faster as final training is skipped)
- **Note:** `--cv_only` and `--skip_cross_validation` are mutually exclusive and cannot be used together

## Feature 2: Best Weight Step Tracking

### Purpose
Provide transparency about when the best model weights were obtained during training. This helps with:
- Understanding model convergence behavior
- Debugging training issues
- Optimizing training duration
- Validating checkpoint saves

### Implementation
All quantum classifier models now track and log:
- The training step at which the lowest loss was achieved
- The best loss value
- This information is saved in checkpoint files (`best_weights.joblib`)

### Output
After training completes, you'll see log messages like:

```
[QML Training] Loaded best weights from step 42 with loss: 0.3456
- Best weights were obtained at step 42 with loss: 0.3456
```

### Model Classes Updated
- `MulticlassQuantumClassifierDR`
- `MulticlassQuantumClassifierDataReuploadingDR`
- `ConditionalMulticlassQuantumClassifierFS`
- `ConditionalMulticlassQuantumClassifierDataReuploadingFS`

### Checkpoint File Format
The `best_weights.joblib` file now includes:
```python
{
    'weights': <weight_arrays>,
    'loss': <best_loss_value>,
    'step': <step_number>  # NEW
}
```

## Examples

### Example 1: Fast Training Without Cross-Validation
```bash
python dre_standard.py \
    --datatypes CNV GeneExpr \
    --skip_cross_validation \
    --steps 50 \
    --verbose
```

### Example 2: Generate Only OOF Predictions for Meta-Learner
```bash
python dre_standard.py \
    --datatypes CNV GeneExpr \
    --cv_only \
    --steps 50 \
    --verbose
```
This will:
- Perform 3-fold cross-validation
- Generate and save OOF predictions
- Skip final model training (saving ~33% of total time)

### Example 3: Time-Based Training with Step Tracking
```bash
python dre_standard.py \
    --datatypes CNV \
    --max_training_time 1.0 \
    --checkpoint_frequency 10 \
    --verbose
```
This will:
- Train for up to 1 hour
- Save checkpoints every 10 steps
- Log the best weight step at the end

### Example 4: Quick Metalearner Training
```bash
python metalearner.py \
    --preds_dir final_ensemble_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --skip_cross_validation \
    --override_steps 100 \
    --verbose
```

## Benefits

### Time Savings
- Skipping 3-fold cross-validation reduces training time by ~66% (use `--skip_cross_validation`)
- Performing only cross-validation reduces training time by ~33% (use `--cv_only`)
- Useful for rapid prototyping and debugging

### Transparency
- Best weight step information helps understand convergence
- Useful for debugging and optimization
- Validates that checkpointing is working correctly

### Flexibility
- Choose between full validation (with CV) or fast training (without CV)
- No changes to existing workflows - backward compatible

## Notes

1. **Best Weights**: Models always use the weights from the step with the lowest training loss, regardless of whether cross-validation is used.

2. **Checkpoint Files**: When using time-based training (`--max_training_time`), best weights are automatically saved to checkpoint files with step information.

3. **Backward Compatibility**: All existing training scripts continue to work without modification. Cross-validation is performed by default unless explicitly skipped.

4. **Output Files**: 
   - When `--skip_cross_validation` is used, OOF prediction files (`train_oof_preds_*.csv`) are not generated, but all other outputs (final model, test predictions, confusion matrices) are created normally.
   - When `--cv_only` is used, only OOF prediction files (`train_oof_preds_*.csv`) are generated. Final models, test predictions, and confusion matrices are not created.

5. **Mutually Exclusive Flags**: The `--skip_cross_validation` and `--cv_only` flags are mutually exclusive. Using both will result in an error.
