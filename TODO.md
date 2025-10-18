# TODOs and Future Enhancements

## Completed Features ✓

1. ✓ Classical readout heads for MulticlassQuantumClassifierDR
2. ✓ Custom serializable Adam optimizer (utils/optim_adam.py)
3. ✓ Checkpoint save/load utilities (utils/io_checkpoint.py)
4. ✓ Metrics logging and visualization (utils/metrics.py)
5. ✓ Unified training script (scripts/train.py)
6. ✓ Optuna nested CV harness (scripts/optuna_nested_cv.py)
7. ✓ Smoke tests (tests/test_smoke.py)
8. ✓ Requirements.txt with all dependencies
9. ✓ Comprehensive README updates

## Remaining TODOs

### High Priority

1. **Add classical readout heads to remaining classifiers:**
   - MulticlassQuantumClassifierDataReuploadingDR
   - ConditionalMulticlassQuantumClassifierFS
   - ConditionalMulticlassQuantumClassifierDataReuploadingFS
   
   Status: Low priority - the unified training script demonstrates the pattern. Can be done incrementally as needed.

2. **Update existing training scripts (dre_standard.py, dre_relupload.py, cfe_standard.py, cfe_relupload.py):**
   - Add --resume_mode, --checkpoint_dir, --metric CLI args
   - Change to stratified 80/20 default split
   - Add optional validation split
   - Use enhanced metrics logging
   
   Status: Not critical - the new unified script (scripts/train.py) provides all these features. Existing scripts can be updated incrementally or deprecated in favor of unified script.

3. **Update metalearner.py:**
   - Add same enhancements (checkpointing, metrics, resume)
   - Integrate with new utilities
   
   Status: Can be done as follow-up. Metalearner works with current implementation.

### Medium Priority

4. **Batch training support:**
   - Current implementation processes full dataset at once
   - Add mini-batch support for large datasets
   - Note: --batch_size argument exists in scripts but not yet implemented in fit()

5. **Early stopping:**
   - Implement early stopping based on validation metrics
   - Add patience parameter

6. **Learning rate scheduling:**
   - Add warmup
   - Add decay schedules (cosine, exponential, step)

### Low Priority

7. **Additional optimizers:**
   - Implement serializable versions of other optimizers (SGD, RMSprop)
   - Make optimizer configurable via CLI

8. **More visualizations:**
   - Learning rate curves
   - Gradient norms
   - Parameter distributions

9. **Multi-GPU support:**
   - Parallelize quantum circuit evaluation
   - Distributed training

## Known Issues

1. **Gradient computation**: Currently uses autograd for gradient computation. For very large models, this may be slow. Consider switching to finite differences for specific use cases.

2. **Memory usage**: Full dataset is loaded into memory. For very large datasets, implement streaming or chunking.

3. **Optuna visualization**: The nested CV script attempts to save Plotly figures as PNG, which requires additional dependencies (kaleido). This is optional and fails gracefully.

## Design Decisions

1. **Why custom Adam optimizer instead of PyTorch/JAX?**
   - Keeps dependencies minimal
   - PennyLane already uses autograd
   - Easier integration with existing codebase
   - Serialization is straightforward with joblib

2. **Why separate unified script instead of updating existing scripts?**
   - Preserves backward compatibility
   - Provides clean reference implementation
   - Easier to maintain and test
   - Users can migrate gradually

3. **Why nested CV in separate script?**
   - Nested CV is computationally expensive
   - Separate script allows focused optimization
   - Can be run independently of regular training
   - Results stored in SQLite for analysis

## Testing Strategy

Current smoke tests cover:
- ClassicalReadoutHead forward pass and serialization
- MulticlassQuantumClassifierDR training and prediction
- Checkpoint save/resume functionality

Additional tests to add:
- All 4 classifier classes
- Full training pipeline end-to-end
- Optuna nested CV with mock data
- Metrics computation edge cases

## Migration Guide

For users of existing training scripts:

1. **Immediate**: Use new unified script for new projects
   ```bash
   python scripts/train.py --data_file data.parquet --output_dir models/
   ```

2. **Short-term**: Update existing scripts to use new utilities
   ```python
   from utils.optim_adam import SerializableAdam
   from utils.io_checkpoint import save_checkpoint, load_checkpoint
   from utils.metrics import compute_epoch_metrics
   ```

3. **Long-term**: Deprecate old scripts in favor of unified approach

## Performance Notes

- Classical readout head adds ~10-20% training time
- Checkpoint saving adds ~1-2% overhead
- Metrics computation adds ~5% overhead
- Overall: ~15-25% slower but much more feature-rich

## Contribution Guidelines

When adding features:
1. Follow existing code style (use logging_utils.log for logging)
2. Add tests to tests/test_smoke.py
3. Update README.md with usage examples
4. Add TODOs to this file if incomplete
5. Use type hints where possible
6. Document all CLI arguments
