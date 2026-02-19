# Performance Extensions Examples

This directory contains example scripts demonstrating the performance enhancement strategies described in [PERFORMANCE_EXTENSIONS.md](../PERFORMANCE_EXTENSIONS.md).

> **📚 Complete Integration Guide:** See [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) for:
> - When to use each approach (decision tree based on dataset size)
> - Handling class imbalance and small datasets
> - Step-by-step integration with QML pipeline
> - Complete workflows with expected results
> - Troubleshooting common issues

## Overview

The performance extensions implement two complementary approaches to improve quantum multimodal cancer classification:

1. **Option 2: Self-Supervised Contrastive Pretraining** (`pretrain_contrastive.py`)
   - **Best for:** 100-500 samples, class imbalance, unlabeled data available
   - **Benefit:** Learn from unlimited unlabeled data, helps minority classes
   
2. **Option 1: Multimodal Transformer Fusion** (`train_transformer_fusion.py`)
   - **Best for:** 500+ samples, missing modalities common (>20%)
   - **Benefit:** Cross-modal attention, native missing modality handling

These can be used independently, with QML, or combined for maximum performance improvement.

## Quick Decision Guide

| Your Situation | Recommended Approach | See Section |
|----------------|---------------------|-------------|
| < 100 samples | QML Only (no extensions) | [Main README](../README.md) |
| 100-500 samples, imbalanced | Contrastive → QML | [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) |
| 500+ samples, missing modalities | Transformer Fusion | Examples below |
| 1000+ samples, GPU available | Full Hybrid (Contrastive → QML + Transformer → Meta-QML) | [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) |

### Understanding Embedding Dimensions

**Key Concept**: The encoders transform variable-dimensional input data into fixed-dimensional embeddings.

**Input Dimensions** (modality-specific, examples):
- Gene Expression (GeneExpr): 500-20,000 features
- miRNA: 200-1,000 features
- Methylation (Meth): 1,000-27,000 features
- Copy Number Variation (CNV): 100-1,000 features
- Protein (Prot): 100-500 features
- Mutation (SNV): 50-500 features

**Output Dimensions** (configurable via `--embed_dim`):
- Default: 256 dimensions
- Can be: 64, 128, 256, 384, 512, or any value
- All modalities share the same output dimension
- Larger = more expressive, slower; Smaller = faster, less expressive

**Example**:
```python
# If GeneExpr has 5000 features and Prot has 200 features:
GeneExpr: (batch, 5000) → Encoder → (batch, 256)
Prot:    (batch, 200)  → Encoder → (batch, 256)

# Both end up in the same 256-dimensional embedding space
# This allows cross-modal comparison and fusion
```

**Why 256?** 
- Balance between expressiveness and computational efficiency
- Common in deep learning (BERT uses 768, ResNet uses 512, SimCLR uses 128-2048)
- Works well on consumer GPUs (4-8GB VRAM)
- Empirically validated for multi-omics data

## Quick Start

### 1. Contrastive Pretraining (Option 2)

Pretrain encoders on unlabeled data using contrastive learning:

```bash
# Basic usage
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100

# With cross-modal contrastive learning (recommended for multi-omics)
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100 \
    --use_cross_modal \
    --temperature 0.07

# Skip certain modalities and use mean imputation
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --skip_modalities SNV Prot \
    --impute_strategy mean

# On GPU with larger batch size
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100 \
    --batch_size 64 \
    --device cuda
```

**Output Structure:**
```
pretrained_models/contrastive/
├── best_model.pt                    # Combined model (all modalities)
├── contrastive_epoch_*.pt           # Periodic checkpoints
├── encoders/                        # Per-modality encoders
│   ├── mRNA_encoder.pt             
│   ├── miRNA_encoder.pt
│   └── ...
├── projections/                     # Projection heads
│   └── ...
├── training_metrics.json            # Loss statistics
└── loss_curve.png                   # Training visualization
```

**Best Model Selection:** Based on **contrastive loss** (lower = better). Lower loss indicates better separation between positive/negative pairs in embedding space.

**Per-Modality Encoder Files Contain:**
- `encoder_state_dict`: Model weights
- `input_dim`: Input feature dimension
- `embed_dim`: Output embedding dimension
- `encoder_type`: 'mlp' or 'transformer'
- `epoch`: Training epoch when saved
- `loss`: Contrastive loss when saved

**Note:** Training behavior is unchanged. Checkpoint saving is enhanced to support loading individual modality encoders.

### 2. Transformer Fusion Training (Option 1)

Train multimodal classifier with cross-modal attention:

```bash
# Training from scratch
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models \
    --num_epochs 50 \
    --num_layers 4 \
    --num_heads 8

# Using pretrained encoders (Combined Approach)
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models_pretrained \
    --pretrained_encoders_dir pretrained_models/contrastive/encoders \
    --num_epochs 50 \
    --num_layers 4 \
    --num_heads 8

# Linear probing (freeze encoders, train only classifier)
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models_linear_probe \
    --pretrained_encoders_dir pretrained_models/contrastive/encoders \
    --freeze_encoders \
    --num_epochs 30
```

**Output:**
- Best model saved to `transformer_models/best_model.pt`
- Config saved to `transformer_models/config.json`
- Training history saved to `transformer_models/training_history.json`
  - Includes comprehensive final metrics: accuracy, precision, recall, F1, specificity (macro/weighted)
- Test metrics saved to `transformer_models/test_metrics.json`
- Confusion matrix saved to `transformer_models/confusion_matrix.csv`
- Normalized confusion matrix saved to `transformer_models/confusion_matrix_normalized.csv`

## Complete Workflow

### 3. Extract Pretrained Features

Extract embeddings from pretrained encoders for use in QML pipeline:

```bash
# Extract embeddings for QML input
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_models/contrastive/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features

# On GPU with larger batch
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_models/contrastive/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features \
    --batch_size 512 \
    --device cuda
```

**Output:**
- `pretrained_features/{modality}_embeddings.npy` - Embeddings per modality
- `pretrained_features/labels.npy` - Class labels
- `pretrained_features/extraction_metadata.json` - Configuration

### 4. Extract Transformer Features

Extract transformer predictions/features for QML meta-learner:

```bash
# Extract in CSV format for direct use with metalearner.py
python examples/extract_transformer_features.py \
    --model_dir transformer_models \
    --data_dir final_processed_datasets \
    --output_dir transformer_predictions \
    --output_format csv

# Extract all feature types in both formats
python examples/extract_transformer_features.py \
    --model_dir transformer_models \
    --data_dir final_processed_datasets \
    --output_dir transformer_predictions \
    --extract_type all \
    --output_format both
```

**Output (CSV format - metalearner compatible):**
- `transformer_predictions/train_oof_preds_Transformer.csv` - Training predictions
- `transformer_predictions/test_preds_Transformer.csv` - Test predictions

**Output (NPY format - for other uses):**
- `transformer_predictions/transformer_probabilities.npy` - Class probabilities
- `transformer_predictions/transformer_predictions.npy` - Class predictions
- `transformer_predictions/labels.npy` - Ground truth labels
- `transformer_predictions/extraction_metadata.json` - Configuration

---

## Complete Workflows

### Combined Approach (Best Performance)

Combining QML base learners AND transformer fusion yields the best results:

```bash
# Step 1: Pretrain encoders with contrastive learning
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100 \
    --use_cross_modal \
    --batch_size 64

# Step 2a: Extract pretrained features for QML base learners
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_models/contrastive/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features

# Step 2b: Train QML base learners on each modality
for modality in GeneExpr miRNA Meth CNV Prot SNV; do
    OUTPUT_DIR=base_learner_outputs python dre_standard.py \
        --use_pretrained_features \
        --pretrained_features_dir pretrained_features \
        --datatypes $modality
done

# Step 3: Train transformer fusion with pretrained encoders
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models \
    --pretrained_encoders_dir pretrained_models/contrastive/encoders \
    --num_epochs 50 \
    --num_layers 4 \
    --num_heads 8 \
    --lr 1e-4

# Step 4: Extract transformer predictions
python examples/extract_transformer_features.py \
    --model_dir transformer_models \
    --data_dir final_processed_datasets \
    --output_dir transformer_predictions \
    --output_format csv

# Step 5: Train QML meta-learner on BOTH QML base learners and transformer
python metalearner.py \
    --preds_dir base_learner_outputs transformer_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train
```

## Command-Line Arguments

### pretrain_contrastive.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `final_processed_datasets` | Directory with parquet data files |
| `--output_dir` | `pretrained_models/contrastive` | Output directory |
| `--encoder_type` | `mlp` | Encoder type: `mlp` (fast) or `transformer` (handles NaN natively) |
| `--embed_dim` | `256` | Embedding dimension |
| `--projection_dim` | `128` | Projection dimension for contrastive loss |
| `--transformer_d_model` | `64` | Transformer model dimension (only for `--encoder_type transformer`) |
| `--transformer_num_heads` | `4` | Transformer attention heads (only for `--encoder_type transformer`) |
| `--transformer_num_layers` | `2` | Transformer encoder layers (only for `--encoder_type transformer`) |
| `--batch_size` | `32` | Batch size (-1 for full batch) |
| `--full_batch` | `False` | Use full batch gradient descent (entire dataset per update) |
| `--num_epochs` | `100` | Number of epochs |
| `--lr` | `1e-3` | Learning rate |
| `--temperature` | `0.5` | Temperature for NT-Xent loss |
| `--use_cross_modal` | `False` | Use cross-modal contrastive loss |
| `--impute_strategy` | auto | Strategy for handling NaN values: `none`, `median`, `mean`, `zero`, or `drop`. **Auto-selects**: `none` for transformer, `median` for MLP |
| `--skip_modalities` | `None` | List of modalities to skip (e.g., `SNV Prot`) |
| `--seed` | `42` | Random seed for reproducibility |
| `--max_grad_norm` | `1.0` | Maximum gradient norm for clipping (0 to disable) |
| `--warmup_epochs` | `10` | Number of epochs for learning rate warmup (prevents gradient explosion) |
| `--checkpoint_interval` | `10` | Save checkpoint every N epochs |
| `--keep_last_n_checkpoints` | `3` | Keep only last N checkpoints + best (0 = keep all) |
| `--resume` | `None` | Path to checkpoint file to resume training from |
| `--device` | `cuda` if available | Device (`cuda` or `cpu`) |

**Ignored Columns:** The encoder automatically excludes metadata columns: `class`, `case_id`

**Example - Transformer encoder with native NaN handling:**
```bash
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --encoder_type transformer \
    --impute_strategy none \
    --num_epochs 100
```

**Example - Full batch gradient descent:**
```bash
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --full_batch \
    --num_epochs 500 \
    --lr 0.01
```

**Example - Skip modalities and use median imputation (for MLP encoder):**
```bash
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --encoder_type mlp \
    --skip_modalities SNV Prot \
    --impute_strategy median
```

### train_transformer_fusion.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `final_processed_datasets` | Directory with parquet data files |
| `--output_dir` | `transformer_models` | Output directory |
| `--pretrained_encoders_dir` | `None` | Directory with pretrained encoders |
| `--embed_dim` | `256` | Embedding dimension |
| `--num_heads` | `8` | Number of attention heads |
| `--num_layers` | `4` | Number of transformer layers |
| `--batch_size` | `32` | Batch size |
| `--num_epochs` | `50` | Number of epochs |
| `--lr` | `1e-3` | Learning rate |
| `--seed` | `42` | Random seed for reproducibility |
| `--max_grad_norm` | `1.0` | Maximum gradient norm for clipping (0 to disable) |
| `--freeze_encoders` | `False` | Freeze pretrained encoders |
| `--test_size` | `0.2` | Test set size (fraction) |
| `--resume` | `None` | Path to checkpoint file to resume training from |
| `--device` | `cuda` if available | Device (`cuda` or `cpu`) |

### extract_pretrained_features.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder_dir` | *required* | Directory containing pretrained encoders |
| `--data_dir` | *required* | Directory with parquet data files |
| `--output_dir` | *required* | Output directory for extracted features |
| `--batch_size` | `256` | Batch size for encoding |
| `--device` | `auto` | Device (`auto`, `cpu`, `cuda`) |

### extract_transformer_features.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | *required* | Directory containing trained transformer model |
| `--data_dir` | *required* | Directory with parquet data files |
| `--output_dir` | *required* | Output directory for extracted features |
| `--extract_type` | `both` | Feature type (`logits`, `probabilities`, `embeddings`, `both`, `all`) |
| `--output_format` | `both` | Output format: `npy`, `csv` (metalearner compatible), or `both` |
| `--batch_size` | `64` | Batch size for inference |
| `--device` | `auto` | Device (`auto`, `cpu`, `cuda`) |

## Expected Performance Improvements

Based on similar multimodal medical AI research:

### Option 2 Only (Contrastive Pretraining)
- **Accuracy improvement**: +5-15% over baseline
- **Data efficiency**: Can achieve similar performance with 50-70% less labeled data
- **Convergence**: 2-3× faster training

### Option 1 Only (Transformer Fusion)
- **Accuracy improvement**: +3-8% over baseline
- **Missing modality handling**: 10-20% better performance when modalities are missing
- **Interpretability**: Attention weights show which modalities contribute to decisions

### Combined (Option 1 + Option 2)
- **Accuracy improvement**: +10-20% over baseline
- **Data efficiency**: Best performance with 60% of labeled data
- **Robustness**: Highly stable, better generalization
- **Missing modality performance**: 15-25% better than baseline

## Data Requirements

### Input Format

The scripts expect parquet files in the data directory with the following naming convention:
- `data_GeneExpr_.parquet` - Gene expression data
- `data_miRNA_.parquet` - miRNA data
- `data_Meth_.parquet` - Methylation data
- `data_CNV_.parquet` - Copy number variation data
- `data_Prot_.parquet` - Protein data
- `data_SNV_.parquet` - Mutation data

Each parquet file should have:
- Feature columns (numeric)
- `class` column (labels, required for supervised training)
- `case_id` column (sample identifier)

### Missing Modalities

Missing data files are handled gracefully:
- Contrastive pretraining: Missing modalities use learnable missing tokens (same as Transformer Fusion)
- Transformer fusion: Missing modalities are replaced with learnable tokens

**Missing Modality Architecture (both use the same approach):**
```python
# Each modality encoder has a learnable missing token
class ModalityEncoder(nn.Module):
    def __init__(self, ...):
        self.missing_token = nn.Parameter(torch.randn(1, embed_dim))
    
    def forward(self, x, is_missing=False):
        if is_missing or x is None:
            return self.missing_token.expand(batch_size, -1)
        return self.encoder(x)
```

## Monitoring Training

### Contrastive Pretraining

Monitor the contrastive loss during training. A decreasing loss indicates the model is learning to:
1. Make augmented views of the same sample more similar
2. Make different samples more dissimilar
3. Align related modalities (if using cross-modal loss)

Typical loss values:
- Initial: 2-4
- After 50 epochs: 0.5-1.5
- After 100 epochs: 0.3-0.8

### Transformer Fusion

Monitor both training and validation metrics:
- **Training accuracy** should steadily increase
- **Validation accuracy** should increase but may plateau
- Watch for overfitting (validation accuracy decreasing while training increases)

Early stopping: The script automatically saves the best model based on validation accuracy.

## Tips for Best Results

1. **Start with contrastive pretraining**: Especially beneficial when labeled data is limited
2. **Use cross-modal loss**: Helps the model learn relationships between modalities
3. **Adjust learning rates**: Use lower LR for fine-tuning pretrained models (e.g., 1e-4 instead of 1e-3)
4. **Experiment with architecture**: Try different numbers of transformer layers and attention heads
5. **Monitor validation**: Use early stopping to prevent overfitting
6. **GPU recommended**: Training is much faster on GPU, especially for transformer models

## Troubleshooting

### NaN Loss or Gradient Explosion

If you see `Warning: NaN/Inf gradient detected` or `Avg Loss: nan`:

**Common Causes:**
1. Temperature too low (< 0.05) with transformer encoder
2. Learning rate too high for early training
3. Data contains extreme values

**Solutions:**
```bash
# Use recommended temperature (0.07)
--temperature 0.07

# Increase warmup epochs
--warmup_epochs 20

# Reduce learning rate
--lr 5e-4

# Ensure gradient clipping is enabled
--max_grad_norm 1.0
```

**Note:** The code automatically skips batches with NaN gradients, but if most batches fail, the model cannot learn effectively.

### Out of Memory

If you encounter OOM errors:
```bash
# Reduce batch size
--batch_size 16

# Reduce model size
--embed_dim 128 --num_layers 2
```

### Poor Convergence

If training doesn't converge:
```bash
# Adjust learning rate
--lr 5e-4

# Increase training epochs
--num_epochs 100
```

### Missing Data Files

Ensure your data directory contains the required parquet files with correct naming.

## Frequently Asked Questions (FAQ)

### Q: What input dimensions does the contrastive encoder accept?

**A:** The encoder accepts **any input dimension** for each modality. Each modality can have a different number of features:
- GeneExpr might have 5,000 features
- Protein might have only 200 features
- Methylation might have 10,000 features

The encoder's job is to map from these variable dimensions to a **fixed embedding dimension** (default 256).

### Q: Can the input dimension be less than 256?

**A:** Yes! Input dimension is completely independent of the embedding dimension:
- Input with 50 features → Encoder → 256-dim embedding ✓
- Input with 5000 features → Encoder → 256-dim embedding ✓

The encoder uses deep layers to transform any input size to the target embedding size.

### Q: Can I change the embedding dimension (embed_dim)?

**A:** Absolutely! The embedding dimension is fully configurable:

```bash
# Smaller embeddings (faster, less memory)
python examples/pretrain_contrastive.py --embed_dim 128

# Default (recommended)
python examples/pretrain_contrastive.py --embed_dim 256

# Larger embeddings (more expressive, slower)
python examples/pretrain_contrastive.py --embed_dim 512
```

**Trade-offs:**
- **64-128**: Fast, low memory, may lose information
- **256-384**: Good balance (recommended)
- **512-1024**: Very expressive, slow, needs more data

### Q: Why is 256 the default embedding dimension?

**A:** Several reasons:

1. **Proven effective**: Common in deep learning research
   - BERT uses 768, ResNet uses 512, SimCLR uses 128-2048
   - Multi-omics papers typically use 128-512

2. **Good balance**: 
   - Large enough to capture complex biological patterns
   - Small enough to train efficiently on consumer GPUs
   - Provides ~256× compression for modalities with >256 features

3. **Computational efficiency**:
   - Transformer attention is O(n² × d) where d is embedding dim
   - 256 keeps this manageable while maintaining expressiveness

4. **Empirical validation**: Works well across various multi-omics datasets

### Q: Must embed_dim be divisible by num_heads for transformers?

**A:** Yes, when using transformer fusion:

```bash
# Valid combinations
--embed_dim 256 --num_heads 8   # 256/8 = 32 per head ✓
--embed_dim 384 --num_heads 8   # 384/8 = 48 per head ✓
--embed_dim 512 --num_heads 8   # 512/8 = 64 per head ✓

# Invalid combinations
--embed_dim 250 --num_heads 8   # 250/8 = 31.25 ✗
--embed_dim 300 --num_heads 7   # 300/7 = 42.86 ✗
```

Common valid pairs:
- embed_dim=128, num_heads=4 or 8
- embed_dim=256, num_heads=4, 8, or 16
- embed_dim=384, num_heads=6, 8, or 12
- embed_dim=512, num_heads=8 or 16

### Q: What is projection_dim and how is it different from embed_dim?

**A:** Two different dimensions serve different purposes:

**embed_dim (e.g., 256)**:
- The main representation dimension
- Kept after pretraining
- Used for downstream tasks
- Configurable and application-dependent

**projection_dim (e.g., 128)**:
- Used only during contrastive pretraining
- Projects embeddings to lower dimension for loss computation
- Discarded after pretraining
- Typically smaller than embed_dim

```python
# During pretraining:
Input → Encoder → Embedding (256-dim) → Projection Head → Projection (128-dim) → Loss

# After pretraining (for downstream tasks):
Input → Encoder → Embedding (256-dim) → [Use for classification]
         ↑                                  [Projection head discarded]
    Keep this
```

### Q: Can different modalities have different embed_dim values?

**A:** Not recommended for contrastive learning. All modalities should share the same embed_dim:

**Why?** Contrastive learning requires computing similarity between modalities in a shared space:
```python
# All modalities to same dimension - can compare ✓
GeneExpr: (batch, 5000) → (batch, 256)
Prot:    (batch, 200)  → (batch, 256)
# Can compute cross-modal similarity

# Different dimensions - can't compare ✗
GeneExpr: (batch, 5000) → (batch, 256)
Prot:    (batch, 200)  → (batch, 128)
# Can't compute cross-modal similarity between 256 and 128
```

### Q: How do I choose the right embed_dim for my data?

**A:** Consider these factors:

1. **Data complexity**: More modalities/classes → larger embed_dim
2. **Available data**: More samples → can support larger embed_dim
3. **Computational resources**: Limited GPU memory → smaller embed_dim
4. **Downstream task**: Complex tasks may benefit from larger embed_dim

**Recommended approach:**
```bash
# Start with default
--embed_dim 256

# If underfitting (low train accuracy):
--embed_dim 384  # or 512

# If overfitting (high train, low val accuracy):
--embed_dim 128  # or add regularization

# If out of memory:
--embed_dim 128  # and/or reduce batch_size
```

## Integration with Existing Pipeline

To integrate these performance extensions with the existing quantum pipeline:

1. Use pretrained encoders as feature extractors before quantum models
2. Replace or augment the meta-learner with the transformer fusion model
3. Combine quantum base learner predictions with transformer-fused features

See `PERFORMANCE_EXTENSIONS.md` for detailed integration strategies.

## References

For more details on the methods, see:
- `PERFORMANCE_EXTENSIONS.md` - Complete technical documentation
- `performance_extensions/` - Implementation modules
- `tests/test_*.py` - Unit tests demonstrating usage
