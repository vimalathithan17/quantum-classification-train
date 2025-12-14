# Performance Extensions Examples

This directory contains example scripts demonstrating the performance enhancement strategies described in `PERFORMANCE_EXTENSIONS.md`.

## Overview

The performance extensions implement two complementary approaches to improve quantum multimodal cancer classification:

1. **Option 2: Self-Supervised Contrastive Pretraining** (`pretrain_contrastive.py`)
2. **Option 1: Multimodal Transformer Fusion** (`train_transformer_fusion.py`)

These can be used independently or combined for maximum performance improvement.

## Quick Start

### 1. Contrastive Pretraining (Option 2)

Pretrain encoders on unlabeled data using contrastive learning:

```bash
# Basic usage
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100

# With cross-modal contrastive learning
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100 \
    --use_cross_modal \
    --temperature 0.5

# On GPU with larger batch size
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100 \
    --batch_size 64 \
    --device cuda
```

**Output:**
- Pretrained encoders saved to `pretrained_models/contrastive/encoders/`
- Training metrics saved to `pretrained_models/contrastive/training_metrics.json`
- Loss curve plot saved to `pretrained_models/contrastive/loss_curve.png`

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
- Training history saved to `transformer_models/training_history.json`

## Complete Workflow

### Combined Approach (Best Performance)

Combining both approaches typically yields the best results:

```bash
# Step 1: Pretrain encoders with contrastive learning
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_models/contrastive \
    --num_epochs 100 \
    --use_cross_modal \
    --batch_size 64

# Step 2: Fine-tune with transformer fusion
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir final_models/combined \
    --pretrained_encoders_dir pretrained_models/contrastive/encoders \
    --num_epochs 50 \
    --num_layers 4 \
    --num_heads 8 \
    --lr 1e-4
```

## Command-Line Arguments

### pretrain_contrastive.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `final_processed_datasets` | Directory with parquet data files |
| `--output_dir` | `pretrained_models/contrastive` | Output directory |
| `--embed_dim` | `256` | Embedding dimension |
| `--projection_dim` | `128` | Projection dimension for contrastive loss |
| `--batch_size` | `32` | Batch size |
| `--num_epochs` | `100` | Number of epochs |
| `--lr` | `1e-3` | Learning rate |
| `--temperature` | `0.5` | Temperature for NT-Xent loss |
| `--use_cross_modal` | `False` | Use cross-modal contrastive loss |
| `--device` | `cuda` if available | Device (`cuda` or `cpu`) |

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
| `--freeze_encoders` | `False` | Freeze pretrained encoders |
| `--test_size` | `0.2` | Test set size (fraction) |
| `--device` | `cuda` if available | Device (`cuda` or `cpu`) |

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
- `data_GeneExp_.parquet` - Gene expression data
- `data_miRNA_.parquet` - miRNA data
- `data_Meth_.parquet` - Methylation data
- `data_CNV_.parquet` - Copy number variation data
- `data_Prot_.parquet` - Protein data
- `data_Mut_.parquet` - Mutation data

Each parquet file should have:
- Feature columns (numeric)
- `class` column (labels, required for supervised training)
- `split` column (optional, can be ignored)

### Missing Modalities

Missing data files are handled gracefully:
- Contrastive pretraining: Only available modalities are used
- Transformer fusion: Missing modalities are replaced with learnable tokens

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
