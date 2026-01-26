# Team Task Assignment - Contrastive Encoder Training

## Prerequisites

**Before starting, complete the Kaggle environment setup:** [KAGGLE_SETUP.md](KAGGLE_SETUP.md)

---

## Overview

Each team member will train a different variant of the contrastive encoder to compare performance. All experiments use the same dataset but different configurations.

**Dataset:** `gbm-lgg-balanced-xgb-reduced` (5 modalities, SNV skipped initially)

**Kaggle Input Path:**
```
/kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced
```

**Shared Settings:**
- `--data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced`
- `--skip_modalities SNV` (skipping SNV for initial training)
- `--use_cross_modal` (always enabled for multi-omics)
- `--checkpoint_interval 10`
- `--seed 42` (for reproducibility)
- `--max_grad_norm 1.0`

---

## Team Member 1: Approach A (Memory-Optimized, embed_dim=64)

**Focus:** Smallest embedding dimension for resource-constrained environments

```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_64dim \
    --embed_dim 64 \
    --projection_dim 32 \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member1_64dim \
    --device cpu
```

**After Training - Extract Features:**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_64dim/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_64dim \
    --device cpu
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_64dim/encoders/` - Trained encoder weights
- `/kaggle/working/pretrained_encoders_64dim/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_64dim/training_metrics.json` - Training metrics
- `/kaggle/working/pretrained_features_64dim/` - Extracted 64-dim features per modality

---

## Team Member 2: Approach B (Standard, embed_dim=256)

**Focus:** Largest embedding dimension for maximum expressiveness

```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_256dim \
    --embed_dim 256 \
    --projection_dim 128 \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member2_256dim \
    --device cpu
```

**Note:** Use `--batch_size 16` to avoid OOM errors with larger embedding dimension.

**After Training - Extract Features:**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_256dim/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_256dim \
    --device cpu
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_256dim/encoders/` - Trained encoder weights
- `/kaggle/working/pretrained_encoders_256dim/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_256dim/training_metrics.json` - Training metrics
- `/kaggle/working/pretrained_features_256dim/` - Extracted 256-dim features per modality

---

## Team Member 3: Approach C (Recommended, embed_dim=128)

**Focus:** Balanced configuration - recommended starting point

```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_128dim \
    --embed_dim 128 \
    --projection_dim 64 \
    --batch_size 32 \
    --num_epochs 150 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member3_128dim \
    --device cpu
```

**After Training - Extract Features:**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_128dim/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_128dim \
    --device cpu
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_128dim/encoders/` - Trained encoder weights
- `/kaggle/working/pretrained_encoders_128dim/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_128dim/training_metrics.json` - Training metrics
- `/kaggle/working/pretrained_features_128dim/` - Extracted 128-dim features per modality

---

## Team Member 4: Temperature Ablation (embed_dim=128, temperature=0.5)

**Focus:** Compare different temperature settings (softer negative mining)

```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_128dim_temp05 \
    --embed_dim 128 \
    --projection_dim 64 \
    --batch_size 32 \
    --num_epochs 150 \
    --lr 1e-3 \
    --temperature 0.5 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member4_128dim_temp05 \
    --device cpu
```

**After Training - Extract Features:**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_128dim_temp05/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_128dim_temp05 \
    --device cpu
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_128dim_temp05/encoders/` - Trained encoder weights
- `/kaggle/working/pretrained_encoders_128dim_temp05/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_128dim_temp05/training_metrics.json` - Training metrics
- `/kaggle/working/pretrained_features_128dim_temp05/` - Extracted 128-dim features per modality

---

## Comparison After Training

### Method 1: Compare Training Metrics

```bash
# View final loss for each variant (run from /kaggle/working)
for dir in /kaggle/working/pretrained_encoders_*; do
    echo "=== $dir ==="
    cat "$dir/training_metrics.json" | python -c "import json,sys; d=json.load(sys.stdin); print(f'Final Loss: {d[\"epoch_losses\"][-1]:.4f}, Best Loss: {d.get(\"best_loss\", \"N/A\")}')"
done
```

### Method 2: Compare on W&B Dashboard

All runs are logged to the same W&B project (`contrastive-team`), so you can compare:
- Training loss curves
- Final loss values
- Training time

### Method 3: Downstream QML Evaluation

After all team members complete training, evaluate each encoder on QML classification:

```bash
# Example: Evaluate Member 3's encoder (128-dim) on GeneExpr
python dre_standard.py \
    --datatypes GeneExpr \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/working/pretrained_features_128dim \
    --n_qbits 14 --n_layers 4 --steps 200 \
    --use_wandb --wandb_project qml-comparison --wandb_run_name geneexpr_128dim \
    --verbose
```

---

## Summary Table

| Member | Variant | embed_dim | temperature | batch_size | epochs | Output Dir |
|--------|---------|-----------|-------------|------------|--------|------------|
| 1 | Memory-Optimized | 64 | 0.07 | 64 | 100 | `/kaggle/working/pretrained_encoders_64dim` |
| 2 | Standard | 256 | 0.07 | 16 | 100 | `/kaggle/working/pretrained_encoders_256dim` |
| 3 | Recommended | 128 | 0.07 | 32 | 150 | `/kaggle/working/pretrained_encoders_128dim` |
| 4 | Temperature Ablation | 128 | 0.5 | 32 | 150 | `/kaggle/working/pretrained_encoders_128dim_temp05` |

**Modalities Trained:** GeneExpr, miRNA, Meth, CNV, Prot (SNV skipped)

---

## Next Steps (After Contrastive Training)

1. **Compare encoder performance** using training metrics and downstream QML evaluation
2. **Select best encoder** based on:
   - Training loss (lower is better)
   - Downstream QML accuracy
   - Memory/time constraints
3. **Proceed to full pipeline** with selected encoder:
   - Train QML base learners for all modalities (GeneExpr, miRNA, Meth, CNV, Prot)
   - Train meta-learner
   - (Optional) Train transformer fusion model

See [RESOURCE_OPTIMIZED_WORKFLOW.md](RESOURCE_OPTIMIZED_WORKFLOW.md) for the complete pipeline.

---

## Kaggle Tips

1. **Save outputs frequently** - Kaggle has a 12-hour runtime limit
2. **Use GPU if available** - Change `--device cpu` to `--device cuda`
3. **Download trained models** - From `/kaggle/working/` before session ends
4. **Commit notebook** - To save outputs permanently as a Kaggle dataset
