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
- `--warmup_epochs 20` (prevents early gradient explosion)

---

## Encoder Types

### MLP Encoder (Default)
- Standard 3-layer MLP encoder
- Requires imputation for NaN values (`--impute_strategy median`)
- Faster training, lower memory usage

### Transformer Encoder (New)
- Treats each feature as a token with self-attention
- Native NaN handling via [MASK] tokens (`--impute_strategy none`)
- Learns to infer missing values from context
- Better for data with missing features

**Key Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--encoder_type` | `mlp` or `transformer` | `mlp` |
| `--impute_strategy` | `median`, `mean`, `zero`, `none`, or auto | auto-selects based on encoder_type |
| `--warmup_epochs` | LR warmup epochs (prevents gradient explosion) | 10 |
| `--transformer_d_model` | Transformer hidden dimension | 64 |
| `--transformer_num_heads` | Number of attention heads | 4 |
| `--transformer_num_layers` | Number of transformer layers | 2 |

---

## Updated Output Structure

After training, models are saved **per-modality**:

```
pretrained_encoders_*/
├── metadata.json              # Training config and modality info
├── encoders/
│   ├── GeneExpr_encoder.pt    # Per-modality encoder weights + metadata
│   ├── Prot_encoder.pt
│   ├── miRNA_encoder.pt
│   ├── Meth_encoder.pt
│   └── CNV_encoder.pt
├── projections/
│   ├── GeneExpr_projection.pt # Per-modality projection heads
│   ├── Prot_projection.pt
│   └── ...
└── checkpoints/
    ├── best_model.pt          # Full model checkpoint
    └── epoch_*.pt             # Periodic checkpoints
```

---

## Team Member 1: MLP Encoder (embed_dim=64, Memory-Optimized)

**Focus:** Smallest embedding dimension with MLP encoder for resource-constrained environments

```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_mlp_64dim \
    --encoder_type mlp \
    --impute_strategy median \
    --embed_dim 64 \
    --projection_dim 32 \
    --batch_size 64 \
    --num_epochs 2500 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 20 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member1_mlp_64dim \
    --device cpu
```

**After Training - Extract Features:**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_mlp_64dim/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_mlp_64dim \
    --device cpu
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_mlp_64dim/encoders/` - Per-modality encoder weights
- `/kaggle/working/pretrained_encoders_mlp_64dim/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_mlp_64dim/metadata.json` - Training metadata
- `/kaggle/working/pretrained_features_mlp_64dim/` - Extracted 64-dim features per modality

---

## Team Member 2: MLP Encoder (embed_dim=256, High Capacity)

**Focus:** Largest embedding dimension with MLP encoder for maximum expressiveness

```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_mlp_256dim \
    --encoder_type mlp \
    --impute_strategy median \
    --embed_dim 256 \
    --projection_dim 128 \
    --batch_size 16 \
    --num_epochs 1000 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 20 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member2_mlp_256dim \
    --device cpu
```

**Note:** Use `--batch_size 16` to avoid OOM errors with larger embedding dimension.

**After Training - Extract Features:**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_mlp_256dim/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_mlp_256dim \
    --device cpu
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_mlp_256dim/encoders/` - Per-modality encoder weights
- `/kaggle/working/pretrained_encoders_mlp_256dim/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_mlp_256dim/metadata.json` - Training metadata
- `/kaggle/working/pretrained_features_mlp_256dim/` - Extracted 256-dim features per modality

---

## Team Member 3: Transformer Encoder (embed_dim=128, Recommended)

**Focus:** Transformer-based encoder with native NaN handling - recommended configuration

```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_transformer_128dim \
    --encoder_type transformer \
    --impute_strategy none \
    --embed_dim 128 \
    --projection_dim 64 \
    --transformer_d_model 64 \
    --transformer_num_heads 4 \
    --transformer_num_layers 2 \
    --batch_size 32 \
    --num_epochs 2500 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 20 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member3_transformer_128dim \
    --device cpu
```

**Key advantage:** No imputation needed! The transformer treats NaN values as [MASK] tokens and learns to infer them from context.

**After Training - Extract Features:**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_transformer_128dim/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_transformer_128dim \
    --device cpu
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_transformer_128dim/encoders/` - Per-modality encoder weights
- `/kaggle/working/pretrained_encoders_transformer_128dim/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_transformer_128dim/metadata.json` - Training metadata
- `/kaggle/working/pretrained_features_transformer_128dim/` - Extracted 128-dim features per modality

---

## Team Member 4: Transformer Encoder Temperature Ablation (embed_dim=256)

**Focus:** Compare different temperature settings with transformer encoder (256-dim for higher capacity)

> **⚠️ Numerical Stability Note:** Very low temperatures (< 0.07) can cause gradient explosion and NaN losses.
> The code now includes automatic NaN detection and batch skipping, but **temperature=0.07 is recommended** for stability.
> If you see NaN losses, try increasing temperature or reducing learning rate.

### Run A: Temperature = 0.07 (Stable, recommended)
```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_transformer_256dim_temp007 \
    --encoder_type transformer \
    --impute_strategy none \
    --embed_dim 256 \
    --projection_dim 128 \
    --transformer_d_model 64 \
    --transformer_num_heads 4 \
    --transformer_num_layers 2 \
    --batch_size 32 \
    --num_epochs 2500 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 20 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member4_transformer_256dim_temp007 \
    --device cuda
```

### Run B: Temperature = 0.05 (Sharper, may need NaN recovery)
```bash
python examples/pretrain_contrastive.py \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_encoders_transformer_256dim_temp005 \
    --encoder_type transformer \
    --impute_strategy none \
    --embed_dim 256 \
    --projection_dim 128 \
    --transformer_d_model 64 \
    --transformer_num_heads 4 \
    --transformer_num_layers 2 \
    --batch_size 32 \
    --num_epochs 2500 \
    --lr 5e-4 \
    --temperature 0.05 \
    --use_cross_modal \
    --skip_modalities SNV \
    --checkpoint_interval 10 \
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 20 \
    --use_wandb --wandb_project contrastive-team --wandb_run_name member4_transformer_256dim_temp005 \
    --device cuda
```

**Note:** Lower learning rate (5e-4 vs 1e-3) helps stability with lower temperature.

**After Training - Extract Features (for best run):**
```bash
python examples/extract_pretrained_features.py \
    --encoder_dir /kaggle/working/pretrained_encoders_transformer_256dim_temp007/encoders \
    --data_dir /kaggle/input/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/pretrained_features_transformer_256dim_temp007 \
    --device cuda
```

**Expected Output:**
- `/kaggle/working/pretrained_encoders_transformer_256dim_temp*/encoders/` - Per-modality encoder weights
- `/kaggle/working/pretrained_encoders_transformer_256dim_temp*/checkpoints/best_model.pt` - Best model checkpoint
- `/kaggle/working/pretrained_encoders_transformer_256dim_temp*/metadata.json` - Training metadata
- `/kaggle/working/pretrained_features_transformer_256dim_temp*/` - Extracted 256-dim features per modality

---

## Comparison After Training

### Method 1: Compare Training Metrics

```bash
# View final loss for each variant (run from /kaggle/working)
for dir in /kaggle/working/pretrained_encoders_*; do
    echo "=== $dir ==="
    if [ -f "$dir/metadata.json" ]; then
        cat "$dir/metadata.json" | python -c "import json,sys; d=json.load(sys.stdin); print(f'Encoder: {d.get(\"encoder_type\", \"mlp\")}, Embed: {d.get(\"embed_dim\", \"N/A\")}')"
    fi
done
```

### Method 2: Compare on W&B Dashboard

All runs are logged to the same W&B project (`contrastive-team`), so you can compare:
- Training loss curves (intra-modal and cross-modal components)
- Final loss values
- Training time
- Encoder type (MLP vs Transformer)

### Method 3: Downstream QML Evaluation

After all team members complete training, evaluate each encoder on QML classification:

```bash
# Example: Evaluate Member 3's transformer encoder (128-dim) on GeneExpr
python dre_standard.py \
    --datatypes GeneExpr \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/working/pretrained_features_transformer_128dim \
    --n_qbits 14 --n_layers 4 --steps 200 \
    --use_wandb --wandb_project qml-comparison --wandb_run_name geneexpr_transformer_128dim \
    --verbose
```

---

## Summary Table

| Member | Encoder | embed_dim | projection_dim | temperature | epochs | Output Dir |
|--------|---------|-----------|----------------|-------------|--------|------------|
| 1 | MLP | 64 | 32 | 0.07 | 2500 | `pretrained_encoders_mlp_64dim` |
| 2 | MLP | 256 | 128 | 0.07 | 1000 | `pretrained_encoders_mlp_256dim` |
| 3 | **Transformer** | 128 | 64 | 0.07 | 2500 | `pretrained_encoders_transformer_128dim` |
| 4a | **Transformer** | 256 | 128 | **0.07** | 2500 | `pretrained_encoders_transformer_256dim_temp007` |
| 4b | **Transformer** | 256 | 128 | 0.05 | 2500 | `pretrained_encoders_transformer_256dim_temp005` |

**Modalities Trained:** GeneExpr, miRNA, Meth, CNV, Prot (SNV skipped)

---

## Key Findings from Previous Experiments

| Run | Embed Dim | Temp | Loss | Notes |
|-----|-----------|------|------|-------|
| member3_128dim | 128 | 0.07 | **0.594** | ✓ Best overall, stable |
| member4_256dim_temp005 | 256 | 0.05 | 0.538 | Good but can cause NaN |
| member2_256dim | 256 | 0.07 | 0.651 | Stable |
| member1_64dim | 64 | 0.07 | 0.935 | - |
| member4_128dim_temp02 | 128 | 0.2 | 1.167 | Too high temp |
| member4_128dim_temp05 | 128 | 0.5 | 2.611 | Too high temp |

**Conclusion:** Temperature 0.07 is recommended for stability. Lower temperatures (0.02-0.05) can achieve better loss but risk NaN/gradient explosion.

---

## Next Steps (After Contrastive Training)

1. **Compare encoder performance** using training metrics and downstream QML evaluation
2. **Select best encoder** based on:
   - Training loss (lower is better)
   - Downstream QML accuracy
   - Memory/time constraints
   - **MLP vs Transformer**: Transformer handles missing data natively, MLP requires imputation
3. **Proceed to full pipeline** with selected encoder:
   - Train QML base learners for all modalities (GeneExpr, miRNA, Meth, CNV, Prot)
   - Train meta-learner
   - (Optional) Train transformer fusion model

See [RESOURCE_OPTIMIZED_WORKFLOW.md](RESOURCE_OPTIMIZED_WORKFLOW.md) for the complete pipeline.

---

## Loading Pretrained Encoders

### Load a Single Modality Encoder

```python
from performance_extensions.training_utils import load_single_modality_encoder

# Load GeneExpr encoder from per-modality checkpoint
encoder, metadata = load_single_modality_encoder(
    encoder_path="pretrained_encoders_transformer_128dim/encoders/GeneExpr_encoder.pt",
    device="cpu"
)

# Use the encoder
x = torch.randn(32, 5000)  # (batch, features)
embedding, valid_mask = encoder(x)  # Returns (batch, 128), (batch,)
```

### Load All Encoders for Downstream Tasks

```python
from performance_extensions.training_utils import load_pretrained_encoders

encoders, metadata = load_pretrained_encoders(
    checkpoint_dir="pretrained_encoders_transformer_128dim",
    device="cpu"
)

# encoders is a dict: {'GeneExpr': encoder1, 'Prot': encoder2, ...}
for modality, encoder in encoders.items():
    embedding, valid_mask = encoder(data[modality])
```

---

## Kaggle Tips

1. **Save outputs frequently** - Kaggle has a 12-hour runtime limit
2. **Use GPU if available** - Change `--device cpu` to `--device cuda`
3. **Download trained models** - From `/kaggle/working/` before session ends
4. **Commit notebook** - To save outputs permanently as a Kaggle dataset
5. **Transformer encoder** - More compute-intensive but handles NaN natively
