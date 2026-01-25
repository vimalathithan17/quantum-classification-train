# Resource-Optimized Workflow for Contrastive Encoder + QML Pipeline

## Overview

This guide addresses integrating the contrastive encoder with QML models under the following resource constraints:

| Constraint | Value |
|------------|-------|
| Available RAM | 30 GB |
| Maximum Qubits | 14 |
| Default Contrastive Encoder Output | 256 dimensions |

**The Core Challenge:** The contrastive encoder outputs 256-dimensional embeddings by default, but your QML circuit can only accept **14 input features** (one per qubit via `AngleEmbedding`). You need a dimensionality bridge between the 256-dim encoder output and the 14-qubit QML input.

**Key Design Decisions:**
- **Always use cross-modal contrastive learning** (`--use_cross_modal`) - it learns relationships between modalities of the same patient, which is essential for multi-omics data
- **UMAP reduction is built into the QML training scripts** (`--dim_reducer umap`) - no separate reduction step needed
- **All scripts support W&B logging** (`--use_wandb`) and checkpointing
- **All training scripts support `--resume`** for resuming interrupted training from checkpoints

---

## Prerequisites: W&B Setup (Recommended)

Before running the pipeline, set up Weights & Biases for experiment tracking:

### Quick Setup

```bash
# 1. Install wandb (already in requirements.txt)
pip install wandb

# 2. Login (one-time setup)
wandb login
# Enter your API key from https://wandb.ai/settings
```

### Understanding W&B Project/Run Names

| Argument | Purpose | Example |
|----------|---------|---------|
| `--wandb_project` | Groups related experiments | `qml_contrastive_128dim` |
| `--wandb_run_name` | Identifies a specific run | `GeneExpr_base_learner` |

**Naming Conventions:**
- **Project**: Use descriptive names like `qml_contrastive_64dim`, `qml_hybrid_pipeline`, `cancer_classification_v2`
- **Run**: Include script + datatype + key params: `dre_CNV_q14_l4`, `metalearner_final`, `pretrain_embed128`

See [README.md](README.md#-weights--biases-wb-setup) for complete W&B setup instructions.

---

## Three Approaches

| Approach | Encoder embed_dim | Reduction Method | Memory Usage | Information Retention |
|----------|-------------------|------------------|--------------|----------------------|
| A | 64 | Built-in UMAP 64→14 | Low (~8-12 GB) | Good |
| B | 256 (default) | Built-in UMAP 256→14 | High (~24-28 GB) | Moderate (large reduction) |
| C | 128 | Built-in UMAP 128→14 | Medium (~14-18 GB) | Best (balanced) |

---

## Approach A: Reduced Embedding Dimension (Best for Memory)

Train the contrastive encoder with a smaller `embed_dim` that is closer to your qubit count. This is the most memory-efficient approach.

### Step 1: Train Contrastive Encoder with embed_dim=64

```bash
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_64dim \
    --embed_dim 64 \
    --projection_dim 32 \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --checkpoint_interval 10 \
    --device cpu
```

**Parameter Explanation:**
- `--embed_dim 64`: Output dimension of each modality encoder (reduced from default 256)
- `--projection_dim 32`: Projection head dimension for contrastive loss (typically half of embed_dim)
- `--batch_size 64`: Safe batch size for 30GB RAM with embed_dim=64
- `--num_epochs 100`: Standard pretraining epochs
- `--lr 1e-3`: Learning rate for Adam optimizer
- `--temperature 0.07`: Lower temperature for harder negative mining (more discriminative features)
- `--use_cross_modal`: **Always enabled** - learns cross-modal relationships between modalities
- `--checkpoint_interval 10`: Save checkpoint every 10 epochs
- `--device cpu`: Use CPU (change to `cuda` if GPU available with sufficient VRAM)

### Step 2: Extract 64-dim Features

```bash
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_encoders_64dim/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features_64dim \
    --batch_size 256 \
    --device cpu
```

### Step 3: Train QML Base Learners with Built-in UMAP Reduction

The QML training scripts have built-in UMAP support via `--dim_reducer umap`. No separate reduction step needed:

```bash
for modality in GeneExpr miRNA Meth CNV Prot SNV; do
    python dre_standard.py \
        --datatypes $modality \
        --use_pretrained_features \
        --pretrained_features_dir pretrained_features_64dim \
        --n_qbits 14 \
        --n_layers 3 \
        --steps 150 \
        --checkpoint_frequency 25 \
        --keep_last_n 3 \
        --use_wandb \
        --wandb_project qml_contrastive_64dim \
        --wandb_run_name ${modality}_base_learner \
        --verbose
done
```

**Note:** UMAP is the default dimensionality reducer (`--dim_reducer umap`) when using pretrained features. It reduces embeddings from `embed_dim` to `n_qubits` dimensions while preserving manifold structure.

### Step 4: Train Meta-Learner

```bash
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --n_trials 30 \
    --use_wandb \
    --wandb_project qml_contrastive_64dim \
    --wandb_run_name metalearner_tuning \
    --verbose
```

**Why Approach A Works:**
- `embed_dim=64` is still expressive but uses ~4x less memory than 256
- Built-in UMAP (64→14) preserves manifold structure better than PCA
- Memory footprint: ~8-12 GB peak during training

---

## Approach B: Keep 256-dim Encoders with Built-in UMAP Reduction

Use the full 256-dimensional representation power with built-in UMAP reduction.

### Step 1: Train Contrastive Encoder with Default embed_dim=256

```bash
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_256dim \
    --embed_dim 256 \
    --projection_dim 128 \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --checkpoint_interval 10 \
    --device cpu
```

**Critical for 30GB RAM:** Use `--batch_size 16` to avoid OOM errors with embed_dim=256.

### Step 2: Extract 256-dim Features

```bash
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_encoders_256dim/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features_256dim \
    --batch_size 256 \
    --device cpu
```

### Step 3: Train QML Base Learners with Built-in UMAP (256→14)

```bash
for modality in GeneExpr miRNA Meth CNV Prot SNV; do
    python dre_standard.py \
        --datatypes $modality \
        --use_pretrained_features \
        --pretrained_features_dir pretrained_features_256dim \
        --n_qbits 14 \
        --n_layers 4 \
        --steps 200 \
        --checkpoint_frequency 25 \
        --keep_last_n 3 \
        --use_wandb \
        --wandb_project qml_contrastive_256dim \
        --wandb_run_name ${modality}_base_learner \
        --verbose
done
```

### Step 4: Train Meta-Learner

```bash
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --n_trials 30 \
    --use_wandb \
    --wandb_project qml_contrastive_256dim \
    --wandb_run_name metalearner_tuning \
    --verbose
```

**Why Approach B:**
- Preserves full 256-dim representation power during pretraining
- Built-in UMAP handles the 256→14 reduction automatically
- Memory footprint: ~24-28 GB peak (tight fit for 30GB)

---

## Approach C: Balanced Hierarchical Approach (Recommended)

Use a moderate embedding size (128) with built-in UMAP for optimal balance.

### Step 1: Train Contrastive Encoder with embed_dim=128

```bash
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_128dim \
    --embed_dim 128 \
    --projection_dim 64 \
    --batch_size 32 \
    --num_epochs 150 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --checkpoint_interval 10 \
    --device cpu
```

**Parameter Notes:**
- `--embed_dim 128`: Balanced between expressiveness and memory efficiency
- `--batch_size 32`: Comfortable fit for 30GB RAM
- `--num_epochs 150`: Slightly more epochs since smaller embedding needs more training
- `--temperature 0.07`: Lower temperature for more discriminative features
- `--use_cross_modal`: **Always enabled** - learns cross-modal relationships

### Step 2: Extract 128-dim Features

```bash
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_encoders_128dim/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features_128dim \
    --batch_size 256 \
    --device cpu
```

### Step 3: Train QML Base Learners with Built-in UMAP (128→14)

```bash
for modality in GeneExpr miRNA Meth CNV Prot SNV; do
    python dre_standard.py \
        --datatypes $modality \
        --use_pretrained_features \
        --pretrained_features_dir pretrained_features_128dim \
        --n_qbits 14 \
        --n_layers 4 \
        --steps 200 \
        --checkpoint_frequency 25 \
        --keep_last_n 3 \
        --use_wandb \
        --wandb_project qml_contrastive_128dim \
        --wandb_run_name ${modality}_base_learner \
        --verbose
done
```

### Step 4: Train Meta-Learner

```bash
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --n_trials 30 \
    --use_wandb \
    --wandb_project qml_contrastive_128dim \
    --wandb_run_name metalearner_tuning \
    --verbose
```

**Why Approach C is Recommended:**
- `embed_dim=128` balances expressiveness with memory efficiency
- Built-in UMAP preserves biological manifold structure
- 128→14 reduction is less aggressive than 256→14
- Memory footprint: ~14-18 GB peak (comfortable for 30GB)

---

## Hyperparameter Tuning

### Tuning QML Base Learners

Use `tune_models.py` to find optimal hyperparameters before training:

```bash
python tune_models.py \
    --datatype GeneExpr \
    --approach 1 \
    --qml_model standard \
    --dim_reducer umap \
    --n_trials 50 \
    --min_qbits 8 \
    --max_qbits 14 \
    --use_wandb \
    --wandb_project qml_tuning \
    --verbose
```

**Key Tuning Parameters:**
- `--approach 1`: DRE (Dimensionality Reduction + Encoding) approach
- `--qml_model standard|reuploading`: Circuit type to tune
- `--dim_reducer umap|pca`: Dimensionality reduction method
- `--n_trials 50`: Number of Optuna trials
- `--min_qbits 8 --max_qbits 14`: Qubit search range

**Note:** Cross-validation uses 3 folds (hardcoded in code for efficiency).

**Tune all modalities:**

```bash
for modality in GeneExpr miRNA Meth CNV Prot SNV; do
    python tune_models.py \
        --datatype $modality \
        --approach 1 \
        --qml_model standard \
        --dim_reducer umap \
        --n_trials 50 \
        --min_qbits 8 \
        --max_qbits 14 \
        --use_wandb \
        --wandb_project qml_tuning \
        --verbose
done
```

### Tuning Meta-Learner

The meta-learner has built-in Optuna tuning:

```bash
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode tune \
    --n_trials 50 \
    --minlayers 3 \
    --maxlayers 6 \
    --use_wandb \
    --wandb_project qml_metalearner_tuning \
    --verbose
```

---

## Transformer Fusion Training

For larger datasets (500+ samples), add transformer fusion to the pipeline.

### Train Transformer with Pretrained Encoders

```bash
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models \
    --pretrained_encoders_dir pretrained_encoders_128dim/encoders \
    --embed_dim 128 \
    --num_heads 8 \
    --num_layers 4 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cpu \
    --checkpoint_interval 10 \
    --keep_last_n 3 \
    --use_wandb \
    --wandb_project cancer-multi-omics \
    --wandb_run_name transformer_128d_8h_4l
```

**Freeze Encoders (Linear Probing):**

```bash
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models_frozen \
    --pretrained_encoders_dir pretrained_encoders_128dim/encoders \
    --freeze_encoders \
    --embed_dim 128 \
    --num_heads 8 \
    --num_layers 4 \
    --batch_size 32 \
    --num_epochs 30 \
    --lr 1e-3 \
    --device cpu \
    --checkpoint_interval 10 \
    --keep_last_n 3 \
    --use_wandb \
    --wandb_project cancer-multi-omics \
    --wandb_run_name transformer_frozen_128d
```
### Extract Transformer Features for Meta-Learner

```bash
python examples/extract_transformer_features.py \
    --model_dir transformer_models \
    --data_dir final_processed_datasets \
    --output_dir transformer_predictions \
    --output_format csv
```

### Combine QML and Transformer in Meta-Learner

```bash
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard transformer_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --n_trials 30 \
    --use_wandb \
    --wandb_project qml_hybrid_pipeline \
    --wandb_run_name metalearner_qml_transformer \
    --verbose
```

---

## Memory Usage Summary

| Configuration | Batch Size | Peak RAM (est.) | Status for 30GB |
|---------------|------------|-----------------|-----------------|
| embed_dim=256, batch=64 | 64 | ~32-40 GB | ❌ OOM |
| embed_dim=256, batch=16 | 16 | ~24-28 GB | ⚠️ Tight |
| embed_dim=128, batch=32 | 32 | ~14-18 GB | ✅ Comfortable |
| embed_dim=64, batch=64 | 64 | ~8-12 GB | ✅ Very safe |

---

## W&B Logging & Checkpointing Reference

### W&B Arguments (All Training Scripts)

| Script | W&B Arguments |
|--------|---------------|
| `dre_standard.py` | `--use_wandb --wandb_project <project> --wandb_run_name <name>` |
| `dre_relupload.py` | `--use_wandb --wandb_project <project> --wandb_run_name <name>` |
| `cfe_standard.py` | `--use_wandb --wandb_project <project> --wandb_run_name <name>` |
| `cfe_relupload.py` | `--use_wandb --wandb_project <project> --wandb_run_name <name>` |
| `tune_models.py` | `--use_wandb --wandb_project <project>` |
| `metalearner.py` | `--use_wandb --wandb_project <project> --wandb_run_name <name>` |
| `pretrain_contrastive.py` | `--use_wandb --wandb_project <project> --wandb_run_name <name>` |
| `train_transformer_fusion.py` | `--use_wandb --wandb_project <project> --wandb_run_name <name>` |

### Checkpointing Arguments

| Script | Checkpoint Arguments |
|--------|---------------------|
| `dre_*.py`, `cfe_*.py` | `--checkpoint_frequency 50 --keep_last_n 3 --checkpoint_fallback_dir <dir> --resume <best/latest/auto>` |
| `metalearner.py` | `--checkpoint_frequency 50 --keep_last_n 3 --resume <best/latest/auto>` |
| `pretrain_contrastive.py` | `--checkpoint_interval 10 --resume <path>` |
| `train_transformer_fusion.py` | `--checkpoint_interval 10 --keep_last_n 3 --resume <path>` |

### Example: Full Pipeline with W&B and Checkpointing

```bash
# 1. Pretrain contrastive encoder
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_128dim \
    --embed_dim 128 \
    --projection_dim 64 \
    --batch_size 32 \
    --num_epochs 150 \
    --lr 1e-3 \
    --temperature 0.07 \
    --use_cross_modal \
    --checkpoint_interval 10 \
    --device cpu

# 2. Extract features
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_encoders_128dim/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features_128dim \
    --device cpu

# 3. Tune hyperparameters (optional but recommended)
for modality in GeneExpr miRNA Meth CNV Prot SNV; do
    python tune_models.py \
        --datatype $modality \
        --approach 1 \
        --qml_model standard \
        --dim_reducer umap \
        --n_trials 30 \
        --min_qbits 8 \
        --max_qbits 14 \
        --use_wandb \
        --wandb_project qml_contrastive_tuning
done

# 4. Train QML base learners with tuned parameters
for modality in GeneExpr miRNA Meth CNV Prot SNV; do
    python dre_standard.py \
        --datatypes $modality \
        --use_pretrained_features \
        --pretrained_features_dir pretrained_features_128dim \
        --checkpoint_frequency 25 \
        --keep_last_n 3 \
        --use_wandb \
        --wandb_project qml_contrastive_128dim \
        --wandb_run_name ${modality}_base_learner \
        --verbose
done

# 5. Tune and train meta-learner
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode tune \
    --n_trials 30 \
    --use_wandb \
    --wandb_project qml_contrastive_128dim \
    --wandb_run_name metalearner_tuning \
    --verbose

# 6. Train final meta-learner with best params
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train \
    --use_wandb \
    --wandb_project qml_contrastive_128dim \
    --wandb_run_name metalearner_final \
    --verbose
```

---

## Quick Reference: Full Pipeline Commands

### Approach A (Memory-Optimized, embed_dim=64)

```bash
# 1. Pretrain with cross-modal contrastive learning
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_64dim \
    --embed_dim 64 --projection_dim 32 --batch_size 64 --num_epochs 100 \
    --temperature 0.07 --use_cross_modal --checkpoint_interval 10 --device cpu

# 2. Extract features
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_encoders_64dim/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features_64dim --device cpu

# 3. Train QML base learners (built-in UMAP 64→14)
for m in GeneExpr miRNA Meth CNV Prot SNV; do
    python dre_standard.py --datatypes $m --use_pretrained_features \
        --pretrained_features_dir pretrained_features_64dim \
        --n_qbits 14 --n_layers 3 --steps 150 \
        --checkpoint_frequency 25 --keep_last_n 3 \
        --use_wandb --wandb_project qml_64dim --wandb_run_name ${m}_base \
        --verbose
done

# 4. Train meta-learner
python metalearner.py --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train --n_trials 30 \
    --use_wandb --wandb_project qml_64dim --wandb_run_name metalearner \
    --verbose
```

### Approach C (Recommended, embed_dim=128)

```bash
# 1. Pretrain with cross-modal contrastive learning
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_128dim \
    --embed_dim 128 --projection_dim 64 --batch_size 32 --num_epochs 150 \
    --temperature 0.07 --use_cross_modal --checkpoint_interval 10 --device cpu

# 2. Extract features
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_encoders_128dim/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features_128dim --device cpu

# 3. Train QML base learners (built-in UMAP 128→14)
for m in GeneExpr miRNA Meth CNV Prot SNV; do
    python dre_standard.py --datatypes $m --use_pretrained_features \
        --pretrained_features_dir pretrained_features_128dim \
        --n_qbits 14 --n_layers 4 --steps 200 \
        --checkpoint_frequency 25 --keep_last_n 3 \
        --use_wandb --wandb_project qml_128dim --wandb_run_name ${m}_base \
        --verbose
done

# 4. Train meta-learner
python metalearner.py --preds_dir base_learner_outputs_app1_standard \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train --n_trials 30 \
    --use_wandb --wandb_project qml_128dim --wandb_run_name metalearner \
    --verbose
```

### Full Hybrid Pipeline (QML + Transformer → Meta-Learner)

```bash
# 1. Pretrain contrastive encoder
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_128dim \
    --embed_dim 128 --projection_dim 64 --batch_size 32 --num_epochs 150 \
    --temperature 0.07 --use_cross_modal --checkpoint_interval 10 --device cpu

# 2. Extract features
python examples/extract_pretrained_features.py \
    --encoder_dir pretrained_encoders_128dim/encoders \
    --data_dir final_processed_datasets \
    --output_dir pretrained_features_128dim --device cpu

# 3. Train QML base learners
for m in GeneExpr miRNA Meth CNV Prot SNV; do
    python dre_standard.py --datatypes $m --use_pretrained_features \
        --pretrained_features_dir pretrained_features_128dim \
        --n_qbits 14 --n_layers 4 --steps 200 \
        --checkpoint_frequency 25 --keep_last_n 3 \
        --use_wandb --wandb_project qml_hybrid --wandb_run_name ${m}_qml \
        --verbose
done

# 4. Train transformer fusion
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models \
    --pretrained_encoders_dir pretrained_encoders_128dim/encoders \
    --embed_dim 128 --num_heads 8 --num_layers 4 \
    --batch_size 32 --num_epochs 50 --lr 1e-3 --device cpu \
    --checkpoint_interval 10 --keep_last_n 3 \
    --use_wandb --wandb_project qml_hybrid --wandb_run_name transformer_128d

# 5. Extract transformer predictions
python examples/extract_transformer_features.py \
    --model_dir transformer_models \
    --data_dir final_processed_datasets \
    --output_dir transformer_predictions \
    --output_format csv

# 6. Train meta-learner on both QML and transformer outputs
python metalearner.py \
    --preds_dir base_learner_outputs_app1_standard transformer_predictions \
    --indicator_file final_processed_datasets/indicator_features.parquet \
    --mode train --n_trials 30 \
    --use_wandb --wandb_project qml_hybrid --wandb_run_name metalearner_hybrid \
    --verbose
```

---

## Resuming Interrupted Training

All training scripts support resuming from checkpoints, which is critical for resource-constrained environments.

### QML Training Scripts (dre_*.py, cfe_*.py, metalearner.py)

```bash
# Resume from best validation checkpoint
python dre_standard.py \
    --datatypes GeneExpr \
    --resume best \
    --use_wandb --wandb_project qml_128dim

# Resume from latest checkpoint (if training was interrupted mid-step)
python dre_standard.py \
    --datatypes GeneExpr \
    --resume latest

# Auto-detect (tries best first, falls back to latest)
python dre_standard.py \
    --datatypes GeneExpr \
    --resume auto
```

### Performance Extension Scripts

```bash
# Resume contrastive pretraining from a specific checkpoint
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders_128dim \
    --resume pretrained_encoders_128dim/checkpoints/epoch_50.pt

# Resume transformer training
python examples/train_transformer_fusion.py \
    --data_dir final_processed_datasets \
    --output_dir transformer_models \
    --resume transformer_models/checkpoints/epoch_25.pt
```

---

## Recommendations

1. **Start with Approach C** (embed_dim=128) - best balance of performance and memory
2. **If memory issues occur**, switch to Approach A (embed_dim=64)
3. **14 qubits is optimal** - empirically 8-16 qubits works well for most classification tasks
4. **Always use `--use_cross_modal`** when training contrastive encoder for multi-omics data
5. **Use `--temperature 0.07`** for more discriminative features (harder negative mining)
6. **Built-in UMAP** (`--dim_reducer umap` in tune_models.py) handles dimensionality reduction automatically
7. **Enable W&B logging** with `--use_wandb --wandb_project <project>` for experiment tracking
8. **Enable checkpointing** with `--checkpoint_frequency` for long training runs
9. **Use `--resume auto`** when restarting after interruptions to continue from the best available checkpoint
