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
- **NaN handling**: Data is imputed during loading (`--impute_strategy median`)

---

## Quick Start: Pre-extracted Features (Skip Encoder Training)

Pre-extracted embeddings from contrastive pretraining are available on Kaggle:

**Kaggle Dataset:** [qml-tcga-pretrained-encoder-extracted-features](https://www.kaggle.com/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features)

**Directory:** `pretrained_features_mlp_264dim`

**Contents:**
- `GeneExpr_embeddings.npy`, `miRNA_embeddings.npy`, `Meth_embeddings.npy`, `CNV_embeddings.npy`, `Prot_embeddings.npy` (264-dim embeddings)
- `case_ids.npy`, `labels.npy`, `extraction_metadata.json`

**Kaggle Input Path:**
```
/kaggle/input/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim
```

**Use directly with tuning and training:**
```bash
# Hyperparameter tuning
python tune_models.py --datatype GeneExpr --approach 1 --qml_model standard \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --n_trials 30 --verbose

# Training with best params
python dre_standard.py --datatypes GeneExpr \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --n_qbits 14 --n_layers 4 --steps 200 --verbose
```

---

## Understanding the Encoder Architecture

### How Encoders Are Created and Saved

**One encoder is created per modality file:**

```
data_GeneExpr_.parquet → encoder_GeneExpr.pt
data_miRNA_.parquet    → encoder_miRNA.pt
data_Meth_.parquet     → encoder_Meth.pt
data_CNV_.parquet      → encoder_CNV.pt
data_Prot_.parquet     → encoder_Prot.pt
data_SNV_.parquet      → encoder_SNV.pt
```

**Output directory structure:**
```
pretrained_models/contrastive/
├── encoders/
│   ├── encoder_GeneExpr.pt   # Encoder weights for GeneExpr
│   ├── encoder_miRNA.pt      # Encoder weights for miRNA
│   ├── encoder_Meth.pt       # Encoder weights for Meth
│   ├── encoder_CNV.pt        # Encoder weights for CNV
│   ├── encoder_Prot.pt       # Encoder weights for Prot
│   ├── encoder_SNV.pt        # Encoder weights for SNV
│   └── metadata.json         # Dimensions, config, NaN stats
├── checkpoints/              # Training checkpoints
├── training_metrics.json     # Loss history and statistics
└── loss_curve.png           # Training visualization
```

### Evaluating and Choosing the Best Contrastive Encoder

After training multiple encoder variants, you need to determine which is "best". There are two approaches:

#### Method 1: Compare Contrastive Loss (Quick but Indirect)

Look at the `loss` column in W&B or `training_metrics.json`. Lower contrastive loss means:
- Positive pairs (same sample, different augmentations) are more similar
- Negative pairs (different samples) are more dissimilar
- The encoder learned better representations

**Metrics to compare:**

| Metric | Meaning | Goal |
|--------|---------|------|
| `loss` | Overall NT-Xent contrastive loss | Lower = better |
| `intra_*` | Same-modality augmentation similarity | Lower = modality encodes well |
| `cross_*_*` | Cross-modal patient alignment | Lower = modalities aligned |

**⚠️ Limitation:** Low contrastive loss does NOT guarantee high downstream classification accuracy!

#### Method 2: Evaluate on Downstream Task (Recommended)

The only reliable way to choose the best encoder is to test on your actual task:

```bash
# For each trained encoder, extract features and run QML
for encoder_name in member1_64dim member2_256dim member3_128dim; do
    # Extract features
    python examples/extract_pretrained_features.py \
        --encoder_dir /path/to/$encoder_name/encoders \
        --data_dir final_processed_datasets \
        --output_dir pretrained_features_$encoder_name
    
    # Run QML with these features
    python dre_standard.py \
        --datatypes GeneExpr \
        --use_pretrained_features \
        --pretrained_features_dir pretrained_features_$encoder_name \
        --use_wandb --wandb_project encoder-comparison
done

# Compare test accuracy / F1 scores across runs
```

**Decision criteria:**
- **Best encoder = highest test F1 score** on your classification task
- If multiple encoders have similar F1, prefer smaller `embed_dim` (faster QML training)

#### Key Hyperparameters That Affect Quality

From empirical testing:

| Parameter | Recommendation | Why |
|-----------|----------------|-----|
| `temperature` | 0.05-0.1 | Lower = harder negatives, sharper representations |
| `num_epochs` | 1000-2500 | Loss continues improving; diminishing returns after ~2000 |
| `embed_dim` | 128-256 | 128 is often sufficient; 256 more expressive but slower |
| `use_cross_modal` | Always `True` | Essential for multi-omics alignment |

### Columns Ignored by Contrastive Encoder

The following columns are automatically excluded from features:
- `class` - Target label (GBM/LGG)
- `case_id` - Sample identifier

**All other columns are used as input features.**

### NaN Handling

The contrastive encoder provides two approaches for handling missing values within features:

**Approach 1: Transformer Encoder (Recommended for data with missing values)**

Use `--encoder_type transformer` for native NaN handling. The transformer uses attention-based masking to learn from context of present features:

```bash
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --encoder_type transformer \
    --impute_strategy none
```

**Approach 2: MLP Encoder (Faster, requires imputation)**

Use `--encoder_type mlp` with an imputation strategy:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `none` | Keep NaN (only for transformer encoder) | Transformer encoder |
| `median` | Replace NaN with column median | Most datasets with MLP |
| `mean` | Replace NaN with column mean | Normally distributed data |
| `zero` | Replace NaN with 0 | Sparse data |
| `drop` | Drop samples with any NaN | Small NaN percentage |

```bash
# MLP encoder with median imputation
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --encoder_type mlp \
    --impute_strategy median

# Full batch gradient descent (uses entire dataset per update)
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --full_batch
```

**Note:** The indicator features (`indicator_features.parquet`) indicate which **entire modalities** are missing for each sample. The transformer encoder handles **feature-level** (column) NaN values within a modality.

### Missing Modality Handling

The contrastive encoder natively handles **missing modalities** using learnable missing tokens:

```python
# Each modality encoder has a learnable missing token
class ModalityEncoder(nn.Module):
    def __init__(self, ...):
        # Learnable token for missing modality
        self.missing_token = nn.Parameter(torch.randn(1, embed_dim))
    
    def forward(self, x, is_missing=False):
        if is_missing or x is None:
            # Return learnable missing token and invalid mask
            embedding = self.missing_token.expand(batch_size, -1)
            valid_mask = torch.zeros(batch_size, dtype=torch.bool)
            return embedding, valid_mask
        embedding = self.encoder(x)
        # Detect all-NaN samples
        valid_mask = ~torch.isnan(x).all(dim=1)
        return embedding, valid_mask
```

**How it works:**
- When a modality is missing for a sample, the encoder returns a **learnable token** instead of encoded features
- The missing token is learned during training to represent "no data available"
- A `valid_mask` is returned to identify samples with valid data (used to exclude all-NaN samples from loss)
- This is the same approach used by Transformer Fusion (`ModalityFeatureEncoder`)

**Use cases:**
- Some patients have missing modality files entirely
- Individual samples have missing values (after `--impute_strategy drop` removes rows with NaN)
- Programmatically excluding modalities during inference

### Skipping Modalities

You can exclude specific modalities from training:

```bash
# Skip SNV and Prot modalities
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --skip_modalities SNV Prot
```

---

## Comparing Encoder Performance

Since contrastive learning is **unsupervised**, you can't directly use classification metrics during pretraining. Here are the evaluation strategies:

### Method 1: Compare Loss Statistics (Quick Check)

After training with different configurations (e.g., temperatures), compare `training_metrics.json`:

```bash
# Compare different temperature runs
cat pretrained_encoders_temp07/training_metrics.json | python -c "import sys,json; d=json.load(sys.stdin); print(f\"Final: {d['loss_statistics']['final_loss']:.4f}, Best: {d['loss_statistics']['min_loss']:.4f}, Improvement: {d['loss_statistics']['improvement_ratio']*100:.1f}%\")"
```

**Key metrics:**
| Metric | Description | Prefer |
|--------|-------------|--------|
| `final_loss` | Loss at end of training | Lower |
| `min_loss` | Best loss achieved | Lower |
| `improvement_ratio` | (initial - final) / initial | Higher |
| `best_epoch` | When best loss occurred | Not too early |

### Method 2: Downstream Task Evaluation (Best Practice)

The **gold standard** is to evaluate on your actual classification task:

```bash
#!/bin/bash
# compare_encoders.sh - Compare different encoder configurations

for config in "temp07" "temp05" "temp03"; do
    echo "=== Evaluating encoder: $config ==="
    
    # 1. Extract features
    python examples/extract_pretrained_features.py \
        --encoder_dir pretrained_encoders_${config}/encoders \
        --data_dir final_processed_datasets \
        --output_dir pretrained_features_${config}
    
    # 2. Train QML on one modality (quick test)
    OUTPUT_DIR=test_outputs_${config} python dre_standard.py \
        --datatypes GeneExpr \
        --use_pretrained_features \
        --pretrained_features_dir pretrained_features_${config} \
        --steps 50 \
        --skip_cross_validation
    
    # 3. Show test metrics
    echo "Test metrics for $config:"
    cat test_outputs_${config}/test_metrics_GeneExpr.json
    echo ""
done
```

### Method 3: W&B Comparison (Recommended for Multiple Runs)

Use Weights & Biases to track and compare experiments:

```bash
# Train with different temperatures, all logged to same project
for temp in 0.03 0.07 0.1 0.5; do
    python examples/pretrain_contrastive.py \
        --data_dir final_processed_datasets \
        --output_dir pretrained_encoders_temp${temp} \
        --temperature $temp \
        --use_cross_modal \
        --use_wandb \
        --wandb_project contrastive-comparison \
        --wandb_run_name temp_${temp}
done
```

Then compare in W&B dashboard at `https://wandb.ai/<your-username>/contrastive-comparison`.

### Temperature Guidelines

| Temperature | Effect | Best For |
|-------------|--------|----------|
| 0.03-0.07 (low) | Harder negatives, more discriminative | Large batches (64+), distinct classes |
| 0.1-0.2 (medium) | Balanced | General use |
| 0.5-1.0 (high) | Softer negatives, smoother gradients | Small batches, noisy data |

**Recommendation:** Start with `--temperature 0.07` for multi-omics data.

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
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 10 \
    --weight_decay 1e-4 \
    --lr_scheduler cosine \
    --use_wandb --wandb_project contrastive-pretrain --wandb_run_name encoder_64dim \
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
- `--warmup_epochs 10`: Gradual LR increase to prevent early gradient explosion
- `--weight_decay 1e-4`: L2 regularization to prevent late-stage weight explosion
- `--lr_scheduler cosine`: Cosine annealing decays LR to 1% to prevent late divergence
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
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 10 \
    --weight_decay 1e-4 \
    --lr_scheduler cosine \
    --use_wandb --wandb_project contrastive-pretrain --wandb_run_name encoder_256dim \
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
    --seed 42 \
    --max_grad_norm 1.0 \
    --warmup_epochs 15 \
    --weight_decay 1e-4 \
    --lr_scheduler cosine \
    --use_wandb --wandb_project contrastive-pretrain --wandb_run_name encoder_128dim \
    --device cpu
```

**Parameter Notes:**
- `--embed_dim 128`: Balanced between expressiveness and memory efficiency
- `--batch_size 32`: Comfortable fit for 30GB RAM
- `--num_epochs 150`: Slightly more epochs since smaller embedding needs more training
- `--temperature 0.07`: Lower temperature for more discriminative features
- `--use_cross_modal`: **Always enabled** - learns cross-modal relationships
- `--warmup_epochs 15`: Gradual LR warmup (10% of epochs is a good rule)
- `--weight_decay 1e-4`: L2 regularization prevents late-stage weight explosion
- `--lr_scheduler cosine`: Decays LR smoothly to prevent divergence in long training

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
    --seed 42 \
    --max_grad_norm 1.0 \
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
| `pretrain_contrastive.py` | `--checkpoint_interval 10 --keep_last_n_checkpoints 3 --resume <path>` |
| `train_transformer_fusion.py` | `--checkpoint_interval 10 --keep_last_n 3 --resume <path>` |

### Contrastive Pretraining Checkpoint Structure

The contrastive pretraining saves models in a structured format for flexible downstream use:

```
pretrained_models/contrastive/
├── best_model.pt                    # Combined model (all modalities)
├── contrastive_epoch_100.pt         # Periodic checkpoint
├── contrastive_epoch_200.pt         # Periodic checkpoint
├── encoders/                        # Per-modality encoders (for downstream use)
│   ├── mRNA_encoder.pt
│   ├── miRNA_encoder.pt
│   ├── DNA_Meth_encoder.pt
│   └── CNV_encoder.pt
└── projections/                     # Projection heads (for continued pretraining)
    ├── mRNA_projection.pt
    └── ...
```

**What's saved:**

| File | Contents | Use Case |
|------|----------|----------|
| `best_model.pt` | Full model state, optimizer, config | Resume training, full inference |
| `encoders/{modality}_encoder.pt` | Single encoder weights + metadata | Load specific modalities |
| `projections/{modality}_projection.pt` | Projection head weights | Continued pretraining |

**Best model selection metric:** **Contrastive Loss (lower = better)**
- Lower loss = better separation of positive/negative pairs
- Indicates embeddings form meaningful clusters
- For downstream performance, evaluate with classification F1 after fine-tuning

**Loading individual modality encoders:**

```python
from performance_extensions.training_utils import load_single_modality_encoder

# Load just the mRNA encoder
encoder, metadata = load_single_modality_encoder(
    "pretrained_models/contrastive/encoders/mRNA_encoder.pt"
)

# metadata contains:
# - input_dim: 5000
# - embed_dim: 256
# - encoder_type: 'transformer' or 'mlp'
# - epoch: 150
# - loss: 0.594
```

**Loading multiple modalities:**

```python
from performance_extensions.training_utils import load_pretrained_encoders

# Load specific modalities only
encoders, metadata = load_pretrained_encoders(
    "pretrained_models/contrastive",
    modalities=['mRNA', 'miRNA']  # Only load these two
)

# Load all modalities
encoders, metadata = load_pretrained_encoders("pretrained_models/contrastive")
```

**Note:** Training behavior is unchanged. Only checkpoint saving is enhanced to support per-modality loading.

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
    --warmup_epochs 15 \
    --weight_decay 1e-4 \
    --lr_scheduler cosine \
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
    --temperature 0.07 --use_cross_modal --checkpoint_interval 10 \
    --seed 42 --max_grad_norm 1.0 --warmup_epochs 10 \
    --weight_decay 1e-4 --lr_scheduler cosine \
    --use_wandb --wandb_project contrastive-pretrain --wandb_run_name encoder_64dim \
    --device cpu

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
    --temperature 0.07 --use_cross_modal --checkpoint_interval 10 \
    --seed 42 --max_grad_norm 1.0 --warmup_epochs 15 \
    --weight_decay 1e-4 --lr_scheduler cosine \
    --use_wandb --wandb_project contrastive-pretrain --wandb_run_name encoder_128dim \
    --device cpu

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
    --temperature 0.07 --use_cross_modal --checkpoint_interval 10 \
    --seed 42 --max_grad_norm 1.0 --warmup_epochs 15 \
    --weight_decay 1e-4 --lr_scheduler cosine \
    --use_wandb --wandb_project qml_hybrid --wandb_run_name contrastive_encoder \
    --device cpu

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
    --batch_size 32 --num_epochs 50 --lr 1e-3 \
    --seed 42 --max_grad_norm 1.0 \
    --device cpu \
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

## Training Stability & Reproducibility

### Reproducibility

All contrastive pretraining runs use seed-based reproducibility by default:

```bash
# Use a specific seed for reproducibility
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders \
    --seed 42
```

The seed controls:
- PyTorch random number generator
- NumPy random number generator
- Python's `random` module
- CUDA random state (if available)
- CuDNN determinism settings

### Gradient Clipping

Gradient clipping prevents training instabilities and exploding gradients:

```bash
# Use gradient clipping (default: 1.0)
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders \
    --max_grad_norm 1.0

# Disable gradient clipping
python examples/pretrain_contrastive.py \
    --data_dir final_processed_datasets \
    --output_dir pretrained_encoders \
    --max_grad_norm 0
```

### Best Model Tracking

The training loop automatically tracks and saves the best model based on the lowest training loss. The best model is saved to `{output_dir}/checkpoints/best_model.pt` in addition to epoch checkpoints.

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
10. **Use `--seed` for reproducible experiments** - always set a seed when comparing configurations
