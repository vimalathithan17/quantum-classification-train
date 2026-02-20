# Team Task Assignment - Contrastive Encoder Training

## Prerequisites

**Before starting, complete the Kaggle environment setup:** [KAGGLE_SETUP.md](KAGGLE_SETUP.md)

---

## Pre-Extracted Features (Quick Start Option)

**Skip encoder training entirely!** Pre-extracted features are available on Kaggle:

**Kaggle Dataset:** [qml-tcga-pretrained-encoder-extracted-features](https://www.kaggle.com/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features)

**Directory:** `pretrained_features_mlp_264dim`

**Contents:**
```
pretrained_features_mlp_264dim/
├── CNV_embeddings.npy        # 264-dim embeddings
├── GeneExpr_embeddings.npy   # 264-dim embeddings
├── Meth_embeddings.npy       # 264-dim embeddings
├── Prot_embeddings.npy       # 264-dim embeddings
├── miRNA_embeddings.npy      # 264-dim embeddings
├── case_ids.npy              # Sample identifiers
├── labels.npy                # Class labels
└── extraction_metadata.json  # Encoder config info
```

**Kaggle Input Path:**
```
/kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim
```

To use these features, jump directly to the **Hyperparameter Tuning** and **Training** sections in each team member's flow below.

---

## Overview

Each team member will train a different variant of the contrastive encoder to compare performance. All experiments use the same dataset but different configurations.

**Dataset:** `gbm-lgg-balanced-xgb-reduced` (5 modalities, SNV skipped initially)

**Kaggle Input Path:**
```
/kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced
```

**Shared Settings:**
- `--data_dir /kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced`
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

## Team Member 1: Transformer Fusion on Raw Features

**Focus:** Tune transformer fusion model on raw multi-omics features (end-to-end learning)

### Complete Flow

#### Step 1: Tune Transformer Fusion Hyperparameters

```bash
python examples/tune_transformer_fusion.py \
    --data_dir /kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --modalities GeneExpr miRNA Meth CNV Prot \
    --n_trials 50 \
    --num_epochs 75 \
    --study_name member1_raw_features \
    --device cuda \
    --use_wandb --wandb_project transformer-tuning --wandb_run_name member1_raw_features \
    --verbose
```

**Tuned Hyperparameters:**
- `embed_dim`: [64, 128, 256, 512]
- `num_heads`: [4, 8]
- `num_layers`: 2-6
- `lr`: 1e-5 to 1e-2 (log scale)
- `batch_size`: [16, 32, 64]
- `dropout`: 0.1-0.5

#### Step 2: Train with Best Hyperparameters

After tuning completes, train final model with best config:

```bash
# Check best params in transformer_tuning_results/member1_raw_features_best_config.json
python examples/train_transformer_fusion.py \
    --data_dir /kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/transformer_models_member1 \
    --embed_dim <best_embed_dim> \
    --num_heads <best_num_heads> \
    --num_layers <best_num_layers> \
    --lr <best_lr> \
    --batch_size <best_batch_size> \
    --num_epochs 100 \
    --device cuda \
    --use_wandb --wandb_project transformer-fusion --wandb_run_name member1_final_model
```

#### Step 3: Extract Transformer Features for Meta-Learner

```bash
python examples/extract_transformer_features.py \
    --model_dir /kaggle/working/transformer_models_member1 \
    --data_dir /kaggle/input/datasets/vimalathithan22i272/gbm-lgg-balanced-xgb-reduced/final_processed_datasets_xgb_balanced \
    --output_dir /kaggle/working/transformer_predictions_member1 \
    --output_format csv
```

**Expected Output:**
- `transformer_tuning_results/member1_raw_features_best_config.json` - Best hyperparameters
- `/kaggle/working/transformer_models_member1/` - Trained model
- `/kaggle/working/transformer_predictions_member1/` - Predictions for meta-learner
- W&B logs with tuning and training metrics

---

## Team Member 2: Transformer Fusion on Pretrained Embeddings

**Focus:** Tune transformer fusion model using pretrained contrastive encoder embeddings (transfer learning)

This approach uses the 264-dim embeddings from pretrained contrastive encoders as input, leveraging learned representations instead of raw features.

### Complete Flow

#### Step 1: Tune Transformer Fusion on Pretrained Embeddings

```bash
python examples/tune_transformer_fusion.py \
    --use_pretrained_embeddings \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --modalities GeneExpr miRNA Meth CNV Prot \
    --n_trials 50 \
    --num_epochs 75 \
    --study_name member2_pretrained_embeddings \
    --device cuda \
    --use_wandb --wandb_project transformer-tuning --wandb_run_name member2_pretrained_emb \
    --verbose
```

**Note:** Since embeddings are already 264-dim, the tuning script automatically limits `embed_dim` to [64, 128, 256] to avoid wasteful over-parameterization.

**Tuned Hyperparameters:**
- `embed_dim`: [64, 128, 256] *(auto-limited for 264-dim input)*
- `num_heads`: [4, 8]
- `num_layers`: 2-6
- `lr`: 1e-5 to 1e-2 (log scale)
- `batch_size`: [16, 32, 64]
- `dropout`: 0.1-0.5

#### Step 2: Train with Best Hyperparameters

```bash
# Check best params in transformer_tuning_results/member2_pretrained_embeddings_best_config.json
python examples/train_transformer_fusion.py \
    --use_pretrained_embeddings \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --output_dir /kaggle/working/transformer_models_member2 \
    --embed_dim <best_embed_dim> \
    --num_heads <best_num_heads> \
    --num_layers <best_num_layers> \
    --lr <best_lr> \
    --batch_size <best_batch_size> \
    --num_epochs 100 \
    --device cuda \
    --use_wandb --wandb_project transformer-fusion --wandb_run_name member2_final_model
```

#### Step 3: Extract Transformer Features for Meta-Learner

```bash
python examples/extract_transformer_features.py \
    --model_dir /kaggle/working/transformer_models_member2 \
    --use_pretrained_embeddings \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --output_dir /kaggle/working/transformer_predictions_member2 \
    --output_format csv
```

**Expected Output:**
- `transformer_tuning_results/member2_pretrained_embeddings_best_config.json` - Best hyperparameters
- `/kaggle/working/transformer_models_member2/` - Trained model using pretrained embeddings
- `/kaggle/working/transformer_predictions_member2/` - Predictions for meta-learner
- W&B logs with tuning and training metrics

**Key Difference from Member 1:** Uses pretrained 264-dim embeddings from contrastive encoders instead of raw features (thousands of dimensions). This tests whether transfer learning from contrastive pretraining improves transformer fusion performance.

---

## Team Member 3: DRE Standard QML Tuning

**Focus:** Tune standard QML (data-reuploading-enhanced) hyperparameters with pretrained features

### Complete Flow

#### Step 1: Hyperparameter Tuning (All Modalities)

Run Optuna hyperparameter search using pretrained features:

```bash
# GeneExpr - Standard QML
python tune_models.py --datatype GeneExpr --approach 1 --qml_model standard \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member3_geneexpr_standard \
    --verbose

# miRNA - Standard QML
python tune_models.py --datatype miRNA --approach 1 --qml_model standard \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member3_mirna_standard \
    --verbose

# Meth - Standard QML
python tune_models.py --datatype Meth --approach 1 --qml_model standard \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member3_meth_standard \
    --verbose

# CNV - Standard QML
python tune_models.py --datatype CNV --approach 1 --qml_model standard \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member3_cnv_standard \
    --verbose

# Prot - Standard QML
python tune_models.py --datatype Prot --approach 1 --qml_model standard \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member3_prot_standard \
    --verbose
```

**Tuned QML Hyperparameters:**
- `n_qubits`: 8-14 (step 2)
- `n_layers`: 2-6
- `scaler`: Standard, MinMax, Robust

#### Step 2: Train with Best Hyperparameters

Use the best parameters from tuning to train final models:

```bash
# Train all modalities with best params from tuning
# Check optuna_studies.db or W&B for best params per modality

python dre_standard.py \
    --datatypes GeneExpr miRNA Meth CNV Prot \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --n_qbits <best_qbits> --n_layers <best_layers> --steps 200 \
    --use_wandb --wandb_project qml-classification --wandb_run_name member3_all_standard \
    --verbose
```

**Expected Output:**
- Optuna study results in SQLite database
- W&B logs with tuning and training metrics
- Base learner predictions for meta-learner

---

## Team Member 4: DRE Reuploading QML Tuning

**Focus:** Tune data-reuploading QML hyperparameters with pretrained features

### Complete Flow

#### Step 1: Hyperparameter Tuning (All Modalities)

Run Optuna hyperparameter search using pretrained features:

```bash
# GeneExpr - Reuploading QML
python tune_models.py --datatype GeneExpr --approach 1 --qml_model reuploading \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member4_geneexpr_reup \
    --verbose

# miRNA - Reuploading QML
python tune_models.py --datatype miRNA --approach 1 --qml_model reuploading \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member4_mirna_reup \
    --verbose

# Meth - Reuploading QML
python tune_models.py --datatype Meth --approach 1 --qml_model reuploading \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member4_meth_reup \
    --verbose

# CNV - Reuploading QML
python tune_models.py --datatype CNV --approach 1 --qml_model reuploading \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member4_cnv_reup \
    --verbose

# Prot - Reuploading QML
python tune_models.py --datatype Prot --approach 1 --qml_model reuploading \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --min_qbits 8 --max_qbits 14 \
    --min_layers 2 --max_layers 6 \
    --steps 75 \
    --n_trials 50 \
    --use_wandb --wandb_project qml-tuning --wandb_run_name member4_prot_reup \
    --verbose
```

**Tuned QML Hyperparameters:**
- `n_qubits`: 8-14 (step 2)
- `n_layers`: 2-6
- `scaler`: Standard, MinMax, Robust

#### Step 2: Train with Best Hyperparameters

Use the best parameters from tuning to train final models:

```bash
# Train all modalities with best params from tuning
python dre_relupload.py \
    --datatypes GeneExpr miRNA Meth CNV Prot \
    --use_pretrained_features \
    --pretrained_features_dir /kaggle/input/datasets/vimalathithan22i272/qml-tcga-pretrained-encoder-extracted-features/pretrained_features_mlp_264dim \
    --n_qbits <best_qbits> --n_layers <best_layers> --steps 200 \
    --use_wandb --wandb_project qml-classification --wandb_run_name member4_all_reup \
    --verbose
```

**Expected Output:**
- Optuna study results in SQLite database
- W&B logs with tuning and training metrics
- Base learner predictions for meta-learner

---

## Team Member Assignments Summary

| Member | Task | Input Data | Model | Device | Trials | Key Focus |
|--------|------|------------|-------|--------|--------|-----------|
| 1 | Transformer Fusion | Raw features | TransformerFusion | CUDA | 50 | End-to-end learning from raw data |
| 2 | Transformer Fusion | Pretrained embeddings (264-dim) | TransformerFusion | CUDA | 50 | Transfer learning with encoder |
| 3 | Standard QML Tuning | Pretrained embeddings | DRE Standard | CPU | 50×5 | All modalities, standard circuit |
| 4 | Reuploading QML Tuning | Pretrained embeddings | DRE Reuploading | CPU | 50×5 | All modalities, data reuploading |

**Key Comparison (Members 1 vs 2):**
- Member 1 tests transformer's ability to learn from high-dimensional raw features
- Member 2 tests whether pretrained contrastive embeddings improve fusion performance

**Common Settings for QML Tuning (Members 3 & 4):**
- `--min_qbits 8 --max_qbits 14`
- `--min_layers 2 --max_layers 6`
- `--steps 75` (75 epochs per trial)

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
