# End-to-End Full Hybrid Quantum-Classical Multi-Omics Classification Pipeline

## Pipeline Overview

```
RAW TCGA DATA → DATA PROCESSING → CONTRASTIVE PRETRAINING → PARALLEL BRANCH (QML + TRANSFORMER) → QML META-LEARNER → FINAL PREDICTIONS
```

---

## STAGE 1: RAW DATA COLLECTION

**Input:** The Cancer Genome Atlas (TCGA) raw multi-omics data

**Data Types (6 Modalities):**
- Gene Expression (GeneExpr): ~20,000 genes, RNA-seq counts
- miRNA Expression (miRNA): ~1,800 microRNAs
- DNA Methylation (Meth): ~450,000 CpG sites (beta values 0-1)
- Copy Number Variation (CNV): ~25,000 genomic segments
- Protein Expression (Prot): ~200 proteins (RPPA data)
- Somatic Mutations (SNV): ~20,000 genes (binary mutation status)

**Output:** Raw parquet files per modality with case_id, features, and cancer type labels

---

## STAGE 2: DATA CLEANING & PREPROCESSING

**Step 2.1: Missing Value Handling**
- Identify samples with missing modalities
- Create indicator features (is_missing_GeneExpr, is_missing_miRNA, etc.)
- Impute within-modality missing values (median/KNN)

**Step 2.2: Feature Normalization**
- Gene Expression: Log2(x+1) transform → Z-score normalization
- miRNA: Log2(x+1) transform → Z-score normalization
- Methylation: Already 0-1 beta values → no transform needed
- CNV: Z-score normalization
- Protein: Z-score normalization
- SNV: Binary (0/1) → no transform needed

**Step 2.3: Feature Selection (per modality)**
- Method: LightGBM or XGBoost feature importance
- Target: Select top N features where N ≤ number of qubits (e.g., 8-16)
- Creates dimensionality-reduced feature sets

**Step 2.4: Train/Test Split**
- Stratified split preserving cancer type distribution
- Typical: 80% train, 20% test
- 5-fold cross-validation for out-of-fold predictions

**Output:** 
- Processed parquet files: data_GeneExpr_.parquet, data_miRNA_.parquet, etc.
- Indicator file: indicator_features.parquet
- Each file contains: case_id, class (cancer type), features

---

## STAGE 3: CONTRASTIVE PRETRAINING (Self-Supervised Learning)

**Purpose:** Learn robust modality-specific representations without labels

**Architecture:**
- 6 Modality Encoders (one per data type)
  - Input: Raw features (variable dimension per modality)
  - Hidden: 2-layer MLP with ReLU activation
  - Output: Shared embedding dimension (e.g., 256)

- 6 Projection Heads (for contrastive loss)
  - Input: 256-dim embeddings
  - Output: 128-dim projections

**Training Process:**
- Create augmented views of each sample using modality-specific augmentations:
  - Gene Expression: Gaussian noise, feature dropout, feature scaling
  - miRNA: Gaussian noise, feature dropout
  - Methylation: Beta noise, feature masking
  - CNV: Segment noise, amplification simulation
  - Protein: Gaussian noise, feature dropout
  - SNV: Mutation flipping, feature masking

- Contrastive Loss (NT-Xent):
  - Positive pairs: Two augmented views of same sample
  - Negative pairs: Views from different samples in batch
  - Temperature parameter: 0.5

- Cross-Modal Contrastive (optional):
  - Align embeddings from different modalities of same patient
  - Encourages learning patient-level representations

**Output:** 
- 6 pretrained modality encoders (GeneExpr_encoder.pt, miRNA_encoder.pt, etc.)
- Encoder metadata (embedding dimensions, architecture)

---

## STAGE 4: FEATURE EXTRACTION FROM PRETRAINED ENCODERS

**Purpose:** Transform raw features into learned embeddings

**Process:**
- Load pretrained encoders from Stage 3
- Pass each modality's data through its encoder
- Extract 256-dimensional embeddings per modality

**Output:**
- Embedding files: GeneExpr_embeddings.npy, miRNA_embeddings.npy, etc.
- Each: (N_samples × 256) dimensional arrays

---

## STAGE 5: PARALLEL PROCESSING BRANCH

This stage has TWO parallel paths that process data simultaneously:

### BRANCH A: QML BASE LEARNERS (Modality-Specific Experts)

**Purpose:** Train quantum classifiers that become experts on each modality

**Architecture (per modality):**
- Dimensionality Reduction: PCA/UMAP to reduce embeddings to N_qubits features
- Quantum Circuit (Variational Quantum Classifier):
  - Data Encoding: Angle embedding of features into qubit rotations
  - Variational Layers: Parameterized rotation gates (RX, RY, RZ) + entangling CNOT gates
  - Measurement: Expectation values of Pauli-Z operators
  - Classical Readout: Linear layer mapping measurements to class logits
- Training: Adam optimizer, Cross-entropy loss

**Models:**
- 4 QML variants available:
  1. MulticlassQuantumClassifierDR (Standard)
  2. MulticlassQuantumClassifierDataReuploadingDR (Data Re-uploading)
  3. ConditionalMulticlassQuantumClassifierFS (Conditional Feature Encoding)
  4. ConditionalMulticlassQuantumClassifierDataReuploadingFS (CFE + Re-uploading)

**Cross-Validation Process:**
- 5-fold stratified CV
- Train on 4 folds, predict on held-out fold
- Collect Out-of-Fold (OOF) predictions for all training samples
- Train final model on all training data for test predictions

**Output (per modality):**
- train_oof_preds_GeneExpr.csv: OOF predictions (case_id, class probabilities)
- test_preds_GeneExpr.csv: Test set predictions
- Model artifacts: trained_model_GeneExpr.joblib

### BRANCH B: TRANSFORMER FUSION (Cross-Modal Attention)

**Purpose:** Learn cross-modal interactions using attention mechanism

**Architecture:**
- Modality Feature Encoders:
  - 6 linear projections from pretrained embeddings (256-dim) to transformer dim
  
- Modality Embeddings:
  - Learnable position-like embeddings for each modality type
  - Added to encoded features to identify modality source

- Multi-Head Self-Attention Transformer:
  - Input: 6 modality tokens (one per data type)
  - Layers: 4-8 transformer encoder layers
  - Heads: 8-16 attention heads
  - Hidden dim: 256-512
  - Attention mask: Handle missing modalities by masking absent tokens

- CLS Token Aggregation:
  - Prepend learnable [CLS] token
  - Use [CLS] output as fused representation

- Classification Head:
  - MLP: 256 → 128 → N_classes
  - Dropout for regularization

**Training:**
- Optimizer: AdamW with weight decay
- Loss: Cross-entropy with label smoothing
- Learning rate: 1e-4 with cosine annealing
- Epochs: 50-100

**Attention Weight Extraction:**
- Save attention matrices for interpretability
- Shows which modalities contribute to each prediction

**Output:**
- train_oof_preds_Transformer.csv: OOF predictions (case_id, class probabilities)
- test_preds_Transformer.csv: Test set predictions
- best_model.pt: Trained transformer weights
- config.json: Model architecture configuration

---

## STAGE 6: QML META-LEARNER (Final Ensemble)

**Purpose:** Combine predictions from all base learners (QML + Transformer) into final prediction

**Input Assembly:**
- Concatenate predictions from all sources:
  - 6 QML base learner OOF predictions (6 modalities × N_classes probabilities)
  - 1 Transformer prediction (N_classes probabilities)
  - Indicator features (6 binary missing modality flags)
- Total meta-features: (6 + 1) × N_classes + 6 indicators

**Preprocessing:**
- Degeneracy removal: Drop one class column per base learner (probabilities sum to 1)
- Indicator conversion: Convert missingness to inclusion masks

**Architecture (Gated Quantum Meta-Learner):**
- Feature Reduction: Linear projection to N_qubits dimensions
- Gating Mechanism: Learnable gates modulate feature importance
- Quantum Circuit:
  - Same variational structure as base learners
  - But operates on fused meta-features
- Classical Readout: Final class probabilities

**Training:**
- Hyperparameter tuning via Optuna:
  - Number of qubits (4-16)
  - Number of layers (3-8)
  - Learning rate
  - Circuit type (standard vs data-reuploading)
- Optimization: Adam with early stopping
- Metric: Weighted F1-score

**Output:**
- final_predictions.csv: Predicted cancer types for test set
- meta_model.joblib: Trained meta-learner
- tuning_results.json: Best hyperparameters

---

## STAGE 7: FINAL PREDICTIONS & EVALUATION

**Prediction Generation:**
- Load trained meta-learner
- Assemble test set meta-features (from test predictions of all base learners)
- Generate class probabilities
- Argmax for final class assignment

**Evaluation Metrics:**
- Accuracy: Overall correct predictions
- Weighted F1-Score: Handles class imbalance
- Confusion Matrix: Per-class performance
- ROC-AUC: Multi-class one-vs-rest

**Interpretability Outputs:**
- Attention heatmaps (from Transformer): Which modalities mattered
- Feature importance (from QML): Which meta-features influenced decision
- Per-modality contribution analysis

---

## COMPLETE PIPELINE FLOW DIAGRAM DESCRIPTION

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    RAW TCGA DATA                                         │
│  (Gene Expression, miRNA, Methylation, CNV, Protein, Mutations - 6 modalities)          │
└─────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA CLEANING & PREPROCESSING                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Missing Value    │→ │ Normalization    │→ │ Feature Selection│→ │ Train/Test Split│  │
│  │ Handling         │  │ (log, z-score)   │  │ (LightGBM/XGBoost)│  │ (Stratified)    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         CONTRASTIVE PRETRAINING (Self-Supervised)                        │
│                                                                                          │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│   │GeneExpr │ │ miRNA   │ │  Meth   │ │  CNV    │ │  Prot   │ │  SNV    │               │
│   │Encoder  │ │Encoder  │ │Encoder  │ │Encoder  │ │Encoder  │ │Encoder  │               │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘               │
│        │           │           │           │           │           │                     │
│        └───────────┴───────────┴─────┬─────┴───────────┴───────────┘                     │
│                                      │                                                   │
│                              Contrastive Loss                                            │
│                          (NT-Xent + Cross-Modal)                                         │
└─────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            PRETRAINED FEATURE EXTRACTION                                 │
│                                                                                          │
│   Raw Features → Pretrained Encoders → 256-dim Embeddings (per modality)                │
└─────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│      BRANCH A: QML BASE LEARNERS      │ │       BRANCH B: TRANSFORMER FUSION            │
│         (Modality Experts)            │ │         (Cross-Modal Attention)               │
│                                       │ │                                               │
│  ┌─────────────────────────────────┐  │ │  ┌─────────────────────────────────────────┐  │
│  │  For each modality:             │  │ │  │  All 6 modality embeddings              │  │
│  │                                 │  │ │  │           ↓                              │  │
│  │  Embeddings → PCA → N_qubits    │  │ │  │  Modality Embeddings + Position Tokens  │  │
│  │         ↓                       │  │ │  │           ↓                              │  │
│  │  ┌─────────────────────────┐    │  │ │  │  ┌─────────────────────────────────┐    │  │
│  │  │  Quantum Circuit (VQC)  │    │  │ │  │  │  Multi-Head Self-Attention      │    │  │
│  │  │  ┌───┐ ┌───┐ ┌───┐     │    │  │ │  │  │  (4-8 layers, 8-16 heads)        │    │  │
│  │  │  │RY │─│RZ │─│CX │ x L │    │  │ │  │  │  with Missing Modality Masking   │    │  │
│  │  │  └───┘ └───┘ └───┘     │    │  │ │  │  └─────────────────────────────────┘    │  │
│  │  └─────────────────────────┘    │  │ │  │           ↓                              │  │
│  │         ↓                       │  │ │  │  [CLS] Token → Classification Head      │  │
│  │  Classical Readout Layer        │  │ │  │           ↓                              │  │
│  │         ↓                       │  │ │  │  Class Probabilities                     │  │
│  │  Class Probabilities            │  │ │  └─────────────────────────────────────────┘  │
│  └─────────────────────────────────┘  │ │                                               │
│                                       │ │                                               │
│  Output: 6 sets of predictions        │ │  Output: 1 set of predictions                │
│  (train_oof + test per modality)      │ │  (train_oof + test for transformer)          │
└───────────────────┬───────────────────┘ └───────────────────────┬───────────────────────┘
                    │                                             │
                    └─────────────────┬───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              QML META-LEARNER (Ensemble)                                 │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                              META-FEATURE ASSEMBLY                                   ││
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐ ││
│  │  │ QML Predictions (6) │ Transformer Predictions (1) │ Indicator Features (6)    │ ││
│  │  │ [P_GeneExpr, P_miRNA, P_Meth, P_CNV, P_Prot, P_SNV] │ [P_Trans] │ [missing_flags] │ ││
│  │  └────────────────────────────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                         GATED QUANTUM META-LEARNER                                   ││
│  │                                                                                      ││
│  │   Meta-features → Gating Layer → Quantum Circuit (VQC) → Classical Readout          ││
│  │                                                                                      ││
│  │   Hyperparameter Tuning via Optuna:                                                  ││
│  │   - N_qubits, N_layers, Learning Rate, Circuit Type                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL PREDICTIONS & EVALUATION                              │
│                                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Class           │  │ Confusion       │  │ F1-Score        │  │ Attention           │ │
│  │ Probabilities   │→ │ Matrix          │  │ (Weighted)      │  │ Interpretability    │ │
│  │                 │  │                 │  │                 │  │ Heatmaps            │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│                                          │                                               │
│                                          ▼                                               │
│                           FINAL CANCER TYPE PREDICTION                                   │
│                    (e.g., BRCA, LUAD, COAD, GBM, ... 33 types)                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## KEY CHARACTERISTICS FOR DIAGRAM

**Color Coding Suggestions:**
- Blue: Data processing stages
- Green: Self-supervised learning (Contrastive)
- Orange: Classical deep learning (Transformer)
- Purple: Quantum machine learning (QML circuits)
- Red: Final ensemble/output

**Arrow Types:**
- Solid arrows: Data flow
- Dashed arrows: Model training/optimization
- Parallel arrows: Simultaneous processing branches

**Box Shapes:**
- Rectangles: Processing steps
- Rounded rectangles: Models/Algorithms
- Diamonds: Decision points (e.g., missing modality check)
- Cylinders: Data storage (parquet files, model weights)

**Key Numbers:**
- 6 modalities throughout
- 256-dim embeddings from contrastive encoders
- 4-16 qubits in quantum circuits
- 4-8 transformer layers
- 7 total prediction sources into meta-learner (6 QML + 1 Transformer)
- 33 cancer types (output classes)

---

## WHY PARALLEL HYBRID? ERROR DIVERSITY EXPLAINED

### What is Error Diversity?

**Error diversity** means that different models in an ensemble make **different mistakes** on different samples. When errors are **uncorrelated**, one model's incorrect predictions are often corrected by another model's correct predictions.

### Why High Error Diversity is Good

#### Example with Low Diversity (Bad):
```
Sample 1: Model A wrong ❌, Model B wrong ❌ → Ensemble wrong ❌
Sample 2: Model A wrong ❌, Model B wrong ❌ → Ensemble wrong ❌
Sample 3: Model A right ✓, Model B right ✓ → Ensemble right ✓
```
Both models fail on the same samples → **No benefit from ensembling**

#### Example with High Diversity (Good):
```
Sample 1: Model A wrong ❌, Model B right ✓ → Ensemble right ✓ (majority vote)
Sample 2: Model A right ✓, Model B wrong ❌ → Ensemble right ✓ (majority vote)
Sample 3: Model A right ✓, Model B right ✓ → Ensemble right ✓
```
Models fail on **different** samples → **Ensemble corrects individual errors**

### Pattern Capture Comparison

| Pattern Type | Base Learners | Transformer | Parallel Captures |
|--------------|---------------|-------------|-------------------|
| Modality-specific (e.g., gene signature → cancer type) | ✅ Excellent | ⚠️ Fused | ✅ Yes |
| Cross-modal (e.g., miRNA-methylation correlation) | ❌ No | ✅ Excellent | ✅ Yes |
| Error diversity (uncorrelated mistakes) | N/A | N/A | ✅ High |

### In Our Pipeline Context

| Approach | Why Errors Are Correlated/Uncorrelated |
|----------|----------------------------------------|
| **QML Base Learners Only** | All see modality-specific data, may miss cross-modal patterns together |
| **Transformer Only** | Single model, single failure mode |
| **Parallel (QML + Transformer)** | QML and Transformer have **different architectures** and see data **differently**, so they make different mistakes |

### Concrete Example:

```
Patient X with subtle cancer subtype:

- QML Base Learner (GeneExpr): Predicts BRCA ❌ (missed mutation signal)
- QML Base Learner (miRNA): Predicts LUAD ❌ (low confidence)
- Transformer: Predicts LUAD ✓ (saw cross-modal correlation)

Meta-learner combines: LUAD ✓ (Transformer's cross-modal insight wins)
```

```
Patient Y with strong single-modality signal:

- QML Base Learner (GeneExpr): Predicts GBM ✓ (strong gene signature)
- Transformer: Predicts COAD ❌ (attention distracted by noise)

Meta-learner combines: GBM ✓ (Base learner's expert knowledge wins)
```

### Mathematical Intuition

If you have N models with accuracy `p` each:

- **Correlated errors**: Ensemble accuracy ≈ `p` (no improvement)
- **Uncorrelated errors**: Ensemble accuracy ≈ `1 - (1-p)^N` (exponentially better)

With 3 models at 80% accuracy:
- Correlated: ~80% ensemble accuracy
- Uncorrelated: ~99.2% ensemble accuracy (if any 2 of 3 are right)

### Summary

**High error diversity = Models fail on different samples = Ensemble can correct individual failures = Better final accuracy**

This is why the parallel hybrid (QML + Transformer → Meta-QML) outperforms either approach alone - the architectural diversity creates uncorrelated errors that the meta-learner can exploit.
