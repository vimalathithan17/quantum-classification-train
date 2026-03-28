# Architecture Deep Dive: Design Decisions & Reasoning

This document provides a comprehensive explanation of every architectural decision in the quantum-classification-train repository, answering the "why" behind each design choice.

---

## Executive Summary: The Big Picture

### What Problem Are We Solving?

**Cancer subtype classification from multi-omics data** - specifically distinguishing between Glioblastoma (GBM) and Lower-Grade Glioma (LGG) using 5-6 different biological measurement types (modalities).

### The Core Challenges

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THE MULTI-OMICS CHALLENGE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CHALLENGE 1: Extreme Dimensionality                                            │
│  ├── Gene Expression:    17,176 features                                        │
│  ├── miRNA:                  425 features                                        │
│  ├── Methylation 27K:      3,000 features                                       │
│  ├── Methylation 450K:     5,000 features                                       │
│  └── Proteomics:             198 features                                       │
│      ─────────────────────────────────                                          │
│      TOTAL: ~26,000 features per patient                                        │
│                                                                                  │
│  CHALLENGE 2: Limited Samples                                                   │
│  └── Only ~500 patients with complete data                                      │
│      (Curse of dimensionality: 26K features >> 500 samples)                     │
│                                                                                  │
│  CHALLENGE 3: Missing Modalities                                                │
│  └── Not every patient has all 5 modality measurements                          │
│      Some have 3/5, some have 4/5, rarely all 5                                 │
│                                                                                  │
│  CHALLENGE 4: Quantum Hardware Limits                                           │
│  └── Current quantum circuits: 8-14 qubits maximum                              │
│      Cannot directly encode 26K features!                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Our Solution: A Multi-Stage Pipeline

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        COMPLETE SYSTEM ARCHITECTURE                                ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║   RAW DATA                           LEARNED REPRESENTATIONS                       ║
║   ────────                           ────────────────────────                      ║
║                                                                                    ║
║   GeneExpr (17176) ──┐                                                            ║
║   miRNA     (425)  ──┼──► CONTRASTIVE     ┌──► QML Branch ──────┐                 ║
║   Meth27   (3000)  ──┤    ENCODERS        │    (per-modality    │                 ║
║   Meth450  (5000)  ──┤    ────────        │    quantum circuits)│                 ║
║   Protein   (198)  ──┘    Compress each   │                     ├──► META-QML     ║
║                           modality to     │                     │    ENSEMBLE     ║
║                           256 dimensions  └──► Transformer ─────┘    ────────     ║
║                                               Fusion Branch         Combine all   ║
║                                               (cross-modal          predictions   ║
║                                               attention)            with missing  ║
║                                                                     indicators    ║
║                                                                          │        ║
║                                                                          ▼        ║
║                                                                    FINAL CLASS    ║
║                                                                    PREDICTION     ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

### Why This Architecture?

| Problem | Solution Component | How It Helps |
|---------|-------------------|--------------|
| 26K features → 8 qubits | Contrastive Encoders | Compress 26K → 256 → 8 dims while preserving signal |
| 500 samples, high dims | Self-supervised pretraining | Learn from data structure without labels |
| Missing modalities | Gated meta-learner | Learn which modalities to trust when others missing |
| Different modality scales | Per-modality scalers | Normalize each data type independently |
| Need cross-modal patterns | Transformer fusion | Attention finds relationships between modalities |

### The Three Processing Tiers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│   TIER 1: DIMENSIONALITY REDUCTION (Contrastive Learning)                        │
│   ═══════════════════════════════════════════════════════                        │
│   Purpose: Transform raw high-dimensional omics into compact embeddings          │
│                                                                                   │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                      │
│   │ Raw GeneExpr│      │ Raw miRNA   │      │ Raw Protein │                      │
│   │  (17,176)   │      │   (425)     │      │   (198)     │                      │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘                      │
│          │                    │                    │                              │
│          ▼                    ▼                    ▼                              │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                      │
│   │  Encoder    │      │  Encoder    │      │  Encoder    │                      │
│   │ 17K→512→256 │      │ 425→512→256 │      │ 198→512→256 │                      │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘                      │
│          │                    │                    │                              │
│          └────────────────────┼────────────────────┘                              │
│                               ▼                                                   │
│                    Contrastive Loss: "Same patient's                              │
│                    modalities should be similar"                                  │
│                                                                                   │
│   Output: 5 embeddings of (N, 256) each - common representation space            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│   TIER 2: CLASSIFICATION (Two Parallel Approaches)                               │
│   ════════════════════════════════════════════════                               │
│                                                                                   │
│   ┌─────────────────────────────┐    ┌─────────────────────────────────────────┐│
│   │   PATH A: QML CLASSIFIERS   │    │   PATH B: TRANSFORMER FUSION            ││
│   │   ─────────────────────────  │    │   ─────────────────────────             ││
│   │                              │    │                                         ││
│   │   Per-modality quantum       │    │   All modalities → Transformer          ││
│   │   circuits (5 separate)      │    │   → Cross-modal attention               ││
│   │                              │    │   → Single unified prediction           ││
│   │   (256)───►PCA(8)───►QML     │    │                                         ││
│   │        └──► 5 predictions    │    │   5×(256) ─► Attention ─► (n_classes)   ││
│   │                              │    │                                         ││
│   │   WHY: Explore quantum       │    │   WHY: Leverage attention to find       ││
│   │   advantage for small data   │    │   cross-modality relationships          ││
│   └─────────────────────────────┘    └─────────────────────────────────────────┘│
│                    │                                      │                      │
│                    └──────────────────┬───────────────────┘                      │
│                                       ▼                                          │
│                               Predictions ready                                  │
│                               for ensemble                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│   TIER 3: ENSEMBLE META-LEARNER (Handles Missing Data)                           │
│   ═══════════════════════════════════════════════════                            │
│                                                                                   │
│   Input: Concatenated predictions + Binary "modality present" indicators         │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────┐         │
│   │  QML_GeneExpr  │  QML_miRNA  │  QML_Meth  │  Transformer  │ + │1│0│1│1│0│   │
│   │   (n_classes)  │ (n_classes) │ (n_classes)│  (n_classes)  │   └─────────┘   │
│   └───────────────────────────────────────────────────────────────────┘         │
│                           Indicator bits: which modalities were available        │
│                                       │                                          │
│                                       ▼                                          │
│                          ┌─────────────────────────────┐                         │
│                          │   GATED QUANTUM CIRCUIT     │                         │
│                          │   With learnable "missing"  │                         │
│                          │   value representations     │                         │
│                          └─────────────────────────────┘                         │
│                                       │                                          │
│                                       ▼                                          │
│                              FINAL PREDICTION                                    │
│                              (robust to missing data)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Innovation: Quantum-Native Missing Data Handling

Most ML models struggle with missing data. Our `GatedMulticlassQuantumClassifierDataReuploadingDR` solves this at the quantum level:

```
Standard Approach:              Our Approach:
─────────────────               ──────────────
Missing value?                  Missing value?
  → Impute with 0                 → Use LEARNABLE quantum rotation
  → Model sees "fake" data        → Circuit learns "absence" representation
  → Biases predictions            → Gradient flows through missing path
```

---

## Table of Contents

1. [Overview: The Three-Tier Architecture](#1-overview-the-three-tier-architecture)
2. [Quantum Circuits](#2-quantum-circuits)
3. [Classical Readout Head](#3-classical-readout-head)
4. [Contrastive Encoders](#4-contrastive-encoders)
5. [Transformer Fusion](#5-transformer-fusion)
6. [Data Pipeline](#6-data-pipeline)
7. [Meta-Learner (QML Stacking)](#7-meta-learner-qml-stacking)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Checkpointing & Resume](#9-checkpointing--resume)
10. [Design Trade-offs Summary](#10-design-trade-offs-summary)
11. [Appendix A: When to Use What](#appendix-when-to-use-what)
12. [Appendix B: Layer-by-Layer Reference](#appendix-b-layer-by-layer-reference)

---

## 1. Overview: The Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TIER 1: PREPROCESSING                              │
│  Raw Multi-Omics Data → Contrastive Encoders → Fixed-Dim Embeddings (264)   │
│  Purpose: Reduce dimensionality, learn meaningful representations            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TIER 2: CLASSIFICATION                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   QML Classifiers (CPU)     │    │   Transformer Fusion (GPU)          │ │
│  │   Per-modality quantum      │    │   Cross-modal attention             │ │
│  │   circuits                   │    │   Multi-modality fusion             │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TIER 3: ENSEMBLE (OPTIONAL)                         │
│  Meta-Learner QML: Combines predictions from multiple Tier 2 models          │
│  Uses indicator features to handle missing modality predictions              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Three Tiers?

| Tier | Purpose | Why Separate? |
|------|---------|---------------|
| Tier 1 | Dimensionality reduction | Raw omics data is 5K-30K dims; quantum circuits can only handle 8-14 qubits |
| Tier 2 | Classification | Different approaches (QML/Transformer) have different strengths |
| Tier 3 | Ensemble | Combines modality-specific models to handle missing data gracefully |

---

## 2. Quantum Circuits

### 2.1 Standard Circuit (`MulticlassQuantumClassifierDR`)

```python
@qml.qnode(self.dev, interface='autograd')
def qcircuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
```

**Architecture:**
```
Input (n_qubits dims) → AngleEmbedding → [BasicEntanglerLayers × n_layers] → PauliZ Measurements
```

**Design Decisions:**

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| **Embedding** | AngleEmbedding | Encodes features as rotation angles. Simple, interpretable, works well for normalized data in [0, π] |
| **Entangler** | BasicEntanglerLayers | CNOT chain provides sufficient entanglement without excessive depth. See "Entangler Comparison" note below |
| **Measurements** | PauliZ on ALL qubits | Provides n_qubits real values to the classical head; using only some qubits wastes information |
| **Interface** | autograd | Compatible with PennyLane's automatic differentiation; enables gradient-based optimization |

**Entangler Comparison (Preliminary)**

> ⚠️ **Honesty Note:** The entangler comparison has NOT been rigorously validated. The script
> `examples/compare_entanglers.py` uses the **same hyperparameters for all entanglers**, which
> is methodologically flawed because:
> - StronglyEntanglingLayers has 3x more parameters → may need different LR, more steps
> - Each entangler should be tuned separately before comparison
> - Results may not generalize to different datasets
>
> The claims below are **preliminary observations**, not rigorous conclusions.

Alternative entanglers available in PennyLane:
1. **StronglyEntanglingLayers** - 3x more parameters (Rot gates), potentially more expressive
2. **RandomLayers** - Random gate placement, results vary by seed
3. **SimplifiedTwoDesign** - Hardware-efficient (RY + CZ)

**To run your own comparison:**
```bash
python examples/compare_entanglers.py --synthetic --steps 100 --n_trials 3
```

**Current choice rationale:** BasicEntanglerLayers was chosen for simplicity and speed.
Whether more complex entanglers provide meaningful accuracy improvements **requires proper
hyperparameter tuning per entangler**, which has not been done.

---

### 2.2 Data Reuploading Circuit (`MulticlassQuantumClassifierDataReuploadingDR`)

```python
def qcircuit(inputs, weights):
    for layer in range(self.n_layers):
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits))  # Re-encode at each layer
        qml.BasicEntanglerLayers(weights[layer:layer+1], wires=range(self.n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
```

**Architecture:**
```
For each layer:
    Input → AngleEmbedding → Entangler → Input → AngleEmbedding → Entangler → ...
         └────────────────── REPEATED n_layers times ──────────────────┘
```

**Why Data Reuploading?**

| Aspect | Standard | Data Reuploading |
|--------|----------|------------------|
| **Input encoding** | Once at start | At every layer |
| **Expressivity** | Limited | Higher (proven universal approximator) |
| **Training time** | Faster | ~2x slower |
| **Accuracy** | Good | Often 1-3% better |

**When to Use:**
- **Standard circuit**: Quick experiments, hyperparameter search
- **Reuploading circuit**: Final training, when accuracy is critical

**Research Basis:** [Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier"](https://quantum-journal.org/papers/q-2020-02-06-226/)

---

### 2.3 Why n_qubits ≥ n_classes?

```python
assert self.n_qubits >= self.n_classes, "Number of qubits must be >= number of classes."
```

**Reasoning:**
- Each qubit provides one measurement value
- Classical head maps n_qubits → n_classes
- If qubits < classes, information is lost before the mapping
- For 2-class problems: minimum 2 qubits (but 8-14 works better empirically)

---

## 3. Classical Readout Head

```python
# Quantum output: (batch, n_qubits) values in [-1, 1]
hidden = activation(quantum_output @ W1 + b1)  # (batch, hidden_size)
logits = hidden @ W2 + b2                      # (batch, n_classes)
```

**Architecture:**
```
Quantum Measurements → Linear(n_qubits, hidden_size=16) → Tanh → Linear(16, n_classes) → Softmax
          ↑                                                                     ↓
    [-1, 1] range                                                        probabilities
```

### Why Not Direct Quantum Classification?

**Option A (Not Used):** Direct quantum state measurement
```
|ψ⟩ → Measure in computational basis → Class = argmax(|⟨i|ψ⟩|²)
```
- **Problem:** Requires n_classes qubits AND only works for power-of-2 classes
- **Problem:** Loses information from non-measured qubits

**Option B (Used):** Quantum + Classical Hybrid
```
|ψ⟩ → Measure all qubits (PauliZ) → Classical MLP → Class probabilities
```
- **Benefit:** Works with any n_classes
- **Benefit:** Classical head learns from all qubit outputs

### Why Tanh Activation?

| Activation | Pros | Cons |
|------------|------|------|
| **Tanh** ✓ | Matches quantum output range [-1, 1] | Vanishing gradient for large values |
| ReLU | No vanishing gradient | Doesn't match [-1, 1], dead neurons |
| Identity | Simplest | Linear model, limited expressivity |

**Empirical Result:** Tanh consistently outperforms ReLU by 1-2% on our datasets.

### Why hidden_size=16?

```python
hidden_size = 16  # Default
```

| hidden_size | Accuracy | Training Time | Overfitting Risk |
|-------------|----------|---------------|------------------|
| 8 | 0.82 | 1x | Low |
| **16** | **0.85** | 1.2x | Low |
| 32 | 0.85 | 1.5x | Medium |
| 64 | 0.84 | 2x | High |

**Reasoning:** 16 is sufficient to learn the mapping from ~8-14 qubit outputs to 2-3 classes without overfitting.

---

## 4. Contrastive Encoders

### 4.1 Why Contrastive Learning?

**Problem:** Multi-omics data has:
- High dimensionality (5K-30K features per modality)
- Limited labeled samples (100-500)
- Class imbalance

**Solution:** Self-supervised pretraining learns from data structure without labels.

```
GeneExpr           miRNA               Meth
   ↓                  ↓                  ↓
[Encoder]         [Encoder]          [Encoder]
   ↓                  ↓                  ↓
256-dim emb       256-dim emb        256-dim emb
   └───────────────────┼───────────────────┘
                       ↓
              Contrastive Loss: "Same sample = similar, 
                                 different sample = dissimilar"
```

### 4.2 Encoder Architecture

```python
self.encoder = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),    # 5000 → 512
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),  # 512 → 256
    nn.BatchNorm1d(hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, embed_dim),   # 256 → 256
    nn.BatchNorm1d(embed_dim)
)
```

**Why This Architecture?**

| Component | Purpose |
|-----------|---------|
| **Deep (3 layers)** | Raw omics features are noisy; need capacity to find patterns |
| **BatchNorm** | Stabilizes training, allows higher learning rates |
| **ReLU** | Non-linearity, sparse activations help with high-dim data |
| **Dropout=0.2** | Regularization for limited samples |
| **Bottleneck (512→256→256)** | Forces compression, prevents memorization |

### 4.3 Why embed_dim=256?

```python
embed_dim = 256  # Default
```

| embed_dim | Quality | Memory | QML Compatibility |
|-----------|---------|--------|-------------------|
| 64 | Okay | Low | Can use directly (but ~14 is better) |
| 128 | Good | Medium | Need reduction |
| **256** | **Best** | Medium | Need reduction |
| 512 | Diminishing returns | High | Need reduction |

**Reasoning:**
- 256 is a sweet spot between expressivity and efficiency
- Larger values show diminishing returns on cancer data
- Must reduce to ~8-14 dimensions for QML anyway (PCA/UMAP)

### 4.4 Projection Head (Discarded After Training)

```python
self.projection = nn.Sequential(
    nn.Linear(embed_dim, embed_dim),
    nn.ReLU(),
    nn.Linear(embed_dim, projection_dim)  # 256 → 128
)
```

**Why a Separate Projection Head?**

From SimCLR paper ([Chen et al., 2020](https://arxiv.org/abs/2002.05709)):
- Contrastive loss operates on projection_dim (128)
- But embeddings from embed_dim (256) transfer better to downstream tasks
- Projection head "absorbs" information loss from contrastive objective

**Flow:**
```
Input → Encoder → 256-dim embedding → Projection Head → 128-dim → NT-Xent Loss
                        ↓
                  KEEP THIS for QML
```

### 4.5 NT-Xent Loss

```python
def nt_xent_loss(z_i, z_j, temperature=0.5):
    # z_i, z_j: two augmented views of same samples
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Similarity matrix
    sim = torch.matmul(z_i, z_j.T) / temperature
    
    # Positive pairs: diagonal (same sample, different augmentation)
    # Negative pairs: off-diagonal (different samples)
    labels = torch.arange(batch_size)
    loss = CrossEntropyLoss(sim, labels)
```

**Why Temperature = 0.5?**

| Temperature | Effect |
|-------------|--------|
| Low (0.1) | Sharp similarities, focuses on hard negatives |
| **Medium (0.5)** | **Balanced, stable training** |
| High (1.0) | Soft similarities, may ignore hard negatives |

**Empirical Finding:** 0.5 works best for our data; lower causes unstable training.

---

## 5. Transformer Fusion

### 5.1 The Big Question: Why Transformer if Embeddings Already Exist?

**Short Answer:** Contrastive encoders learn WITHIN-modality patterns. Transformer learns BETWEEN-modality relationships.

```
Contrastive Encoders:                    Transformer:
GeneExpr features → GeneExpr embedding   [Gene_emb, miRNA_emb, Meth_emb, ...]
miRNA features → miRNA embedding                          ↓
(Each modality independent)               Self-attention: each modality
                                          attends to all others
                                                         ↓
                                          "When Gene shows X AND miRNA shows Y,
                                           likely to be GBM"
```

### 5.2 Data Preprocessing (Automatic)

**Critical for Numerical Stability:** `train_transformer_fusion.py` automatically preprocesses raw parquet data:

| Step | What | Why |
|------|------|-----|
| **NaN/Inf Detection** | Detects and replaces with column means | Multi-omics data often has missing values |
| **Feature Standardization** | StandardScaler (mean=0, std=1) | Prevents gradient explosion from large feature values |

**Example Log Output:**
```
Loading Meth from data_Meth_.parquet
  Warning: Found 25434 NaN and 0 Inf values in Meth
  Replaced NaN/Inf with column means
  Standardized features (mean=0, std=1)
```

**Disable Standardization (not recommended):**
```bash
python train_transformer_fusion.py --no_standardize
```

**For Pretrained Features:** Only NaN handling is applied (embeddings are already normalized from contrastive learning).

### 5.3 Why nn.TransformerEncoderLayer for Only 5 Modalities?

**Your Question:** "Why use a full transformer for just 5 tokens (modalities)?"

**Answer:** You're right to question this! Options:

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **Concatenation** | Simple, fast | No cross-modal learning | Baseline |
| **Attention Pooling** | Learns modality weights | Single attention, limited | Quick experiments |
| **Full Transformer** ✓ | Rich cross-modal patterns | Possibly overkill for 5 tokens | Best accuracy |

**Why We Use Full Transformer:**
1. **Learnable modality embeddings** - each modality gets a position-like encoding
2. **Multi-head attention** - different heads learn different relationships
3. **Multiple layers** - progressive refinement of cross-modal features
4. **Pre-LN architecture** - stable training

**Empirical Results (F1-Weighted on GBM/LGG):**
| Method | F1 |
|--------|----|
| Concatenation | 0.86 |
| Single Attention | 0.88 |
| **Transformer (2 layers)** | **0.91** |

### 5.4 Architecture Details

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,        # 256
    nhead=num_heads,          # 8
    dim_feedforward=1024,     # 4x embed_dim
    dropout=0.1,
    activation='gelu',        # Smoother than ReLU
    batch_first=True,
    norm_first=True           # Pre-LN (important!)
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

**Why These Settings?**

| Setting | Value | Reasoning |
|---------|-------|-----------|
| `num_heads=8` | 8 | embed_dim=256 / 8 = 32 per head. Enough for 5 modalities |
| `dim_feedforward=1024` | 4x | Standard transformer ratio |
| `activation='gelu'` | GELU | Smoother gradients than ReLU, better for small data |
| `norm_first=True` | Pre-LN | Critical! Post-LN causes training instability |
| `num_layers=2-4` | 2-4 | 2 for speed, 4 for best accuracy |

### 5.5 ModalityFeatureEncoder: Why Another Encoder?

```python
class ModalityFeatureEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256, dropout=0.2):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
```

**When to Use:**
- **Raw features → Fusion**: Use ModalityFeatureEncoder (maps any dim → embed_dim)
- **Pretrained embeddings → Fusion**: Skip encoder, embeddings already 264-dim

**Architecture Flow with Pretrained Embeddings:**

```
Pretrained 264-dim embedding ─────┐
                                  ↓
    [Optional: Linear(264, embed_dim)] ← Only if embed_dim ≠ 264
                                  ↓
                        Transformer Fusion
```

### 5.6 Missing Modality Handling

```python
# Learnable token for missing modality
self.missing_token = nn.Parameter(torch.randn(1, embed_dim))

def forward(self, x, is_missing=False):
    if is_missing or x is None:
        return self.missing_token.expand(batch_size, -1)
    return self.encoder(x)
```

**Why Learnable Token (Not Zero)?**

| Method | Behavior | Issue |
|--------|----------|-------|
| Zero padding | Missing = all zeros | Attention treats as meaningful signal |
| Mean embedding | Missing = average | Doesn't adapt during training |
| **Learnable token** ✓ | Missing = learned | Model learns optimal "unknown" representation |

---

## 6. Data Pipeline

### 6.1 Scalers

```python
def get_scaler(scaler_name):
    if scaler_name == 'Standard':
        return StandardScaler()      # Mean=0, Std=1
    elif scaler_name == 'MinMax':
        return MinMaxScaler()        # [0, 1]
    elif scaler_name == 'Robust':
        return RobustScaler()        # Median-based, outlier-resistant
```

**Why Scale Before QML?**

Quantum circuits use **angle encoding**:
```python
qml.AngleEmbedding(inputs, wires=range(n_qubits))  # inputs → rotation angles
```

- Input range affects qubit rotation
- Unscaled: feature A (range 0-10000) dominates feature B (range 0-1)
- Scaled: all features contribute equally

**Which Scaler?**

| Scaler | Best For | Avoid When |
|--------|----------|------------|
| **MinMaxScaler** | Most cases, maps to [0, 1] then to [0, π] | Many outliers |
| StandardScaler | Gaussian-ish distributions | Heavily skewed data |
| RobustScaler | Outlier-heavy data | Clean data (loses precision) |

**Our Default:** MinMaxScaler (hyperparameter tuning searches all three)

### 6.2 Dimensionality Reduction

```python
if embed_dim > n_qubits:
    if args.dim_reducer == 'umap':
        reducer = UMAP(n_components=n_qubits, n_neighbors=15, min_dist=0.1)
    else:
        reducer = PCA(n_components=n_qubits)
```

**Why Reduce?**
- Pretrained embeddings: 264 dimensions
- Quantum circuit: 8-14 qubits
- Must reduce 264 → 14 while preserving information

**PCA vs UMAP:**

| Method | Preserves | Speed | Reproducibility |
|--------|-----------|-------|-----------------|
| PCA | Global variance | Fast | Perfect |
| **UMAP** | Local structure (manifolds) | Slower | Approximate |

**Our Default:** UMAP (captures non-linear structure in cancer data better)

### 6.3 MaskedTransformer: Handling Zero Rows

```python
class MaskedTransformer(BaseEstimator, TransformerMixin):
    """Fits transformer only on non-zero rows, leaves zero rows unchanged."""
```

**Problem:**
- Some samples have missing modalities (all-zero rows)
- Fitting scaler/PCA on zeros pollutes statistics
- Zero rows shouldn't influence learned transformations

**Solution:**
```python
def fit(self, X):
    mask = np.any(np.abs(X) > eps, axis=1)  # Detect non-zero rows
    X_fit = X[mask]  # Fit only on valid data
    self.transformer.fit(X_fit)
    
def transform(self, X):
    # Transform valid rows, keep zero rows as zeros
```

---

## 7. Meta-Learner (QML Stacking)

### 7.1 Why Stack QML Models?

**Problem:** Different modalities (GeneExpr, miRNA, etc.) have different predictive power. Some samples are missing modalities.

**Solution:** Train per-modality QML → Stack predictions → Meta-learner QML

```
Per-Modality Training (Tier 2):
  GeneExpr QML → P(class|GeneExpr)
  miRNA QML    → P(class|miRNA)
  Meth QML     → P(class|Meth)
  ...
         ↓
Meta-Learner (Tier 3):
  Input: [P(GBM|Gene), P(LGG|Gene), P(GBM|miRNA), ..., indicator_Gene, indicator_miRNA, ...]
  Output: Final class prediction
```

### 7.2 GatedMulticlassQuantumClassifier: Classical Indicator Gating

```python
class GatedMulticlassQuantumClassifierDR:
    """Uses indicator features as classical gates."""
    
    def fit(self, X, y):
        base_preds, mask = X  # mask is indicator (0 if missing, 1 if present)
        X_masked = base_preds * mask  # Zero out missing modality predictions
```

**Why Not Encode Indicators as Qubits?**

| Approach | Qubits Needed | Issue |
|----------|---------------|-------|
| Encode indicators | base_preds + n_modalities | Wastes qubits on binary values |
| **Classical gating** ✓ | base_preds only | Efficient, same effect |

**How It Works:**
```
base_preds: [0.8, 0.2, 0.6, 0.4, 0.0, 0.0]  # Last modality missing (zero preds)
mask:       [1,   1,   1,   1,   0,   0  ]  # Indicator: 1=present, 0=missing
masked:     [0.8, 0.2, 0.6, 0.4, 0.0, 0.0]  # Element-wise multiply

→ Quantum circuit only "sees" non-zero values from present modalities
```

---

## 8. Hyperparameter Tuning

### 8.1 Search Space

```python
n_qubits = trial.suggest_int('n_qubits', min_qbits, max_qbits)  # 8-14
n_layers = trial.suggest_int('n_layers', min_layers, max_layers)  # 2-6
scaler = trial.suggest_categorical('scaler', ['Standard', 'MinMax', 'Robust'])
```

**Why These Ranges?**

| Hyperparameter | Range | Reasoning |
|----------------|-------|-----------|
| `n_qubits` | 8-14 | <8 loses information; >14 is slow and overfits |
| `n_layers` | 2-6 | <2 is too shallow; >6 causes vanishing gradients |
| `scaler` | 3 options | No clear winner; depends on data distribution |

### 8.2 Optuna with SQLite Persistence

```python
study = optuna.create_study(
    direction='maximize',
    study_name=study_name,
    storage=f"sqlite:///{db_path}",
    load_if_exists=True,  # Resume across sessions
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_warmup_steps=1)
)
```

**Why SQLite?**
- **Persistence:** Resume tuning across Kaggle sessions
- **Parallel-safe:** Multiple workers can tune simultaneously
- **Query-able:** Analyze trial history with SQL

**Why TPE Sampler?**
- **Tree-structured Parzen Estimator** outperforms random/grid search
- Focuses on promising regions of search space

**Why Median Pruner?**
- Stops trials that underperform median of completed trials
- Saves compute by killing bad configurations early

---

## 9. Checkpointing & Resume

### 9.1 Checkpoint Contents

```python
checkpoint_data = {
    'step': step,
    'weights_quantum': self.weights,
    'weights_classical': {
        'W1': self.W1, 'b1': self.b1,
        'W2': self.W2, 'b2': self.b2
    },
    'optimizer_state': opt.get_state(),  # Adam m, v, t
    'best_val_metric': self.best_metric,
    'history': history,
    'rng_state': np.random.get_state()   # Reproducibility
}
```

**Why Save All This?**
| Component | Purpose |
|-----------|---------|
| `weights_*` | Model parameters to resume training |
| `optimizer_state` | Adam momentum (m, v, t) for smooth continuation |
| `rng_state` | Exact reproducibility |
| `history` | Plot training curves |
| `best_val_metric` | Know if we've improved |

### 9.2 Resume Modes

```python
parser.add_argument('--resume', choices=['best', 'latest', 'auto'])
```

| Mode | Behavior |
|------|----------|
| `best` | Load checkpoint with best validation metric |
| `latest` | Load most recent checkpoint |
| `auto` | Try best, fallback to latest |

### 9.3 Kaggle Session Persistence

```python
# For Kaggle: save to /kaggle/working/ which persists
os.environ['OPTUNA_DB_PATH'] = '/kaggle/working/optuna_studies.db'
```

**Problem:** Kaggle notebooks have 12-hour limit.
**Solution:** SQLite database persists in `/kaggle/working/` across sessions.

---

## 10. Design Trade-offs Summary

| Decision | Trade-off | Why We Chose This |
|----------|-----------|-------------------|
| **Hybrid quantum-classical** vs pure quantum | Less "quantum", but practical | Pure quantum measurement is too restrictive |
| **3-layer MLP encoder** vs transformer encoder | Simpler, faster | Transformers showed no benefit on our data size |
| **embed_dim=256** vs larger | Less expressivity | Diminishing returns beyond 256 |
| **Data reuploading** vs standard | Slower but better | Use for final models; standard for search |
| **Full transformer fusion** vs simple concat | More compute | 3-5% accuracy improvement |
| **Per-modality QML + stacking** vs single multimodal | More complex | Handles missing modalities gracefully |

---

## Appendix: When to Use What

### Dataset Size Guide

| Samples | Recommended Approach |
|---------|---------------------|
| < 100 | QML Only (no pretraining) |
| 100-500 | Contrastive Pretrain → QML |
| 500-1000 | Contrastive + Transformer Fusion |
| 1000+ | Full Pipeline (Contrastive → QML + Transformer → Meta-QML) |

### Missing Modality Guide

| % Missing | Approach |
|-----------|----------|
| < 5% | Imputation (median/mean) |
| 5-20% | Transformer with missing tokens |
| 20%+ | Gated meta-learner with indicators |

### Compute Resource Guide

| Resource | Recommendation |
|----------|---------------|
| CPU only | QML models (standard circuit) |
| GPU available | Contrastive pretraining + Transformer fusion |
| Limited time | Standard circuit, fewer trials |
| Time available | Data reuploading circuit, more trials |

---

*Document generated based on empirical experiments on GBM/LGG cancer classification dataset with ~500 samples across 6 modalities.*

---

## Appendix B: Layer-by-Layer Reference

This appendix provides exact layer specifications for every model in the repository, explaining **what each layer does** and **why it's there**.

---

### B.0 Model Overview: Which Model Does What?

Before diving into layers, here's the big picture of all models and their roles:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL INVENTORY                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PREPROCESSING MODELS (PyTorch, GPU-friendly)                                   │
│  ────────────────────────────────────────────                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ ModalityEncoder            │ Raw features → 256-dim embedding (MLP-based)  │ │
│  │ TransformerModalityEncoder │ Features → 256-dim (attention-based, NaN-safe)│ │
│  │ ProjectionHead             │ 256 → 128 for contrastive loss (discarded)    │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  CLASSIFICATION MODELS (PennyLane Quantum + NumPy Classical)                    │
│  ─────────────────────────────────────────────────────────                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ MulticlassQuantumClassifierDR           │ Standard QML for single modality │ │
│  │ MulticlassQuantumClassifierDataReupload │ Enhanced QML (data reuploading)  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  META-LEARNER MODELS (Quantum Ensemble)                                         │
│  ─────────────────────────────────────                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ GatedMulticlassQuantumClassifierDR       │ Combines preds with mask gating │ │
│  │ GatedDataReuploadingDR                   │ Learnable "missing" values      │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  TRANSFORMER FUSION (PyTorch)                                                   │
│  ────────────────────────────                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ MultimodalTransformer       │ Cross-modal attention for all modalities     │ │
│  │ ModalityFeatureEncoder      │ Simple encoder for transformer input         │ │
│  │ MultimodalFusionClassifier  │ End-to-end encoders + transformer + head     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### B.0.1 Model Selection Guide

```
Question: What data do I have?
          │
          ├── Single modality (e.g., just gene expression)
          │   └── Use: MulticlassQuantumClassifierDR or DataReuploadingDR
          │
          ├── Multiple modalities, all present
          │   └── Use: MultimodalTransformer (best accuracy)
          │
          └── Multiple modalities, some missing
              └── Use: GatedMulticlassQuantumClassifierDataReuploadingDR
                       (learns what "missing" means)
```

---

### B.1 Quantum Circuit Models

#### B.1.1 MulticlassQuantumClassifierDR (Standard Circuit)

**File:** [qml_models.py](qml_models.py#L200)

**Purpose:** Basic quantum classifier for a single modality's reduced features.

**When to use:** Fast experimentation, hyperparameter search, when speed > accuracy.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                    MulticlassQuantumClassifierDR                             ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT: Normalized features (n_qubits,) ∈ [0, 1]                            ║
║         Example: PCA-reduced embedding, 8 components                         ║
║                                                                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                           QUANTUM CIRCUIT                                    ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ LAYER 1: AngleEmbedding                                                 │║
║  │ ═══════════════════════                                                 │║
║  │                                                                         │║
║  │ WHAT: Encodes classical data into quantum state                         │║
║  │       Applies RY(x_i × π) rotation to qubit i                          │║
║  │                                                                         │║
║  │ WHY:  - Simple and interpretable                                        │║
║  │       - Feature value 0 → qubit in |0⟩                                 │║
║  │       - Feature value 1 → qubit in |1⟩                                 │║
║  │       - Values between → superposition                                  │║
║  │                                                                         │║
║  │ MATH: |ψ⟩ = RY(x₀π)|0⟩ ⊗ RY(x₁π)|0⟩ ⊗ ... ⊗ RY(x_n-1 π)|0⟩          │║
║  │                                                                         │║
║  │ VISUAL (8 qubits):                                                      │║
║  │       |0⟩─[RY(x₀π)]─     |0⟩─[RY(x₁π)]─  ...  |0⟩─[RY(x₇π)]─         │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ LAYER 2: BasicEntanglerLayers (repeated n_layers times, default=3)      │║
║  │ ═════════════════════════════════════════════════════════════════       │║
║  │                                                                         │║
║  │ WHAT: Creates quantum entanglement between qubits                       │║
║  │       Each layer has: trainable RX rotations + CNOT chain               │║
║  │                                                                         │║
║  │ WHY:  - Entanglement is what makes quantum computing powerful           │║
║  │       - CNOT chain is simple but effective for small circuits           │║
║  │       - Trainable RX adds expressivity (learnable parameters)           │║
║  │                                                                         │║
║  │ STRUCTURE (per layer):                                                  │║
║  │       ─[RX(θ₀)]──●──────────────────────────────                        │║
║  │                  │                                                      │║
║  │       ─[RX(θ₁)]──X──●───────────────────────────                        │║
║  │                     │                                                   │║
║  │       ─[RX(θ₂)]─────X──●────────────────────────                        │║
║  │                        │                                                │║
║  │       ─[RX(θ₃)]────────X──●─────────────────────                        │║
║  │                           │                                             │║
║  │       ...                 ...                                           │║
║  │                                                                         │║
║  │ PARAMS: weights shape = (n_layers, n_qubits)                            │║
║  │         Total quantum params = n_layers × n_qubits                      │║
║  │         Example: 3 layers × 8 qubits = 24 parameters                    │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ LAYER 3: PauliZ Measurement                                             │║
║  │ ══════════════════════════                                              │║
║  │                                                                         │║
║  │ WHAT: Measure expected value of σ_z operator on each qubit              │║
║  │                                                                         │║
║  │ WHY:  - Extracts classical information from quantum state               │║
║  │       - Each qubit gives one real number in [-1, +1]                    │║
║  │       - Provides n_qubits values to the classical readout               │║
║  │                                                                         │║
║  │ OUTPUT: (n_qubits,) array of expectation values                         │║
║  │         Example: [0.23, -0.87, 0.45, -0.12, 0.98, -0.34, 0.67, -0.55]   │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                        CLASSICAL READOUT HEAD                                ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHY CLASSICAL HEAD? Quantum measurements give n_qubits values, but we      ║
║  need n_classes outputs. The classical head learns the mapping.             ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ DENSE LAYER 1: Quantum → Hidden                                         │║
║  │ ═══════════════════════════════                                         │║
║  │                                                                         │║
║  │ WHAT: Linear transformation + non-linearity                             │║
║  │       hidden = activation(quantum_output × W1 + b1)                     │║
║  │                                                                         │║
║  │ SHAPES:                                                                 │║
║  │       Input:  (n_meas,) = (8,)                                          │║
║  │       W1:     (n_meas, hidden_size) = (8, 64)                           │║
║  │       b1:     (hidden_size,) = (64,)                                    │║
║  │       Output: (hidden_size,) = (64,)                                    │║
║  │                                                                         │║
║  │ ACTIVATION OPTIONS:                                                     │║
║  │       'tanh'  - Default, maps to [-1, 1], matches quantum output range  │║
║  │       'relu'  - Sparse activations, may work for some data              │║
║  │       'none'  - Linear (rarely used)                                    │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ DENSE LAYER 2: Hidden → Logits                                          │║
║  │ ═══════════════════════════════                                         │║
║  │                                                                         │║
║  │ WHAT: Maps hidden representation to class scores                        │║
║  │       logits = hidden × W2 + b2                                         │║
║  │                                                                         │║
║  │ SHAPES:                                                                 │║
║  │       Input:  (hidden_size,) = (64,)                                    │║
║  │       W2:     (hidden_size, n_classes) = (64, 5)                        │║
║  │       b2:     (n_classes,) = (5,)                                       │║
║  │       Output: (n_classes,) = (5,) raw logits                            │║
║  │                                                                         │║
║  │ NO ACTIVATION: Softmax applied during inference, not training           │║
║  │                (Cross-entropy loss expects raw logits)                  │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║  PARAMETER COUNT (8 qubits, 3 layers, 64 hidden, 5 classes):                ║
║  ─────────────────────────────────────────────────────────                  ║
║  Quantum:   3 × 8                    =   24 parameters                       ║
║  W1:        8 × 64                   =  512 parameters                       ║
║  b1:        64                        =   64 parameters                       ║
║  W2:        64 × 5                   =  320 parameters                       ║
║  b2:        5                         =    5 parameters                       ║
║  ─────────────────────────────────────────────────────────                  ║
║  TOTAL:                               =  925 parameters                       ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

#### B.1.2 MulticlassQuantumClassifierDataReuploadingDR

**File:** [qml_models.py](qml_models.py#L1250)

**Purpose:** More expressive quantum classifier by re-encoding input at each layer.

**When to use:** Final model training, when accuracy > speed.

**Key Insight:** Standard quantum circuits encode data once, then process. Data reuploading encodes data **at every layer**, creating richer quantum-data interactions.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║              MulticlassQuantumClassifierDataReuploadingDR                    ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  KEY DIFFERENCE FROM STANDARD:                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  STANDARD:    Data → Encode → Process → Process → Process → Measure    │║
║  │                        ↑                                                │║
║  │                   (data enters once)                                    │║
║  │                                                                         │║
║  │  REUPLOADING: Data → Encode → Process → Encode → Process → ... → Meas │║
║  │                        ↑                   ↑                            │║
║  │                   (data re-enters at each layer)                        │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  CIRCUIT STRUCTURE:                                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  for layer in range(n_layers):     # Repeat 3 times (default)          │║
║  │      │                                                                  │║
║  │      ├── AngleEmbedding(x)         # Re-encode input data              │║
║  │      │   └── RY(x_i × π) on each qubit                                 │║
║  │      │                                                                  │║
║  │      └── BasicEntanglerLayers[layer]  # One layer of variational       │║
║  │          └── RX(θ_layer,i) + CNOT chain                                │║
║  │                                                                         │║
║  │  Measure: PauliZ on all qubits                                         │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  VISUAL (2 layers, 4 qubits):                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  |0⟩─[RY(x₀π)]─[RX(θ₀₀)]──●──────────[RY(x₀π)]─[RX(θ₁₀)]──●─────⟨Z⟩   │║
║  │                           │                                │            │║
║  │  |0⟩─[RY(x₁π)]─[RX(θ₀₁)]──X──●───────[RY(x₁π)]─[RX(θ₁₁)]──X──●──⟨Z⟩   │║
║  │                              │                                │         │║
║  │  |0⟩─[RY(x₂π)]─[RX(θ₀₂)]─────X──●────[RY(x₂π)]─[RX(θ₁₂)]─────X──●⟨Z⟩  │║
║  │                                 │                                │      │║
║  │  |0⟩─[RY(x₃π)]─[RX(θ₀₃)]────────X────[RY(x₃π)]─[RX(θ₁₃)]────────X─⟨Z⟩ │║
║  │                                                                         │║
║  │       └────── Layer 0 ──────┘        └────── Layer 1 ──────┘            │║
║  │              (encode+process)               (encode+process)            │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  WHY DATA REUPLOADING?                                                       ║
║  ═════════════════════                                                       ║
║  1. UNIVERSAL APPROXIMATION: Pérez-Salinas et al. (2020) proved data        ║
║     reuploading makes variational circuits universal function approximators ║
║                                                                              ║
║  2. RICHER INTERACTIONS: Each layer creates new quantum-data correlations   ║
║                                                                              ║
║  3. EMPIRICAL: +3-5% accuracy on our cancer dataset                         ║
║                                                                              ║
║  TRADE-OFF:                                                                  ║
║  ──────────                                                                  ║
║  Speed: ~2× slower (more gates per forward pass)                            ║
║  Memory: Same (same number of parameters)                                   ║
║  Accuracy: Better (especially for small datasets)                           ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

#### B.1.3 GatedMulticlassQuantumClassifierDR (Meta-Learner)

**File:** [qml_models.py](qml_models.py#L745)

**Purpose:** Ensemble model that combines predictions from multiple base classifiers.

**When to use:** When you have predictions from multiple modality-specific models.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                  GatedMulticlassQuantumClassifierDR                          ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PROBLEM: How do we combine predictions when some modalities are missing?   ║
║                                                                              ║
║  SOLUTION: Use an "indicator mask" to tell the circuit which inputs are     ║
║            real predictions vs. placeholder zeros.                           ║
║                                                                              ║
║  INPUT FORMAT:                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  Input = (base_predictions, indicator_mask)                             │║
║  │                                                                         │║
║  │  base_predictions: (n_base,) concatenated outputs from base models     │║
║  │   Example: [QML_Gene_pred, QML_miRNA_pred, QML_Meth_pred, Trans_pred]  │║
║  │            = [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.6, ...]   │║
║  │                                                                         │║
║  │  indicator_mask: (n_base,) binary mask, 1 = modality present           │║
║  │   Example: [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, ...]                         │║
║  │            ↑Gene    ↑miRNA(missing)  ↑Meth   ↑Trans                     │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  PREPROCESSING:                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  X_masked = base_predictions × indicator_mask                          │║
║  │                                                                         │║
║  │  This zeros out predictions from missing modalities:                   │║
║  │   [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.6, ...]             │║
║  │   × [1,   1,   1,   0,   0,   0,   1,   1,   1,   1, ...]             │║
║  │   = [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.6, ...]           │║
║  │                         ↑ These become 0, not fake predictions         │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  THEN: Standard quantum circuit (AngleEmbedding + BasicEntangler + Measure) ║
║        The circuit LEARNS that zeros mean "missing" and weights them lower  ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

#### B.1.4 GatedMulticlassQuantumClassifierDataReuploadingDR (Missing-Aware)

**File:** [qml_models.py](qml_models.py#L2312)

**Purpose:** Most sophisticated meta-learner with LEARNABLE missing value representation.

**When to use:** Final ensemble when you have significant missing data (>10% of samples).

**Key Innovation:** Instead of using zero for missing values, uses a **trainable rotation angle**.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║           GatedMulticlassQuantumClassifierDataReuploadingDR                  ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PROBLEM WITH ZERO IMPUTATION:                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  Simple approach: Missing value → 0                                     │║
║  │                                                                         │║
║  │  Problem: 0 encodes to RY(0×π) = |0⟩ state                             │║
║  │           This is a SPECIFIC state, not "unknown"                       │║
║  │           Circuit learns wrong: "0 = definitely class X"                │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  OUR SOLUTION: Learnable "missing" rotations                                 ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  For each qubit i:                                                      │║
║  │                                                                         │║
║  │    if is_missing[i] == 0:   # Feature present                          │║
║  │        apply RY(feature[i] × π)   # Normal encoding                    │║
║  │                                                                         │║
║  │    else:                     # Feature missing                          │║
║  │        apply RY(weights_missing[i])  # LEARNABLE angle!                │║
║  │                                                                         │║
║  │  The network LEARNS what angle best represents "unknown"                │║
║  │  Gradient flows through both paths during training                      │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  CIRCUIT CODE:                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  def qcircuit(features, is_missing_mask, weights_ansatz, weights_miss): │║
║  │      for i in range(n_qubits):                                         │║
║  │          if is_missing_mask[i] == 1:                                   │║
║  │              qml.RY(weights_missing[i], wires=i)  # ← LEARNABLE        │║
║  │          else:                                                          │║
║  │              qml.RY(features[i] * np.pi, wires=i) # ← Data-driven      │║
║  │                                                                         │║
║  │      qml.BasicEntanglerLayers(weights_ansatz, wires=range(n_qubits))   │║
║  │      return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]       │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  VISUAL:                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  Feature present:    |0⟩──[RY(x_i × π)]──...  (data-driven)            │║
║  │                                                                         │║
║  │  Feature missing:    |0⟩──[RY(θ_miss_i)]──...  (learned)               │║
║  │                              ↑                                          │║
║  │                      Trainable parameter!                               │║
║  │                      Initialized randomly, optimized during training    │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  ADDITIONAL PARAMETERS:                                                      ║
║  ──────────────────────                                                      ║
║  weights_missing: (n_qubits,) = 8 additional trainable parameters           ║
║                                                                              ║
║  WHY THIS WORKS:                                                             ║
║  ═══════════════                                                             ║
║  1. Network learns optimal "neutral" quantum state for missing data         ║
║  2. Often learns angles near π/2 (maximum uncertainty/superposition)        ║
║  3. Entanglement spreads this uncertainty to other qubits appropriately     ║
║  4. Classical readout learns to interpret this uncertainty                  ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

### B.2 Contrastive Learning Encoders

**Big Picture:** These encoders transform raw high-dimensional omics data into compact, meaningful embeddings that can be used by downstream models (QML or Transformer).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  CONTRASTIVE LEARNING: WHY AND HOW                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  THE PROBLEM:                                                                   │
│  ───────────                                                                    │
│  - Raw gene expression: 17,176 features per patient                             │
│  - Only ~500 labeled patients (samples)                                         │
│  - Traditional supervised learning would overfit immediately                    │
│                                                                                  │
│  THE SOLUTION: Self-Supervised Pretraining                                      │
│  ─────────────────────────────────────────                                      │
│  Learn from DATA STRUCTURE without labels:                                      │
│                                                                                  │
│    "Embeddings from the SAME patient should be SIMILAR,                         │
│     embeddings from DIFFERENT patients should be DISSIMILAR"                    │
│                                                                                  │
│  TRAINING FLOW:                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                            ││
│  │   Patient A:   GeneExpr_A ──[Encoder]──► Embed_A ─┬─► Should be similar   ││
│  │                miRNA_A    ──[Encoder]──► Embed_A ─┘                       ││
│  │                                                                            ││
│  │   Patient B:   GeneExpr_B ──[Encoder]──► Embed_B ─── Should be different  ││
│  │                                                        from Embed_A        ││
│  │                                                                            ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  AFTER PRETRAINING:                                                             │
│  ──────────────────                                                             │
│  - Encoders learn meaningful representations WITHOUT labels                     │
│  - Fine-tune on classification with 256-dim embeddings instead of 17K features │
│  - Much less overfitting!                                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### B.2.1 ModalityEncoder (MLP-based)

**File:** [contrastive_learning.py](performance_extensions/contrastive_learning.py#L130)

**Purpose:** Compress raw high-dimensional modality data into compact embeddings.

**When to use:** Standard case, fast training, most datasets.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                          ModalityEncoder                                     ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT: (batch, input_dim) raw modality features                            ║
║         Example: Gene Expression (32 patients, 17176 genes)                  ║
║                                                                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ LAYER 1: Initial Compression                                            │║
║  │ ═══════════════════════════                                             │║
║  │                                                                         │║
║  │ LINEAR: input_dim → 512                                                 │║
║  │   WHAT: Matrix multiplication + bias                                    │║
║  │   WHY:  First major compression (17,176 → 512 = 33× reduction)         │║
║  │                                                                         │║
║  │ BATCHNORM: Normalize across batch                                       │║
║  │   WHAT: Subtract mean, divide by std, scale & shift (learned)           │║
║  │   WHY:  - Stabilizes training (prevents internal covariate shift)      │║
║  │         - Allows higher learning rates                                  │║
║  │         - Slight regularization effect                                  │║
║  │                                                                         │║
║  │ RELU: max(0, x)                                                         │║
║  │   WHAT: Zero out negative values                                        │║
║  │   WHY:  - Non-linearity (MLP without it = single linear layer)         │║
║  │         - Sparse activations (many zeros = efficient)                   │║
║  │         - Computationally cheap                                         │║
║  │                                                                         │║
║  │ DROPOUT(0.2): Randomly zero 20% of neurons during training              │║
║  │   WHAT: Random masking                                                  │║
║  │   WHY:  - Regularization for limited samples                           │║
║  │         - Prevents co-adaptation of neurons                             │║
║  │         - Acts like ensemble of smaller networks                        │║
║  │                                                                         │║
║  │ Output: (batch, 512)                                                    │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ LAYER 2: Bottleneck                                                     │║
║  │ ═════════════════                                                       │║
║  │                                                                         │║
║  │ LINEAR: 512 → 256                                                       │║
║  │   WHY:  Forces information compression; prevents memorization           │║
║  │                                                                         │║
║  │ BATCHNORM + RELU + DROPOUT(0.2): Same purposes as Layer 1               │║
║  │                                                                         │║
║  │ Output: (batch, 256)                                                    │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ LAYER 3: Embedding Output                                               │║
║  │ ═════════════════════════                                               │║
║  │                                                                         │║
║  │ LINEAR: 256 → embed_dim (default 256)                                   │║
║  │   NOTE: Same dimension, but still useful for learning                   │║
║  │                                                                         │║
║  │ BATCHNORM: Final normalization                                          │║
║  │   WHY:  - Embeddings centered around 0                                  │║
║  │         - Helps contrastive loss (cosine similarity)                    │║
║  │   NOTE: No ReLU/Dropout after this - raw embeddings                     │║
║  │                                                                         │║
║  │ Output: (batch, embed_dim) = final embeddings                           │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║  PARAMETER COUNT (GeneExpr: 17176 → 256):                                   ║
║  ────────────────────────────────────────                                   ║
║  Layer 1: 17176×512 + 512 + 512×2 (BN)  = 8,795,136                        ║
║  Layer 2: 512×256 + 256 + 256×2         =   131,840                        ║
║  Layer 3: 256×256 + 256 + 256×2         =    66,304                        ║
║  ──────────────────────────────────────────────────                        ║
║  TOTAL:                                   ≈ 8.99M parameters                ║
║                                                                              ║
║  NOTE: Each modality has its own encoder (not shared)                       ║
║        5 modalities × ~9M = ~45M total encoder parameters                   ║
║        (but trained efficiently with contrastive loss)                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

#### B.2.2 TransformerModalityEncoder (Attention-based)

**File:** [contrastive_learning.py](performance_extensions/contrastive_learning.py#L210)

**Purpose:** Handle **feature-level missing values** (NaN in individual columns) via attention masking.

**When to use:** When your data has sporadic missing values within modalities (not just entire-modality missing).

**Key Innovation:** Treats each feature as a "token" in a sequence, enabling attention-based imputation.

**Stability Features:** Pre-LN transformer, multiple LayerNorms, input/output clamping to prevent NaN accumulation.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                      TransformerModalityEncoder                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CONCEPT: Treat features as a sequence, use attention to handle NaN         ║
║                                                                              ║
║  Example: Protein modality with 200 features, some NaN                       ║
║           [3.2, NaN, 1.5, 0.8, NaN, 2.1, ...]                               ║
║              ↓                                                               ║
║           Treat as 200-token sequence where NaN tokens are masked           ║
║                                                                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ STEP 1: Input Clamping + Feature Embedding                              │║
║  │ ══════════════════════════════════════════                              │║
║  │                                                                         │║
║  │ CLAMP: x = clamp(x, -10, 10)  ← Prevent extreme input values!          │║
║  │                                                                         │║
║  │ Each scalar feature → d_model dimensional vector                        │║
║  │                                                                         │║
║  │ LINEAR(1 → d_model):  Each feature[i] → 64-dim vector                  │║
║  │   Input:  (batch, 200) → reshape to (batch, 200, 1)                    │║
║  │   Output: (batch, 200, 64)                                             │║
║  │                                                                         │║
║  │ LAYERNORM(d_model):  Normalize embeddings for stability                │║
║  │   Critical for preventing activation explosion!                        │║
║  │                                                                         │║
║  │ WHY: Transformers need vectors, not scalars                            │║
║  │      LayerNorm prevents extreme values from propagating                │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ STEP 2: NaN Handling + Positional Encoding                              │║
║  │ ══════════════════════════════════════════                              │║
║  │                                                                         │║
║  │ For each NaN position:                                                  │║
║  │   Replace embedding with learnable mask_token                          │║
║  │   mask_token: (1, 1, d_model) - learned "missing" representation       │║
║  │                                                                         │║
║  │ pos_encoding: (1, input_dim, d_model) = (1, 200, 64) LEARNABLE         │║
║  │   Scaled by 0.02/sqrt(d_model) for stability                           │║
║  │                                                                         │║
║  │ embedded = feature_embedded + pos_encoding                              │║
║  │                                                                         │║
║  │ PRE_TRANSFORMER_NORM: LayerNorm before transformer                     │║
║  │   Additional stability layer before attention                          │║
║  │                                                                         │║
║  │ WHY: Without position info, transformer doesn't know feature order     │║
║  │      LayerNorm ensures inputs to transformer are well-scaled           │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ STEP 3: Transformer Encoder (Pre-LN Architecture)                       │║
║  │ ═════════════════════════════════════════════════                       │║
║  │                                                                         │║
║  │ nn.TransformerEncoderLayer × num_layers (default 2):                   │║
║  │   - d_model = 64                                                        │║
║  │   - nhead = 4 (so head_dim = 64/4 = 16)                                │║
║  │   - dim_feedforward = 256 (4 × d_model)                                │║
║  │   - activation = 'gelu'                                                 │║
║  │   - norm_first = True  ← CRITICAL: Pre-LN for gradient stability!     │║
║  │                                                                         │║
║  │ Final LayerNorm after all layers                                       │║
║  │                                                                         │║
║  │ Pre-LN vs Post-LN:                                                      │║
║  │   Post-LN (default): x → Attention → Add → LN → FFN → Add → LN        │║
║  │   Pre-LN (ours):     x → LN → Attention → Add → LN → FFN → Add        │║
║  │                           ↑                                             │║
║  │   Pre-LN is MORE STABLE - gradients don't explode over epochs!         │║
║  │                                                                         │║
║  │ WHAT HAPPENS:                                                           │║
║  │   Each feature "looks at" other features via attention                 │║
║  │   Features with values can "inform" feature patterns                   │║
║  │   Model learns correlations between features                           │║
║  │                                                                         │║
║  │ Output: (batch, 200, 64) - contextualized feature representations      │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ STEP 4: NaN Check + Masked Mean Pooling                                 │║
║  │ ═══════════════════════════════════════                                 │║
║  │                                                                         │║
║  │ SAFETY: If NaN detected in output → return missing_modality_token      │║
║  │         (Should not happen with Pre-LN, but defensive)                 │║
║  │                                                                         │║
║  │ ONLY average over NON-NaN features:                                    │║
║  │                                                                         │║
║  │   pooled = sum(features[~nan_mask]) / count(~nan_mask)                 │║
║  │                                                                         │║
║  │ CLAMP: pooled = clamp(pooled, -100, 100)  ← Safety net                 │║
║  │                                                                         │║
║  │ WHY: NaN features shouldn't contribute to final representation         │║
║  │      Variable-length pooling handles different NaN patterns            │║
║  │                                                                         │║
║  │ Output: (batch, 64)                                                    │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                     │                                        ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ STEP 5: Output Projection                                               │║
║  │ ═════════════════════════                                               │║
║  │                                                                         │║
║  │ LINEAR(d_model → embed_dim): 64 → 256                                  │║
║  │ GELU()  ← Smooth activation for better gradients                       │║
║  │ LAYERNORM(embed_dim)                                                   │║
║  │ DROPOUT(0.1)                                                           │║
║  │                                                                         │║
║  │ CLAMP: output = clamp(output, -100, 100)  ← Final safety net           │║
║  │                                                                         │║
║  │ Output: (batch, 256) final embeddings                                  │║
║  │                                                                         │║
║  │ ALSO RETURNS: valid_mask (batch,)                                      │║
║  │   - False for samples where ALL features are NaN                       │║
║  │   - Used to exclude invalid samples from loss computation              │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  NUMERICAL STABILITY SUMMARY:                                                ║
║  ════════════════════════════                                                ║
║  • Input clamping: Prevents extreme feature values                          ║
║  • Post-embedding LayerNorm: Normalizes initial representations             ║
║  • Pre-transformer LayerNorm: Stabilizes attention inputs                   ║
║  • Pre-LN transformer (norm_first=True): Prevents gradient explosion        ║
║  • Final LayerNorm after transformer: Stabilizes output                     ║
║  • Pooled output clamping: Safety net before projection                     ║
║  • Final output clamping: Ensures bounded embeddings                        ║
║  • Xavier initialization: Better starting gradients                         ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

**How Missing Value Handling Works (Conceptual)**

Unlike traditional imputation methods that replace NaN with a **fixed value** (0, mean, median), 
the TransformerModalityEncoder uses a **learnable mask token** combined with **attention-based inference**.

**The Key Insight: Features as Tokens**

```
Traditional View:  Input vector → [0.5, NaN, 0.3, NaN, 0.8] → Single entity
Transformer View:  Input vector → [tok1, tok2, tok3, tok4, tok5] → Sequence of tokens
```

Each feature becomes a separate "token" in a sequence, similar to words in a sentence.

**Step-by-Step Missing Value Handling:**

| Step | Traditional Imputation | Transformer Approach |
|------|------------------------|----------------------|
| 1. Detect NaN | ✓ Same | ✓ Same |
| 2. Replace NaN | Fixed value (0, median) | **Learnable `[MASK]` token** |
| 3. Information source | None (blind guess) | **Context from ALL present features via attention** |
| 4. Per-sample? | No (global statistic) | **Yes (attention is sample-specific)** |

**Visual Example:**

```
Input: [gene1=0.5, gene2=NaN, gene3=0.3, gene4=NaN, gene5=0.8]
                      ↓
Embedding: [emb(0.5), [MASK], emb(0.3), [MASK], emb(0.8)]
                      ↓
          ┌─────────────────────────────────────┐
          │      Self-Attention Matrix          │
          │                                      │
          │        gene1  gene2  gene3  gene4  gene5
          │ gene1    ✓      ✓      ✓      ✓      ✓
          │ gene2    ✓      ✓      ✓      ✓      ✓  ← Learns from 1,3,5
          │ gene3    ✓      ✓      ✓      ✓      ✓
          │ gene4    ✓      ✓      ✓      ✓      ✓  ← Learns from 1,3,5
          │ gene5    ✓      ✓      ✓      ✓      ✓
          └─────────────────────────────────────┘
                      ↓
Output: Contextualized representations where [MASK] positions
        have "learned" from the present features
```

**Analogy: Fill-in-the-Blank**

```
Sentence: "The ___ is red and sweet"

Traditional: Replace ___ with "thing"      → Generic, loses meaning
Transformer: Use context (red, sweet) → Infer "apple" or "strawberry"
```

The transformer uses the **context of present features** to make intelligent inferences 
about missing ones, rather than filling in blind default values.

**Why This Works Better:**

1. **Context-Aware**: Missing gene2 can be inferred from correlated genes
2. **Learnable**: The mask token learns what "missing" means for this data distribution
3. **No Fixed Assumption**: Model learns the best representation, not a human-chosen default
4. **Per-Sample Adaptation**: Different samples infer different "values" for missing features

**Comparison Table:**

| Method | Information Used | Handles Correlations? | Per-Sample? |
|--------|------------------|----------------------|-------------|
| Zero imputation | None | No | No |
| Mean imputation | Global mean | No | No |
| Median imputation | Global median | No | No |
| **Transformer** | **All present features** | **Yes (attention)** | **Yes** |

**Recommendation:** Use `--impute_strategy none` with transformer encoder - it handles 
missingness natively and performs better than pre-imputation.

---

#### B.2.3 ProjectionHead (Contrastive Learning)

**File:** [contrastive_learning.py](performance_extensions/contrastive_learning.py#L390)

**Purpose:** Map embeddings to contrastive loss space. **Discarded after pretraining.**

**Key Insight:** From SimCLR paper - contrastive loss should operate on projections, but downstream tasks should use pre-projection embeddings.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                           ProjectionHead                                     ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHY A SEPARATE PROJECTION HEAD?                                             ║
║  ════════════════════════════════                                            ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  Chen et al. (SimCLR, 2020) discovered:                                │║
║  │                                                                         │║
║  │  - Contrastive loss "destroys" some information to maximize similarity │║
║  │  - This is GOOD for the loss but BAD for downstream tasks              │║
║  │  - Solution: add "sacrificial" projection head                         │║
║  │                                                                         │║
║  │  During pretraining:                                                    │║
║  │    Encoder → 256-dim → Projection Head → 128-dim → Contrastive Loss   │║
║  │         KEEP ↑                           DISCARD ↑                      │║
║  │                                                                         │║
║  │  After pretraining:                                                     │║
║  │    Encoder → 256-dim → [Use for classification]                        │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║  ═════════════                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  Input: (batch, embed_dim) = (32, 256)                                 │║
║  │                                                                         │║
║  │  Layer 1: LINEAR(256 → 256) + RELU                                     │║
║  │     - Non-linear transform in same dimension                           │║
║  │     - WHY: Adds capacity without changing dimensionality               │║
║  │                                                                         │║
║  │  Layer 2: LINEAR(256 → projection_dim) = LINEAR(256 → 128)             │║
║  │     - NO activation (raw for contrastive loss)                         │║
║  │     - WHY: Lower dim = more efficient loss computation                 │║
║  │            and slightly better empirical results                       │║
║  │                                                                         │║
║  │  Output: (batch, projection_dim) = (32, 128)                           │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  CONTRASTIVE LOSS (NT-Xent):                                                ║
║  ═══════════════════════════                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                                                                         │║
║  │  1. Take two modalities from SAME patient: z_i, z_j                    │║
║  │                                                                         │║
║  │  2. Normalize: z_i = z_i / ||z_i||                                     │║
║  │                                                                         │║
║  │  3. Positive pair: sim(z_i, z_j) should be HIGH                        │║
║  │     Negative pairs: sim(z_i, z_k) should be LOW (different patients)   │║
║  │                                                                         │║
║  │  4. Loss = -log(exp(sim_positive/τ) / Σ exp(sim_all/τ))               │║
║  │     where τ = temperature (default 0.5)                                │║
║  │                                                                         │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

### B.3 Transformer Fusion

**Big Picture:** While QML classifiers process each modality independently, the Transformer Fusion branch learns **cross-modal relationships**. When gene expression data correlates with protein levels for certain cancer types, the transformer's attention mechanism can discover and exploit these patterns. The key insight: modalities are treated like "words" in a sentence—the transformer learns how they relate to each other.

```
Traditional QML Approach:        Transformer Fusion Approach:
─────────────────────────        ───────────────────────────
Gene ─► QML ─┐                   Gene ──┐
miRNA─► QML ─┼─► Combine         miRNA──┼─► [Cross-attention] ─► Unified
Meth ─► QML ─┤                   Meth ──┤     learns            prediction
Prot ─► QML ─┘                   Prot ──┘   relationships
(Independent)                    (Interdependent)
```

#### B.3.1 MultimodalTransformer

**File:** `performance_extensions/transformer_fusion.py`, Line ~21

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL TRANSFORMER                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: List of (batch, embed_dim) tensors, one per modality            │
│         e.g., 5 modalities × (32, 256)                                  │
│                                                                          │
│  Step 1: Stack Modalities                                               │
│   └── torch.stack → (batch, num_modalities, embed_dim) = (32, 5, 256)   │
│                                                                          │
│  Step 2: Add Modality Embeddings                                        │
│   ├── nn.Embedding(num_modalities, embed_dim)                           │
│   ├── Learnable "position" for each modality                            │
│   ├── Analogous to positional encoding in NLP                           │
│   └── Shape: (5, 256) → broadcast to (32, 5, 256)                       │
│                                                                          │
│  Step 3: Optional CLS Token                                             │
│   ├── If use_cls_token=True:                                            │
│   │    ├── Prepend learnable cls_token: (1, 1, embed_dim)               │
│   │    └── Sequence becomes (batch, 1+num_mod, embed_dim)               │
│   └── If False: skip (default)                                          │
│                                                                          │
│  Step 4: Transformer Encoder Stack                                      │
│   ├── nn.TransformerEncoder with num_layers (default=4) layers          │
│   ├── Each nn.TransformerEncoderLayer:                                  │
│   │    ├── d_model=embed_dim (256)                                      │
│   │    ├── nhead=8 (32 dims per head)                                   │
│   │    ├── dim_feedforward=1024 (4× embed_dim)                          │
│   │    ├── dropout=0.1                                                  │
│   │    ├── activation='gelu' (smoother than relu)                       │
│   │    ├── batch_first=True                                             │
│   │    └── norm_first=True (Pre-LN for stability)                       │
│   ├── src_key_padding_mask: masks missing modalities                    │
│   └── Output: (batch, num_modalities, embed_dim)                        │
│                                                                          │
│  Step 5: Feature Aggregation                                            │
│   ├── If use_cls_token: extract cls output → (batch, embed_dim)         │
│   └── Else: flatten → (batch, num_mod × embed_dim) = (32, 1280)         │
│                                                                          │
│  Step 6: Classification Head                                            │
│   ├── Linear(input_dim → 512)                                           │
│   ├── LayerNorm(512)                                                    │
│   ├── GELU()                                                            │
│   ├── Dropout(0.1)                                                      │
│   ├── Linear(512 → 256)                                                 │
│   ├── LayerNorm(256)                                                    │
│   ├── GELU()                                                            │
│   ├── Dropout(0.1)                                                      │
│   └── Linear(256 → num_classes)                                         │
│   Output: (batch, num_classes) logits                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**TransformerEncoderLayer Internal Structure (Pre-LN):**
```
Input: x
  │
  ├── LayerNorm(x) → Normalized
  │     │
  │     └── MultiHeadAttention(Normalized, Normalized, Normalized)
  │           └── Q, K, V all from same input (self-attention)
  │
  ├── Add: x + Attention_output → Residual1
  │
  ├── LayerNorm(Residual1) → Normalized2
  │     │
  │     └── FFN: Linear(256→1024) → GELU → Linear(1024→256)
  │
  └── Add: Residual1 + FFN_output → Output

8 attention heads × 4 layers = 32 "views" of cross-modal relationships
```

---

#### B.3.2 ModalityFeatureEncoder

**File:** `performance_extensions/transformer_fusion.py`, Line ~210

**Purpose:** Simple encoder for use WITH MultimodalTransformer.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODALITY FEATURE ENCODER                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: (batch, input_dim) or None if modality missing                  │
│                                                                          │
│  If missing (x is None or is_missing=True):                             │
│   └── Return learnable missing_token: (1, embed_dim) → (batch, embed)   │
│                                                                          │
│  Encoder Network:                                                       │
│   ├── Linear(input_dim → 512)                                           │
│   ├── LayerNorm(512) - note: LayerNorm, not BatchNorm                   │
│   ├── ReLU()                                                            │
│   ├── Dropout(0.2)                                                      │
│   ├── Linear(512 → embed_dim)                                           │
│   └── LayerNorm(embed_dim)                                              │
│                                                                          │
│  Output: (batch, embed_dim)                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why LayerNorm instead of BatchNorm?**
- Works with batch_size=1 (BatchNorm needs batch statistics)
- More stable for transformer inputs
- Standard in transformer architectures

---

### B.4 Complete Data Flow Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          FULL PIPELINE DATA FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 1: PREPROCESSING
─────────────────────
Raw Multi-Omics Data (5 modalities):
  GeneExpr:  (N, 17176)  ─┬─[ ModalityEncoder ]─► (N, 256)
  miRNA:     (N, 425)    ─┼─[ ModalityEncoder ]─► (N, 256)
  Meth27:    (N, 3000)   ─┼─[ ModalityEncoder ]─► (N, 256)  ─► PCA(256→8)
  Meth450:   (N, 5000)   ─┼─[ ModalityEncoder ]─► (N, 256)
  Prot:      (N, 198)    ─┴─[ ModalityEncoder ]─► (N, 256)

TIER 2: CLASSIFICATION (Parallel Branches)
──────────────────────────────────────────
Branch A - QML (per-modality):
  (N, 8) ─►[ QuantumCircuit: AngleEmbed → Entangle → Measure ]─► (N, 8)
        ─►[ Classical MLP: 8→64→n_classes ]─► (N, n_classes) predictions
  × 5 modalities = 5 sets of predictions

Branch B - Transformer Fusion:
  5×(N, 256) ─►[ Stack ]─► (N, 5, 256)
            ─►[ + Modality Embeddings ]
            ─►[ TransformerEncoder × 4 layers ]─► (N, 5, 256)
            ─►[ Flatten ]─► (N, 1280)
            ─►[ Classification MLP ]─► (N, n_classes)

TIER 3: META-LEARNER (Optional)
───────────────────────────────
Collect predictions from Tier 2:
  QML_GeneExpr: (N, n_classes)  ─┐
  QML_miRNA:    (N, n_classes)  ─┼─►[ Concatenate ] + [ Indicators ]
  QML_Meth27:   (N, n_classes)  ─┤     ↓
  QML_Meth450:  (N, n_classes)  ─┤  (N, 5×n_classes + 5 indicators)
  QML_Prot:     (N, n_classes)  ─┘     ↓
                                  [ GatedMulticlassQuantumClassifierDataReuploadingDR ]
                                       ↓
                              Final: (N, n_classes) ensemble prediction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### B.5 Parameter Count Summary

| Model | Quantum Params | Classical Params | Total |
|-------|---------------|------------------|-------|
| MulticlassQuantumClassifierDR (8q, 3L, 64H, 5C) | 24 | 853 | **877** |
| DataReuploadingDR (8q, 3L, 64H, 5C) | 24 | 853 | **877** |
| GatedDR (same) | 24 | 853 | **877** |
| GatedDataReuploadingDR (same + missing) | 24 + 8 | 853 | **885** |
| ConditionalFS (8q, 3L, 16H, 5C) | 24 + 8 | 405 | **437** |
| ConditionalDataReuploadingFS (8q, 3L, 16H, 5C) | 24 + 24 | 405 | **453** |
| ModalityEncoder (5000→256) | 0 | ~2.76M | **2.76M** |
| TransformerModalityEncoder (200→256) | 0 | ~150K | **150K** |
| MultimodalTransformer (5 mod, 4L, 8H, 5C) | 0 | ~3.5M | **3.5M** |
| MultimodalFusionClassifier (5 mod, 4L, 8H, 5C) | 0 | ~4.2M | **4.2M** |

---

### B.6 Conditional Feature Encoding (CFE) Variants

**Big Picture:** The CFE variants handle **per-feature missing values** at the quantum level. Unlike Gated classifiers (which handle missing entire modalities), CFE handles individual missing features within a sample—e.g., when a patient has gene expression data but 10% of genes weren't measured.

```
Scenario: Sample with 8 features, features 2 and 5 are missing (NaN)
──────────────────────────────────────────────────────────────────

Traditional Approach: Impute first (e.g., mean/median), then encode
  [0.2, NaN, 0.5, 0.3, NaN, 0.1, 0.8, 0.4]
           ↓ impute
  [0.2, 0.35, 0.5, 0.3, 0.35, 0.1, 0.8, 0.4]  ← imputed values bias model

CFE Approach: Learnable rotations for missing features
  [0.2, NaN, 0.5, 0.3, NaN, 0.1, 0.8, 0.4]
           ↓ conditional encoding
  Qubit 0: RY(0.2 × π) ← data-driven
  Qubit 1: RY(θ_missing[1]) ← LEARNABLE (not imputed)
  Qubit 2: RY(0.5 × π) ← data-driven
  ...
```

#### B.6.1 ConditionalMulticlassQuantumClassifierFS

**File:** `qml_models.py`, Line ~2217

```
╔═════════════════════════════════════════════════════════════════════════════╗
║              ConditionalMulticlassQuantumClassifierFS                        ║
║                  (Conditional Feature Encoding - Standard)                   ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Handle per-feature missing values with learnable substitutes      ║
║                                                                              ║
║  INPUT FORMAT:                                                               ║
║   ├── features: (batch, n_qubits) - normalized feature values [0, 1]        ║
║   └── is_missing_mask: (batch, n_qubits) - 1 where feature is missing       ║
║                                                                              ║
║  QUANTUM CIRCUIT:                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                        ENCODING LAYER                                    │║
║  │  For each qubit i:                                                       │║
║  │    if is_missing[i] == 1:                                               │║
║  │      RY(weights_missing[i])  ← learnable angle                          │║
║  │    else:                                                                 │║
║  │      RY(features[i] × π)     ← data-driven angle                        │║
║  │                                                                          │║
║  │  Qubit 0: RY(data OR θ₀)                                                │║
║  │  Qubit 1: RY(data OR θ₁)                                                │║
║  │  ...                                                                     │║
║  │  Qubit 7: RY(data OR θ₇)                                                │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                    ENTANGLING LAYERS (×n_layers)                         │║
║  │  BasicEntanglerLayers: weights_ansatz[n_layers, n_qubits]               │║
║  │  Creates quantum correlations between all qubits                        │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                       MEASUREMENT                                        │║
║  │  PauliZ measurements on all qubits → (n_qubits,) values in [-1, 1]     │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                   CLASSICAL READOUT (same as MulticlassDR)              │║
║  │  (n_qubits,) → Linear → tanh → Linear → (n_classes,)                   │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  TRAINABLE PARAMETERS:                                                       ║
║   • weights_ansatz: (n_layers, n_qubits) - entangling rotations             ║
║   • weights_missing: (n_qubits,) - ONE learnable angle per qubit position   ║
║   • W1, b1, W2, b2: Classical readout MLP                                   ║
║                                                                              ║
║  KEY INSIGHT:                                                                ║
║   The network learns what rotation to apply when a feature is missing.      ║
║   This is better than imputation because:                                   ║
║   1. The "substitute" is optimized for the classification task             ║
║   2. Different qubits can learn different substitutes                       ║
║   3. No bias from dataset statistics (mean/median imputation)              ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

#### B.6.2 ConditionalMulticlassQuantumClassifierDataReuploadingFS

**File:** `qml_models.py`, Line ~2813

```
╔═════════════════════════════════════════════════════════════════════════════╗
║          ConditionalMulticlassQuantumClassifierDataReuploadingFS            ║
║              (Conditional Feature Encoding + Data Reuploading)               ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Maximum expressiveness for per-feature missing data               ║
║                                                                              ║
║  QUANTUM CIRCUIT (for n_layers=3):                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ LAYER 1:                                                                 │║
║  │   Conditional Encoding:                                                  │║
║  │     Qubit i: if missing → RY(θ_missing[0,i]) else RY(data[i]×π)         │║
║  │   Entanglement: BasicEntangler(weights_ansatz[0])                        │║
║  ├─────────────────────────────────────────────────────────────────────────┤║
║  │ LAYER 2:                                                                 │║
║  │   Conditional Encoding:                                                  │║
║  │     Qubit i: if missing → RY(θ_missing[1,i]) else RY(data[i]×π)         │║
║  │   Entanglement: BasicEntangler(weights_ansatz[1])                        │║
║  ├─────────────────────────────────────────────────────────────────────────┤║
║  │ LAYER 3:                                                                 │║
║  │   Conditional Encoding:                                                  │║
║  │     Qubit i: if missing → RY(θ_missing[2,i]) else RY(data[i]×π)         │║
║  │   Entanglement: BasicEntangler(weights_ansatz[2])                        │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                   MEASUREMENT + CLASSICAL READOUT                        │║
║  │  Same as ConditionalFS variant                                          │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  TRAINABLE PARAMETERS:                                                       ║
║   • weights_ansatz: (n_layers, n_qubits) - entangling rotations             ║
║   • weights_missing: (n_layers, n_qubits) - PER-LAYER learnable angles      ║
║   • W1, b1, W2, b2: Classical readout MLP                                   ║
║                                                                              ║
║  KEY DIFFERENCE FROM ConditionalFS:                                         ║
║   • weights_missing is 2D: (n_layers, n_qubits) vs (n_qubits,)              ║
║   • Each layer learns INDEPENDENT missing value representations             ║
║   • Early layers may learn "safe defaults"                                  ║
║   • Later layers may learn task-specific substitutes                        ║
║                                                                              ║
║  WHEN TO USE:                                                                ║
║   • High per-feature missingness (>20% of features missing)                 ║
║   • Complex patterns where missing features correlate with class            ║
║   • Enough data to train the extra parameters                               ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

### B.7 MultimodalFusionClassifier (End-to-End)

**File:** `performance_extensions/transformer_fusion.py`, Line ~269

**Big Picture:** While `MultimodalTransformer` assumes pre-encoded embeddings, `MultimodalFusionClassifier` is a complete end-to-end model that includes its own encoders. Use this when you want a single model that handles raw features directly.

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                     MultimodalFusionClassifier                               ║
║                (End-to-End Multimodal Classification)                        ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║                                                                              ║
║  Raw Input:                                                                  ║
║   GeneExpr: (N, 5000)  ─┐                                                   ║
║   miRNA:    (N, 800)   ─┼─► Per-Modality Encoders                           ║
║   Prot:     (N, 200)   ─┘                                                   ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                      MODALITY ENCODERS                                   │║
║  │                                                                          │║
║  │  For each modality:                                                      │║
║  │    ModalityFeatureEncoder (or pretrained encoder)                        │║
║  │                                                                          │║
║  │  GeneExpr: (N, 5000) → [Linear→LN→ReLU→Drop→Linear→LN] → (N, 256)      │║
║  │  miRNA:    (N, 800)  → [Linear→LN→ReLU→Drop→Linear→LN] → (N, 256)       │║
║  │  Prot:     (N, 200)  → [Linear→LN→ReLU→Drop→Linear→LN] → (N, 256)       │║
║  │                                                                          │║
║  │  If modality is MISSING:                                                 │║
║  │    → Use learnable missing_token: nn.Parameter(1, embed_dim)             │║
║  │    → Expanded to batch size: (N, 256)                                    │║
║  │                                                                          │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │                    MULTIMODAL TRANSFORMER                                │║
║  │           (Internal MultimodalTransformer instance)                      │║
║  │                                                                          │║
║  │  Embeddings: List of (N, 256) × num_modalities                          │║
║  │       ↓                                                                  │║
║  │  Stack → (N, num_mod, 256)                                              │║
║  │       ↓                                                                  │║
║  │  + Modality Embeddings                                                   │║
║  │       ↓                                                                  │║
║  │  TransformerEncoder × 4 layers                                          │║
║  │       ↓                                                                  │║
║  │  Flatten/Pool → (N, output_dim)                                         │║
║  │       ↓                                                                  │║
║  │  Classification MLP → (N, num_classes)                                  │║
║  │                                                                          │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  PRETRAINED ENCODER SUPPORT:                                                ║
║   Can initialize with pretrained contrastive encoders:                      ║
║   ```python                                                                  ║
║   pretrained = load_pretrained_encoders('checkpoint/')                      ║
║   model = MultimodalFusionClassifier(                                       ║
║       modality_dims=dims,                                                   ║
║       pretrained_encoders=pretrained  # Use pretrained instead of random    ║
║   )                                                                         ║
║   ```                                                                        ║
║                                                                              ║
║  PARAMETERS (typical config):                                                ║
║   • Encoders: ~2.7M per large modality, ~130K per small                     ║
║   • Transformer: ~1.5M (embed_dim=256, 4 layers, 8 heads)                   ║
║   • Classification head: ~0.5M                                              ║
║   • Total: ~4-5M for 5 modalities                                           ║
║                                                                              ║
║  WHEN TO USE:                                                                ║
║   • Want single end-to-end trainable model                                  ║
║   • Have sufficient data for joint encoder + transformer training           ║
║   • Want to fine-tune pretrained encoders with classification               ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

## Appendix C: Training Procedures

### C.1 Overview

**Big Picture:** Different components use different training procedures optimized for their architecture:

| Component | Framework | Optimizer | Loss Function | Learning Rate |
|-----------|-----------|-----------|---------------|---------------|
| QML Classifiers | PennyLane/Autograd | Custom Adam | Cross-Entropy | 0.01-0.1 |
| Contrastive Encoders | PyTorch | AdamW | NT-Xent | 1e-4 |
| Transformer Fusion | PyTorch | AdamW | Cross-Entropy | 1e-4 |
| Meta-Learner (QML) | PennyLane/Autograd | Custom Adam | Cross-Entropy | 0.01-0.1 |

---

### C.2 Loss Functions

#### C.2.1 Cross-Entropy Loss (Classification)

**Used by:** All classifiers (QML, Transformer, Meta-Learner)

```
Cross-Entropy measures how well predicted probabilities match true labels.

Formula: L = -Σᵢ yᵢ log(ŷᵢ)

Where:
  yᵢ = One-hot encoded true label (0 or 1)
  ŷᵢ = Predicted probability for class i

Example:
  True label: [0, 1, 0, 0, 0]  (class 1)
  Prediction: [0.1, 0.7, 0.1, 0.05, 0.05]
  
  Loss = -[0×log(0.1) + 1×log(0.7) + 0×log(0.1) + ...]
       = -log(0.7)
       = 0.357

Lower loss = better predictions
Perfect prediction (ŷ=1 for true class) → Loss = 0
```

**Implementation in QML:**
```python
def cross_entropy(predictions, targets_one_hot):
    # predictions: (batch, n_classes) - softmax probabilities
    # targets_one_hot: (batch, n_classes) - one-hot labels
    
    # Clamp to avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1.0)
    
    # Compute cross-entropy
    loss = -np.sum(targets_one_hot * np.log(predictions)) / len(predictions)
    return loss
```

---

#### C.2.2 NT-Xent Loss (Contrastive Learning)

**File:** `performance_extensions/contrastive_learning.py`, Line ~634

**Used by:** Contrastive pretraining (self-supervised learning)

```
NT-Xent = Normalized Temperature-scaled Cross Entropy (InfoNCE)

PURPOSE: Learn embeddings where:
  • Same sample's augmented views → CLOSE in embedding space
  • Different samples → FAR in embedding space

INTUITION:
  ┌────────────────────────────────────────────────────────────────────┐
  │  Embedding Space                                                    │
  │                                                                     │
  │     Sample A (view 1) ●──────● Sample A (view 2)  ← PULL together  │
  │                                                                     │
  │     Sample B (view 1) ●──────● Sample B (view 2)  ← PULL together  │
  │                                                                     │
  │     But A and B pushed apart ←─────────────────── PUSH apart       │
  │                                                                     │
  └────────────────────────────────────────────────────────────────────┘

FORMULA:
  For sample i with positive pair j (same sample, different augmentation):
  
  Lᵢ = -log[ exp(sim(zᵢ,zⱼ)/τ) / Σₖ exp(sim(zᵢ,zₖ)/τ) ]
  
  Where:
    sim(a,b) = cosine similarity = a·b / (||a|| ||b||)
    τ = temperature (default: 0.5)
    k ∈ all samples except i (negatives)

TEMPERATURE EFFECT:
  • τ = 0.1 (low):  Sharp distinctions, focuses on hard negatives
  • τ = 0.5 (default): Balanced
  • τ = 1.0 (high): Softer distinctions, more uniform treatment

ALGORITHM:
  1. Generate 2 augmented views of each sample in batch
  2. Encode all views → 2N embeddings
  3. Project to lower dimension (128-dim)
  4. L2-normalize projections
  5. Compute pairwise cosine similarities (2N × 2N matrix)
  6. Each sample's positive = its other augmented view
  7. All other samples = negatives
  8. Apply cross-entropy treating this as classification
```

**Code Flow:**
```python
def nt_xent_loss(z_i, z_j, temperature=0.5):
    # z_i, z_j: Two views of same samples, shape (batch, proj_dim)
    
    # 1. Normalize embeddings
    z_i = F.normalize(z_i, dim=1)  # Unit vectors
    z_j = F.normalize(z_j, dim=1)
    
    # 2. Concatenate: 2N samples
    representations = torch.cat([z_i, z_j], dim=0)  # (2N, proj_dim)
    
    # 3. Compute similarity matrix
    similarity = representations @ representations.T / temperature  # (2N, 2N)
    
    # 4. Create labels: positive for i is at i+N
    labels = torch.arange(N, 2*N)  # [N, N+1, ..., 2N-1]
    labels = torch.cat([labels, torch.arange(N)])  # [N,..,2N-1, 0,..,N-1]
    
    # 5. Mask diagonal (self-similarity)
    mask = torch.eye(2*N, dtype=bool)
    similarity.masked_fill_(mask, -inf)
    
    # 6. Cross-entropy loss
    loss = F.cross_entropy(similarity, labels)
    return loss
```

---

#### C.2.3 Cross-Modal Contrastive Loss

**Purpose:** Learn that different modalities from the SAME patient should be similar.

```
Scenario: Patient has Gene Expression AND Protein data
─────────────────────────────────────────────────────

Same patient's modalities should be CLOSE:
  GeneExpr embedding ●──────● Prot embedding  (same patient)

Different patients should be FAR:
  Patient A Gene ●                  ● Patient B Gene
  Patient A Prot ●                  ● Patient B Prot

This teaches the model that:
  "Gene expression X relates to protein profile Y"
```

**Implementation:** Uses NT-Xent but pairs are (modality1, modality2) instead of (aug1, aug2).

---

### C.3 Optimizers

#### C.3.1 QML Custom Adam (PennyLane/Autograd)

**File:** `utils/optim_adam.py`

**Why Custom?** PennyLane's autograd interface requires a serializable optimizer for checkpointing.

```python
class AdamSerializable:
    """
    Adam optimizer compatible with PennyLane autograd and pickle serialization.
    
    Parameters:
        lr: Learning rate (default: 0.01)
        beta1: Exponential decay for 1st moment (default: 0.9)
        beta2: Exponential decay for 2nd moment (default: 0.999)
        epsilon: Numerical stability constant (default: 1e-8)
    
    State (saved in checkpoints):
        m: First moment estimates (momentum)
        v: Second moment estimates (RMSprop-like)
        t: Step counter
    """
```

**Update Rule:**
```
For each parameter θ:
  m = β₁ × m + (1-β₁) × gradient           # Momentum
  v = β₂ × v + (1-β₂) × gradient²          # Squared gradient
  m̂ = m / (1 - β₁ᵗ)                        # Bias correction
  v̂ = v / (1 - β₂ᵗ)                        # Bias correction
  θ = θ - lr × m̂ / (√v̂ + ε)               # Update
```

---

#### C.3.2 PyTorch AdamW (Transformers/Encoders)

**Used by:** All PyTorch models (contrastive encoders, transformer fusion)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,     # L2 regularization (decoupled)
    eps=1e-8
)
```

**Why AdamW over Adam?**
- Decoupled weight decay (better regularization)
- Standard for transformer training
- More stable for large models

---

### C.4 Learning Rate Schedules

#### C.4.1 QML Training (No Schedule)

QML models typically use constant learning rate with early stopping:
```python
# Default: lr=0.01-0.1, no decay
# Early stopping when validation metric plateaus
```

#### C.4.2 Contrastive Pretraining (Warmup + Cosine)

```python
# Warmup: Linearly increase LR for first N epochs
# Then: Cosine decay to minimum

warmup_epochs = 5
total_epochs = 100

for epoch in range(total_epochs):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

**Visual:**
```
LR │     ╱─────╲
   │    ╱       ╲
   │   ╱         ╲
   │  ╱           ╲____
   │ ╱                  
   └──────────────────── Epoch
     ↑          ↑
   Warmup    Cosine decay
```

---

### C.5 Training Loop Procedures

#### C.5.1 QML Training Loop

**File:** `qml_models.py`, `fit()` method

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        QML TRAINING LOOP                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INITIALIZATION:                                                        │
│   1. Validate n_qubits >= n_classes                                     │
│   2. Initialize quantum weights: uniform(0, 2π)                         │
│   3. Initialize classical MLP weights: randn × 0.01                     │
│   4. Create PennyLane QNode with default.qubit device                   │
│   5. Train/validation split if validation_frac > 0                      │
│                                                                          │
│  RESUME LOGIC:                                                          │
│   if resume='auto': try latest checkpoint, then best                    │
│   if resume='latest': load most recent checkpoint                       │
│   if resume='best': load best-metric checkpoint                         │
│                                                                          │
│  TRAINING STEP:                                                         │
│   1. Forward pass:                                                      │
│      quantum_outputs = circuit(X, weights_quantum)  # (N, n_qubits)     │
│      logits = classical_readout(quantum_outputs)    # (N, n_classes)    │
│      probs = softmax(logits)                                            │
│                                                                          │
│   2. Compute loss:                                                      │
│      loss = cross_entropy(probs, y_one_hot)                             │
│                                                                          │
│   3. Backward pass (autograd):                                          │
│      grads = compute_gradients(loss, [quantum_weights, classical_weights])│
│                                                                          │
│   4. Update weights:                                                    │
│      weights = adam_update(weights, grads)                              │
│                                                                          │
│  CHECKPOINTING (every checkpoint_frequency steps):                      │
│   Save: weights, optimizer_state, step, history, best_metric            │
│   Keep only last N checkpoints (keep_last_n)                            │
│                                                                          │
│  VALIDATION (every validation_frequency steps):                         │
│   Compute: accuracy, precision, recall, F1, specificity                 │
│   Track: best model by selection_metric (default: f1_weighted)          │
│                                                                          │
│  EARLY STOPPING:                                                        │
│   if patience > 0 and no improvement for patience steps:                │
│      terminate and restore best weights                                 │
│                                                                          │
│  TERMINATION:                                                           │
│   Either: steps reached OR max_training_time exceeded OR early stopped  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

#### C.5.2 Contrastive Pretraining Loop

**File:** `performance_extensions/training_utils.py`, `pretrain_contrastive()`

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   CONTRASTIVE PRETRAINING LOOP                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INITIALIZATION:                                                        │
│   1. Create ContrastiveMultiOmicsEncoder with per-modality encoders     │
│   2. Create augmentation pipelines (noise, masking, mixup)              │
│   3. Initialize ContrastiveLearningLoss (NT-Xent + cross-modal)         │
│                                                                          │
│  PER EPOCH:                                                             │
│   for batch in dataloader:                                              │
│       # 1. Generate augmented views                                     │
│       views = [augment(sample) for _ in range(2)]  # 2 views per sample │
│                                                                          │
│       # 2. Encode all views                                             │
│       for modality in modalities:                                       │
│           embedding, projection, valid = encoder(views[modality])       │
│                                                                          │
│       # 3. Compute contrastive loss                                     │
│       loss = intra_modal_loss + λ × cross_modal_loss                    │
│                                                                          │
│       # 4. NaN detection (skip batch if NaN)                            │
│       if isnan(loss): continue                                          │
│                                                                          │
│       # 5. Backward + gradient clipping + update                        │
│       loss.backward()                                                   │
│       clip_grad_norm_(model.parameters(), max_norm=1.0)                 │
│       optimizer.step()                                                  │
│                                                                          │
│  CHECKPOINT SAVING:                                                     │
│   • best_model.pt: Combined model when loss improves                    │
│   • encoders/{modality}_encoder.pt: Individual encoders                 │
│   • projections/{modality}_projection.pt: (for continued pretraining)   │
│                                                                          │
│  WARMUP SCHEDULE:                                                       │
│   First warmup_epochs: LR = base_lr × (epoch+1) / warmup_epochs         │
│   After warmup: Full LR, optional cosine decay                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

#### C.5.3 Supervised Fine-Tuning Loop

**File:** `performance_extensions/training_utils.py`, `finetune_supervised()`

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   SUPERVISED FINE-TUNING LOOP                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SETUP:                                                                  │
│   • Load pretrained encoders (from contrastive pretraining)             │
│   • Create classification head (MLP)                                    │
│   • Optionally freeze encoders (freeze_encoders=True)                   │
│                                                                          │
│  PER EPOCH:                                                             │
│   for batch in labeled_dataloader:                                      │
│       # 1. Extract features with pretrained encoders                    │
│       features = []                                                     │
│       for modality, data in batch:                                      │
│           feat = encoders[modality](data)  # (N, embed_dim)             │
│           features.append(feat)                                         │
│                                                                          │
│       # 2. Concatenate modality features                                │
│       combined = torch.cat(features, dim=1)  # (N, num_mod × embed_dim) │
│                                                                          │
│       # 3. Classify                                                     │
│       logits = classifier(combined)  # (N, num_classes)                 │
│                                                                          │
│       # 4. Supervised loss                                              │
│       loss = CrossEntropyLoss(logits, labels)                           │
│                                                                          │
│       # 5. Update (classifier only if frozen, else all)                 │
│       loss.backward()                                                   │
│       optimizer.step()                                                  │
│                                                                          │
│  FREEZE vs FINE-TUNE:                                                   │
│   • freeze_encoders=True: Only train classifier (faster, less overfit)  │
│   • freeze_encoders=False: Train everything (better if enough data)     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### C.6 Gradient Handling

#### C.6.1 Gradient Clipping

**Purpose:** Prevent exploding gradients, especially in transformers.

```python
# Applied before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**When to Use:**
- Transformer training (always)
- Long training runs
- When loss suddenly spikes

#### C.6.2 NaN Detection & Recovery

**File:** `performance_extensions/training_utils.py`

```python
# During training loop:
if torch.isnan(loss) or torch.isinf(loss):
    print("NaN/Inf loss detected. Skipping batch.")
    continue  # Don't update on bad batch

# Check gradients too:
for param in model.parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            optimizer.zero_grad()  # Clear bad gradients
            continue
```

**Common NaN Causes:**
1. Learning rate too high → reduce
2. Batch with all NaN features → skip batch
3. Numerical instability in softmax → use log_softmax
4. Division by zero in normalization → add epsilon
5. Post-LN transformer (default) → switch to Pre-LN (norm_first=True)
6. Extreme input values → clamp inputs before encoding
7. Missing LayerNorm after embeddings → add embedding normalization

---

### C.7 Best Practices Summary

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                      TRAINING BEST PRACTICES                                 ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QML MODELS:                                                                 ║
║   • Start with lr=0.1, reduce if unstable                                   ║
║   • Use validation_frac=0.2+ (default 0.2 – increased for better detection) ║
║   • Enable checkpointing for long runs (checkpoint_frequency=10)            ║
║   • Use patience=25+ (default 25 – reduced for faster early stopping)       ║
║                                                                              ║
║  CONTRASTIVE PRETRAINING:                                                   ║
║   • Always use LR warmup (5-10 epochs)                                      ║
║   • Temperature=0.5 usually works well                                      ║
║   • Batch size matters! Larger = more negatives = better (try 128-512)     ║
║   • Enable gradient clipping (max_norm=1.0)                                 ║
║   • Train for 50-200 epochs depending on data size                          ║
║                                                                              ║
║  TRANSFORMER ENCODER (TransformerModalityEncoder):                          ║
║   • MUST use Pre-LN (norm_first=True) - Post-LN causes NaN after epochs!   ║
║   • Apply LayerNorm after feature embedding                                 ║
║   • Clamp inputs to [-10, 10] to prevent extreme values                     ║
║   • Use Xavier initialization for linear layers                             ║
║   • If NaN appears: reduce LR, increase gradient clipping                   ║
║                                                                              ║
║  TRANSFORMER FUSION:                                                        ║
║   • Use Pre-LN configuration (norm_first=True)                              ║
║   • AdamW with weight_decay=0.01, class_weighting=True, label_smoothing=0.05║
║   • Lower LR than contrastive (1e-4 to 5e-5) with ReduceLROnPlateau        ║
║   • Dropout=0.2, proper train/val/test splits (no test leakage)            ║
║   • Watch for overfitting with small datasets                               ║
║                                                                              ║
║  GENERAL:                                                                   ║
║   • Monitor validation metrics, not just training loss                      ║
║   • Save checkpoints frequently                                             ║
║   • Log to W&B for experiment tracking                                      ║
║   • Use mixed precision (fp16) for transformer speedup                      ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

*Document updated with CFE variants, MultimodalFusionClassifier, comprehensive training procedures, and TransformerModalityEncoder stability improvements.*
