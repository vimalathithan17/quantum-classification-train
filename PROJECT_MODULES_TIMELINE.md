# Project Modules & 2-Month Implementation Timeline

---

## MODULE BREAKDOWN

### Module 1: Data Collection & Processing
**Files:** `data_collection_and_processing/data-process.ipynb`, `data_collection_and_processing/feature-extraction-xgb.ipynb`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 1.1 TCGA Data Download | Fetch raw multi-omics data from GDC portal | None |
| 1.2 Data Cleaning | Handle missing values, outliers, duplicates | 1.1 |
| 1.3 Normalization | Log transform, z-score per modality | 1.2 |
| 1.4 Feature Selection | LightGBM/XGBoost importance ranking | 1.3 |
| 1.5 Train/Test Split | Stratified splitting, CV fold creation | 1.4 |
| 1.6 Indicator Generation | Missing modality flags per sample | 1.2 |

**Output:** `final_processed_datasets/` with parquet files

---

### Module 2: Contrastive Pretraining
**Files:** `performance_extensions/contrastive_learning.py`, `performance_extensions/augmentations.py`, `examples/pretrain_contrastive.py`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 2.1 Modality Encoders | 6 MLP encoders (input → 256-dim) | Module 1 |
| 2.2 Projection Heads | 6 projection networks (256 → 128-dim) | 2.1 |
| 2.3 Augmentation Engine | Modality-specific data augmentations | 2.1 |
| 2.4 Contrastive Loss | NT-Xent loss implementation | 2.2 |
| 2.5 Cross-Modal Loss | Align same-patient different-modality | 2.2 |
| 2.6 Training Loop | Epoch training with logging | 2.1-2.5 |

**Output:** `pretrained_models/encoders/` with encoder weights

---

### Module 3: Feature Extraction
**Files:** `examples/extract_pretrained_features.py`, `performance_extensions/training_utils.py`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 3.1 Encoder Loading | Load pretrained encoder weights | Module 2 |
| 3.2 Batch Inference | Extract embeddings in batches | 3.1 |
| 3.3 Embedding Storage | Save embeddings as .npy files | 3.2 |

**Output:** `pretrained_features/` with embedding files

---

### Module 4: QML Base Learners
**Files:** `qml_models.py`, `dre_standard.py`, `dre_relupload.py`, `cfe_standard.py`, `cfe_relupload.py`, `tune_models.py`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 4.1 Quantum Circuits | VQC implementations (4 variants) | PennyLane |
| 4.2 Classical Readout | Linear layer after quantum measurement | 4.1 |
| 4.3 Dimensionality Reduction | PCA/feature selection to N_qubits | Module 3 |
| 4.4 Training Pipeline | CV training with OOF predictions | 4.1-4.3 |
| 4.5 Hyperparameter Tuning | Optuna-based circuit optimization | 4.4 |
| 4.6 Model Persistence | Save/load trained QML models | 4.4 |

**Output:** `base_learner_outputs/` with predictions and models

---

### Module 5: Transformer Fusion
**Files:** `performance_extensions/transformer_fusion.py`, `examples/train_transformer_fusion.py`, `examples/extract_transformer_features.py`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 5.1 Modality Embeddings | Learnable position tokens per modality | Module 2 |
| 5.2 Feature Encoders | Project embeddings to transformer dim | 5.1 |
| 5.3 Transformer Encoder | Multi-head self-attention layers | 5.2 |
| 5.4 Missing Modality Masking | Attention masks for absent modalities | 5.3 |
| 5.5 CLS Aggregation | Classification token for prediction | 5.3 |
| 5.6 Classification Head | MLP for final class logits | 5.5 |
| 5.7 Prediction Export | CSV output for meta-learner | 5.6 |

**Output:** `transformer_predictions/` with CSV predictions

---

### Module 6: QML Meta-Learner
**Files:** `metalearner.py`, `utils/io_checkpoint.py`, `utils/optim_adam.py`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 6.1 Meta-Feature Assembly | Concatenate all base learner predictions | Modules 4, 5 |
| 6.2 Degeneracy Removal | Drop redundant probability columns | 6.1 |
| 6.3 Gating Mechanism | Learnable feature importance gates | 6.2 |
| 6.4 Quantum Meta-Circuit | VQC operating on meta-features | 6.3 |
| 6.5 Hyperparameter Tuning | Optuna optimization for meta-learner | 6.4 |
| 6.6 Checkpointing | Save/resume training state | 6.4 |

**Output:** `final_model_and_predictions/` with final predictions

---

### Module 7: Inference & Evaluation
**Files:** `inference.py`, `utils/metrics_utils.py`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 7.1 Model Loading | Load all trained components | Modules 4-6 |
| 7.2 End-to-End Inference | Full pipeline prediction | 7.1 |
| 7.3 Metrics Calculation | F1, accuracy, confusion matrix | 7.2 |
| 7.4 Interpretability | Attention visualization, importance | 7.2 |

**Output:** Evaluation reports and visualizations

---

### Module 8: Testing & Validation
**Files:** `tests/test_*.py`, `run_tests.py`

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| 8.1 Unit Tests | Component-level testing | All modules |
| 8.2 Integration Tests | Module interaction testing | All modules |
| 8.3 E2E Tests | Full pipeline validation | All modules |

---

## 2-MONTH IMPLEMENTATION TIMELINE

### Timeline Overview (8 Weeks)

```
Week 1-2: Data Processing + Contrastive Pretraining Setup
Week 3-4: Contrastive Training + Feature Extraction + QML Base Learners
Week 5-6: Transformer Fusion + Integration
Week 7:   Meta-Learner + Full Pipeline Integration
Week 8:   Testing, Optimization, Documentation
```

---

### Detailed Gantt Chart Description

```
WEEK 1 (Days 1-7)
├── Day 1-2: Module 1.1-1.2 | Data Download & Cleaning
│   └── [██████████] Download TCGA, handle missing values
├── Day 3-4: Module 1.3-1.4 | Normalization & Feature Selection
│   └── [██████████] Log transform, z-score, LightGBM selection
├── Day 5-6: Module 1.5-1.6 | Splitting & Indicators
│   └── [██████████] Stratified split, CV folds, indicator generation
└── Day 7: Validation checkpoint
    └── [█████] Verify data pipeline outputs

WEEK 2 (Days 8-14)
├── Day 8-9: Module 2.1-2.2 | Encoder Architecture
│   └── [██████████] Implement modality encoders + projection heads
├── Day 10-11: Module 2.3 | Augmentation Engine
│   └── [██████████] Modality-specific augmentation functions
├── Day 12-13: Module 2.4-2.5 | Contrastive Losses
│   └── [██████████] NT-Xent loss, cross-modal contrastive
└── Day 14: Module 2.6 | Training Loop Setup
    └── [█████] Initial training loop, logging setup

WEEK 3 (Days 15-21)
├── Day 15-18: Module 2 Training | Contrastive Pretraining
│   └── [████████████████████] Train encoders (100-200 epochs)
├── Day 19-20: Module 3 | Feature Extraction
│   └── [██████████] Extract embeddings from pretrained encoders
└── Day 21: Validation checkpoint
    └── [█████] Verify embedding quality (t-SNE visualization)

WEEK 4 (Days 22-28)
├── Day 22-24: Module 4.1-4.3 | QML Circuit Setup
│   └── [███████████████] Quantum circuits, readout layers, dim reduction
├── Day 25-27: Module 4.4 | QML Base Learner Training
│   └── [███████████████] Train 6 modality-specific QML models (5-fold CV)
└── Day 28: Module 4.6 | Model Persistence
    └── [█████] Save OOF predictions, trained models

WEEK 5 (Days 29-35)
├── Day 29-31: Module 5.1-5.3 | Transformer Architecture
│   └── [███████████████] Modality embeddings, encoders, attention layers
├── Day 32-33: Module 5.4-5.5 | Masking & Aggregation
│   └── [██████████] Missing modality handling, CLS token
├── Day 34: Module 5.6 | Classification Head
│   └── [█████] MLP classifier on top of transformer
└── Day 35: Module 5.7 | Prediction Export
    └── [█████] CSV export for meta-learner compatibility

WEEK 6 (Days 36-42)
├── Day 36-39: Module 5 Training | Transformer Training
│   └── [████████████████████] Train transformer fusion (50-100 epochs)
├── Day 40-41: Integration Testing | Parallel Branches
│   └── [██████████] Verify QML + Transformer outputs compatible
└── Day 42: Validation checkpoint
    └── [█████] Compare individual model performances

WEEK 7 (Days 43-49)
├── Day 43-44: Module 6.1-6.2 | Meta-Feature Assembly
│   └── [██████████] Concatenate predictions, degeneracy removal
├── Day 45-46: Module 6.3-6.4 | Gated Meta-Learner
│   └── [██████████] Implement gating mechanism + quantum meta-circuit
├── Day 47-48: Module 6.5 | Hyperparameter Tuning
│   └── [██████████] Optuna optimization (50-100 trials)
└── Day 49: Module 6.6 | Checkpointing
    └── [█████] Save final trained meta-learner

WEEK 8 (Days 50-56)
├── Day 50-51: Module 7 | Inference Pipeline
│   └── [██████████] End-to-end inference, metrics calculation
├── Day 52-53: Module 8 | Testing Suite
│   └── [██████████] Unit tests, integration tests, E2E validation
├── Day 54-55: Optimization | Performance Tuning
│   └── [██████████] Speed optimizations, memory efficiency
└── Day 56: Documentation & Release
    └── [█████] Final documentation, README updates
```

---

### Visual Gantt Chart (Text Format)

```
                        WEEK 1    WEEK 2    WEEK 3    WEEK 4    WEEK 5    WEEK 6    WEEK 7    WEEK 8
                        1234567   1234567   1234567   1234567   1234567   1234567   1234567   1234567
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: DATA PROCESSING                                                                           │
│ 1.1 Data Download     ████                                                                          │
│ 1.2 Data Cleaning       ████                                                                        │
│ 1.3 Normalization         ████                                                                      │
│ 1.4 Feature Selection       ████                                                                    │
│ 1.5 Train/Test Split          ████                                                                  │
│ 1.6 Indicator Gen               ████                                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MODULE 2: CONTRASTIVE PRETRAINING                                                                   │
│ 2.1 Modality Encoders           ████                                                                │
│ 2.2 Projection Heads              ████                                                              │
│ 2.3 Augmentations                   ████                                                            │
│ 2.4 Contrastive Loss                  ████                                                          │
│ 2.5 Cross-Modal Loss                    ██                                                          │
│ 2.6 Training Loop                         ████████████                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MODULE 3: FEATURE EXTRACTION                                                                        │
│ 3.1-3.3 Extract Embeddings                          ████                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MODULE 4: QML BASE LEARNERS                                                                         │
│ 4.1 Quantum Circuits                                    ██████                                      │
│ 4.2 Classical Readout                                       ██                                      │
│ 4.3 Dim Reduction                                           ██                                      │
│ 4.4 Training Pipeline                                         ██████                                │
│ 4.5 HP Tuning (optional)                                            ████                            │
│ 4.6 Model Persistence                                                 ██                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MODULE 5: TRANSFORMER FUSION                                                                        │
│ 5.1 Modality Embeddings                                                 ██████                      │
│ 5.2 Feature Encoders                                                        ██                      │
│ 5.3 Transformer Encoder                                                       ████                  │
│ 5.4 Missing Masking                                                             ██                  │
│ 5.5-5.6 CLS + Head                                                                ██                │
│ 5.7 Training                                                                        ████████        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MODULE 6: QML META-LEARNER                                                                          │
│ 6.1-6.2 Meta Assembly                                                                     ████      │
│ 6.3-6.4 Gated Meta-QML                                                                      ████    │
│ 6.5 HP Tuning                                                                                 ████  │
│ 6.6 Checkpointing                                                                               ██  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MODULE 7: INFERENCE                                                                                 │
│ 7.1-7.4 Full Pipeline                                                                           ████│
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MODULE 8: TESTING                                                                                   │
│ 8.1-8.3 All Tests                                                                             ██████│
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

LEGEND:
████ = Active development
░░░░ = Buffer/contingency
──── = Dependency line
```

---

### Milestones & Deliverables

| Milestone | Week | Deliverable | Success Criteria |
|-----------|------|-------------|------------------|
| **M1: Data Ready** | 1 | Processed parquet files | 6 modality files + indicators |
| **M2: Encoders Trained** | 3 | Pretrained encoder weights | Contrastive loss < 1.0 |
| **M3: Embeddings Extracted** | 3 | Embedding .npy files | t-SNE shows clustering |
| **M4: QML Base Learners** | 4 | 6 trained QML models | OOF F1 > 0.75 per modality |
| **M5: Transformer Trained** | 6 | Transformer model + predictions | Test F1 > 0.85 |
| **M6: Meta-Learner** | 7 | Final ensemble model | Test F1 > 0.90 |
| **M7: Full Pipeline** | 8 | End-to-end inference | <1s per sample inference |
| **M8: Release Ready** | 8 | All tests passing | 100% test coverage |

---

### Resource Requirements

| Phase | Compute | GPU Memory | Time Estimate |
|-------|---------|------------|---------------|
| Data Processing | CPU only | N/A | 2-4 hours |
| Contrastive Training | GPU recommended | 8GB | 8-16 hours |
| QML Training (6 models) | CPU (simulator) | N/A | 12-24 hours |
| Transformer Training | GPU required | 8-12GB | 6-12 hours |
| Meta-Learner Training | CPU (simulator) | N/A | 2-4 hours |
| **Total** | Mixed | 12GB peak | ~40-60 hours compute |

---

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| QML training slow | High | Medium | Use fewer qubits, parallelize modalities |
| GPU memory overflow | Medium | High | Reduce batch size, gradient checkpointing |
| Contrastive collapse | Low | High | Monitor loss, use temperature scheduling |
| Missing data issues | Medium | Medium | Robust indicator handling, masking |
| Integration bugs | Medium | High | Incremental testing, CI/CD pipeline |

---

### Team Allocation Suggestions (If Applicable)

| Role | Modules | Weeks |
|------|---------|-------|
| Data Engineer | 1, 3 | 1-3 |
| ML Engineer (Classical) | 2, 5 | 2-6 |
| QML Researcher | 4, 6 | 4-7 |
| MLOps/Testing | 7, 8 | 7-8 |

---

### Dependencies Graph

```
Module 1 (Data)
    │
    ▼
Module 2 (Contrastive)
    │
    ▼
Module 3 (Feature Extraction)
    │
    ├─────────────────────────┐
    ▼                         ▼
Module 4 (QML Base)      Module 5 (Transformer)
    │                         │
    └───────────┬─────────────┘
                ▼
         Module 6 (Meta-Learner)
                │
                ▼
         Module 7 (Inference)
                │
                ▼
         Module 8 (Testing) ← [Runs throughout]
```
