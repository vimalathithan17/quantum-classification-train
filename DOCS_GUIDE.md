# Documentation Guide

Quick reference to find information and choose your approach.

---

## Quick Start (2 Minutes)

| Your Situation | Start Here | Time |
|---|---|---|
| **New user** | [README.md](README.md) | 10 min |
| **Need to choose approach** | [Decision Tree](#decision-tree) below | 5 min |
| **Ready to integrate extensions** | [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | 20 min |
| **Understand data pipeline** | [DATA_PROCESSING.md](DATA_PROCESSING.md) | 15 min |

---

## Decision Tree

```
START: How much labeled data do you have?
│
├─→ < 100 samples
│   └─→ Use QML Only → [README.md](README.md)
│       • No GPU needed, 2-4 hours training
│       • QML excels on small datasets
│
├─→ 100-500 samples
│   ├─→ Have unlabeled data?
│   │   └─→ YES: Contrastive + QML → [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#pattern-3)
│   │       • +5-10% F1, uses unlabeled data
│   │
│   └─→ Labeled only?
│       ├─→ Balanced: Data-Reuploading QML or Ensemble
│       │   • ⚠️ DON'T use contrastive (−6% F1 on small data!)
│       └─→ Imbalanced: QML + Class Weighting
│           → [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#class-imbalance)
│
└─→ 500+ samples
    ├─→ Missing modalities > 20%?
    │   └─→ Transformer Fusion → [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#pattern-2)
    │       • Native attention masking for missing data
    │
    └─→ Complete data + GPU available?
        └─→ Full Hybrid Pipeline → [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#pattern-1)
            • Contrastive → Transformer → QML Meta-learner
            • Best performance, 16-24 hours
```

---

## Document Map

```
README.md                    ← Start here: QML pipeline, setup, commands
│
├── DOCS_GUIDE.md            ← This file: navigation & decision tree
│
├── INTEGRATION_GUIDE.md     ← Integration patterns, class imbalance, troubleshooting
│
├── DATA_PROCESSING.md       ← Data pipeline: notebooks, outputs, formats
│
├── ARCHITECTURE.md          ← Technical deep-dive: quantum circuits, design decisions
│
├── PERFORMANCE_EXTENSIONS.md ← Transformer fusion & contrastive learning specs
│
├── examples/README.md       ← Running example scripts
│
└── COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md ← Test inventory
```

---

## Common Questions

### "I have 200 samples and class imbalance. What do I do?"
→ Use QML with class weighting. If you have unlabeled data, add contrastive pretraining.
→ [INTEGRATION_GUIDE.md - Class Imbalance](INTEGRATION_GUIDE.md#class-imbalance-and-small-dataset-considerations)

### "Should I use contrastive pretraining on my small labeled dataset?"
→ **No.** Empirical tests show −6% F1 on small labeled-only data. Use QML or ensemble instead.
→ [INTEGRATION_GUIDE.md - Scenario B](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples)

### "How do I handle missing modalities?"
→ Use Transformer Fusion (native attention masking) or Conditional QML with indicators.
→ [INTEGRATION_GUIDE.md - Missing Modality Strategy](INTEGRATION_GUIDE.md#architecture-decision-tree)

### "I'm getting OOM errors"
→ Reduce batch size, use gradient accumulation, reduce model dimensions.
→ [INTEGRATION_GUIDE.md - Troubleshooting](INTEGRATION_GUIDE.md#issue-1-out-of-memory-during-transformer-training)

### "What's the data flow through the notebooks?"
```
data-process.ipynb (10 stages)
└── final_filtered_datasets/ (312 cases, full features)
    │
    └── feature-extraction-xgb.ipynb (MI → XGBoost)
        └── final_processed_datasets_xgb_balanced/ (312 cases, 500 features/modality)
```
→ [DATA_PROCESSING.md](DATA_PROCESSING.md)

---

## Reading Paths

### Path 1: QML Only (30 min)
1. [README.md](README.md) - Overview & commands
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Quantum circuits
3. Run example commands

### Path 2: With Extensions (1.5 hours)
1. [README.md](README.md) - Overview
2. This guide - Choose pattern
3. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration details
4. [examples/README.md](examples/README.md) - Run examples

### Path 3: Full Deep-Dive (3 hours)
1. [README.md](README.md)
2. [ARCHITECTURE.md](ARCHITECTURE.md)
3. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
4. [PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md)
5. [DATA_PROCESSING.md](DATA_PROCESSING.md)
