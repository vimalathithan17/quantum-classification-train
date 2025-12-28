# ⚠️ DEPRECATED - This file has been consolidated

**This document has been merged into [DOCS_GUIDE.md](DOCS_GUIDE.md).**

Please use the new consolidated documentation:
- **[DOCS_GUIDE.md](DOCS_GUIDE.md)** - Navigation guide with decision trees
- **[README.md](README.md)** - Main entry point

**This file can be safely deleted.**

### Decision Point 1: Dataset Size < 100 Samples

```
Small Dataset (< 100 samples)
│
└─→ RECOMMENDATION: QML Only
    │
    ├─ WHY:
    │  • QML is sample-efficient by design
    │  • Exponential Hilbert space with few qubits
    │  • Proven advantages on small data
    │  • No GPU needed
    │
    ├─ WHEN NOT TO USE:
    │  • Need cross-modal fusion (use Transformer)
    │  • Severe class imbalance (use Contrastive)
    │  • Want 95%+ accuracy (consider ensemble)
    │
    ├─ DOCUMENTATION:
    │  → [README.md](README.md)
    │  → [ARCHITECTURE.md](ARCHITECTURE.md)
    │
    ├─ EXPECTED RESULTS:
    │  • Training: 2-4 hours
    │  • GPU: Not needed
    │  • F1 Score: 0.75-0.85
    │
    └─ NEXT: Go to [README.md](README.md) for setup
```

### Decision Point 2: Dataset Size 100-500 Samples

```
Medium Dataset (100-500 samples)
│
├─→ Do you have unlabeled data available?
│   │
│   ├─ YES (Branch A1)
│   │  │
│   │  └─→ RECOMMENDATION: Contrastive Pretraining → QML
│   │     │
│   │     ├─ WHY:
│   │     │  • Contrastive learns from unlabeled data
│   │     │  • Class-agnostic learning (helps imbalance)
│   │     │  • Better features than PCA/UMAP
│   │     │  • Minimal pipeline changes
│   │     │
│   │     ├─ EXPECTED IMPROVEMENT:
│   │     │  • F1 gain: +5-10%
│   │     │  • Special case: Unlabeled helps minorities most
│   │     │
│   │     ├─ DOCUMENTATION:
│   │     │  → [INTEGRATION_GUIDE.md - Pattern 3](INTEGRATION_GUIDE.md#pattern-3-contrastive-pretraining--qml-pipeline)
│   │     │  → [INTEGRATION_GUIDE.md - Workflow A](INTEGRATION_GUIDE.md#workflow-a-adding-contrastive-pretraining-to-existing-qml-pipeline)
│   │     │
│   │     └─ NEXT: Go to [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#pattern-3-contrastive-pretraining--qml-pipeline)
│   │
│   └─ NO (Branch A2)
│      │
│      └─→ Do you have balanced or imbalanced data?
│         │
│         ├─ BALANCED (Branch A3)
│         │  │
│         │  └─→ RECOMMENDATION: Data-Reuploading QML or QML Ensemble
│         │     │
│         │     ├─ OPTION 1: Data-Reuploading QML
│         │     │  • F1 improvement: +3%
│         │     │  • Training: 3 hours
│         │     │  • Best single model
│         │     │
│         │     ├─ OPTION 2: QML Ensemble (4 models)
│         │     │  • F1 improvement: +5%
│         │     │  • Training: 4 hours
│         │     │  • Best overall performance
│         │     │
│         │     ├─ ⚠️ WARNING: DON'T use Contrastive Pretraining
│         │     │  • Empirical evidence: -6% F1 (WORSE!)
│         │     │  • Only 320-400 samples insufficient for contrastive
│         │     │  • No unlabeled data advantage
│         │     │  • High overfitting risk
│         │     │
│         │     ├─ DOCUMENTATION:
│         │     │  → [INTEGRATION_GUIDE.md - Scenario B](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples)
│         │     │  → [INTEGRATION_GUIDE.md - Small Dataset Strategy](INTEGRATION_GUIDE.md#small-dataset-strategy--100-samples)
│         │     │
│         │     └─ NEXT: Go to [INTEGRATION_GUIDE.md - Scenario B](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples)
│         │
│         └─ IMBALANCED (Branch A4)
│            │
│            └─→ RECOMMENDATION: Contrastive + QML with Class Weighting
│               │
│               ├─ WHY:
│               │  • Contrastive is class-agnostic (helps minorities)
│               │  • Class weighting amplifies minority signal
│               │  • Combined approach most effective
│               │
│               ├─ EXPECTED IMPROVEMENT:
│               │  • Minority class F1: +20-30%
│               │  • Overall macro F1: +10-15%
│               │
│               ├─ DOCUMENTATION:
│               │  → [INTEGRATION_GUIDE.md - Solution 2](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)
│               │  → [INTEGRATION_GUIDE.md - Solution 3](INTEGRATION_GUIDE.md#solution-3-combined-qml--contrastive-for-best-results)
│               │
│               └─ NEXT: Go to [INTEGRATION_GUIDE.md - Solution 2](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)
```

### Decision Point 3: Dataset Size 500-1000 Samples

```
Medium-Large Dataset (500-1000 samples)
│
├─→ Do you have > 20% missing modalities?
│   │
│   ├─ YES
│   │  └─→ RECOMMENDATION: Transformer Fusion (+ Contrastive)
│   │     │
│   │     ├─ WHY:
│   │     │  • Transformers natively handle missing modalities
│   │     │  • Attention masking elegantly handles gaps
│   │     │  • Cross-modal fusion learns interactions
│   │     │
│   │     ├─ DOCUMENTATION:
│   │     │  → [INTEGRATION_GUIDE.md - Pattern 2](INTEGRATION_GUIDE.md#pattern-2-transformer-fusion-replacing-qml-base-learners)
│   │     │
│   │     └─ NEXT: See Pattern 2
│   │
│   └─ NO
│      └─→ Check next criteria
│
└─→ Is there class imbalance?
    │
    ├─ YES
    │  └─→ RECOMMENDATION: Contrastive → Transformer or QML Meta-Learner
    │     │
    │     ├─ EXPECTED IMPROVEMENT:
    │     │  • F1 gain: +8-15%
    │     │
    │     └─ DOCUMENTATION:
    │        → [INTEGRATION_GUIDE.md - Solution 2](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)
    │
    └─ NO
       └─→ RECOMMENDATION: Transformer Fusion (simple path)
          │
          ├─ EXPECTED IMPROVEMENT:
          │  • F1 gain: +3-8%
          │
          └─ DOCUMENTATION:
             → [INTEGRATION_GUIDE.md - Pattern 2](INTEGRATION_GUIDE.md#pattern-2-transformer-fusion-replacing-qml-base-learners)
```

### Decision Point 4: Dataset Size 1000+ Samples

```
Large Dataset (1000+ samples)
│
└─→ RECOMMENDATION: Full Hybrid Pipeline
   │
   ├─ STAGES:
   │  1. Contrastive Pretraining (200 epochs)
   │  2. Transformer Fusion (50 epochs)
   │  3. QML Meta-Learner (final)
   │
   ├─ EXPECTED RESULTS:
   │  • F1 improvement: +15-25%
   │  • Training time: 16-24 hours (GPU required)
   │  • Best overall performance
   │
   ├─ DOCUMENTATION:
   │  → [INTEGRATION_GUIDE.md - Pattern 1](INTEGRATION_GUIDE.md#pattern-1-qml-as-meta-learner-with-deep-learning-base-learners)
   │
   └─ NEXT: Go to [INTEGRATION_GUIDE.md - Pattern 1](INTEGRATION_GUIDE.md#pattern-1-qml-as-meta-learner-with-deep-learning-base-learners)
```

---

## 🎯 Problem-Specific Decision Trees

### Problem: Class Imbalance

```
Class Imbalance Detected
│
├─→ SOLUTION 1: Class Weighting in QML
│   ├─ Cost: Minimal (modify loss function)
│   ├─ Improvement: +5-10%
│   ├─ Time: 2-4 hours
│   └─ Documentation: [INTEGRATION_GUIDE.md - Solution 1](INTEGRATION_GUIDE.md#solution-1-qml-pipeline-with-weighted-loss)
│
├─→ SOLUTION 2: Contrastive Pretraining
│   ├─ Cost: High (requires GPU, 8-12 hours)
│   ├─ Improvement: +10-20% (especially minorities)
│   ├─ Time: 8-12 hours
│   ├─ Requirement: Unlabeled data or enough labeled data
│   └─ Documentation: [INTEGRATION_GUIDE.md - Solution 2](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)
│
└─→ SOLUTION 3: Combined Approach
    ├─ Cost: High
    ├─ Improvement: +15-25% (best results)
    ├─ Time: 12-24 hours
    ├─ Combination: Contrastive + Class Weighting + QML Ensemble
    └─ Documentation: [INTEGRATION_GUIDE.md - Solution 3](INTEGRATION_GUIDE.md#solution-3-combined-qml--contrastive-for-best-results)
```

### Problem: Missing Modalities (> 20%)

```
Missing Modalities Problem
│
├─→ STRATEGY: Transformer Fusion
│   ├─ Why: Native attention masking support
│   ├─ How: Masks missing modalities automatically
│   ├─ Improvement: +5-10%
│   └─ Documentation: [INTEGRATION_GUIDE.md - Pattern 2](INTEGRATION_GUIDE.md#pattern-2-transformer-fusion-replacing-qml-base-learners)
│
├─→ ALTERNATIVE: QML with Conditional Encoding
│   ├─ Why: Handles missing via indicators
│   ├─ How: CFE approach with indicator features
│   ├─ Improvement: +2-5%
│   └─ Documentation: [README.md](README.md)
│
└─→ COMBINE: Transformer + QML Meta-Learner
    ├─ Why: Best flexibility
    ├─ Improvement: +8-15%
    └─ Documentation: [INTEGRATION_GUIDE.md - Pattern 1](INTEGRATION_GUIDE.md#pattern-1-qml-as-meta-learner-with-deep-learning-base-learners)
```

---

## 💡 Quick Reference

### By Improvement Priority
- **-6% F1**: Contrastive on balanced 100-400 sample labeled-only data ❌
- **+3% F1**: Data-Reuploading QML on small balanced data
- **+5-10% F1**: Contrastive + QML on imbalanced or with unlabeled
- **+8-15% F1**: Transformer Fusion
- **+10-25% F1**: Full Hybrid Pipeline

### By Computational Cost
- **Lowest**: QML Only (CPU, 2-4h)
- **Medium**: Contrastive + QML (GPU, 8-12h)
- **High**: Transformer Fusion (GPU, 6-12h)
- **Highest**: Full Hybrid (GPU cluster, 16-24h)

### By Data Requirements
- **Least data needed**: QML Only (50+ samples)
- **Medium data needed**: Contrastive + QML (100+ samples + unlabeled)
- **More data needed**: Transformer Fusion (500+ samples)
- **Most data needed**: Full Hybrid (1000+ samples)

---

## 📍 Where to Find Help

**Decision made? Go to:**
→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for implementation details

**Still unsure?**
→ [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for more guidance

**Ran into error?**
→ [INTEGRATION_GUIDE.md - Troubleshooting](INTEGRATION_GUIDE.md#troubleshooting)

**Want to understand everything?**
→ [NAVIGATION_SITEMAP.md](NAVIGATION_SITEMAP.md)

---

**Last Updated:** December 28, 2024  
**Decision Tree Completeness:** ✅ 100%
