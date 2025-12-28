# ⚠️ DEPRECATED - This file has been consolidated

**This document has been merged into [DOCS_GUIDE.md](DOCS_GUIDE.md).**

Please use the new consolidated documentation:
- **[DOCS_GUIDE.md](DOCS_GUIDE.md)** - Navigation guide with decision trees
- **[README.md](README.md)** - Main entry point

**This file can be safely deleted.**

---

## 🎲 Dataset Size Decision

### I have < 100 samples
→ **Use QML Only**  
📖 [README.md](README.md)  
⏱️ 2-4 hours training  
💻 No GPU needed  

### I have 100-500 samples
→ **Which option applies?**
- **Have unlabeled data?** → [Contrastive + QML](INTEGRATION_GUIDE.md#pattern-3-contrastive-pretraining--qml-pipeline)
- **Balanced, labeled-only?** → [Data-Reuploading QML](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples)
- **Imbalanced?** → [Contrastive + QML + Class Weights](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)

### I have 500+ samples
→ **Use Transformer Fusion or Full Hybrid**  
📖 [INTEGRATION_GUIDE.md - Pattern 1 or 2](INTEGRATION_GUIDE.md#integration-patterns)  
⏱️ 6-24 hours training  
💻 GPU required  

---

## 🔍 I Have a Specific Problem

### Class Imbalance
→ [INTEGRATION_GUIDE.md - Class Imbalance](INTEGRATION_GUIDE.md#class-imbalance-and-small-dataset-considerations)

**Quick solutions:**
- Solution 1: Class weighting in QML loss
- Solution 2: Contrastive pretraining (learns class-agnostic representations)
- Solution 3: Combined approach (best results)

### Missing Modalities (> 20% samples incomplete)
→ [INTEGRATION_GUIDE.md - Missing Modality Strategy](INTEGRATION_GUIDE.md#architecture-decision-tree)

**Quick solution:** Use Transformer Fusion (native attention masking support)

### Out of Memory (OOM) Errors
→ [INTEGRATION_GUIDE.md - Issue 1](INTEGRATION_GUIDE.md#issue-1-out-of-memory-during-transformer-training)

**Quick solutions:**
- Reduce batch size
- Reduce model size (fewer layers, smaller embeddings)
- Use gradient accumulation

### Poor Minority Class Performance
→ [INTEGRATION_GUIDE.md - Issue 3](INTEGRATION_GUIDE.md#issue-3-poor-performance-on-minority-classes)

**Quick solutions:**
- Use class weighting
- Use contrastive pretraining
- Oversample minority classes with SMOTE

### QML Training Too Slow
→ [INTEGRATION_GUIDE.md - Issue 4](INTEGRATION_GUIDE.md#issue-4-qml-training-too-slow)

**Quick solutions:**
- Reduce circuit complexity (fewer qubits/layers)
- Use standard QML instead of data-reuploading
- Sample subset of data for training

---

## 🔄 I Want to Add an Extension

### Add Contrastive Pretraining to My QML Pipeline
→ [INTEGRATION_GUIDE.md - Workflow A](INTEGRATION_GUIDE.md#workflow-a-adding-contrastive-pretraining-to-existing-qml-pipeline)

**Steps:**
1. Pretrain encoders on all data
2. Extract pretrained features
3. Replace PCA/UMAP with pretrained embeddings
4. Train QML as normal

**Expected improvement:** +5-10% F1 (if class imbalance or unlabeled data available)

### Add Transformer Fusion
→ [INTEGRATION_GUIDE.md - Workflow B](INTEGRATION_GUIDE.md#workflow-b-adding-transformer-fusion)

**Steps:**
1. Train transformer with pretrained or random encoders
2. Extract transformer predictions
3. Use as features for QML meta-learner

**Expected improvement:** +3-8% F1 (especially for missing modalities)

### Use Full Hybrid Pipeline
→ [INTEGRATION_GUIDE.md - Pattern 1](INTEGRATION_GUIDE.md#pattern-1-qml-as-meta-learner-with-deep-learning-base-learners)

**Steps:**
1. Contrastive pretraining (100-200 epochs)
2. Transformer fusion training (50 epochs)
3. QML meta-learner on transformer predictions

**Expected improvement:** +10-20% F1 (best for large datasets)

---

## 📚 All Documents

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [README.md](README.md) | QML-only pipeline | 10 min |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) ⭐ | Integration guide, decision tree, examples | 30 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Quantum models, design details | 20 min |
| [PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md) | Transformer/contrastive specs | 15 min |
| [WORKFLOW_INTEGRATION_GUIDE.md](WORKFLOW_INTEGRATION_GUIDE.md) | Workflow comparisons | 25 min |
| [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) | Navigation hub | 5 min |
| [examples/README.md](examples/README.md) | Example scripts | 5 min |
| [COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md](COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md) | Testing info | 10 min |

---

## 🎓 Recommended Reading Order

### Path 1: QML Only (30 min)
1. [README.md](README.md) - Overview
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Understanding models
3. Run example commands

### Path 2: With Extensions (1.5 hours)
1. [README.md](README.md) - Overview
2. [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Navigation
3. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Choose pattern
4. [examples/README.md](examples/README.md) - Run examples

### Path 3: Everything (3 hours)
1. [README.md](README.md)
2. [ARCHITECTURE.md](ARCHITECTURE.md)
3. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
4. [PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md)
5. [WORKFLOW_INTEGRATION_GUIDE.md](WORKFLOW_INTEGRATION_GUIDE.md)
6. [examples/README.md](examples/README.md)

---

## 💡 Pro Tips

✅ **Read the right docs for your dataset size** - Don't spend time on approaches you won't use

✅ **Check [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) first** - It has decision trees and empirical evidence

✅ **Look at Scenario B** - If you have 100-400 balanced labeled samples, [this section](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples) explains why contrastive pretraining may NOT help (-6% F1!)

✅ **Use troubleshooting guide** - [INTEGRATION_GUIDE.md - Troubleshooting](INTEGRATION_GUIDE.md#troubleshooting) has 5 common issues with solutions

✅ **Check performance matrix** - [INTEGRATION_GUIDE.md - Performance Trade-offs](INTEGRATION_GUIDE.md#performance-trade-offs) shows empirical results

---

## 🆘 Still Lost?

1. **Completely new?** → Start with [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md#-quick-start-5-minutes)
2. **Need specific help?** → Go to [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md#-finding-specific-topics)
3. **Want code examples?** → Check [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#practical-examples)
4. **Have errors?** → See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#troubleshooting)

---

**Last Updated:** December 28, 2024  
**Status:** ✅ Production Ready
