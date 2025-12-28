# ⚠️ DEPRECATED - This file has been consolidated

**This document has been merged into [DOCS_GUIDE.md](DOCS_GUIDE.md).**

Please use the new consolidated documentation:
- **[DOCS_GUIDE.md](DOCS_GUIDE.md)** - Navigation guide with decision trees
- **[README.md](README.md)** - Main entry point

**This file can be safely deleted.**

---

## 📊 Architecture Decision Tree

```
START: What's your use case?
│
├─→ QML-Only Pipeline (simplest)
│   └─→ [README.md](README.md) + [ARCHITECTURE.md](ARCHITECTURE.md)
│
├─→ Add Deep Learning (transformers/contrastive)
│   └─→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) ⭐
│       ├─→ Pattern 1: QML Meta-Learner with Deep Learning Base Learners
│       ├─→ Pattern 2: Transformer Fusion
│       └─→ Pattern 3: Contrastive Pretraining + QML
│
├─→ Class Imbalance Problem?
│   └─→ [INTEGRATION_GUIDE.md - Class Imbalance](INTEGRATION_GUIDE.md#class-imbalance-and-small-dataset-considerations)
│       ├─→ Solution 1: Class Weighting
│       ├─→ Solution 2: Contrastive Pretraining
│       └─→ Solution 3: Combined QML + Contrastive
│
└─→ Small Dataset (< 100 samples)?
    └─→ [INTEGRATION_GUIDE.md - Small Dataset](INTEGRATION_GUIDE.md#small-dataset-strategy--100-samples)
        └─→ Stick with QML (sample-efficient by design)
```

---

## 🎯 Finding Answers Fast

### Common Questions

**"I have 200 samples and class imbalance. What should I do?"**
→ [INTEGRATION_GUIDE.md - Scenario B Analysis](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples) + [Solution 2](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)

**"I'm getting OOM errors. How do I fix this?"**
→ [INTEGRATION_GUIDE.md - Troubleshooting](INTEGRATION_GUIDE.md#issue-1-out-of-memory-during-transformer-training)

**"Should I use contrastive pretraining on my small labeled dataset?"**
→ [INTEGRATION_GUIDE.md - Scenario B](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples) (Short answer: Probably not, -6% F1 in tests)

**"How do I add transformers to my QML pipeline?"**
→ [INTEGRATION_GUIDE.md - Workflow B](INTEGRATION_GUIDE.md#workflow-b-adding-transformer-fusion)

**"What's the best approach for 1000+ samples?"**
→ [INTEGRATION_GUIDE.md - Pattern 1 (Full Hybrid)](INTEGRATION_GUIDE.md#pattern-1-qml-as-meta-learner-with-deep-learning-base-learners)

**"How do I handle missing modalities?"**
→ [INTEGRATION_GUIDE.md - Missing Modality Strategy](INTEGRATION_GUIDE.md#architecture-decision-tree)

---

## 🔄 Understanding the 2-Step Preprocessing Funnel

The pipeline uses a **two-notebook data preparation** flow:

1. **Notebook 1**: `data-process.ipynb` (collect → align → merge → split → create indicators → select cases)
2. **Notebook 2**: `feature-extraction-xgb.ipynb` (Mutual Information → XGBoost to 500 features per modality)

Output: `final_processed_datasets_xgb_balanced/` (set `SOURCE_DIR` to use these)

**Training-time preprocessing (inside model scripts):**
- **Approach 1 (DRE):** Imputation → Dimensionality Reduction (PCA/UMAP)  
- **Approach 2 (CFE):** Imputation → Feature Selection (LightGBM/XGBoost)

→ See [DATA_PROCESSING_SUMMARY.md](DATA_PROCESSING_SUMMARY.md) and [ARCHITECTURE.md](ARCHITECTURE.md)---

## 📋 Document Purpose Matrix

| Document | What It Covers | When to Use | Time |
|----------|---|---|---|
| **[README.md](README.md)** | QML-only pipeline, 2-step funnel, commands | First time setup, QML-only approach | 10 min |
| **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** ⭐ | Integration patterns, decision tree, class imbalance, troubleshooting | **Main reference** for all integration questions | 30 min |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | QML models, quantum circuits, design choices | Understanding algorithm details | 20 min |
| **[PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md)** | Transformer, contrastive learning, technical specs | Understanding extension architectures | 15 min |
| **[WORKFLOW_INTEGRATION_GUIDE.md](WORKFLOW_INTEGRATION_GUIDE.md)** | Complete workflows end-to-end, comparisons | Detailed workflow understanding | 25 min |
| **[examples/README.md](examples/README.md)** | Running example scripts | Executing example code | 5 min |
| **[COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md](COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md)** | Test inventory, coverage metrics | Validation, production readiness | 10 min |

---

## 🎯 Dataset Size Guide

### QML Only Pipeline
**Best For:** < 100 samples  
**Advantages:** No GPU needed, fast training, handles small data well  
**Time:** 2-4 hours  
→ [README.md](README.md)

### Medium Dataset (100-500 samples)
**Best For:** Balanced or imbalanced data  
**Options:**
- **With unlabeled data** → Contrastive + QML → [INTEGRATION_GUIDE.md - Pattern 3](INTEGRATION_GUIDE.md#pattern-3-contrastive-pretraining--qml-pipeline)
- **Balanced, labeled-only** → Data-Reuploading QML or QML Ensemble → [INTEGRATION_GUIDE.md - Scenario B](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples)
- **Imbalanced** → Contrastive + QML with class weights → [INTEGRATION_GUIDE.md - Solution 2](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)

**Time:** 8-12 hours (with pretraining)  
→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)

### Large Dataset (500+ samples)
**Best For:** Maximum performance  
**Options:**
- **500-1000 samples** → Transformer Fusion
- **1000+ samples** → Full Hybrid (Contrastive → Transformer → QML Meta-learner)

**Time:** 16-24 hours  
→ [INTEGRATION_GUIDE.md - Pattern 1](INTEGRATION_GUIDE.md#pattern-1-qml-as-meta-learner-with-deep-learning-base-learners)

---

## 🔍 Finding Specific Topics

### By Problem

**Class Imbalance**
- Overview: [INTEGRATION_GUIDE.md - Class Imbalance](INTEGRATION_GUIDE.md#class-imbalance-and-small-dataset-considerations)
- Solution 1: [Weighted Loss](INTEGRATION_GUIDE.md#solution-1-qml-pipeline-with-weighted-loss)
- Solution 2: [Contrastive Pretraining](INTEGRATION_GUIDE.md#solution-2-contrastive-pretraining-for-imbalanced-data)
- Solution 3: [Combined Approach](INTEGRATION_GUIDE.md#solution-3-combined-qml--contrastive-for-best-results)

**Small Dataset**
- [INTEGRATION_GUIDE.md - Small Dataset Strategy](INTEGRATION_GUIDE.md#small-dataset-strategy--100-samples)
- Key insight: QML is sample-efficient by design

**Medium Dataset (100-400 labeled samples)**
- [INTEGRATION_GUIDE.md - Scenario B: Balanced, Labeled Only](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples)
- Empirical evidence: Contrastive doesn't help (-6% F1), use Data-Reuploading QML instead (+3% F1)

**Missing Modalities**
- [INTEGRATION_GUIDE.md - Missing Modality Strategy](INTEGRATION_GUIDE.md#architecture-decision-tree)
- Recommendation: Transformer Fusion (native support)

### By Technology

**QML (Quantum Machine Learning)**
- [README.md](README.md) - Basic pipeline
- [ARCHITECTURE.md](ARCHITECTURE.md) - Model details
- [qml_models.py](qml_models.py) - Implementation

**Transformer Fusion**
- [PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md) - Technical specs
- [INTEGRATION_GUIDE.md - Pattern 2](INTEGRATION_GUIDE.md#pattern-2-transformer-fusion-replacing-qml-base-learners) - Integration
- [examples/train_transformer_fusion.py](examples/train_transformer_fusion.py) - Code

**Contrastive Pretraining**
- [PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md) - Technical specs
- [INTEGRATION_GUIDE.md - Pattern 3](INTEGRATION_GUIDE.md#pattern-3-contrastive-pretraining--qml-pipeline) - Integration
- [examples/pretrain_contrastive.py](examples/pretrain_contrastive.py) - Code

### By Task

**Getting Started**
1. [README.md](README.md) - Overview
2. [INTEGRATION_GUIDE.md - Decision Tree](INTEGRATION_GUIDE.md#architecture-decision-tree) - Choose approach
3. [Your chosen path](#-reading-paths)

**Integrating Extensions**
- [INTEGRATION_GUIDE.md - Integration Patterns](INTEGRATION_GUIDE.md#integration-patterns)
- Choose Pattern 1, 2, or 3 based on your needs

**Adding Contrastive Pretraining**
- [INTEGRATION_GUIDE.md - Workflow A](INTEGRATION_GUIDE.md#workflow-a-adding-contrastive-pretraining-to-existing-qml-pipeline)

**Adding Transformer Fusion**
- [INTEGRATION_GUIDE.md - Workflow B](INTEGRATION_GUIDE.md#workflow-b-adding-transformer-fusion)

**Handling Class Imbalance**
- [INTEGRATION_GUIDE.md - Class Imbalance Solutions](INTEGRATION_GUIDE.md#class-imbalance-and-small-dataset-considerations)

**Troubleshooting**
- [INTEGRATION_GUIDE.md - Troubleshooting](INTEGRATION_GUIDE.md#troubleshooting)
- 5 common issues with solutions

---

## 🔗 Cross-References

All documents are tightly cross-referenced:

- README.md → Links to ARCHITECTURE.md for details
- ARCHITECTURE.md → Links to INTEGRATION_GUIDE.md for real-world use
- INTEGRATION_GUIDE.md → Links to examples for implementation
- PERFORMANCE_EXTENSIONS.md → Links to INTEGRATION_GUIDE.md for practical usage
- examples/README.md → Links back to INTEGRATION_GUIDE.md for theory

---

## 📊 Performance Comparison

Want to know which approach is best for your scenario?

→ [INTEGRATION_GUIDE.md - Performance Trade-offs](INTEGRATION_GUIDE.md#performance-trade-offs)

Contains:
- Computational cost analysis
- Performance improvement matrix (benchmarked)
- Pros/cons summary for each approach

---

## 🧪 Testing & Validation

→ [COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md](COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md)

- 116 passing tests
- 95% code coverage
- Production-ready validation

---

## 💻 Command Reference

### QML-Only Pipeline
```bash
# Tune hyperparameters
python tune_models.py --datatype GeneExp --approach 1 --qml_model standard

# Train base learners (Approach 1: DRE)
python dre_standard.py --datatype GeneExp --verbose

# Train base learners (Approach 2: CFE)
python cfe_standard.py --datatype GeneExp --verbose

# Train meta-learner
python metalearner.py --preds_dir base_learner_outputs --indicator_file indicators.parquet
```

→ See [README.md](README.md) for full commands

### With Performance Extensions
```bash
# Contrastive pretraining
python examples/pretrain_contrastive.py --data_dir data --output_dir encoders

# Transformer training
python examples/train_transformer_fusion.py --data_dir data --output_dir models

# Run inference
python inference.py --model_dir trained_models --patient_data_dir new_patient
```

→ See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for detailed workflows

---

## 🎓 Learning Order

### For First-Time Users
1. **5 min:** [README.md](README.md) - Get overview
2. **10 min:** [INTEGRATION_GUIDE.md - Decision Tree](INTEGRATION_GUIDE.md#architecture-decision-tree) - Choose your approach
3. **15 min:** [Your chosen documentation section](INTEGRATION_GUIDE.md#integration-patterns)
4. **30 min:** Run [example scripts](examples/README.md)

### For Experienced Users
1. **5 min:** [INTEGRATION_GUIDE.md - Decision Tree](INTEGRATION_GUIDE.md#architecture-decision-tree)
2. **10 min:** Skim [integration pattern](INTEGRATION_GUIDE.md#integration-patterns)
3. **20 min:** Implement using [step-by-step workflow](INTEGRATION_GUIDE.md#step-by-step-integration-workflows)

### For Advanced Users
1. **2 min:** [INTEGRATION_GUIDE.md - Quick Decision Guide](#-quick-start-5-minutes)
2. **5 min:** Jump to [specific workflow](INTEGRATION_GUIDE.md#step-by-step-integration-workflows)
3. **1 hour:** Implement with [example code](INTEGRATION_GUIDE.md#practical-examples)

---

## 🆘 Help! I'm Lost

1. **Don't know where to start?** → Go to [Quick Start](#-quick-start-5-minutes)
2. **Can't find what I need?** → Use [Finding Specific Topics](#-finding-specific-topics)
3. **Looking for specific code?** → Check [Module Documentation](#-module-documentation)
4. **Still stuck?** → See [Common Questions → Documentation](#-common-questions--documentation)

---

**Last Updated:** December 28, 2024

**Project Status:** ✅ Production-Ready  
**Test Coverage:** 95%  
**Documentation:** Comprehensive
- Directory layout
- Full QML workflow commands with **2-step funnel**
- CLI arguments reference

## Quick Decision Guide

### I want to...
- **Understand the 2-step funnel** → [ARCHITECTURE.md](ARCHITECTURE.md) (Stage 3)
- **See feature selection options** → [ARCHITECTURE.md](ARCHITECTURE.md) (Feature Selection Options)
- **See detailed integration examples** → [ARCHITECTURE.md](ARCHITECTURE.md) (Integration with QML)
- **Understand how everything works** → [WORKFLOW_INTEGRATION_GUIDE.md](WORKFLOW_INTEGRATION_GUIDE.md)
- **Run a quick experiment** → [examples/README.md](examples/README.md)
- **Understand the theory** → [PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md)
- **See the existing QML pipeline** → [README.md](README.md)
- **Understand QML architecture** → [ARCHITECTURE.md](ARCHITECTURE.md)

### I'm looking for...
- **2-step funnel explanation** → ARCHITECTURE.md (Stage 3) & README.md (Key Feature)
- **LightGBM vs XGBoost comparison** → ARCHITECTURE.md (Feature Selection Options)
- **Hybrid combining methods** → ARCHITECTURE.md (Integration Examples 3 & 4)
- **Workflow diagrams** → WORKFLOW_INTEGRATION_GUIDE.md (Part 1 & 2)
- **Integration strategies** → WORKFLOW_INTEGRATION_GUIDE.md (Part 4)
- **Code examples** → WORKFLOW_INTEGRATION_GUIDE.md (Part 7) & ARCHITECTURE.md
- **Metrics explanation** → WORKFLOW_INTEGRATION_GUIDE.md (Part 6)
- **Performance expectations** → WORKFLOW_INTEGRATION_GUIDE.md (Summary)
- **Research references** → PERFORMANCE_EXTENSIONS.md (References section)

### I need to know...
- **How does the 2-step funnel work?** → See ARCHITECTURE.md & README.md
- **Should I use LightGBM or XGBoost?** → Start with LightGBM (faster), try XGBoost if needed
- **How to use hybrid methods?** → See ARCHITECTURE.md (Examples 3, 4, 5)
- **Can I use without QML?** → YES! See WORKFLOW_INTEGRATION_GUIDE.md Part 3
- **How to integrate with QML?** → WORKFLOW_INTEGRATION_GUIDE.md Part 4 & ARCHITECTURE.md
- **Which approach for my case?** → WORKFLOW_INTEGRATION_GUIDE.md (Summary > Decision Tree)
- **Expected accuracy improvement?** → +5-20%, see WORKFLOW_INTEGRATION_GUIDE.md Part 6
- **Training time?** → 12 hours to 4 days, see WORKFLOW_INTEGRATION_GUIDE.md Part 7
- **What are input/output dimensions for contrastive encoder?** → PERFORMANCE_EXTENSIONS.md FAQ & examples/README.md FAQ
- **Can input dimension be different from 256?** → YES! See PERFORMANCE_EXTENSIONS.md FAQ Q1-Q2
- **Why is embed_dim 256 by default?** → See PERFORMANCE_EXTENSIONS.md FAQ Q4 & examples/README.md
- **How to change embed_dim?** → See PERFORMANCE_EXTENSIONS.md FAQ Q3 & examples/README.md FAQ

---

**Last Updated:** December 15, 2024
**Key Updates:** 
- Added comprehensive FAQ sections for embedding dimensions (input vs output, why 256, configurability)
- Enhanced documentation for contrastive encoder architecture
- Clarified dimension flow in both contrastive pretraining and transformer fusion
- Added 2-step funnel documentation, feature selection options (LightGBM/XGBoost/Hybrid), detailed integration examples
