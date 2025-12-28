# ⚠️ DEPRECATED - This file has been consolidated

**This document has been merged into [DOCS_GUIDE.md](DOCS_GUIDE.md).**

Please use the new consolidated documentation:
- **[DOCS_GUIDE.md](DOCS_GUIDE.md)** - Navigation guide with decision trees
- **[README.md](README.md)** - Main entry point

**This file can be safely deleted.**
│  ├─ 2-step training-time preprocessing funnel
│  ├─ Installation & setup
│  └─ Basic commands
│
├─ DOCUMENTATION_INDEX.md 📍
│  ├─ Decision tree
│  ├─ Quick decision guide (by dataset size)
│  ├─ Common questions & answers
│  ├─ Reading paths
│  ├─ Module documentation
│  └─ FAQ
│
├─ INTEGRATION_GUIDE.md ⭐⭐⭐ (MAIN REFERENCE)
│  ├─ Architecture decision tree
│  ├─ Integration patterns (3 options)
│  │  ├─ Pattern 1: QML Meta-Learner + Deep Learning Base Learners
│  │  ├─ Pattern 2: Transformer Fusion
│  │  └─ Pattern 3: Contrastive Pretraining + QML
│  ├─ Class imbalance solutions (3 options)
│  ├─ Small dataset strategy
│  ├─ Medium dataset strategies (2 scenarios)
│  ├─ Large dataset strategy
│  ├─ Step-by-step workflows (2 workflows)
│  ├─ Real-world examples (3 examples)
│  ├─ Performance trade-offs & matrices
│  ├─ Pros/cons comparison
│  └─ Troubleshooting (5 issues + solutions)
│
├─ ARCHITECTURE.md
│  ├─ Quantum circuit design
│  ├─ Model descriptions
│  ├─ 2-step training-time preprocessing funnel
│  ├─ Design decisions & trade-offs
│  └─ Performance metrics
│
├─ PERFORMANCE_EXTENSIONS.md
│  ├─ Transformer fusion technical specs
│  ├─ Contrastive learning framework
│  ├─ Augmentation strategies
│  └─ Loss functions
│
├─ WORKFLOW_INTEGRATION_GUIDE.md
│  ├─ Existing QML workflow
│  ├─ Performance extensions workflow
│  ├─ Standalone vs integrated usage
│  ├─ Integration strategies
│  ├─ Implementation details
│  ├─ Metrics & evaluation
│  └─ Complete usage examples
│
├─ examples/README.md
│  ├─ Embedding dimensions explained
│  ├─ Performance extensions overview
│  └─ Example code usage
│
└─ COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md
   ├─ 116 passing tests
   ├─ 95% code coverage
   ├─ Test categories
   └─ Validation coverage matrix
```

---

## 🔍 By Task/Problem

### Getting Started
```
START
 ├─→ New to project?
 │   └─→ [QUICK_START.md](QUICK_START.md)
 │
 └─→ Want to run QML only?
     └─→ [README.md](README.md)
```

### Choosing an Approach
```
Need guidance?
 ├─→ Quick overview (5 min)
 │   └─→ [QUICK_START.md](QUICK_START.md)
 │
 ├─→ Dataset size guide (10 min)
 │   └─→ [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md#-dataset-size-guide)
 │
 └─→ Complete decision tree (20 min)
     └─→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#architecture-decision-tree)
```

### Integration Questions
```
Want to integrate extensions?
 ├─→ Which patterns exist?
 │   └─→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#integration-patterns)
 │
 ├─→ Step-by-step workflow
 │   └─→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#step-by-step-integration-workflows)
 │
 └─→ Real-world examples
     └─→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#practical-examples)
```

### Specific Problems
```
Class Imbalance?
 └─→ [INTEGRATION_GUIDE.md - Solutions](INTEGRATION_GUIDE.md#class-imbalance-and-small-dataset-considerations)

Missing Modalities?
 └─→ [INTEGRATION_GUIDE.md - Strategy](INTEGRATION_GUIDE.md#architecture-decision-tree)

OOM Errors?
 └─→ [INTEGRATION_GUIDE.md - Troubleshooting](INTEGRATION_GUIDE.md#issue-1-out-of-memory-during-transformer-training)

Small Dataset?
 └─→ [INTEGRATION_GUIDE.md - Strategy](INTEGRATION_GUIDE.md#small-dataset-strategy--100-samples)

Imbalanced + Small?
 └─→ [INTEGRATION_GUIDE.md - Scenario B](INTEGRATION_GUIDE.md#scenario-b-you-have-only-labeled-data-no-unlabeled-balanced-classes--400-samples)
```

### Technical Details
```
Want deep understanding?
 ├─→ Quantum models
 │   └─→ [ARCHITECTURE.md](ARCHITECTURE.md)
 │
 ├─→ Transformer/Contrastive
 │   └─→ [PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md)
 │
 └─→ All workflows
     └─→ [WORKFLOW_INTEGRATION_GUIDE.md](WORKFLOW_INTEGRATION_GUIDE.md)
```

### Running Code
```
Need examples?
 ├─→ Quick start script
 │   └─→ [examples/README.md](examples/README.md)
 │
 └─→ Integration workflows
     └─→ [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#practical-examples)
```

### Validation/Testing
```
Want to validate?
 └─→ [COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md](COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md)
     - 116 tests
     - 95% coverage
     - Production readiness
```

---

## 📊 Dataset Size Navigation

```
< 100 samples
 ├─ README.md (QML Only)
 └─ Time: 2-4 hours

100-400 balanced samples (NO unlabeled)
 ├─ INTEGRATION_GUIDE.md - Scenario B
 ├─ Find: Data-Reuploading QML or QML Ensemble
 ├─ Avoid: Contrastive pretraining (empirically worse -6% F1)
 └─ Time: 3-4 hours

100-500 WITH unlabeled data
 ├─ INTEGRATION_GUIDE.md - Pattern 3
 ├─ Find: Contrastive Pretraining + QML
 └─ Time: 8-12 hours

500-1000 samples
 ├─ INTEGRATION_GUIDE.md - Pattern 2
 ├─ Find: Transformer Fusion
 └─ Time: 6-12 hours

1000+ samples
 ├─ INTEGRATION_GUIDE.md - Pattern 1
 ├─ Find: Full Hybrid (Contrastive → Transformer → QML)
 └─ Time: 16-24 hours
```

---

## 🎓 Learning Paths

### Path 1: QML Only (30 min)
```
README.md (10 min)
    ↓
ARCHITECTURE.md (10 min)
    ↓
Try running commands (10 min)
```

### Path 2: With Extensions (1.5 hours)
```
QUICK_START.md (5 min)
    ↓
README.md (10 min)
    ↓
INTEGRATION_GUIDE.md (30 min)
    ↓
examples/README.md (5 min)
    ↓
Try implementation (30 min)
```

### Path 3: Comprehensive (3 hours)
```
QUICK_START.md (5 min)
    ↓
README.md (10 min)
    ↓
ARCHITECTURE.md (15 min)
    ↓
INTEGRATION_GUIDE.md (45 min)
    ↓
PERFORMANCE_EXTENSIONS.md (15 min)
    ↓
WORKFLOW_INTEGRATION_GUIDE.md (25 min)
    ↓
examples/README.md (5 min)
    ↓
COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md (10 min)
```

---

## 🔗 Cross-References

### From README.md
- Links to: ARCHITECTURE, INTEGRATION_GUIDE
- Used by: First-time users wanting QML-only

### From INTEGRATION_GUIDE.md
- Links to: README, ARCHITECTURE, PERFORMANCE_EXTENSIONS, examples
- Used by: Users wanting to integrate extensions, handle imbalance, troubleshoot

### From ARCHITECTURE.md
- Links to: README, INTEGRATION_GUIDE, PERFORMANCE_EXTENSIONS
- Used by: Users wanting to understand models deeply

### From PERFORMANCE_EXTENSIONS.md
- Links to: INTEGRATION_GUIDE, examples, README
- Used by: Users implementing transformer/contrastive components

### From DOCUMENTATION_INDEX.md
- Links to: All documents
- Used by: Finding specific topics, navigation

### From QUICK_START.md
- Links to: All documents
- Used by: First-time users wanting quick guidance

---

## ✅ Document Completeness Checklist

- ✅ README.md - QML-only pipeline guide
- ✅ ARCHITECTURE.md - Detailed model descriptions
- ✅ INTEGRATION_GUIDE.md - Complete integration reference (2000+ lines)
- ✅ PERFORMANCE_EXTENSIONS.md - Technical specifications
- ✅ WORKFLOW_INTEGRATION_GUIDE.md - Workflow comparisons
- ✅ examples/README.md - Example explanations
- ✅ COMPREHENSIVE_TEST_COVERAGE_SUMMARY.md - Test validation
- ✅ DOCUMENTATION_INDEX.md - Navigation hub
- ✅ QUICK_START.md - Quick guidance (new!)
- ✅ NAVIGATION_SITEMAP.md - This file (new!)

---

## 🎯 Key Decision Points

**Q: Which document do I read first?**
- First time? → QUICK_START.md
- Want fast answer? → DOCUMENTATION_INDEX.md
- Want full guidance? → INTEGRATION_GUIDE.md
- Want QML only? → README.md

**Q: How long to understand everything?**
- Quick overview: 5-10 min (QUICK_START.md)
- Basic understanding: 30 min (README + QUICK_START)
- Full understanding: 2 hours (all docs)
- Implementation: 1-2 days (depends on complexity)

**Q: Which doc has what I need?**
→ Check [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md#-common-questions--documentation)

---

**Last Updated:** December 28, 2024  
**Navigation Completeness:** ✅ 100%  
**Cross-References:** ✅ Comprehensive
