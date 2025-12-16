# Documentation Index

Quick reference to find the information you need.

## 🔄 Understanding the 2-Step Preprocessing Funnel

Both Approach 1 (DRE) and Approach 2 (CFE) use a **2-step funnel** to prepare data for quantum circuits:

**Approach 1**: Imputation → Dimensionality Reduction (PCA/UMAP)
**Approach 2**: Imputation → Feature Selection (LightGBM/XGBoost/Hybrid)

See [ARCHITECTURE.md](ARCHITECTURE.md) and [WORKFLOW_INTEGRATION_GUIDE.md](WORKFLOW_INTEGRATION_GUIDE.md) for detailed explanations.

---

## Need to Understand the Workflows?
→ **[WORKFLOW_INTEGRATION_GUIDE.md](WORKFLOW_INTEGRATION_GUIDE.md)** ⭐ START HERE
- Complete explanation of existing QML workflow vs performance extensions
- **2-step funnel**: Imputation → Dimensionality Reduction/Feature Selection
- Standalone vs integrated usage
- Integration strategies
- Complete usage examples

## Need Technical Details on Performance Extensions?
→ **[PERFORMANCE_EXTENSIONS.md](PERFORMANCE_EXTENSIONS.md)**
- Deep technical specification
- Research paper references
- Implementation roadmap
- Challenges and mitigation

## Need Quick Start Guide for Extensions?
→ **[examples/README.md](examples/README.md)**
- Quick start examples
- Command-line reference
- Expected performance
- Troubleshooting

## Need to Understand QML Architecture?
→ **[ARCHITECTURE.md](ARCHITECTURE.md)** ⭐ UPDATED WITH 2-STEP FUNNEL
- **2-step preprocessing funnel** (Imputation → Feature Selection/Reduction)
- **Feature selection options**: LightGBM, XGBoost, Hybrid methods
- **Detailed integration examples** with QML
- Deep dive into quantum models
- Nested cross-validation
- Advanced training features
- Classical design decisions

## Need Main Project README?
→ **[README.md](README.md)** ⭐ UPDATED WITH 2-STEP FUNNEL
- Project overview with **2-step funnel** explanation
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
