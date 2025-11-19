# Plasma Multi-Omics and Interpretable Deep Neural Networks for Early Dementia Prediction

This repository contains the implementation of the framework described in the paper:

> **Plasma Multi-Omics and Interpretable Deep Neural Networks Enable Early Prediction of Dementia and Its Subtypes**  
> M. Mursil, et al.

The project builds an **interpretable deep neural network (DNN)** pipeline for classifying dementia status from **plasma multi-omics data**. It combines:

- rigorous **pre-processing and normalization**,
- **feature selection** with univariate tests and **Sequential Forward Selection (SFS)**,
- a **regularized DNN** with batch normalization and dropout,
- and **explainability analyses** using permutation importance and **SHAP**.

---

## Features

- **Multi-omics input** (e.g. proteomics) with missingness filtering and KNN imputation.
- **Rank-based inverse normal transform** to stabilize distributions.
- **Feature selection pipeline**:
  - ANOVA F-test (`SelectKBest`)
  - Sequential Forward Selection (SFS) with cross-validated AUC.
- **Class imbalance handling** via **RandomUnderSampler** and **SMOTE**.
- **Regularized DNN** classifier with batch normalization and dropout.
- **Explainability**:
  - Permutation feature importance
  - SHAP summary plots for global interpretability.
- Optional **logistic association analysis** for conventional statistical comparisons.

---

## Directory Structure

The repository is organised as follows:

```text
<repo-root>/
│
├── src/
│   ├── analysis/
│   │   ├── explainability.py        # Permutation importance & SHAP analyses
│   │   └── logistic_association.py  # Logistic / association analyses for selected features
│   │
│   ├── data/
│   │   ├── preprocessing.py         # Missingness filters, KNN imputation, train/test split
│   │   └── transforms.py            # Inverse normal transform and other utility transforms
│   │
│   ├── models/
│   │   └── models.py                # Regularized DNN and PyTorch model wrappers
│   │
│   └── training/
│       ├── feature_selection.py     # SelectKBest + Sequential Forward Selection (SFS)
│       └── training.py              # Training loop, early stopping, evaluation metrics
│
├── main.py                          # Entry point: runs full pipeline end-to-end
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Files/directories ignored by Git

> **Requirements**

This project uses Python 3.x and the following main libraries:

torch

pandas

numpy

scikit-learn

imblearn

matplotlib

scipy

shap


> **Reproducibility**

Several random seeds are fixed inside the code (for data shuffling, resampling, and cross-validation).
For strict reproducibility across environments, you may also explicitly set:
import numpy as np
import torch
import random

seed = 44
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


> **Citation**

If you use this repository in your research, please cite the associated paper:
@article{mursil2025multiomics_dnn,
  author  = {Mursil, M. and Rashwan, H. A. and Scarmeas, N. and Murphy, M. M. and Papandreou, C. and Puig, D. and others},
  title   = {Plasma Multi-Omics and Interpretable Deep Neural Networks Enable Early Prediction of Dementia and Its Subtypes},
  year    = {2025},
  journal = {To appear},
}
