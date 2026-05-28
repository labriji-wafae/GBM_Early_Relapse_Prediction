# Multimodal Machine Learning for Early GBM Relapse Prediction at Baseline (M0)

A Machine Learning pipeline designed to predict **early local tumor relapse** in high-grade brain tumor **Glioblastoma (GBM)** patients at baseline (fixed timepoint M0). 

This repository leverages advanced clinical features extracted from multiple processing modalities—combining conventional structural **3D MRI** with advanced quantitative **Magnetic Resonance Spectroscopy Imaging (MRSI)**. The core software is architected to tackle the three major hurdles of computer-aided clinical diagnosis: massive class imbalance, high-dimensional multimodality redundancy, and strict subject-level separation.

---

## 📐 Project Architecture

The code repository is refactored into a standardized, clean, flat layout where each sub-module bears a single and clear software engineering responsibility:

```text
GBM_Early_Relapse_Prediction/
│
├── configs/                    # Extensible training hyperparameters configurations
│
├── data/
│   ├── __init__.py
│   └── data_balancing.py       # Robust class imbalance sampler with automated mathematical fallbacks
│
├── features/
│   ├── __init__.py
│   └── selection.py            # Supervised predictors optimization (Boruta, Lasso RFE, Fisher scoring)
│
├── models/
│   ├── __init__.py
│   ├── gridsearch_custom.py    # Custom Scikit-Learn GridSearch with dynamic threshold (tau) tuning
│   └── evaluation.py           # Strict Leave-One-Group-Out (LOGO) patient validation split
│
├── notebooks/                  # Preserved and cleaned unsupervised explorative prototypes
│   ├── 1_spatial_unsupervised_clustering.ipynb
│   └── 2_predictive_modeling_benchmarks.ipynb
│
├── run_prediction_pipeline.py  # Main production orchestration entrypoint (argparse managed CLI)
├── .gitignore                  # Protection filters against large clinical .pkl and .h5 files
├── requirements.txt            # Unified environment packages
└── README.md                   # Main descriptive documentation
```
---

## ⚡ Highlights

* **Dynamic Decision Threshold Tuning ($\tau$):** In heavy clinical class imbalance scenarios, standard probability cutoff thresholds ($0.5$) drastically collapse sensitivity. Our custom `GridSearchLOOCV` class extends the Scikit-Learn estimator API to actively optimize hyper-parameters alongside a dynamic decision boundary threshold ($\tau$) mapped against F1-score maximizations in internal loops.
* **Strict Patient-Wise Data Isolation:** To counter data leakage artifacts (where voxels belonging to the same subject cross into both train and validation subsets, leading to over-optimistic test records), evaluation is entirely wrapped into a rigid **Leave-One-Group-Out (LOGO)** cross-validation matrix. Each patient acts as a pure, unseen external testing volume.
* **Redundancy & Multi-collinearity Scrubbing:** Multimodal feature spaces often carry severe spatial correlations. The feature engineering module includes deterministic filtering blocks to systematically discard multi-collinear predictors above customizable metrics ($>0.85$), prior to executing non-linear robust selectors such as **Boruta (Random Forest shadows)**.
* **Adaptive Sampling Resiliency:** Unsupervised spatial cluster noise can distort raw synthetic updates. The data balancing block features protective logical wrappers: if a mathematical corner case trips an error during advanced samplings (e.g., `KMeansSMOTE`), it dynamically defaults back to deterministic Scikit-Learn resampling routines to secure persistent loop pipelines.

---

## 🚀 Installation & Setup

We recommend setting up a dedicated virtual environment or a standalone conda block:

### 1. Clone the repository

```bash
git clone [https://github.com/labriji-wafae/GBM_Early_Relapse_Prediction.git](https://github.com/labriji-wafae/GBM_Early_Relapse_Prediction.git)
cd GBM_Early_Relapse_Prediction

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

---

## 💻 Running the End-to-End Pipeline

The core framework exposes a fully configurable Command-Line Interface (CLI). You can test benchmarks across varying balancing strategies or switch feature engineering algorithms on-the-fly.

### Standard Boruta Selection + SMOTE Balancing

```bash
python run_prediction_pipeline.py --data_path ./dataframe/MsrGB_features_M0.pkl --fs_method boruta --balancing smote

```

### Lasso Regularization + Naive Resampling

```bash
python run_prediction_pipeline.py --data_path ./dataframe/MsrGB_features_M0.pkl --fs_method lasso --balancing resample

```

### Explore CLI Parameters help:

```bash
python run_prediction_pipeline.py --help

```

---

## 📊 Evaluation Outputs & Audit Reports

Every automated cycle displays standard clinical classification matrices to the console and exports a subject-by-subject assessment report (`patient_evaluation_report.csv`) compiling:

* Voxel counts and localized micro-relapse baseline event ratios.
* Subject-specific performance metrics (**Sensitivity, Specificity, AUC, F1, AUPRC**).

```

---