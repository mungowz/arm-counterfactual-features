# Global Feature Importance from Local Counterfactuals

This repository implements an end-to-end Explainable AI (XAI) pipeline that extracts **Global Feature Importance** from **local** counterfactual explanations.

The methodology is based on the approach described in the paper *"From local counterfactuals to global feature importance: efficient, robust, and model-agnostic explanations for brain connectivity networks"*. It has been deliberately adapted and optimized to handle purely categorical data and is applied to the **ACS Income 2024** (American Community Survey) demographic dataset.

## Overview and Key Features

The primary goal of this project is to identify which combinations of features (e.g., *Educational Attainment*, *Usual Hours Worked*, *Age*) drive machine learning prediction flips at the decision boundary.

- **Native Categorical Data Handling:** Utilizes Hamming distance and a custom, highly optimized version of the BoCSoR algorithm (`CategoricalBoCSoR`).
- **Extreme Performance (Apple M-Series Ready):** Counterfactual perturbation computations are fully vectorized using NumPy (eliminating Python `for` loops) and parallelized via `joblib` (using the `loky` backend) to maximize physical CPU core utilization.
- **Auto-Calibrated Association Rule Mining (ARM):** The FP-Growth rule extraction process automatically calibrates Minimum Support and Confidence parameters based on dataset sparsity. It intelligently filters out "neutral" correlations (Lift $\approx$ 1.0) while preserving analytically significant positive and negative correlations.
- **Adaptive Regional Thresholds:** The dataset is rigorously balanced via stratified undersampling and uses differentiated income thresholds based on regional cost of living (Northeast: $110,000, South: $90,000).

---

## Pipeline Structure

The project is structured into three sequentially executable modules:

### 1. `create_dataset.py` (ETL & Preprocessing)

Downloads 2024 PUMS microdata via the `folktables` library.

- Applies human-readable mappings to standard Census Bureau categorical codes.
- Performs continuous-to-categorical binning for `AGEP` (Age) and `WKHP` (Usual Hours Worked) adhering to standard demographic guidelines (e.g., *Mid-Career*, *Full-Time*).
- Saves the final, balanced, and classification-ready datasets into the `data/` directory.

### 2. `feature_importance.py` (Local Counterfactuals)

Trains a `CatBoost` classifier and builds `BallTree` spatial search structures to find opposite-class neighbors.

- Extracts local explanations by injecting counterfactual feature values into the original boundary instances to verify prediction switches (drivers).
- Explores different neighborhood sizes ($k \in [1, 3, 5, 7]$).
- Exports the extracted local drivers into transactional CSV formats within the `results/` directory.

### 3. `macroscopic_experiment_association_rules.py` (Global ARM)

Aggregates the local counterfactual explanations and applies the **FP-Growth** algorithm to extract global association rules.

- Executes a parallelized grid search to uncover the most frequent feature patterns.
- Automatically generates diagnostic heatmaps (Support vs. Confidence vs. Lift) and comprehensive calibration logs.

---

## Installation

Ensure you have Python 3.9+ installed on your system. Clone this repository and install the required dependencies.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy folktables catboost scikit-learn joblib mlxtend matplotlib
```

## Usage Guide

Execute the scripts in the following order to run the full experimental pipeline:

### Step 1: Data Generation

```bash
python create_dataset.py
```

This script will take a few minutes to download the raw data via the Census API. It generates acs_income_northeast_2024.csv and acs_income_south_2024.csv in the data/ folder.

### Step 2: Local Counterfactual Extraction

```bash
python feature_importance.py
```

This step trains the CatBoost model and evaluates spatial boundary perturbations. Raw and aggregated transactional results for each k value will be saved to results/<region>/important_features/.

### Step 3: Global Rule Extraction (FP-Growth)

```bash
python macroscopic_experiment_association_rules.py
```

Calculates frequent itemsets, auto-calibrates optimal FP-Growth parameters, and exports the final association rules along with visual heatmaps to results/<region>/association_rules/.

## Output Directory Structure (results/)

Running the full pipeline generates a directory tree similar to the following:

Plaintext
results/
├── northeast/
│   ├── important_features/
│   │   ├── feature_importance_log.txt
│   │   ├── k_1/  (and k_3, k_5, k_7)
│   │   │   ├── aggregated_labels_by_sample.csv  <-- Target input for ARM
│   │   │   ├── transactions_values.csv          <-- Local Explanations
│   │   │   └── ...
│   └── association_rules/
│       └── auto_sup=auto_d0.02_conf=0.05.../
│           ├── experiment_log.txt
│           ├── k_comparison/
│           │   ├── k_comparison_summary.csv
│           │   ├── heatmap_k_support.png
│           │   └── ...
│           └── k_1/  (and k_3, k_5, k_7)
│               ├── calibration_log.txt
│               ├── summary.csv
│               └── heatmaps/
└── south/
    └── ... (Matches Northeast structure)

## References

Original Methodology: From local counterfactuals to global feature importance: efficient, robust, and model-agnostic explanations for brain connectivity networks.

Data Source: American Community Survey (ACS) 2024 PUMS via Folktables
