# MATH3310-FALL-2025-P2-G2

## Abstract
This project studies imbalanced binary classification on the Mammography dataset. The workflow compares baseline and tuned classifiers, tests imbalance-handling strategies (class weighting, SMOTE + undersampling), and uses model interpretability tools (MDI, permutation importance, SHAP) to explain learned feature effects.

## Problem Statement
Predict mammography class labels (`-1` majority, `1` minority) from six numerical attributes (`attr1`-`attr6`) under strong class imbalance.

The central objective is not only high overall accuracy, but improved minority-class detection quality using imbalance-aware metrics and validation strategies.

## Mathematical/Statistical Framework
1. Binary classification with severe imbalance
- Dataset ratio is approximately 42:1 (`-1`: 10,923 vs `1`: 260).

2. Model families evaluated
- Decision Tree
- Random Forest
- Logistic Regression

3. Preprocessing and model composition
- Median imputation + standardization for numeric features
- Optional categorical encoding block retained in pipeline architecture
- End-to-end `Pipeline` / `imblearn.Pipeline`

4. Evaluation metrics
- Accuracy
- Balanced accuracy
- Minority-class F1
- Error rate
- G-mean (`geometric_mean_score`)
- Confusion matrix and classification report

5. Validation design
- Initial train/test split with stratification
- 5-fold stratified CV for baseline
- GridSearchCV for model/hyperparameter selection
- Additional sensitivity checks using shuffled `KFold` and alternative random seeds

## Implementation Overview
### 1) Main entry point files
- `scripts/project2_repro.py` (reproducible command-line runner)
- `notebooks/project2.ipynb` (exploratory notebook)

### 2) Core modeling assets
- `scripts/project2_repro.py`
- `notebooks/project2.ipynb`

### 3) Dataset used
- `notebooks/Mammography.csv`
- Shape: 11,183 rows x 7 columns
- Features: `attr1`-`attr6`
- Target: `class`
- Missing values detected in CSV: 0

### 4) Inferred problem framing
A supervised imbalanced binary classification problem focused on improving minority-class detection while preserving acceptable global performance.

### 5) Evaluation methodology implemented
- Baseline Random Forest with stratified train/test split and stratified CV
- Grid search over Decision Tree, Random Forest, Logistic Regression
- Comparison against class-weighted Random Forest
- Comparison against SMOTE + random undersampling + Random Forest
- Post-hoc feature attribution via impurity and permutation importance

## Results
Key notebook outputs report:

1. Baseline Random Forest
- Test accuracy: 0.9861
- Test balanced accuracy: 0.7395
- Minority-class precision/recall/F1: 0.86 / 0.48 / 0.62

2. Tuned model (GridSearchCV best)
- Best estimator: RandomForestClassifier (`n_estimators=1000`)
- Test accuracy: 0.9870
- Test balanced accuracy: 0.7587
- Confusion matrix (test): `[[2181, 4], [25, 27]]`

3. Aggregate metric summaries (reported in notebook)
- Baseline RF: Balanced Accuracy 0.7395, Error 0.0139, F1 0.6173, G-Mean 0.6927
- Tuned model: Balanced Accuracy 0.7587, Error 0.0130, F1 0.6506, G-Mean 0.7199
- Shuffle-seed variant: Balanced Accuracy 0.7782, Error 0.0116, F1 0.6905, G-Mean 0.7463
- SMOTE pipeline (reported): Balanced Accuracy 0.9073, Error 0.0161, F1 0.7049, G-Mean 0.9037

## Reproducibility
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Use a compatible Python version:
- `>=3.10,<3.13` (aligned with `pyproject.toml`)

3. Run reproducible pipeline (recommended):
```bash
python scripts/project2_repro.py
```
This writes a metrics artifact to `reports/results/project2_metrics.json`.

4. Optional exploratory notebook run:
```bash
jupyter notebook notebooks/project2.ipynb
```

5. Makefile shortcut:
```bash
make run_repro
```

## Limitations
- The reproducible script and notebook are aligned conceptually, but only the script guarantees a single consistent split/evaluation path.
- Some reported metrics depend strongly on random seeds and split strategy.
- Metric tracking is print-based; no persisted experiment registry.
- No unit/integration tests for data processing or model evaluation.
- Notebook-side SMOTE result paths should still be interpreted carefully due to cell-order dependence.
