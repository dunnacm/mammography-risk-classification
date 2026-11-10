# Methods, data contract and reproduction

## Question and observation

Can recorded class +1 be distinguished from class -1 using six numeric attributes? This is a supervised learning exercise. The outcome is the CSV's recorded label, not a calibrated probability of disease or a verified patient-level outcome.

Source: `notebooks/Mammography.csv`, 11,183 rows, numeric `attr1` through `attr6`, target `class` in {-1, +1}. Positive means +1. The course reference describes +1 as calcification and -1 as non-calcification; neither label is a cancer diagnosis. Labels are retained as saved (-1/+1) rather than silently recoded. Counts: 10,923 negative and 260 positive. There are no missing cells in this saved input. The runner rejects unrecognized columns, invalid targets, infinities and all-missing features; partial numeric missingness is median-imputed within training folds. Units, patient identifiers and detailed acquisition information are not supplied by the CSV. Original project materials identify the dataset as Mammography from OpenML; clinical semantics should not be inferred from anonymous attribute names.

SHA-256: `a5475b88aa0373c2dd530c165ea5e6805a93d4e0a870e982479dc59723e9ee7b`.

All source rows are retained. There are 7,848 unique exact six-feature profiles, 3,335 rows beyond the first occurrence of each profile, and one feature profile with conflicting class labels. Identical features do not prove duplicate patients. The conflict is reported, not silently relabeled or dropped.

## Evaluation protocol

Default: stratified 80/20 row split with seed 42; five training-only stratified folds with seed 100. Median imputation and scaling are fitted inside each fold. Scaling is useful for logistic regression and SMOTE's distance calculations; it is not needed for forest thresholds. SMOTE runs only within each training fold and is never applied to validation/test observations.

Candidate comparison uses mean training-CV balanced accuracy. The best of baseline RF, weighted RF, SMOTE RF and the grid winner is frozen before any test metrics are computed. The modest grid contains two tree depths, two forest sizes and three logistic regularization values. Its CV selection score is optimistic after choosing the best candidate; it is not an independent performance estimate. Other test results are descriptive comparisons and must not be used for further tuning.

The recorded data and historical test outputs have already been inspected. This run is reproducibility and methodological illustration, not a fresh confirmatory experiment. No threshold optimization, calibration study, external validation, clinical cost model or deployment claim is made. This focused repair is not a complete reimplementation of the assignment: fresh EDA charts, hyperparameter-response plots, shuffled-KFold experiments and MDI/SHAP comparisons remain teaching exercises; historical outputs are retained with their limitations.

Optional sensitivity:

```powershell
python scripts/reviewed_analysis.py --split-strategy profile --output-json reports/results/profile_metrics.json
```

This groups exact feature profiles together in both outer and inner folds. It measures generalization to previously unseen exact profiles, a different question from the default row split. It does **not** establish patient independence or greater real-world validity. The first of five deterministic (unshuffled) stratified group folds is held out; the realized fraction and class counts are recorded because groups prevent an exact 20% split. Row/profile results are not directly comparable evidence of an algorithm improvement.

## Metrics and interpretation

Confusion matrix rows are actual labels and columns predicted labels, both ordered [-1, +1]: [[TN, FP], [FN, TP]].

- Accuracy = (TP + TN) / all observations.
- Positive recall = TP / (TP + FN); specificity = TN / (TN + FP).
- Balanced accuracy = (positive recall + specificity) / 2.
- Positive precision = TP / (TP + FP); positive F1 is its harmonic mean with recall.
- G-mean = square root of positive recall times specificity.
- Macro F1 averages class-specific F1 values.
- ROC AUC and average precision use the +1 probability score; they evaluate ranking, not calibration. Average precision depends on class prevalence.
- Zero positive predictions produce precision and F1 of zero by explicit convention.
- CV fold standard deviation describes fold variation; it is not a confidence interval.

The always-negative model provides a meaningful imbalance reference. Each metric includes denominator/support information. Prediction indices must match target indices exactly before evaluation. Companion CSVs permit separate confusion-matrix reconstruction.

Permutation importance measures decrease in **balanced accuracy**, five repeats, on the reported split for the preselected model. It is descriptive model dependence, not causality or medical feature meaning. Correlation can redistribute importance; its standard deviation is not inferential uncertainty.

## Environment and commands

Tested with Python 3.12.14, NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, scikit-learn 1.7.2 and imbalanced-learn 0.14.2. `requirements-reviewed-lock.txt` records the tested environment; `requirements.txt` supplies compatible direct-dependency ranges. Use a new virtual environment, never modify a global Python installation.

```text
python -m pip install -r requirements-reviewed-lock.txt
python -m unittest discover -s tests -v
python scripts/reviewed_analysis.py --output-json reports/results/2026-11-10_row.json
python scripts/reviewed_analysis.py --split-strategy profile --output-json reports/results/2026-11-10_profile.json
python scripts/validate_results.py reports/results/2026-11-10_row.json reports/results/2026-11-10_profile.json
```

The old editable-install directive was removed: the original Flit metadata names a package absent from this repository. The maintained workflow runs scripts directly; `pip install -e .` is not supported. Original packaging metadata is preserved for provenance. Optional notebook dependencies are in `requirements-notebook.txt`, but installing them does not repair the historical notebook's local ARFF path or invalid evaluation logic. Do not rerun and publish its outputs without correcting those issues in a separately identified copy.

## Sources

- [Original group repository](https://github.com/bpickett14/MATH3310_FA_2025_P02_G02_PICKETT_CARDONA), read-only comparison at commit d01a5401b2229563069c5a699393e83a2e6aa536.
- [imbalanced-learn: common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html), training-only resampling and pipelines.
- [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), classification metric definitions.
- [scikit-learn: StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), group exclusion and approximate stratification.

Consulted November 10, 2026. Private course instructions were used for context and are not redistributed.
