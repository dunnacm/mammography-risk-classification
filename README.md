# Mammography: learning to evaluate imbalanced classification

This educational group project asks how different classifiers detect the recorded minority class when it represents only 260 of 11,183 observations (2.32%). Overall accuracy alone is insufficient: predicting the majority class for every row achieves about 97.68% accuracy while detecting no minority examples.

**The original notebook is historical evidence, not a validated performance report.** Its SMOTE evaluation uses predictions and target labels from different test splits; the reported SMOTE balanced accuracy 0.9073 and F1 0.7049 must not be cited as valid model performance. The train/test-seed variant also reuses the wrong target array. See the [review and corrections](docs/review_2026-11-10.md).

## Current analysis

The maintained entry point is [scripts/reviewed_analysis.py](scripts/reviewed_analysis.py). It checks the input schema, keeps preprocessing and SMOTE inside training folds, selects a model by training cross-validation, checks prediction row identities, and exports metrics with a source hash, split details and environment versions.

It compares an always-negative reference, random forests, class weighting, SMOTE and a small grid of tree, forest and logistic models. Reported test results are an **illustrative rerun on previously examined data**, not new untouched evidence, clinical validation or proof of screening effectiveness. The CSV does not establish patient identity, feature units or clinical feature meanings.

## Run

Use Python 3.12 in an isolated environment, from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python scripts/reviewed_analysis.py
```

On macOS/Linux use `.venv/bin/python`. Results go to `reports/results/reviewed_metrics.json` with a companion predictions CSV. See [methods and reproduction](docs/methods.md) for exact tested versions, metric definitions and the optional exact-profile sensitivity analysis.

## Preserved material and attribution

[notebooks/project2.ipynb](notebooks/project2.ipynb), the bundled [CSV](notebooks/Mammography.csv), and [scripts/project2_repro.py](scripts/project2_repro.py) retain their original bytes. The earlier runner is a historical, generally consistent single-split implementation, separate from the notebook's evaluation defects; it is not the maintained entry point.

The original group work is by **Brenna Pickett and Dunn Cardona**, preserved from [the group repository](https://github.com/bpickett14/MATH3310_FA_2025_P02_G02_PICKETT_CARDONA). This review does not establish individual or unaided authorship. The November 10, 2026 audit, reviewed runner, tests and documentation were prepared with AI assistance. Original notebook content and attribution remain unchanged. No new license is asserted for third-party data or the original project.
