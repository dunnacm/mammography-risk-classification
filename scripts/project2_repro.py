import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible Project 2 runner")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("notebooks") / "Mammography.csv",
        help="Path to Mammography.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=100)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports") / "results" / "project2_metrics.json",
        help="Where to write metrics JSON",
    )
    return parser.parse_args()


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=["number", "float64", "int64"]).columns.tolist()
    categorical_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "error_rate": float(1.0 - accuracy_score(y_true, y_pred)),
        "minority_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "minority_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "minority_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "g_mean": float(geometric_mean_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"\n=== {name} ===")
    print(f"accuracy:           {metrics['accuracy']:.4f}")
    print(f"balanced_accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"error_rate:         {metrics['error_rate']:.4f}")
    print(f"minority_precision: {metrics['minority_precision']:.4f}")
    print(f"minority_recall:    {metrics['minority_recall']:.4f}")
    print(f"minority_f1:        {metrics['minority_f1']:.4f}")
    print(f"g_mean:             {metrics['g_mean']:.4f}")
    print("confusion_matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("classification_report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    df = pd.read_csv(data_path)

    if "class" not in df.columns:
        raise ValueError("Expected a 'class' column in the dataset.")

    x = df.drop(columns=["class"])
    y = df["class"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.split_seed,
    )

    preprocessor = make_preprocessor(x)
    cv = StratifiedKFold(
        n_splits=args.cv_splits,
        shuffle=True,
        random_state=args.cv_seed,
    )

    baseline_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )

    baseline_cv = cross_val_score(
        baseline_pipe,
        x_train,
        y_train,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    baseline_pipe.fit(x_train, y_train)
    y_pred_baseline = baseline_pipe.predict(x_test)

    search_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=args.split_seed)),
        ]
    )

    param_grid = [
        {
            "classifier": [DecisionTreeClassifier(random_state=args.split_seed)],
            "classifier__max_depth": [2, 3, 4],
        },
        {
            "classifier": [RandomForestClassifier(random_state=args.split_seed, n_jobs=-1)],
            "classifier__n_estimators": [300, 500, 1000],
        },
        {
            "classifier": [LogisticRegression(random_state=args.split_seed, max_iter=1000)],
            "classifier__C": [0.01, 1, 100],
        },
    ]

    grid = GridSearchCV(
        estimator=search_pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    best_pipe = grid.best_estimator_
    y_pred_tuned = best_pipe.predict(x_test)

    weighted_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=args.split_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    weighted_cv = cross_val_score(
        weighted_pipe,
        x_train,
        y_train,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    weighted_pipe.fit(x_train, y_train)
    y_pred_weighted = weighted_pipe.predict(x_test)

    smote_pipe = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("oversample", SMOTE(random_state=args.split_seed)),
            ("undersample", RandomUnderSampler(random_state=args.split_seed)),
            (
                "classifier",
                RandomForestClassifier(random_state=args.split_seed, n_jobs=-1),
            ),
        ]
    )
    smote_cv = cross_val_score(
        smote_pipe,
        x_train,
        y_train,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    smote_pipe.fit(x_train, y_train)
    y_pred_smote = smote_pipe.predict(x_test)

    results = {
        "data_path": str(data_path),
        "split_seed": args.split_seed,
        "cv_seed": args.cv_seed,
        "cv_splits": args.cv_splits,
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
        "cv": {
            "baseline_rf_balanced_accuracy": {
                "folds": baseline_cv.tolist(),
                "mean": float(np.mean(baseline_cv)),
                "std": float(np.std(baseline_cv)),
            },
            "weighted_rf_balanced_accuracy": {
                "folds": weighted_cv.tolist(),
                "mean": float(np.mean(weighted_cv)),
                "std": float(np.std(weighted_cv)),
            },
            "smote_pipeline_balanced_accuracy": {
                "folds": smote_cv.tolist(),
                "mean": float(np.mean(smote_cv)),
                "std": float(np.std(smote_cv)),
            },
            "grid_best_score": float(grid.best_score_),
            "grid_best_params": {k: str(v) for k, v in grid.best_params_.items()},
        },
        "test_metrics": {
            "baseline_rf": evaluate_model("Baseline RandomForest", y_test, y_pred_baseline),
            "tuned_model": evaluate_model("Tuned GridSearch Model", y_test, y_pred_tuned),
            "weighted_rf": evaluate_model("Class-weighted RandomForest", y_test, y_pred_weighted),
            "smote_pipeline": evaluate_model("SMOTE + UnderSampling Pipeline", y_test, y_pred_smote),
        },
    }

    # Optional interpretability summary for tree-based tuned model.
    tuned_model = best_pipe.named_steps["classifier"]
    if hasattr(tuned_model, "feature_importances_"):
        feature_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
        importances = tuned_model.feature_importances_
        top_mdi = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )[:10]

        perm = permutation_importance(
            best_pipe,
            x_test,
            y_test,
            scoring="balanced_accuracy",
            n_repeats=10,
            random_state=args.split_seed,
            n_jobs=-1,
        )
        top_perm_idx = np.argsort(perm.importances_mean)[::-1][:10]
        top_perm = [
            {
                "feature": x.columns[i],
                "mean": float(perm.importances_mean[i]),
                "std": float(perm.importances_std[i]),
            }
            for i in top_perm_idx
        ]

        results["interpretability"] = {
            "top_mdi_features": [
                {"feature": f, "importance": float(v)} for f, v in top_mdi
            ],
            "top_permutation_features": top_perm,
        }

    output_path = args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"Data: {data_path}")
    print(f"Baseline CV balanced accuracy mean: {np.mean(baseline_cv):.4f}")
    print(f"Weighted CV balanced accuracy mean: {np.mean(weighted_cv):.4f}")
    print(f"SMOTE CV balanced accuracy mean: {np.mean(smote_cv):.4f}")
    print(f"Grid best score: {grid.best_score_:.4f}")
    print(f"Grid best params: {grid.best_params_}")
    print(f"Saved metrics JSON to: {output_path}")


if __name__ == "__main__":
    main()
