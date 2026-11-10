"""Check saved metrics against source rows using only the Python standard library."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def check(result_path, data_path):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with data_path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    with result_path.with_suffix(".predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == result["data_sha256"]
    assert len(source) == result["rows"]
    ids = [int(row["row_id"]) for row in rows]
    assert len(ids) == len(set(ids)) == result["partition"]["test_rows"]
    assert all(0 <= i < len(source) for i in ids)
    assert result["partition"]["train_rows"] + len(ids) == len(source)
    for row in rows:
        assert int(row["actual"]) == int(source[int(row["row_id"])]["class"])
    test_ids = set(ids)
    profile = lambda row: tuple(float(row[f"attr{i}"]) for i in range(1, 7))
    train_profiles = {profile(row) for i, row in enumerate(source) if i not in test_ids}
    test_profiles = {profile(source[i]) for i in ids}
    assert len(train_profiles & test_profiles) == result["partition"]["overlapping_profile_groups"]
    if result["partition"]["method"] == "profile":
        assert not train_profiles & test_profiles
    for model, expected in result["test_metrics"].items():
        tn = sum(int(row["actual"]) == -1 and int(row[model]) == -1 for row in rows)
        fp = sum(int(row["actual"]) == -1 and int(row[model]) == 1 for row in rows)
        fn = sum(int(row["actual"]) == 1 and int(row[model]) == -1 for row in rows)
        tp = sum(int(row["actual"]) == 1 and int(row[model]) == 1 for row in rows)
        assert [[tn, fp], [fn, tp]] == expected["confusion_matrix"], model
        assert expected["confusion_matrix_labels"] == [-1, 1]
        assert expected["n"] == len(rows)
        assert expected["positive_support"] == tp + fn
        recall, specificity = tp / (tp + fn), tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        negative_f1 = 2 * tn / (2 * tn + fn + fp)
        derived = {"accuracy": (tp + tn) / len(rows), "balanced_accuracy": (recall + specificity) / 2,
                   "positive_precision": precision, "positive_recall": recall, "specificity": specificity,
                   "positive_f1": f1, "g_mean": math.sqrt(recall * specificity), "macro_f1": (f1 + negative_f1) / 2}
        for name, value in derived.items():
            assert math.isclose(value, expected[name], abs_tol=1e-12), (model, name)
    print(json.dumps({"file": result_path.as_posix(), "rows_verified": len(rows),
                      "models_verified": len(result["test_metrics"]), "status": "passed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "notebooks/Mammography.csv")
    args = parser.parse_args()
    for result in args.results:
        check(result, args.data)
