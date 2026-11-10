"""Independent hand cases for evaluation and split invariants."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("reviewed", Path(__file__).parents[1] / "scripts/reviewed_analysis.py")
reviewed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewed)

class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_confusion_and_rates(self):
        actual = pd.Series([-1, -1, -1, 1, 1], index=[10, 20, 30, 40, 50])
        prediction = pd.Series([-1, -1, 1, 1, -1], index=actual.index)
        metrics = reviewed.evaluate(actual, prediction)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [1, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 3/5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 7/12)
        self.assertAlmostEqual(metrics["positive_f1"], .5)
        self.assertAlmostEqual(metrics["g_mean"], math.sqrt(1/3))

    def test_majority_baseline_exposes_accuracy_trap(self):
        actual = pd.Series([-1] * 98 + [1] * 2)
        metrics = reviewed.evaluate(actual, pd.Series([-1] * 100))
        self.assertEqual(metrics["accuracy"], .98)
        self.assertEqual(metrics["balanced_accuracy"], .5)
        self.assertEqual(metrics["positive_recall"], 0)
        self.assertEqual(metrics["positive_f1"], 0)

    def test_wrong_row_ids_rejected_even_when_lengths_match(self):
        actual = pd.Series([-1, 1], index=[100, 200])
        with self.assertRaisesRegex(ValueError, "row IDs"):
            reviewed.evaluate(actual, pd.Series([-1, 1], index=[200, 100]))

    def test_invalid_label_not_silently_converted(self):
        frame = pd.DataFrame([[1] * 6 + [-1], [2] * 6 + [.5]], columns=reviewed.FEATURES + ["class"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "labels"):
                reviewed.load_data(path)

    def test_profile_groups_never_cross_folds(self):
        groups = np.repeat(np.arange(20), 2)
        x = pd.DataFrame({"value": groups})
        y = pd.Series(np.repeat([-1, 1] * 10, 2))
        splits = reviewed.group_splits(x, y, groups, 2, 42)
        held_out = []
        for train, valid in splits:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            held_out.extend(valid)
        self.assertEqual(sorted(held_out), list(range(len(x))))

if __name__ == "__main__":
    unittest.main()
