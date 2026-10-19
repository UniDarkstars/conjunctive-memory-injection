"""Independent archived references, paired identities and frozen MC protocol."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.extended_comparisons import build_results, did_permutation, _queries, _methods, _detector


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ExtendedComparisonsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mita_extended_")
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)/"data"
        shutil.copytree(ROOT/"data/processed", self.data)

    def test_all_results_match_independent_archived_outputs(self):
        actual = build_results(self.data)
        reference = json.loads((ROOT/"reference/extended_comparisons.json").read_text(encoding="utf-8"))
        self.assertEqual(set(actual), set(reference["results"]))
        for name, expected in reference["results"].items():
            self.assertEqual(len(actual[name]), len(expected), name)
            for index, (row, saved) in enumerate(zip(actual[name], expected)):
                for field, value in saved.items():
                    with self.subTest(result=name, row=index, field=field):
                        if isinstance(row[field], (int, float)):
                            self.assertIsInstance(value, (int, float), "Strict shared verifier requires typed numeric references")
                            self.assertAlmostEqual(row[field], value, delta=reference["tolerances"].get(name, 1e-12))
                        else:
                            self.assertEqual(row[field], value)

    def test_missing_query_task_is_not_counted_as_failure(self):
        path = self.data/"e19q_tasks.csv"
        rows = read_rows(path)
        write_rows(path, rows[1:])
        with self.assertRaisesRegex(ValueError, "Expected 30 tasks"):
            _queries(self.data)

    def test_method_model_mismatch_is_rejected_before_pairing(self):
        path = self.data/"e17_episodes.csv"
        rows = read_rows(path)
        rows[0]["model_id"] = "different-model"
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "model/configuration mismatch"):
            _methods(self.data)

    def test_missing_detector_window_does_not_become_negative(self):
        path = self.data/"e15_coverage_windows.csv"
        rows = read_rows(path)
        write_rows(path, rows[1:])
        with self.assertRaisesRegex(ValueError, "window/source inventory disagreement"):
            _detector(self.data)

    def test_complete_coverage_misses_recomputed_from_joint_rows(self):
        path = self.data/"e15_coverage_episodes.csv"
        rows = read_rows(path)
        selected = next(r for r in rows if r["lang"] == "en" and r["config"] == "mem_prefix" and r["case"] == "both" and r["detected"] == "0")
        selected["detected"] = "1"
        write_rows(path, rows)
        labels_path = self.data/"e15_coverage_labels.csv"
        labels = read_rows(labels_path)
        identity_fields = ("platform", "scenario", "case", "lang", "config", "task_id")
        next(r for r in labels if all(r[k] == selected[k] for k in identity_fields))["detected"] = "1"
        write_rows(labels_path, labels)
        joint = _detector(self.data)["e15_coverage_joint"][0]
        self.assertEqual((joint["complete"], joint["positive"], joint["complete_missed"]), (120, 32, 88))

    def test_chunked_permutation_preserves_original_random_stream(self):
        import numpy as np
        rows = [{"y_fm_base": 1, "y_fm_high": 0, "y_ext_base": 0, "y_ext_high": i % 2} for i in range(7)]
        n_mc, seed = 257, 20261008
        arrays = [np.array([r[k] for r in rows], dtype=np.int8) for k in ("y_fm_base", "y_fm_high", "y_ext_base", "y_ext_high")]
        fb, fh, eb, eh = arrays
        observed = abs((fh-fb).mean()-(eh-eb).mean())
        rng, n_ge = np.random.default_rng(seed), 0
        for _ in range(n_mc):
            swap = rng.random(len(rows)) < .5
            fbase, fhigh = np.where(swap, eb, fb), np.where(swap, eh, fh)
            ebase, ehigh = np.where(swap, fb, eb), np.where(swap, fh, eh)
            n_ge += abs((fhigh-fbase).mean()-(ehigh-ebase).mean()) >= observed-1e-12
        self.assertEqual(did_permutation(rows, n_mc, seed)["p_perm"], (int(n_ge)+1)/(n_mc+1))


if __name__ == "__main__":
    unittest.main()
