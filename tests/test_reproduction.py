"""Checks against the manuscript, analytic statistics and deliberately damaged inputs."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("reproduction", ROOT / "scripts" / "reproduce_tables.py")
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)
from src.statistics_exact import clopper_pearson, holm, mcnemar_exact_p, paired_risk_difference


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mita_offline_test_")
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name) / "processed"
        shutil.copytree(ROOT / "data" / "processed", self.data)
        self.reference = json.loads((ROOT / "reference" / "manuscript_tables.json").read_text(encoding="utf-8"))

    def test_all_tables_match_independent_manuscript_reference(self):
        tables = reproduction.calculate_tables(self.data)
        receipts = reproduction.verify_tables(tables, self.reference)
        self.assertEqual(len(receipts), 6)
        self.assertEqual(sum(receipt["rows_checked"] for receipt in receipts), 50)

    def test_changed_observation_changes_result_and_fails_reference_check(self):
        path = self.data / "table1_components.csv"
        fields, rows = read_csv(path)
        row = next(row for row in rows if row["condition"] == "clean" and row["executed"] == "0")
        row["executed"], row["export_calls"] = "1", "1"
        write_csv(path, fields, rows)
        tables = reproduction.calculate_tables(self.data)
        with self.assertRaisesRegex(ValueError, "Table 1, row"):
            reproduction.verify_tables(tables, self.reference)

    def test_duplicate_and_missing_query_task_rejected(self):
        path = self.data / "table3_queries.csv"
        fields, rows = read_csv(path)
        row = rows[0]
        other = next(r for r in rows if r["scenario"] == row["scenario"]
                     and r["layout"] == row["layout"] and r["query_mode"] == row["query_mode"]
                     and r["task_id"] != row["task_id"])
        other["task_id"] = row["task_id"]
        write_csv(path, fields, rows)
        with self.assertRaisesRegex(ValueError, "Missing/duplicate/unpaired"):
            reproduction.calculate_tables(self.data)

    def test_restoration_pair_requires_same_source_snapshot(self):
        path = self.data / "table4_episodes.csv"
        fields, rows = read_csv(path)
        row = next(row for row in rows if row["arm"] == "restore")
        row["snapshot_sha256"] = "0" * 64
        write_csv(path, fields, rows)
        with self.assertRaisesRegex(ValueError, "frozen inputs differ"):
            reproduction.calculate_tables(self.data)

    def test_method_batches_cannot_be_pooled(self):
        path = self.data / "table5_episodes.csv"
        fields, rows = read_csv(path)
        row = next(row for row in rows if row["batch"] == "concurrent" and row["method"] == "fully_merged")
        row["batch"] = "original"
        write_csv(path, fields, rows)
        with self.assertRaisesRegex(ValueError, "duplicate Table V"):
            reproduction.calculate_tables(self.data)

    def test_missing_detector_record_is_not_a_negative_label(self):
        path = self.data / "table6_windows.csv"
        fields, rows = read_csv(path)
        write_csv(path, fields, rows[1:])
        with self.assertRaisesRegex(ValueError, "window records disagree"):
            reproduction.calculate_tables(self.data)

    def test_changed_detector_label_changes_table(self):
        path = self.data / "table6_episodes.csv"
        fields, rows = read_csv(path)
        source = next(r for r in rows if r["saved_per_record_positive"] == "0")
        source["saved_per_record_positive"] = "1"
        write_csv(path, fields, rows)
        windows_path = self.data / "table6_windows.csv"
        window_fields, windows = read_csv(windows_path)
        window = next(w for w in windows if w["language"] == source["language"] and w["source_id"] == source["source_id"])
        window["positive"] = "1"
        write_csv(windows_path, window_fields, windows)
        tables = reproduction.calculate_tables(self.data)
        with self.assertRaisesRegex(ValueError, "Table 6, row"):
            reproduction.verify_tables(tables, self.reference)

    def test_portable_copy_has_no_external_file_or_network_access(self):
        isolated = Path(self.temporary.name) / "isolated_release"
        for directory in ("src", "data", "reference"):
            shutil.copytree(ROOT / directory, isolated / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        (isolated / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts" / "reproduce_tables.py", isolated / "scripts" / "reproduce_tables.py")
        # Audit all actual file opens, including Python code loads; permit only
        # this isolated package and the interpreter's standard library.
        harness = r'''
import os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
allowed = (root, pathlib.Path(sys.base_prefix).resolve())
def audit(event, args):
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
        raise RuntimeError('Network/process access denied in offline check: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        if not any(path.is_relative_to(base) for base in allowed):
            raise RuntimeError('External file access denied: ' + str(path))
        flags = args[2]
        writing = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and not path.is_relative_to(root / 'outputs'):
            raise RuntimeError('Write outside outputs denied: ' + str(path))
sys.dont_write_bytecode = True
sys.addaudithook(audit)
sys.argv = [str(root / 'scripts/reproduce_tables.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", harness, str(isolated)],
                                cwd=self.temporary.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS: 6/6 tables", result.stdout)
        self.assertTrue((isolated / "outputs" / "tables.md").is_file())


class ExactStatisticsTests(unittest.TestCase):
    def test_binomial_boundary_intervals_against_closed_form(self):
        alpha, n = 0.05, 30
        lo, hi = clopper_pearson(0, n, alpha)
        self.assertEqual(lo, 0)
        self.assertAlmostEqual(hi, 1 - (alpha / 2)**(1 / n), places=13)
        lo, hi = clopper_pearson(n, n, alpha)
        self.assertAlmostEqual(lo, (alpha / 2)**(1 / n), places=13)
        self.assertEqual(hi, 1)

    def test_exact_mcnemar_small_cases(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1)
        self.assertEqual(mcnemar_exact_p(3, 0), 0.25)
        self.assertEqual(mcnemar_exact_p(5, 1), 0.21875)

    def test_holm_preserves_input_order(self):
        for actual, expected in zip(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(actual, expected)

    def test_reverse_contrast_reverses_interval(self):
        rd, lo, hi = paired_risk_difference(17, 2, 30)
        reverse, rlo, rhi = paired_risk_difference(2, 17, 30)
        self.assertAlmostEqual(reverse, -rd)
        self.assertAlmostEqual(rlo, -hi)
        self.assertAlmostEqual(rhi, -lo)


if __name__ == "__main__":
    unittest.main()
