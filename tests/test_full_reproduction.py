"""Integration checks for all reported results, without online experiment calls."""
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
spec = importlib.util.spec_from_file_location("full_reproduction", ROOT / "scripts/reproduce_all.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


class FullReproductionTests(unittest.TestCase):
    def test_reference_cannot_hide_missing_extra_or_changed_results(self):
        reference = {"results": {"example": [{"task": "a", "count": 2, "p": 0.125}]}}
        correct = {"example": [{"task": "a", "count": 2, "p": 0.125, "detail": 1}]}
        self.assertEqual(full.verify_extended(correct, reference, "example")[0]["rows_checked"], 1)
        for changed in ({}, {"example": []}, {"example": correct["example"] * 2},
                        {"example": [{"task": "a", "count": 3, "p": 0.125}]},
                        {"example": [{"task": "a", "count": 2, "p": float("nan")}]},
                        {**correct, "unexpected": []}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    full.verify_extended(changed, reference, "example")

    def test_rounded_archived_values_use_explicit_tolerance(self):
        reference = {"tolerances": {"rounded": 5.1e-7},
                     "results": {"rounded": [{"value": 0.333333}]}}
        full.verify_extended({"rounded": [{"value": 1 / 3}]}, reference, "rounded")
        with self.assertRaises(ValueError):
            full.verify_extended({"rounded": [{"value": 0.333334}]}, reference, "rounded")

    def test_full_portable_copy_without_original_workspace_or_network(self):
        with tempfile.TemporaryDirectory(prefix="mita_full_test_") as temporary:
            isolated = Path(temporary) / "release"
            for directory in ("src", "data", "reference", "scripts"):
                shutil.copytree(ROOT / directory, isolated / directory,
                                ignore=shutil.ignore_patterns("__pycache__"))
            harness = r'''
import os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
# Import installed numeric dependencies before restricting external file reads.
# Thereafter only this copy and the installed interpreter may be accessed.
import numpy
allowed = (root, pathlib.Path(sys.base_prefix).resolve())
def audit(event, args):
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
        raise RuntimeError('Network/process access denied: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        if not any(path.is_relative_to(base) for base in allowed):
            raise RuntimeError('External file access denied: ' + str(path))
        flags = args[2]
        writing = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and not path.is_relative_to(root / 'outputs'):
            raise RuntimeError('Write outside copy outputs denied: ' + str(path))
sys.dont_write_bytecode = True
sys.addaudithook(audit)
sys.argv = [str(root / 'scripts/reproduce_all.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
            result = subprocess.run([sys.executable, "-I", "-B", "-c", harness, str(isolated)],
                                    cwd=temporary, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS: all released numerical results", result.stdout)
            receipt = json.loads((isolated / "outputs/all_verification.json").read_text(encoding="utf-8"))
            self.assertFalse(receipt["online_experiments_run"])
            self.assertTrue((isolated / "outputs/supplement_s1.csv").is_file())
            self.assertTrue((isolated / "outputs/supplement_s2.csv").is_file())


if __name__ == "__main__":
    unittest.main()
