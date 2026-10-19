"""Independent-reference, pairing, and condition-integrity checks."""
import csv
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from src.extended_core import build_results
from src.statistics_exact import holm


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("table1_components.csv", "table2_layouts.csv", "table3_queries.csv", "e14_episodes.csv")


class ExtendedCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual = build_results(ROOT / "data/processed")
        cls.reference = json.loads((ROOT / "reference/extended_core.json").read_text(encoding="utf-8"))

    def copied_inputs(self):
        temp_base = ROOT / "outputs/_test_tmp"
        temp_base.mkdir(parents=True, exist_ok=True)
        # Default ACL inheritance also works under the Windows restricted token;
        # tempfile's private-directory ACL is not accessible in that sandbox.
        directory = temp_base / ("core_" + uuid4().hex)
        directory.mkdir()
        def cleanup():
            if directory.resolve().parent != temp_base.resolve():
                raise ValueError("Refusing cleanup outside the test directory")
            shutil.rmtree(directory)
        self.addCleanup(cleanup)
        for name in INPUTS:
            shutil.copyfile(ROOT / "data/processed" / name, directory / name)
        return directory

    @staticmethod
    def edit_rows(directory, name, edit):
        path = directory / name
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        edit(rows)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_all_fields_match_independent_archived_statistics_and_tex(self):
        expected = self.reference["results"]
        self.assertEqual(set(self.actual), set(expected))
        for key, expected_rows in expected.items():
            tolerance = self.reference["tolerances"].get(key, self.reference["tolerances"]["default"])
            self.assertEqual(len(self.actual[key]), len(expected_rows), key)
            for index, (actual, reference) in enumerate(zip(self.actual[key], expected_rows)):
                self.assertEqual(set(actual), set(reference), (key, index))
                for field, value in reference.items():
                    with self.subTest(output=key, row=index, field=field):
                        if isinstance(value, float):
                            self.assertAlmostEqual(actual[field], value, delta=tolerance)
                        else:
                            self.assertEqual(actual[field], value)

    def test_pairing_changes_statistics_even_when_group_counts_do_not_change(self):
        directory = self.copied_inputs()
        def move_success(rows):
            match = {r["task_id"]: r for r in rows if r["platform"] == "LangMem"
                     and r["scenario"] == "s1" and r["condition"] == "kmatch"}
            mismatch = [r for r in rows if r["platform"] == "LangMem"
                        and r["scenario"] == "s1" and r["condition"] == "kmismatch"]
            source = next(r for r in mismatch if r["executed"] == "1" and match[r["task_id"]]["executed"] == "1")
            target = next(r for r in mismatch if r["executed"] == "0" and match[r["task_id"]]["executed"] == "0")
            source["executed"] = source["export_calls"] = "0"
            target["executed"] = target["export_calls"] = "1"
        self.edit_rows(directory, "e14_episodes.csv", move_success)
        changed = build_results(directory)
        self.assertEqual(changed["e14_cells"], self.actual["e14_cells"])
        original, modified = self.actual["e14_pairs"][2], changed["e14_pairs"][2]
        self.assertEqual(original["rd"], modified["rd"])
        self.assertNotEqual((original["b"], original["c"]), (modified["b"], modified["c"]))
        self.assertNotEqual(original["p_exact"], modified["p_exact"])

    def test_e14_rejects_wrong_return_code(self):
        directory = self.copied_inputs()
        self.edit_rows(directory, "e14_episodes.csv", lambda rows: rows[0].update(condition_rc="9"))
        with self.assertRaisesRegex(ValueError, "return code"):
            build_results(directory)

    def test_e14_rejects_missing_condition_content(self):
        directory = self.copied_inputs()
        self.edit_rows(directory, "e14_episodes.csv", lambda rows: rows[0].update(all_requests_full="0"))
        with self.assertRaisesRegex(ValueError, "full exposure"):
            build_results(directory)

    def test_e14_rejects_missing_paired_task(self):
        directory = self.copied_inputs()
        self.edit_rows(directory, "e14_episodes.csv", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "task set"):
            build_results(directory)

    def test_joint_family_is_retained_when_selecting_supplement_s2(self):
        pairs = self.actual["e13_pairs"]
        self.assertEqual(len(pairs), 16)
        self.assertEqual({r["path"] for r in pairs}, {"natural", "full"})
        self.assertEqual([r["p_holm"] for r in pairs], holm([r["p_exact"] for r in pairs]))
        self.assertEqual([r["p_holm"] for r in self.actual["supplement_s2"]],
                         [r["p_holm"] for r in pairs if r["path"] == "full"])
        primary = self.actual["e19x_pairs"][:4]
        exploratory = self.actual["e19x_pairs"][4:]
        self.assertEqual([r["p_holm"] for r in primary], holm([r["p_exact"] for r in primary]))
        self.assertEqual([r["p_holm"] for r in exploratory], holm([r["p_exact"] for r in exploratory]))
        self.assertNotEqual(primary[0]["family"], exploratory[0]["family"])


if __name__ == "__main__":
    unittest.main()
