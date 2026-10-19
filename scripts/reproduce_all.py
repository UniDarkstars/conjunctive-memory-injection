"""Recompute the MiTA tables, supplementary statistics and reported diagnostics.

Only released observations are read. No model, detector or network is called.
The six-table-only entry remains scripts/reproduce_tables.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_tables import calculate_tables, markdown_table, verify_tables


def check_value(actual, expected, location, tolerance):
    """Expected dictionaries may select independently archived fields."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{location}: expected a mapping")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{location}: missing field {key}")
            check_value(actual[key], value, f"{location}.{key}", tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: result row count differs")
        for i, (value, reference) in enumerate(zip(actual, expected)):
            check_value(value, reference, f"{location}[{i}]", tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
                actual, expected, abs_tol=tolerance, rel_tol=0):
            raise ValueError(f"{location}: computed {actual!r}, reference {expected!r}")
    elif actual != expected:
        raise ValueError(f"{location}: computed {actual!r}, reference {expected!r}")


def verify_extended(results, reference, label):
    expected = reference["results"]
    if set(results) != set(expected):
        raise ValueError(f"{label}: result groups differ: {set(results) ^ set(expected)}")
    tolerances = reference.get("tolerances", {})
    receipts = []
    for key, rows in expected.items():
        tolerance = tolerances.get(key, tolerances.get("default", 1e-12))
        check_value(results[key], rows, key, tolerance)
        receipts.append({"result": key, "rows_checked": len(rows),
                         "absolute_tolerance": tolerance, "match": True})
    return receipts


def output_path(name):
    directory = ROOT / "outputs"
    if directory.is_symlink() or directory.resolve().parent != ROOT:
        raise ValueError("outputs/ must be a real directory in this release")
    directory.mkdir(exist_ok=True)
    target = directory / name
    if target.is_symlink() or target.resolve().parent != directory.resolve():
        raise ValueError(f"Refusing redirected output: {name}")
    return target


def write_rows(name, rows):
    if not rows:
        raise ValueError(f"No observations calculated for {name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output_path(name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check without writing outputs")
    args = parser.parse_args()
    from src.extended_core import build_results as core_results
    from src.extended_comparisons import build_results as comparison_results
    import numpy as np

    data = ROOT / "data" / "processed"
    # Reference values are opened only after the observation-based calculations.
    tables = calculate_tables(data)
    groups = {"extended_core": core_results(data),
              "extended_comparisons": comparison_results(data)}
    table_reference = json.loads((ROOT / "reference/manuscript_tables.json").read_text(encoding="utf-8"))
    receipts = {"main_tables": verify_tables(tables, table_reference)}
    for label, results in groups.items():
        reference = json.loads((ROOT / "reference" / f"{label}.json").read_text(encoding="utf-8"))
        receipts[label] = verify_extended(results, reference, label)
    keys = [key for results in groups.values() for key in results]
    if len(keys) != len(set(keys)):
        raise ValueError("Extended modules produced duplicate output names")
    verification = {
        "status": "PASS", "scope": "Six main tables, S1/S2 and additional reported experiment statistics",
        "online_experiments_run": False, "python_version": sys.version.split()[0],
        "numpy_version": np.__version__, "checks": receipts,
        "input_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(data.glob("*.csv"))},
        "reference_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT / "reference").glob("*.json"))},
    }
    if not args.verify_only:
        for table, expected in zip(tables, table_reference["tables"]):
            with output_path(f"table_{table['number']:02d}_{expected['slug']}.csv").open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(table["columns"])
                writer.writerows(table["rows"])
        write_rows("table_04_statistics.csv", next(t for t in tables if t["number"] == 4)["statistics"])
        for results in groups.values():
            for name, rows in results.items():
                write_rows(name + ".csv", rows)
        output_path("tables.md").write_text(
            "# Recomputed MiTA main tables\n\n" + "\n".join(markdown_table(t) for t in tables), encoding="utf-8")
        for name, value in (("tables.json", tables), ("all_results.json", groups),
                            ("all_verification.json", verification)):
            output_path(name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    for group, checks in receipts.items():
        print(f"PASS {group}: {len(checks)} result groups; {sum(c['rows_checked'] for c in checks)} rows")
    print("PASS: all released numerical results match their independent references; no online experiments were run.")
    if not args.verify_only:
        print(f"Outputs: {ROOT / 'outputs'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
