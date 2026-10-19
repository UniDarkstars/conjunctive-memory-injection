"""Recompute all six MiTA tables offline and check them against the manuscript.

Run from any directory. Inputs are under this release's data/processed/;
outputs are written only under this release's outputs/. No model calls occur.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def calculate_tables(data_dir):
    from src.tables_1_3 import build_tables as first_three
    from src.table_4 import build_tables as restoration
    from src.tables_5_6 import build_tables as last_two

    tables = sorted(first_three(data_dir) + restoration(data_dir) + last_two(data_dir),
                    key=lambda table: table["number"])
    if [table["number"] for table in tables] != list(range(1, 7)):
        raise ValueError("Expected exactly one result for each table I-VI")
    return tables


def verify_tables(tables, reference):
    """The reference is consulted only after all outcomes have been calculated."""
    receipts = []
    for actual, expected in zip(tables, reference["tables"], strict=True):
        number = expected["number"]
        if actual["number"] != number or actual["columns"] != expected["columns"]:
            raise ValueError(f"Table {number}: column or table identity mismatch")
        if len(actual["rows"]) != len(expected["rows"]):
            raise ValueError(f"Table {number}: row count differs from the manuscript")
        for row_number, (row, expected_row) in enumerate(zip(actual["rows"], expected["rows"], strict=True), 1):
            if row != expected_row:
                raise ValueError(f"Table {number}, row {row_number} differs from manuscript:\n"
                                 f"computed: {row}\nreference: {expected_row}")
        receipts.append({"table": number, "label": expected["label"],
                         "rows_checked": len(actual["rows"]), "match": True})
    return receipts


def markdown_table(table):
    def escape(cell):
        return str(cell).replace("|", "\\|").replace("\n", " ")
    lines = [f"## Table {table['number']}: {table['caption']}", "",
             "| " + " | ".join(map(escape, table["columns"])) + " |",
             "| " + " | ".join(["---"] * len(table["columns"])) + " |"]
    lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in table["rows"])
    lines.extend(["", table["notes"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Calculate and verify without writing outputs")
    args = parser.parse_args()

    data_dir = ROOT / "data" / "processed"
    tables = calculate_tables(data_dir)
    reference_path = ROOT / "reference" / "manuscript_tables.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    receipts = verify_tables(tables, reference)
    verification = {
        "status": "PASS", "scope": "Six main-text MiTA tables, recomputed from copied task-level outcomes",
        "network_required": False, "third_party_packages_required": False,
        "python_version": sys.version.split()[0], "tables": receipts,
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"):
                         hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(data_dir.glob("*.csv"))},
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    if not args.verify_only:
        output_dir = ROOT / "outputs"
        # Protect copied inputs and the official workspace from a redirected output folder.
        if output_dir.is_symlink() or output_dir.resolve().parent != ROOT:
            raise ValueError("outputs/ must be a real directory directly inside this release")
        output_dir.mkdir(exist_ok=True)
        for table, expected in zip(tables, reference["tables"], strict=True):
            target = output_dir / f"table_{table['number']:02d}_{expected['slug']}.csv"
            if target.is_symlink():
                raise ValueError(f"Refusing redirected output: {target.name}")
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(table["columns"])
                writer.writerows(table["rows"])
        outputs = {
            "tables.md": "# Recomputed MiTA tables\n\n" + "\n".join(markdown_table(t) for t in tables),
            "tables.json": json.dumps(tables, ensure_ascii=False, indent=2) + "\n",
            "verification.json": json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        }
        for name, content in outputs.items():
            target = output_dir / name
            if target.is_symlink():
                raise ValueError(f"Refusing redirected output: {name}")
            target.write_text(content, encoding="utf-8")
        statistics = next(table for table in tables if table["number"] == 4).get("statistics", [])
        if statistics:
            target = output_dir / "table_04_statistics.csv"
            if target.is_symlink():
                raise ValueError("Refusing redirected statistics output")
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
                writer.writeheader()
                writer.writerows(statistics)
        print(f"Outputs: {output_dir}")
    for receipt in receipts:
        print(f"PASS Table {receipt['table']}: {receipt['rows_checked']} rows match manuscript")
    print("PASS: 6/6 tables; all calculations completed without model or network access.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
