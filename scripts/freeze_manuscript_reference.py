"""Packaging utility: freeze table cells from a supplied manuscript, not experiment data.

This is not called by reproduce_tables.py. It only writes the release's reference
file and never edits the manuscript. Re-freezing is a deliberate packaging step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    (1, "components", "tab:components", ["Memory content", "ADK Supplier", "ADK Release", "LangMem Supplier", "LangMem Release"], []),
    (2, "layout", "tab:layout", ["Platform", "Scenario", "Layout", "Complete delivery", "Natural execution", "Supplied execution"], [0, 1]),
    (3, "query", "tab:query", ["Scenario", "Layout", "Whole task (joint / execution)", "Per object (joint / execution)"], [0]),
    (4, "restoration", "tab:restore", ["Model", "Platform", "Scenario", "Replay", "Restore", "Benign", "Restore - benign [95% CI] (pp)", "Holm p"], [0]),
    (5, "methods", "tab:methods", ["Platform", "Method", "Supplier delivery", "Supplier execution", "Release delivery", "Release execution"], [0]),
    (6, "detection", "tab:detection", ["Text", "Platform", "Scenario", "Attack memory prefix", "Attack per record", "Clean memory prefix", "Clean per record"], [0]),
]


def parse_tables(sections):
    tables = []
    for number, slug, label, columns, fill_columns in SPECS:
        path = sections / ("discussion.tex" if number == 6 else "results.tex")
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.S)
        matches = [block for block in blocks if "\\label{" + label + "}" in block]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {label} in {path}")
        block = matches[0]
        data = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        rows = []
        for line in data.splitlines():
            line = line.strip()
            if not line or line.startswith((r"\midrule", r"\cmidrule")):
                continue
            if not line.endswith(r"\\"):
                raise ValueError(f"Unrecognized table row: {line}")
            line = line[:-2]
            line = re.sub(r"\\multirow\{[^{}]+\}\{[^{}]+\}\{([^{}]+)\}", r"\1", line)
            cells = [cell.replace("$", "").strip() for cell in line.split("&")]
            if len(cells) != len(columns):
                raise ValueError(f"{label}: expected {len(columns)} cells, got {cells}")
            for col in fill_columns:
                if not cells[col]:
                    if not rows:
                        raise ValueError(f"Missing leading label in {label}")
                    cells[col] = rows[-1][col]
            rows.append(cells)
        tables.append({"number": number, "slug": slug, "label": label,
                       "columns": columns, "rows": rows})
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True,
                        help="Directory containing results.tex and discussion.tex")
    args = parser.parse_args()
    tables = parse_tables(args.sections)
    reference = {
        "description": "Numeric table reference extracted directly from the English MiTA manuscript; used only for verification, never to calculate results.",
        "source_sha256": {name: hashlib.sha256((args.sections / name).read_bytes()).hexdigest()
                          for name in ("results.tex", "discussion.tex")},
        "tables": tables,
    }
    output = ROOT / "reference" / "manuscript_tables.json"
    output.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Froze {len(tables)} tables ({sum(len(t['rows']) for t in tables)} rows) into {output}")


if __name__ == "__main__":
    main()
