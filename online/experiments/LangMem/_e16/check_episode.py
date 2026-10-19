"""Validate saved LangMem E16 runs; no model or embedding calls."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode

import sys

_THIS = Path(__file__).resolve().parent
for _path in (_THIS, _THIS.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import arms_core
from _e16_evaluation import evaluate as evaluate_saved, request_blocks
from _e16_runtime import atomic_write_json, atomic_write_text


def _blocks(llm_requests: list[dict]) -> list[str]:
    return [block for group in request_blocks(llm_requests, "langmem") for block in group]


def evaluate(run_dir: Path) -> dict:
    return evaluate_saved(run_dir, "langmem", arms_core)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E16 episode evaluator")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    validate_output_args(args)
    verdicts = []
    for run_dir in args.run_dirs:
        verdict = evaluate(run_dir)
        verdicts.append(verdict)
        atomic_write_json(run_dir / "verdicts.json", verdict)
        print(json.dumps(verdict, ensure_ascii=False))
    if args.csv and verdicts:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(verdicts[0]))
        writer.writeheader()
        writer.writerows(verdicts)
        atomic_write_text(args.csv, buffer.getvalue())
        print(f"rolled up -> {args.csv}", file=sys.stderr)
    if any(not verdict["valid_run"] for verdict in verdicts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
