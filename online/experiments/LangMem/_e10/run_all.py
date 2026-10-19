"""Batch driver for LangMem E10: 2 scenarios x 4 variants x 30 unique instances."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


HERE = Path(__file__).resolve().parent

SCENARIOS = ["s1", "s2"]
VARIANTS = ["clean", "T-only", "keys-only", "both"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E10 batch runner")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of unique task instances per cell (default 30)")
    parser.add_argument("--start-index", type=int, default=1,
                        help="1-based first instance of this batch")
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e10"))
    parser.add_argument("--only-scenario", choices=SCENARIOS, default=None)
    parser.add_argument("--only-variant", choices=VARIANTS, default=None)
    parser.add_argument("--only-index", type=int, default=None)
    parser.add_argument("--model", default=None,
                        help="Override LLM id (else NORMAL_CHAT_MODEL env or DeepSeek default)")
    args = parser.parse_args()
    validate_output_args(args)

    run_root = Path(args.run_root)
    scenarios = [args.only_scenario] if args.only_scenario else SCENARIOS
    variants = [args.only_variant] if args.only_variant else VARIANTS

    for scenario in scenarios:
        for variant in variants:
            for i in range(args.start_index, args.episodes + 1):
                if args.only_index is not None and i != args.only_index:
                    continue
                out_dir = run_root / scenario / variant / f"epi-{i:02d}"
                print(f"=== {scenario}/{variant} episode {i} -> {out_dir}", flush=True)
                cmd = [
                    sys.executable,
                    str(HERE / "run_episode.py"),
                    "--scenario", scenario,
                    "--variant", variant,
                    "--episode-index", str(i),
                    "--run-root", str(run_root),
                ]
                if args.model:
                    cmd += ["--model", args.model]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                print(proc.stdout.strip(), flush=True)
                if proc.returncode != 0:
                    print(proc.stderr[-2000:], file=sys.stderr, flush=True)
                    raise SystemExit(f"episode failed: {scenario}/{variant}#{i}")

    dirs = sorted(
        d
        for scenario_dir in run_root.iterdir()
        if scenario_dir.is_dir()
        for variant_dir in scenario_dir.iterdir()
        if variant_dir.is_dir()
        for d in variant_dir.iterdir()
        if d.is_dir() and d.name.startswith("epi-")
    )

    print("=== evaluating", flush=True)
    proc = subprocess.run(
        [sys.executable, str(HERE / "check_episode.py"), *map(str, dirs),
         "--csv", str(run_root / "matrix.csv")],
        capture_output=True,
        text=True,
    )
    print(proc.stdout, flush=True)
    print(proc.stderr[-2000:], file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise SystemExit("evaluation failed")


if __name__ == "__main__":
    main()
