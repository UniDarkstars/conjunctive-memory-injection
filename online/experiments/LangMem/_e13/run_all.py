"""Batch driver for LangMem E13: 2 scenarios x 2 paths x 3 layouts x 30
instances.

Paths: natural (actual semantic retrieval at the frozen W) and full (R-full
diagnostic block assembled at the same <memories> injection position).
Layouts: B0 distributed / B1 merged / B2 burst, E11 attack atoms + the E13
competitive condition (N_COMPETITORS same-domain competitors, frozen after
precheck.py). Execution order is appended to <run-root>/order.jsonl;
--shuffle-seed block-randomises the job order.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
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
PATHS = ["natural", "full"]
BASELINES = ["B0", "B1", "B2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E13 batch runner")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of unique task instances per cell (default 30)")
    parser.add_argument("--start-index", type=int, default=1,
                        help="1-based first instance of this batch")
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e13"))
    parser.add_argument("--only-scenario", choices=SCENARIOS, default=None)
    parser.add_argument("--only-path", choices=PATHS, default=None)
    parser.add_argument("--only-baseline", choices=BASELINES, default=None)
    parser.add_argument("--only-index", type=int, default=None)
    parser.add_argument("--model", default=None,
                        help="Override LLM id (else NORMAL_CHAT_MODEL env or DeepSeek default)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Randomise the condition/episode execution order")
    args = parser.parse_args()
    validate_output_args(args)

    run_root = Path(args.run_root)
    scenarios = [args.only_scenario] if args.only_scenario else SCENARIOS
    paths = [args.only_path] if args.only_path else PATHS
    baselines = [args.only_baseline] if args.only_baseline else BASELINES

    jobs = [
        (scenario, path, baseline, i)
        for scenario in scenarios
        for path in paths
        for baseline in baselines
        for i in range(args.start_index, args.episodes + 1)
        if args.only_index is None or i == args.only_index
    ]
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(jobs)

    order_path = run_root / "order.jsonl"
    order_path.parent.mkdir(parents=True, exist_ok=True)

    for scenario, path, baseline, i in jobs:
        out_dir = run_root / scenario / path / baseline / f"epi-{i:02d}"
        print(f"=== {scenario}/{path}/{baseline} episode {i} -> {out_dir}", flush=True)
        order_path.open("a", encoding="utf-8").write(json.dumps({
            "ts": time.time(), "scenario": scenario, "path": path,
            "baseline": baseline, "episode_index": i,
        }, ensure_ascii=False) + "\n")
        cmd = [
            sys.executable,
            str(HERE / "run_episode.py"),
            "--scenario", scenario,
            "--path", path,
            "--baseline", baseline,
            "--episode-index", str(i),
            "--run-root", str(run_root),
        ]
        if args.model:
            cmd += ["--model", args.model]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout.strip(), flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr, flush=True)
            raise SystemExit(f"episode failed: {scenario}/{path}/{baseline}#{i}")

    dirs = sorted(
        d
        for scenario_dir in run_root.iterdir()
        if scenario_dir.is_dir()
        for path_dir in scenario_dir.iterdir()
        if path_dir.is_dir()
        for baseline_dir in path_dir.iterdir()
        if baseline_dir.is_dir()
        for d in baseline_dir.iterdir()
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
