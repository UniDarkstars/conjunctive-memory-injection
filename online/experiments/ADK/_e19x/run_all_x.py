"""Task-block driver for E19x: the four cells (layout x query mode) run
CONCURRENTLY within one batch - same tasks, same materials, same model,
same return budget; old-batch execution rates are never used as controls
(author requirement).

Scheduling mirrors the e17/e17mg contract: task blocks with per-scenario
Latin rotation of the four conditions, block order shuffled with the
frozen seed (20261016), order.jsonl appended per execution, schedule
dumped once and REUSED on resume; episodes with ok=true and a matching
config signature are skipped; failed dirs are re-invoked only with
--retry-failed (prior attempts archived under attempts/ by the runner).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
E17_DIR = HERE.parent / "_e17"
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from _e17_materials import instances_s1, instances_s2, methods  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}
SEED = 20261016
CONDITIONS = [(layout, mode)
              for layout in ("distributed", "fully_merged")
              for mode in ("whole", "per_object")]


def _cfg() -> dict:
    return json.loads((E17_DIR / "study_config.json").read_text(encoding="utf-8"))


def _sig(sc, layout, mode, tid, m, model_id, w):
    task = _MODULES[sc].by_id(tid)
    kit = methods.build_method_kit(sc, layout, task, m)
    h = methods.kit_signature(kit, model_id)
    return hashlib.sha256(f"{h}|mode={mode}|w={w}".encode()).hexdigest()[:16]


def schedule_blocks(jobs, seed):
    rng = random.Random(seed)
    by_scenario: dict[str, dict[str, list[str]]] = {}
    for sc, cond, tid in jobs:
        by_scenario.setdefault(sc, {}).setdefault(tid, []).append(cond)
    blocks = []
    for sc, tasks in by_scenario.items():
        keys = sorted(tasks)
        rng.shuffle(keys)
        for i, tid in enumerate(keys):
            conds = tasks[tid]
            rot = (i * 1) % len(conds)
            blocks.append({"scenario": sc, "task_id": tid,
                           "conditions": conds[rot:] + conds[:rot],
                           "block_index": i})
    rng.shuffle(blocks)
    return blocks


def run_batch(batch, jobs, args) -> int:
    run_root = Path(args.run_root)
    order_path = run_root / "order.jsonl"
    run_root.mkdir(parents=True, exist_ok=True)

    sched_path = run_root / f"schedule_{batch}.json"
    if sched_path.exists():
        sched = json.loads(sched_path.read_text(encoding="utf-8"))
        if sched.get("seed") != args.seed:
            raise SystemExit("saved schedule seed mismatch; resume with it")
        blocks = sched["blocks"]
    else:
        blocks = schedule_blocks(jobs, args.seed)
        sched_path.write_text(json.dumps(
            {"seed": args.seed, "batch": batch, "blocks": blocks},
            ensure_ascii=False, indent=2), encoding="utf-8")
    planned = {(b["scenario"], tuple(c), b["task_id"])
               for b in blocks for c in b["conditions"]}
    if planned != set(jobs):
        raise SystemExit("saved schedule does not match planned job set")

    cfg = _cfg()
    m = cfg["m_levels"]["base"]
    w = int(cfg.get("W", 10))
    model_id = args.model or os.environ.get(
        "NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731")

    failures: list[str] = []
    n_done = n_skipped = n_retried = 0
    for block in blocks:
        for cond in block["conditions"]:
            layout, mode = cond  # tuple (fresh) or list (from saved JSON)
            sc, tid = block["scenario"], block["task_id"]
            out_dir = run_root / batch / sc / layout / mode / tid
            status = out_dir / "_status.json"
            st = {}
            if status.exists():
                try:
                    st = json.loads(status.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    st = {}
            if st.get("ok"):
                sig = st.get("config_sig")
                if sig is None:
                    # compat: pre-fix episodes keep the signature only in
                    # input_config.json - fall back there
                    try:
                        sig = json.loads(
                            (out_dir / "input_config.json").read_text(
                                encoding="utf-8")).get("config_sig")
                    except (OSError, json.JSONDecodeError):
                        sig = None
                if sig == _sig(sc, layout, mode, tid, m, model_id, w):
                    n_skipped += 1
                    continue
                failures.append(f"config-mismatch (manual review): {out_dir}")
                continue
            if (out_dir.exists() or status.exists()) and not args.retry_failed:
                failures.append(f"needs --retry-failed: {out_dir}")
                continue
            if out_dir.exists():
                n_retried += 1
            order_path.open("a", encoding="utf-8").write(json.dumps({
                "ts": time.time(), "batch": batch, "scenario": sc,
                "layout": layout, "mode": mode, "task_id": tid,
                "block_index": block["block_index"],
            }, ensure_ascii=False) + "\n")
            cmd = [sys.executable, str(HERE / "run_exec.py"),
                   "--scenario", sc, "--layout", layout, "--mode", mode,
                   "--task-id", tid, "--batch", batch,
                   "--run-root", str(run_root), "--m-level", "base"]
            if args.model:
                cmd += ["--model", args.model]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append(f"{sc}/{layout}/{mode}/{tid}: "
                                f"{proc.stderr[-300:]}")
                print(f"FAIL {sc}/{layout}/{mode}/{tid}",
                      file=sys.stderr, flush=True)
            else:
                try:
                    st2 = json.loads(status.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    st2 = {}
                if st2.get("ok"):
                    n_done += 1
                    print(proc.stdout.strip().splitlines()[-1]
                          if proc.stdout.strip() else "done", flush=True)
                else:
                    failures.append(f"{sc}/{layout}/{mode}/{tid}: ok=false")
    print(f"=== batch {batch}: ran {n_done}, retried {n_retried}, "
          f"skipped {n_skipped}, failures {len(failures)}")
    if failures:
        (run_root / f"failures_{batch}.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="E19x batch driver")
    parser.add_argument("--batch", default="x-main")
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e19x_v1"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="4 conditions x 2 dev tasks (s1/s2 dev-01)")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    validate_output_args(args)

    if args.smoke:
        jobs = [(sc, cond, "dev-01") for sc in ("s1", "s2")
                for cond in CONDITIONS]
    else:
        jobs = [(sc, cond, t["task_id"])
                for sc in ("s1", "s2")
                for t in _MODULES[sc].TEST_INSTANCES
                for cond in CONDITIONS]

    if args.plan_only:
        cells = {}
        for sc, cond, tid in jobs:
            cells[f"{sc}/{cond[0]}/{cond[1]}"] = \
                cells.get(f"{sc}/{cond[0]}/{cond[1]}", 0) + 1
        print(json.dumps({"episodes": len(jobs), "cells": cells},
                         ensure_ascii=False, indent=2))
        return
    sys.exit(run_batch(args.batch, jobs, args))


if __name__ == "__main__":
    main()
