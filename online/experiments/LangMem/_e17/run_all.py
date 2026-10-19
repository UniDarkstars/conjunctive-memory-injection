"""Task-block batch driver for E17 (one platform; self-detects ADK/LangMem).

Formal batches (experiment.md E17 §2/§5): one batch per M level
(m-base / m-mid / m-stress), 2 scenarios x 5 methods x 30 test tasks =
300 episodes per batch. Scheduling contract (§9):
  - jobs grouped into task blocks (scenario, task_id); the five methods
    run adjacently within a block;
  - block order shuffled with the frozen seed; method order inside a
    block rotates (Latin rotation) so every method appears in every
    position across a batch - position balance;
  - every execution appended to <run-root>/order.jsonl; the full frozen
    schedule dumped to <run-root>/schedule_<batch>.json;
  - resume: episodes whose _status.json has ok=true are skipped; a run
    dir without ok status aborts the batch unless --force (rerun);
  - --plan-only enumerates the planned identities offline (no LLM,
    no embedding, external payload not required) and checks dev/test
    family disjointness.

Smoke (§9 step 3): 2 dev tasks per scenario x 5 methods (batch "smoke")
plus the same dev tasks x focal methods (fully_merged, external) at
m-stress (batch "smoke-stress").
"""
from __future__ import annotations

import argparse
import os
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
PLATFORM = "adk" if HERE.parent.name.upper() == "ADK" else "langmem"
_EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(_EXPERIMENTS))

from _e17_materials import instances_s1, instances_s2, methods  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}
SCENARIOS = ["s1", "s2"]
FOCAL_SMOKE_METHODS = ["fully_merged", "external"]


def smoke_task_ids(scenario: str) -> list[str]:
    # One dev task per scenario: smoke covers every method, both scenarios
    # and the M-scaling mechanics with 10 + 4 episodes per platform.
    return [t["task_id"] for t in _MODULES[scenario].DEV_INSTANCES[:1]]


def formal_jobs() -> list[tuple[str, str, str]]:
    return [
        (sc, m, t["task_id"])
        for sc in SCENARIOS
        for t in _MODULES[sc].TEST_INSTANCES
        for m in methods.METHODS
    ]


def smoke_jobs(stress: bool) -> list[tuple[str, str, str]]:
    ms = FOCAL_SMOKE_METHODS if stress else methods.METHODS
    return [
        (sc, m, tid)
        for sc in SCENARIOS
        for tid in smoke_task_ids(sc)
        for m in ms
    ]


def schedule_blocks(jobs: list[tuple[str, str, str]], seed: int) -> list[dict]:
    """Group jobs into (scenario, task) blocks. Within EACH scenario the
    five methods rotate by that scenario's block index (Latin rotation),
    so every method occupies every position equally often per scenario;
    the combined block list is then shuffled with the frozen seed."""
    rng = random.Random(seed)
    by_scenario: dict[str, list[str]] = {}
    for sc, m, tid in jobs:
        by_scenario.setdefault(sc, {})
        by_scenario[sc].setdefault(tid, [])
        by_scenario[sc][tid].append(m)
    blocks: list[dict] = []
    for sc, tasks in by_scenario.items():
        keys = sorted(tasks)
        rng.shuffle(keys)
        for i, tid in enumerate(keys):
            ms = tasks[tid]
            rot = i % len(ms)
            blocks.append({
                "scenario": sc, "task_id": tid,
                "methods": ms[rot:] + ms[:rot],
                "block_index": i,
            })
    rng.shuffle(blocks)
    return blocks


def _sig(sc: str, m: str, tid: str, m_count: int, model_id: str,
         width: int | None) -> str:
    """Config signature a resumed episode must match (model, M, width,
    material texts) - experiment.md E17 §9: a resume never silently mixes
    changed conditions."""
    task = _MODULES[sc].by_id(tid)
    kit = methods.build_method_kit(
        sc, m, task, m_count,
        external_dir=HERE / "external" / "payloads",
        allow_missing_external=True,
    )
    return methods.kit_signature(kit, model_id, width)


def episode_dir(run_root: Path, batch: str, sc: str, m: str, tid: str) -> Path:
    return run_root / batch / sc / m / tid


def check_external_payloads() -> None:
    """A scenario counts as missing when its dry-run reports a placeholder
    external kit (no frozen payload)."""
    missing = []
    for sc in SCENARIOS:
        probe = subprocess.run(
            [sys.executable, str(HERE / "run_episode.py"),
             "--scenario", sc, "--method", "external",
             "--task-id", _MODULES[sc].TEST_INSTANCES[0]["task_id"],
             "--m-count", "1", "--batch", "_probe", "--dry-run",
             "--run-root", str(run_dir_for(__file__) / "_probe")],
            capture_output=True, text=True,
        )
        if '"placeholder_external": true' in probe.stdout or probe.returncode != 0:
            missing.append(sc)
    if missing:
        raise SystemExit(
            f"external payload not frozen for scenarios: {missing}; "
            "freeze external/payloads first (experiment.md E17 §9 step 1/3)"
        )


def run_batch(batch: str, jobs: list[tuple[str, str, str]], args, extra: list[str]) -> int:
    run_root = Path(args.run_root)
    order_path = run_root / "order.jsonl"
    run_root.mkdir(parents=True, exist_ok=True)

    blocks = schedule_blocks(jobs, args.seed)
    (run_root / f"schedule_{batch}.json").write_text(
        json.dumps({"seed": args.seed, "batch": batch, "blocks": blocks},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Resume guard inputs: the batch's M, width (LangMem) and model.
    cfg = json.loads((HERE / "study_config.json").read_text(encoding="utf-8")) \
        if (HERE / "study_config.json").exists() else {}
    model_id = args.model or os.environ.get(
        "NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731"
    )
    m_level = {"m-base": "base", "m-mid": "mid", "m-stress": "stress"}.get(batch)
    m_count = cfg.get("m_levels", {}).get(m_level) if m_level else None
    width = cfg.get("W") if PLATFORM == "langmem" else None

    failures: list[str] = []
    n_done = n_skipped = 0
    for block in blocks:
        for m in block["methods"]:
            sc, tid = block["scenario"], block["task_id"]
            out_dir = episode_dir(run_root, batch, sc, m, tid)
            status = out_dir / "_status.json"
            if status.exists():
                try:
                    st = json.loads(status.read_text(encoding="utf-8"))
                    sig_ok = (m_count is None or st.get("config_sig") is None or
                              st.get("config_sig") == _sig(sc, m, tid, m_count,
                                                           model_id, width))
                    if st.get("ok") and sig_ok:
                        n_skipped += 1
                        continue
                    if st.get("ok") and not sig_ok and not args.force:
                        failures.append(
                            f"config-mismatch (needs --force): {out_dir}")
                        continue
                except json.JSONDecodeError:
                    pass
            if out_dir.exists() and not args.force:
                failures.append(f"exists-without-status: {out_dir}")
                continue
            order_path.open("a", encoding="utf-8").write(json.dumps({
                "ts": time.time(), "batch": batch, "platform": PLATFORM,
                "scenario": sc, "method": m, "task_id": tid,
                "block_index": block["block_index"],
            }, ensure_ascii=False) + "\n")
            cmd = [
                sys.executable, str(HERE / "run_episode.py"),
                "--scenario", sc, "--method", m, "--task-id", tid,
                "--batch", batch, "--run-root", str(run_root),
                *extra,
            ]
            if args.model:
                cmd += ["--model", args.model]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append(f"{sc}/{m}/{tid}: {proc.stderr[-400:]}")
                print(f"FAIL {sc}/{m}/{tid}\n{proc.stderr[-600:]}", file=sys.stderr,
                      flush=True)
            else:
                # A clean exit is not a valid episode: read the runner's
                # own verdict; ok=false (crash, length, loop-limit) goes
                # to the failure list instead of the done count.
                try:
                    st = json.loads(status.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    st = {}
                if st.get("ok"):
                    n_done += 1
                    print(proc.stdout.strip(), flush=True)
                else:
                    failures.append(
                        f"{sc}/{m}/{tid}: episode finished with "
                        f"ok=false ({st.get('end_status') or st.get('error')})")
                    print(f"FAIL(status) {sc}/{m}/{tid}", file=sys.stderr,
                          flush=True)

    print(f"=== batch {batch}: ran {n_done}, skipped {n_skipped}, "
          f"failures {len(failures)}")
    if failures:
        (run_root / f"failures_{batch}.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1
    return 0


def plan_only(args) -> None:
    plans = {}
    for batch, jobs, extra in [
        ("m-base", formal_jobs(), ["--m-level", "base"]),
        ("m-mid", formal_jobs(), ["--m-level", "mid"]),
        ("m-stress", formal_jobs(), ["--m-level", "stress"]),
        ("dev", [(sc, m, t["task_id"])
                 for sc in SCENARIOS
                 for t in _MODULES[sc].DEV_INSTANCES
                 for m in methods.METHODS], ["--m-level", "base"]),
        ("smoke", smoke_jobs(stress=False), ["--m-level", "base"]),
        ("smoke-stress", smoke_jobs(stress=True), ["--m-level", "stress"]),
    ]:
        cells: dict[tuple, int] = {}
        for sc, m, tid in jobs:
            cells[(sc, m)] = cells.get((sc, m), 0) + 1
        plans[batch] = {
            "episodes": len(jobs),
            "cells": {f"{sc}/{m}": n for (sc, m), n in sorted(cells.items())},
            "runner_args": extra,
        }
    for sc in SCENARIOS:
        mod = _MODULES[sc]
        assert not (set(mod.DEV_FAMILIES) & set(mod.TEST_FAMILIES))
        plans[f"family_disjointness_{sc}"] = {
            "dev": mod.DEV_FAMILIES, "test": mod.TEST_FAMILIES,
        }
    plans["formal_total_per_batch"] = len(formal_jobs())
    plans["formal_three_batches_total"] = 3 * len(formal_jobs())
    out = Path(args.run_root) / "plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if k.startswith("formal") or k.startswith("family")
                          else {"episodes": v["episodes"], "cells": len(v["cells"])})
                      for k, v in plans.items()}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"E17 batch driver ({PLATFORM})")
    parser.add_argument("--batch", default=None,
                        choices=["base", "mid", "stress", "dev", "smoke",
                                 "smoke-stress", "all"],
                        help="formal M batches map to m-<level> run dirs")
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e17_v1"))
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true",
                        help="rerun episode dirs that exist without ok status")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-missing-external", action="store_true",
                        help="run batches whose external cells are not yet "
                             "frozen (pre-freeze partial runs only)")
    args = parser.parse_args()
    validate_output_args(args)

    if args.plan_only:
        plan_only(args)
        return
    if not args.batch:
        raise SystemExit("pass --batch or --plan-only")

    if not args.allow_missing_external:
        check_external_payloads()
    extra_by_batch = {
        "base": ["--m-level", "base"],
        "mid": ["--m-level", "mid"],
        "stress": ["--m-level", "stress"],
        "dev": ["--m-level", "base"],
        "smoke": ["--m-level", "base"],
        "smoke-stress": ["--m-level", "stress"],
    }
    jobs_by_batch = {
        "base": formal_jobs(), "mid": formal_jobs(), "stress": formal_jobs(),
        "dev": [(sc, m, t["task_id"])
                for sc in SCENARIOS
                for t in _MODULES[sc].DEV_INSTANCES
                for m in methods.METHODS],
        "smoke": smoke_jobs(stress=False),
        "smoke-stress": smoke_jobs(stress=True),
    }
    batches = (["base", "mid", "stress"] if args.batch == "all" else [args.batch])
    rc = 0
    for b in batches:
        rc |= run_batch(f"m-{b}" if b in ("base", "mid", "stress") else b,
                        jobs_by_batch[b], args, extra_by_batch[b])
    sys.exit(rc)


if __name__ == "__main__":
    main()
