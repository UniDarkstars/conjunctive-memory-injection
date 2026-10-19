"""Task-block batch driver for e17mg_v1 (one platform; self-detects ADK/LangMem).

Batches (plan §6, 480 formal episodes across BOTH platforms; this driver
runs one platform's 240):
  mg-base    2 scenarios x 60 test tasks x {memorygraft, fully_merged} = 120;
             the two methods run adjacently inside each task block
             (interleaved, Latin rotation) so the primary FM-MG contrast
             is same-batch concurrent;
  mg-mid     memorygraft only x 60 test tasks (descriptive levels);
  mg-stress  memorygraft only x 60 test tasks;
  smoke      memorygraft on DEV tasks only: 10 s1 + 4 s2 = 14 episodes,
             saved under runs/smoke_e17mg_v1 (independent directory).

Scheduling contract mirrors _e17/run_all.py: task blocks with per-scenario
Latin rotation of the block's method list, block order shuffled with the
frozen seed (20261013), every execution appended to order.jsonl, full
schedule dumped per batch. Resume: episodes with ok=true and a matching
config signature are skipped; an existing dir WITHOUT ok status is
reported as a failure and never overwritten (there is no --force in this
batch, plan §10). No external-payload gate (no external method here).
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
PLATFORM = "adk" if HERE.parent.name.upper() == "ADK" else "langmem"
E17_DIR = HERE.parent / "_e17"
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from _e17_materials import instances_s1, instances_s2, methods  # noqa: E402
from _e17mg_materials import mg_methods  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}
SCENARIOS = ["s1", "s2"]
SEED = 20261013

FORMAL_METHODS = {
    "base": ["memorygraft", "fully_merged"],
    "mid": ["memorygraft"],
    "stress": ["memorygraft"],
}


def smoke_jobs() -> list[tuple[str, str, str]]:
    # Dev tasks ONLY (plan §10): all 10 s1 dev + first 4 s2 dev, memorygraft.
    jobs = [
        (sc, "memorygraft", t["task_id"])
        for sc, n in (("s1", 10), ("s2", 4))
        for t in _MODULES[sc].DEV_INSTANCES[:n]
    ]
    return jobs


def formal_jobs(level: str) -> list[tuple[str, str, str]]:
    return [
        (sc, m, t["task_id"])
        for sc in SCENARIOS
        for t in _MODULES[sc].TEST_INSTANCES
        for m in FORMAL_METHODS[level]
    ]


def schedule_blocks(jobs: list[tuple[str, str, str]], seed: int) -> list[dict]:
    """Same contract as _e17/run_all.schedule_blocks: group by (scenario,
    task); within each scenario rotate the block's method list by the
    block index (position balance); shuffle task keys per scenario, then
    shuffle the combined block list - all with the frozen seed."""
    rng = random.Random(seed)
    by_scenario: dict[str, dict[str, list[str]]] = {}
    for sc, m, tid in jobs:
        by_scenario.setdefault(sc, {}).setdefault(tid, []).append(m)
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


def _cfg() -> dict:
    p = E17_DIR / "study_config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


FREEZE_SECTIONS = ("new_files", "reused_from_e17_v1",
                   "shared_e17_materials", "e11_scene_templates")


def _verify_freeze(args) -> None:
    """Fail closed unless the freeze receipt exists and every dependency
    hash matches (review fix 4): a post-freeze prompt edit or a different
    seed can never mix configs into one batch. Older receipts lacking the
    gap-fill sections verify only the sections they carry."""
    fp = Path(args.freeze)
    if not fp.exists():
        raise SystemExit(
            f"freeze receipt not found: {fp}. Run freeze_mg.py after the "
            "dev selection (plan §5 step 5), or pass --allow-unfrozen for "
            "explicitly pre-freeze partial runs only."
        )
    f = json.loads(fp.read_text(encoding="utf-8"))
    if f.get("seed") != args.seed:
        raise SystemExit(
            f"seed mismatch: freeze receipt seed {f.get('seed')} != "
            f"--seed {args.seed}; a batch must run under its frozen seed"
        )

    def resolve(rel: str) -> Path:
        r = rel.replace("\\", "/")
        if not r.startswith("experiments/"):
            raise SystemExit(
                f"freeze receipt key is not repo-relative: {rel} "
                "(regenerate the receipt with the current freeze_mg.py)"
            )
        return EXPERIMENTS.parent / r

    mismatches = []
    for section in FREEZE_SECTIONS:
        for rel, want in f.get(section, {}).items():
            p = resolve(rel)
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()[:16] != want:
                mismatches.append(rel)
    if mismatches:
        raise SystemExit(
            "frozen dependencies changed since the receipt - refusing to "
            f"run: {mismatches}"
        )


def _load_or_write_schedule(run_root: Path, batch: str,
                            jobs: list[tuple[str, str, str]], seed: int) -> list[dict]:
    """Resume must REUSE the saved schedule (review fix 4): never
    regenerate or overwrite it; a different seed is refused; the saved
    identities must exactly match the planned job set."""
    p = run_root / f"schedule_{batch}.json"
    if p.exists():
        sched = json.loads(p.read_text(encoding="utf-8"))
        if sched.get("seed") != seed:
            raise SystemExit(
                f"saved schedule {p.name} has seed {sched.get('seed')}, "
                f"--seed is {seed}; resume with the saved seed"
            )
        blocks = sched["blocks"]
    else:
        blocks = schedule_blocks(jobs, seed)
        p.write_text(json.dumps({"seed": seed, "batch": batch, "blocks": blocks},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    planned = {(b["scenario"], m, b["task_id"])
               for b in blocks for m in b["methods"]}
    if planned != set(jobs):
        raise SystemExit(
            f"saved schedule {p.name} does not match the planned job set; "
            "refusing to mix schedules"
        )
    return blocks


def _sig(sc: str, m: str, tid: str, m_count: int, model_id: str,
         width: int | None) -> str:
    task = _MODULES[sc].by_id(tid)
    kit = mg_methods.build_mg_kit(sc, m, task, m_count)
    return mg_methods.kit_signature(kit, model_id, width)


def run_batch(batch: str, jobs: list[tuple[str, str, str]], args, extra: list[str]) -> int:
    if not args.allow_unfrozen:
        _verify_freeze(args)
    run_root = Path(args.run_root) if args.run_root else (
        run_dir_for(__file__) / ("smoke_e17mg_v1" if batch == "smoke" else "e17mg_v1")
    )
    order_path = run_root / "order.jsonl"
    run_root.mkdir(parents=True, exist_ok=True)

    blocks = _load_or_write_schedule(run_root, batch, jobs, args.seed)

    cfg = _cfg()
    model_id = args.model or os.environ.get(
        "NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731"
    )
    m_level = {"mg-base": "base", "mg-mid": "mid", "mg-stress": "stress"}.get(batch)
    m_count = cfg.get("m_levels", {}).get(m_level) if m_level else \
        cfg.get("m_levels", {}).get("base")
    width = cfg.get("W") if PLATFORM == "langmem" else None

    failures: list[str] = []
    n_done = n_skipped = n_retried = 0
    for block in blocks:
        for m in block["methods"]:
            sc, tid = block["scenario"], block["task_id"]
            out_dir = run_root / batch / sc / m / tid
            status = out_dir / "_status.json"
            st = {}
            if status.exists():
                try:
                    st = json.loads(status.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    st = {}
            if st.get("ok"):
                if st.get("config_sig") == _sig(
                        sc, m, tid, m_count, model_id, width):
                    n_skipped += 1
                    continue
                # Config drift on a completed episode is never retried.
                failures.append(
                    f"config-mismatch (frozen config changed; needs manual "
                    f"review): {out_dir}")
                continue
            if (out_dir.exists() or status.exists()) and not args.retry_failed:
                # Evidence is preserved: the episode runner archives prior
                # files under attempts/ before rerunning. Retry only when
                # explicitly requested (review fix: failure recovery).
                failures.append(
                    f"needs --retry-failed (prior attempt archived under "
                    f"attempts/ on rerun): {out_dir}")
                continue
            if out_dir.exists():
                n_retried += 1
            order_path.open("a", encoding="utf-8").write(json.dumps({
                "ts": time.time(), "batch": batch, "platform": PLATFORM,
                "scenario": sc, "method": m, "task_id": tid,
                "block_index": block["block_index"],
            }, ensure_ascii=False) + "\n")
            cmd = [
                sys.executable, str(HERE / "run_episode_mg.py"),
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
                    print(f"FAIL(status) {sc}/{m}/{tid}", file=sys.stderr, flush=True)

    print(f"=== batch {batch} [{PLATFORM}]: ran {n_done}, retried {n_retried}, "
          f"skipped {n_skipped}, failures {len(failures)}")
    if failures:
        (run_root / f"failures_{batch}.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1
    return 0


def plan_only(args) -> None:
    plans = {}
    for batch, jobs, extra in [
        ("mg-base", formal_jobs("base"), ["--m-level", "base"]),
        ("mg-mid", formal_jobs("mid"), ["--m-level", "mid"]),
        ("mg-stress", formal_jobs("stress"), ["--m-level", "stress"]),
        ("smoke", smoke_jobs(), ["--m-level", "base"]),
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
    plans["formal_total_per_platform"] = sum(
        plans[b]["episodes"] for b in ("mg-base", "mg-mid", "mg-stress"))
    plans["formal_total_both_platforms"] = 2 * plans["formal_total_per_platform"]
    plans["smoke_uses_dev_tasks_only"] = True
    out = Path(args.run_root or (run_dir_for(__file__) / "_plan")) / "plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {"episodes": v["episodes"], "cells": v["cells"]})
                      for k, v in plans.items()
                      if not k.startswith("family_disjointness")},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"e17mg_v1 batch driver ({PLATFORM})")
    parser.add_argument("--batch", default=None,
                        choices=["base", "mid", "stress", "smoke", "all", "plan"])
    parser.add_argument("--run-root", default=None,
                        help="default: runs/e17mg_v1 (smoke: runs/smoke_e17mg_v1)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--freeze", default=str(HERE / "freeze_mg.json"),
                        help="freeze receipt path verified before running")
    parser.add_argument("--allow-unfrozen", action="store_true",
                        help="pre-freeze partial runs only (skips receipt "
                             "verification)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-invoke episodes whose _status is missing or "
                             "ok=false; the runner archives the prior attempt "
                             "under attempts/ (evidence preserved)")
    args = parser.parse_args()
    validate_output_args(args)

    if args.plan_only or args.batch == "plan":
        plan_only(args)
        return
    if not args.batch:
        raise SystemExit("pass --batch or --plan-only")

    extra_by_batch = {
        "base": ["--m-level", "base"],
        "mid": ["--m-level", "mid"],
        "stress": ["--m-level", "stress"],
        "smoke": ["--m-level", "base"],
    }
    jobs_by_batch = {
        "base": formal_jobs("base"),
        "mid": formal_jobs("mid"),
        "stress": formal_jobs("stress"),
        "smoke": smoke_jobs(),
    }
    batches = (["base", "mid", "stress"] if args.batch == "all" else [args.batch])
    rc = 0
    for b in batches:
        rc |= run_batch(f"mg-{b}" if b in ("base", "mid", "stress") else b,
                        jobs_by_batch[b], args, extra_by_batch[b])
    sys.exit(rc)


if __name__ == "__main__":
    main()
