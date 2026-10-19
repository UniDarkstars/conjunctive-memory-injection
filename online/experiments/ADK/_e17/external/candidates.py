"""Dev-candidate selection driver for the E17 external group.

Stages candidate payloads (payload.py output) into a run-ready directory,
runs DEV episodes for the external group through the platform runner,
appends every run to the dev ledger (cap: 100 full agent runs per method
per scenario, experiment.md E17 §3), and on --freeze copies the chosen
candidate into payloads/ as the frozen external kit.
"""
from __future__ import annotations

import argparse
import json
import shutil
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
CONSTRUCTION = run_dir_for(__file__) / "construction"
LEDGER = CONSTRUCTION / "dev_ledger.jsonl"
CAP_PER_SCENARIO = 100


def ledger_count(method: str, scenario: str) -> int:
    if not LEDGER.exists():
        return 0
    n = 0
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["method"] == method and r["scenario"] == scenario:
            n += 1
    return n


def stage_candidate(args) -> Path:
    staging = CONSTRUCTION / "candidates" / f"staging_{args.tag}"
    staging.mkdir(parents=True, exist_ok=True)
    for sc in args.scenarios:
        src = Path(args.candidate_dir) / f"frozen_payload_{sc}.json"
        if not src.exists():
            raise SystemExit(f"missing staged payload for {sc}: {src}")
        shutil.copy2(src, staging / f"frozen_payload_{sc}.json")
    return staging


def run_dev(args, staging: Path) -> None:
    from _e17_materials import instances_s1, instances_s2

    modules = {"s1": instances_s1, "s2": instances_s2}
    m_args = (["--m-count", str(args.m_count)] if args.m_count is not None
              else ["--m-level", args.m_level or "base"])
    # Per-tag run root: candidates never share episode dirs, so one
    # candidate's success evidence cannot leak into another's.
    run_root = run_dir_for(__file__) / f"e17_dev_{args.tag}"
    for sc in args.scenarios:
        used = ledger_count("external", sc)
        if used >= CAP_PER_SCENARIO:
            print(f"cap reached for external/{sc}: {used}/{CAP_PER_SCENARIO}",
                  file=sys.stderr)
            continue
        for task in modules[sc].DEV_INSTANCES:
            if ledger_count("external", sc) >= CAP_PER_SCENARIO:
                print(f"cap hit mid-batch for {sc}", file=sys.stderr)
                break
            cmd = [
                sys.executable, str(HERE.parent / "run_episode.py"),
                "--scenario", sc, "--method", "external",
                "--task-id", task["task_id"], "--batch", "dev",
                "--run-root", str(run_root),
                "--payload-dir", str(staging),
                *m_args,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            LEDGER.open("a", encoding="utf-8").write(json.dumps({
                "ts": __import__("time").time(), "tag": args.tag,
                "method": "external", "scenario": sc,
                "task_id": task["task_id"], "ok": proc.returncode == 0,
            }, ensure_ascii=False) + "\n")
            print(proc.stdout.strip() or proc.stderr[-300:], flush=True)


def freeze(args) -> None:
    (CONSTRUCTION / "payloads").mkdir(parents=True, exist_ok=True)
    src = CONSTRUCTION / "candidates" / f"staging_{args.tag}"
    for sc in args.scenarios:
        shutil.copy2(
            src / f"frozen_payload_{sc}.json",
            CONSTRUCTION / "payloads" / f"frozen_payload_{sc}.json",
        )
    record = {
        "frozen_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag, "scenarios": args.scenarios,
        "note": args.note or "",
    }
    (CONSTRUCTION / "payloads" / "freeze_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="E17 external candidate driver")
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"],
                        choices=["s1", "s2"])
    parser.add_argument("--tag", required=True, help="candidate tag (dir name)")
    parser.add_argument("--candidate-dir", required=True,
                        help="dir holding frozen_payload_<sc>.json candidates")
    parser.add_argument("--run-dev", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--note", default="")
    parser.add_argument("--m-level", default=None,
                        choices=["base", "mid", "stress"])
    parser.add_argument("--m-count", type=int, default=None,
                        help="explicit M for dev runs before study_config "
                             "is frozen (overrides --m-level)")
    args = parser.parse_args()
    validate_output_args(args)

    if Path(args.tag).name != args.tag or any(c in args.tag for c in "/\\:") or args.tag in (".", ".."):
        parser.error("tag must be a simple directory name")
    staging = stage_candidate(args)
    if args.run_dev:
        sys.path.insert(0, str(HERE.parents[2]))
        run_dev(args, staging)
    if args.freeze:
        freeze(args)


if __name__ == "__main__":
    main()
