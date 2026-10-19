"""Episode runner for batch e17mg_v1, ADK side (MemoryGraft adaptation +
concurrent Fully merged base arm).

Delegation contract: the episode body (memory writes, request capture,
simulated tools, attempt archiving, _status verdict) is executed by the
frozen e17_v1 runner loaded from ../_e17/run_episode.py via importlib.
NOTHING under _e17/ is modified, so the e17_v1 freeze.json hashes stay
valid. This wrapper only differs in:

  --method choices are ["memorygraft", "fully_merged"] and the kit is
      built by _e17mg_materials (fully_merged delegates to the frozen
      e17_v1 constructor byte-for-byte);
  default run root runs/e17mg_v1;
  after the episode, input_config.json/_status.json are relabelled
      experiment=e17mg (the base body hardcodes "e17") and the wrapper
      revision is recorded.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import RUNS, run_dir_for, validate_output_args, prepare_fresh_episode


HERE = Path(__file__).resolve().parent
E17_DIR = HERE.parent / "_e17"
EXPERIMENTS = HERE.parents[1]
for _p in (str(EXPERIMENTS), str(E17_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _e17_materials import instances_s1, instances_s2, templates  # noqa: E402
from _e17mg_materials import mg_methods  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}
WRAPPER_REVISION = "e17mg-adk-wrap-1"

_spec = importlib.util.spec_from_file_location(
    "e17_base_runner_adk", E17_DIR / "run_episode.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

DEV_LEDGER = RUNS / "e17mg_development" / "dev_ledger.jsonl"
DEV_LOCK = DEV_LEDGER.with_suffix(".lock")
PLATFORM_TAG = "adk" if HERE.parent.name.upper() == "ADK" else "langmem"
SEED = 20261013


def _resolved_batch(args) -> str:
    """The run's REAL batch label (mirrors the base runner's dir logic):
    an omitted --batch on a dev task resolves to 'dev', so it is still
    accounted (review fix 5a)."""
    return args.batch or ("dev" if args.split == "dev" else f"m-{args.m_level}")


def _verify_formal_freeze(args) -> None:
    """Any invocation whose resolved batch is NOT 'dev' requires a valid
    freeze receipt - default-deny, so a hand-typed batch label like
    'm-base' cannot bypass the driver's verification (review fix 1b;
    tightened after the D6 counterexample)."""
    label = _resolved_batch(args)
    if label == "dev" or args.allow_unfrozen:
        return
    spec = importlib.util.spec_from_file_location(
        "run_all_mg_mod", HERE / "run_all_mg.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    ra._verify_freeze(argparse.Namespace(freeze=args.freeze, seed=args.seed))


def _dev_ledger(args, model_id: str, batch_label: str) -> None:
    """Shared dev-run accounting (plan §12 item 5): ONE ledger for both
    platforms, hard cap 100 recorded runs per scenario (combined). The
    cap check and the append happen inside ONE exclusive lock (review fix
    5b: two platforms reading 99 simultaneously can no longer both pass).
    Recorded runs: resolved batch 'dev' (with or without --batch) or any
    invocation carrying an explicit --purpose; formal/smoke not recorded."""
    import os
    import time

    if batch_label != "dev" and args.purpose == "dev_evaluation":
        return
    DEV_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(400):
        try:
            fd = os.open(str(DEV_LOCK),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(0.05)
            continue
        try:
            entries = []
            if DEV_LEDGER.exists():
                entries = [
                    json.loads(line)
                    for line in DEV_LEDGER.read_text(
                        encoding="utf-8").splitlines()
                    if line.strip()
                ]
            n_sc = sum(1 for e in entries
                       if e.get("scenario") == args.scenario)
            if n_sc >= 100 and not args.dev_over_budget:
                raise SystemExit(
                    f"dev budget cap reached for {args.scenario}: {n_sc} "
                    "runs recorded (cap 100 per scenario, both platforms "
                    "combined); pass --dev-over-budget to exceed explicitly"
                )
            with DEV_LEDGER.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": time.time(), "platform": PLATFORM_TAG,
                    "scenario": args.scenario, "method": args.method,
                    "task_id": args.task_id, "batch": batch_label,
                    "purpose": args.purpose, "model_id": model_id,
                    "over_budget": bool(args.dev_over_budget),
                }, ensure_ascii=False) + "\n")
            return
        finally:
            os.close(fd)
            DEV_LOCK.unlink(missing_ok=True)
    raise SystemExit(f"dev ledger lock timeout: {DEV_LOCK}")


def _relabel(run_dir: Path) -> None:
    """Mark the episode as e17mg after the base runner wrote its config."""
    for name in ("input_config.json", "_status.json"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data["experiment"] = "e17mg"
            data["wrapper_revision"] = WRAPPER_REVISION
            p.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADK e17mg episode runner (MemoryGraft adaptation + FM base)"
    )
    parser.add_argument("--scenario", required=True, choices=["s1", "s2"])
    parser.add_argument("--method", required=True, choices=mg_methods.MG_METHODS)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--m-level", choices=["base", "mid", "stress"], default=None)
    parser.add_argument("--m-count", type=int, default=None)
    parser.add_argument("--batch", default=None,
                        help="run subdirectory (default: m-<level> or dev/smoke)")
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e17mg_v1"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--purpose", default="dev_evaluation",
                        help="recorded in the shared dev ledger")
    parser.add_argument("--dev-over-budget", action="store_true",
                        help="explicitly exceed the 100-per-scenario dev cap")
    parser.add_argument("--freeze", default=str(HERE / "freeze_mg.json"),
                        help="freeze receipt verified for formal/smoke runs")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="must match the freeze receipt seed")
    parser.add_argument("--allow-unfrozen", action="store_true",
                        help="skip receipt verification (pre-freeze use only)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_output_args(args)

    task = _MODULES[args.scenario].by_id(args.task_id)
    args.split = task["split"]
    m_count = _base.resolve_m_count(args)

    kit = mg_methods.build_mg_kit(args.scenario, args.method, task, m_count)
    if not kit["budget"]["within_budget"] and not args.dry_run:
        raise SystemExit(f"budget violation: {kit['budget']['reason']}")

    if args.dry_run:
        _base.dry_run(args, kit)
        return

    _verify_formal_freeze(args)

    import os
    model_id = args.model or os.environ.get(
        "NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731"
    )
    _dev_ledger(args, model_id, _resolved_batch(args))

    _base.load_secrets_env()
    tpl = templates.get("adk", args.scenario)
    run_dir = asyncio.run(_base.run_episode(args, kit, tpl))
    _relabel(run_dir)
    print(f"episode dir: {run_dir}")


if __name__ == "__main__":
    main()
