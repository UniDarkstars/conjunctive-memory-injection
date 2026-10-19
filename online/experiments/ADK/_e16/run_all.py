"""Run a versioned E16 batch; existing observations are never overwritten."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode

import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _e16_runtime import (RUNTIME_REVISION, atomic_write_json,
                         ensure_batch_manifest, exclusive_lock, runtime_metadata)

PLATFORM = "adk"
SCENARIOS = ("s1", "s2")
ARMS = ("replay", "restore", "sham")
SCHED_SEED = 20261004
RETRIEVAL_CONFIG = {"search_limit": 10, "n_competitors": 12}
MODEL_CONFIG = {"temperature_forwarded": False, "max_tokens_forwarded": False}


def schedule_assignments(n_tasks: int, rng: random.Random) -> list[tuple[str, ...]]:
    permutations = list(itertools.permutations(ARMS))
    assignments = permutations * (n_tasks // 6) + permutations[:n_tasks % 6]
    rng.shuffle(assignments)
    return assignments


def full_schedule(episodes: int) -> dict:
    rng = random.Random(SCHED_SEED)
    return {"seed": SCHED_SEED, "episodes": episodes,
            "assignments": {scenario: [list(a) for a in schedule_assignments(episodes, rng)]
                            for scenario in SCENARIOS}}


def snapshot_identities(snap_dir: Path, episodes: int) -> dict:
    identities = {}
    for scenario in SCENARIOS:
        for index in range(1, episodes + 1):
            path = snap_dir / scenario / f"epi-{index:02d}.json"
            snap = json.loads(path.read_text(encoding="utf-8"))
            if (snap.get("platform"), snap.get("scenario"), snap.get("episode_index")) != (
                    PLATFORM, scenario, index):
                raise ValueError(f"Snapshot identity mismatch: {path}")
            if snap.get("config") != RETRIEVAL_CONFIG:
                raise ValueError(f"Snapshot retrieval config mismatch: {path}")
            raw = {key: value for key, value in snap.items() if key != "snapshot_sha256"}
            digest = hashlib.sha256(json.dumps(
                raw, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            if snap.get("snapshot_sha256") != digest:
                raise ValueError(f"Snapshot hash mismatch: {path}")
            identities[f"{scenario}/epi-{index:02d}"] = digest
    return identities


def batch_expected(args) -> dict:
    snap_dir = Path(args.snap_dir).resolve()
    root = Path(args.run_root).resolve()
    return {"experiment": "e16", "platform": PLATFORM, "model_id": args.model,
            "runtime_revision": RUNTIME_REVISION, "batch_id": root.name,
            "episodes": args.episodes, "snap_dir": str(snap_dir),
            "retrieval_config": dict(RETRIEVAL_CONFIG),
            "model_config": dict(MODEL_CONFIG),
            "runtime_metadata": runtime_metadata(),
            "snapshots": snapshot_identities(snap_dir, args.episodes)}


def child_environment() -> dict:
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def run_child(command):
    return subprocess.run(command, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          env=child_environment())


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        f"e16_batch_checker_{PLATFORM}", HERE / "check_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def existing_episode_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("s*/*/epi-*"))


def verify_existing(run_dir: Path, expected: dict, evaluate) -> None:
    scenario, arm = run_dir.parents[1].name, run_dir.parent.name
    try:
        index = int(run_dir.name.removeprefix("epi-"))
    except ValueError as exc:
        raise ValueError(f"Unexpected episode directory: {run_dir}") from exc
    if scenario not in SCENARIOS or arm not in ARMS or not 1 <= index <= expected["episodes"]:
        raise ValueError(f"Episode outside this batch plan: {run_dir}")
    cfg = json.loads((run_dir / "input_config.json").read_text(encoding="utf-8"))
    required = {"platform": PLATFORM, "scenario": scenario, "arm": arm,
                "episode_index": index, "model_id": expected["model_id"],
                "batch_id": expected["batch_id"], "runtime_revision": RUNTIME_REVISION,
                "config": RETRIEVAL_CONFIG,
                "snapshot_sha256": expected["snapshots"][f"{scenario}/epi-{index:02d}"]}
    # run_episode spreads runtime_metadata() flat into input_config.json; accept the
    # nested form too so both writer schemas verify against the same values.
    recorded = cfg.get("runtime_metadata")
    if recorded is None:
        recorded = {key: cfg.get(key) for key in expected["runtime_metadata"]}
    if recorded != expected["runtime_metadata"] or any(
            cfg.get(key) != value for key, value in required.items()):
        raise ValueError(f"Episode configuration mismatch; use a new run root: {run_dir}")
    if Path(cfg.get("snapshot", "")).resolve() != (
            Path(expected["snap_dir"]) / scenario / f"epi-{index:02d}.json"):
        raise ValueError(f"Episode snapshot path mismatch: {run_dir}")
    for key, value in MODEL_CONFIG.items():
        if key in ("temperature", "max_tokens") and cfg.get(key) != value:
            raise ValueError(f"Episode model parameter mismatch ({key}): {run_dir}")
    verdict = evaluate(run_dir)
    if not verdict.get("valid_run") or not verdict.get("completion_verified"):
        raise ValueError(f"Failed or incomplete episode; use a new run root: {run_dir}")


def selected_runs(args, schedule: dict):
    for scenario in SCENARIOS:
        if args.only_scenario and scenario != args.only_scenario:
            continue
        for index, arms in enumerate(schedule["assignments"][scenario], 1):
            if args.only_index is not None and index != args.only_index:
                continue
            for arm in arms:
                if args.only_arm and arm != args.only_arm:
                    continue
                yield scenario, index, arm, arms


def execute_batch(args) -> int:
    root = Path(args.run_root).resolve()
    expected = batch_expected(args)
    manifest_path = root / "batch_manifest.json"
    if args.resume and not manifest_path.exists():
        raise ValueError("--resume requires an existing versioned batch manifest")
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".batch.lock"):
        if manifest_path.exists() and not args.resume:
            raise FileExistsError("Batch already exists; use --resume for complete same-config episodes")
        ensure_batch_manifest(root, expected)
        schedule_path = root / "schedule.json"
        generated = full_schedule(args.episodes)
        if args.resume:
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            if schedule != generated:
                raise ValueError("Saved schedule differs from the frozen complete batch schedule")
            evaluate = load_evaluator()
            for run_dir in existing_episode_dirs(root):
                verify_existing(run_dir, expected, evaluate)
        else:
            schedule = generated
            atomic_write_json(schedule_path, schedule)
        failures = 0
        executed = skipped = 0
        for scenario, index, arm, arms in selected_runs(args, schedule):
            run_dir = root / scenario / arm / f"epi-{index:02d}"
            if args.resume and run_dir.exists():
                skipped += 1
                continue
            append_jsonl(root / "order.jsonl", {
                "ts": time.time(), "scenario": scenario, "arm": arm,
                "episode_index": index, "block": arms, "model_id": args.model,
                "batch_id": root.name, "runtime_revision": RUNTIME_REVISION})
            command = [sys.executable, "-B", str(HERE / "run_episode.py"),
                       "--scenario", scenario, "--episode-index", str(index),
                       "--reference-snapshot",
                       str(Path(expected["snap_dir"]) / scenario / f"epi-{index:02d}.json"),
                       "--intervention", arm, "--run-root", str(root),
                       "--model", args.model]
            print(f"running {scenario}/{arm}/epi-{index:02d}", flush=True)
            result = run_child(command)
            executed += 1
            if result.stdout.strip():
                print(result.stdout.strip(), flush=True)
            if result.returncode:
                failures += 1
                append_jsonl(root / "failures.jsonl", {
                    "ts": time.time(), "scenario": scenario, "arm": arm,
                    "episode_index": index, "phase": "episode",
                    "returncode": result.returncode})
                print(f"{scenario}/{arm}/epi-{index:02d}: child exited {result.returncode}",
                      file=sys.stderr, flush=True)
        directories = existing_episode_dirs(root)
        if directories:
            result = run_child([sys.executable, "-B", str(HERE / "check_episode.py"),
                                *map(str, directories), "--csv", str(root / "matrix.csv")])
            if result.stdout.strip():
                print(result.stdout[-3000:].strip(), flush=True)
            if result.returncode:
                failures += 1
                append_jsonl(root / "failures.jsonl", {
                    "ts": time.time(), "phase": "evaluation",
                    "returncode": result.returncode})
                print(f"Evaluation failed with exit {result.returncode}; matrix is not current",
                      file=sys.stderr, flush=True)
        else:
            failures += 1
        # Detect a changed shared snapshot even if every child returned successfully.
        if snapshot_identities(Path(expected["snap_dir"]), args.episodes) != expected["snapshots"]:
            raise ValueError("Frozen snapshots changed during this batch")
        print(f"executed={executed} skipped={skipped} failures={failures}", flush=True)
        return int(bool(failures))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"{PLATFORM} E16 versioned batch runner")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--episodes", type=int, choices=range(1, 31), default=30)
    parser.add_argument("--snap-dir", default=str(HERE / "snapshots"))
    parser.add_argument("--run-root")
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-scenario", choices=SCENARIOS)
    parser.add_argument("--only-arm", choices=ARMS)
    parser.add_argument("--only-index", type=int)
    args = parser.parse_args(argv)
    validate_output_args(args)
    if args.only_index is not None and not 1 <= args.only_index <= args.episodes:
        parser.error("--only-index must fall within --episodes")
    if args.collect:
        if args.resume:
            parser.error("--collect cannot be combined with --resume")
        result = run_child([sys.executable, "-B", str(HERE / "collect_snapshot.py"),
                            "--episodes", str(args.episodes), "--snap-dir", args.snap_dir,
                            "--scenarios", *SCENARIOS])
        if result.stdout.strip():
            print(result.stdout.strip())
        return int(bool(result.returncode))
    if not args.run_root or not args.model:
        parser.error("Agent runs require explicit --model and --run-root")
    try:
        return execute_batch(args)
    except (ValueError, FileExistsError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
