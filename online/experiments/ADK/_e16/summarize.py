"""Offline E16 statistics with model, batch, validity and family checks."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode

import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS))
from _e16_runtime import RUNTIME_REVISION, atomic_write_text

ARMS = ("replay", "restore", "sham")
SETTINGS = (("adk", "s1"), ("adk", "s2"), ("langmem", "s1"), ("langmem", "s2"))


def cp_bound(x: int, n: int, alpha: float) -> tuple[float, float]:
    if not 0 <= x <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid CP arguments")

    def invert(k, target):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            p = (lo + hi) / 2
            cdf = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                      for i in range(k + 1))
            if cdf > target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return (0.0 if x == 0 else invert(x - 1, 1 - alpha / 2),
            1.0 if x == n else invert(x, alpha / 2))


def mcnemar_exact_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(b + c, i)
                           for i in range(min(b, c) + 1)) / 2 ** (b + c))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def boolean(value, default=False) -> bool:
    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0", "", None):
        return default if value in ("", None) else False
    raise ValueError(f"Invalid boolean field: {value!r}")


def evaluator_for(platform: str):
    directory = "ADK" if platform == "adk" else "LangMem"
    source = EXPERIMENTS / directory / "_e16/check_episode.py"
    spec = importlib.util.spec_from_file_location(f"e16_summary_checker_{platform}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def load_matrix(path: Path) -> list[dict]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    manifest_path = path.parent / "batch_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    evaluators = {}
    result = []
    booleans = {"technical_error", "input_checks_ok", "P3", "task_proxy",
                "all_atoms_all_requests", "valid_run", "P3_observed",
                "completion_verified", "artifacts_complete"}
    integers = {"episode_index", "missing_count", "n_requests", "n_blocks",
                "export_calls", "benign_calls"}
    checked_fields = ("platform", "scenario", "arm", "episode_index",
                      "snapshot_sha256", "technical_error", "status",
                      "input_checks_ok", "n_requests", "n_blocks", "export_calls",
                      "P3", "benign_calls", "task_proxy", "all_atoms_all_requests",
                      "model_id", "batch_id", "runtime_revision", "valid_run",
                      "P3_observed", "completion_verified", "artifacts_complete")
    for saved_row in saved:
        row = dict(saved_row)
        platform = row.get("platform", "")
        scenario, arm = row.get("scenario"), row.get("arm")
        if platform not in ("adk", "langmem") or scenario not in ("s1", "s2") or arm not in ARMS:
            raise ValueError(f"Invalid episode identity in {path}")
        index = int(row["episode_index"])
        run_dir = path.parent / scenario / arm / f"epi-{index:02d}"
        cfg_path = run_dir / "input_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if (cfg.get("platform"), cfg.get("scenario"), cfg.get("arm"),
                cfg.get("episode_index")) != (platform, scenario, arm, index):
            raise ValueError(f"Matrix/episode directory identity mismatch: {run_dir}")
        model = row.get("model_id") or cfg.get("model_id")
        if not model:
            raise ValueError(f"Model provenance missing: {run_dir}")
        if model != cfg.get("model_id"):
            raise ValueError(f"Matrix/model configuration mismatch: {run_dir}")
        batch = row.get("batch_id") or cfg.get("batch_id") or path.parent.name
        if cfg.get("batch_id") and batch != cfg["batch_id"]:
            raise ValueError(f"Matrix/batch configuration mismatch: {run_dir}")
        if row.get("snapshot_sha256") != cfg.get("snapshot_sha256"):
            raise ValueError(f"Matrix/snapshot configuration mismatch: {run_dir}")
        if manifest and any(manifest.get(k) != v for k, v in (
                ("model_id", model), ("platform", platform), ("batch_id", batch))):
            raise ValueError(f"Matrix/batch manifest mismatch: {path}")
        revision = row.get("runtime_revision") or cfg.get("runtime_revision") or "legacy"
        if cfg.get("runtime_revision") and revision != cfg["runtime_revision"]:
            raise ValueError(f"Matrix/runtime configuration mismatch: {run_dir}")
        if manifest and (manifest.get("runtime_revision") != revision
                         or cfg.get("runtime_revision") != revision):
            raise ValueError(f"Matrix/runtime manifest mismatch: {run_dir}")
        legacy = revision != RUNTIME_REVISION
        if not legacy and batch != path.parent.name:
            raise ValueError(f"Matrix/batch directory identity mismatch: {run_dir}")
        if manifest.get("retrieval_config") and cfg.get("config") != manifest["retrieval_config"]:
            raise ValueError(f"Episode/batch retrieval config mismatch: {run_dir}")
        for key in ("temperature", "max_tokens"):
            if key in manifest.get("model_config", {}) and cfg.get(key) != manifest["model_config"][key]:
                raise ValueError(f"Episode/batch model config mismatch ({key}): {run_dir}")
        if not legacy and any(key not in row for key in (
                "valid_run", "P3_observed", "completion_verified", "artifacts_complete")):
            raise ValueError(f"New-runtime matrix is missing validity fields: {path}")
        if platform not in evaluators:
            evaluators[platform] = evaluator_for(platform)
        # Both legacy and new matrices are indices. Saved episode artifacts,
        # rather than an old matrix row, establish completion and execution.
        verified = evaluators[platform](run_dir)
        for key in checked_fields:
            if key not in row or key not in verified:
                continue
            left, right = row[key], verified[key]
            if key in booleans:
                agree = boolean(left) == boolean(right)
            elif key in integers:
                agree = int(left or 0) == int(right or 0)
            else:
                agree = left == right
            if not agree:
                raise ValueError(f"Stale matrix/artifact disagreement ({key}): {run_dir}")
        for key in checked_fields + ("invalid_reason", "termination_reason"):
            if key in verified:
                row[key] = verified[key]
        requests, blocks = int(row.get("n_requests") or 0), int(row.get("n_blocks") or 0)
        input_valid = (boolean(row.get("input_checks_ok")) and requests > 0
                       and blocks == requests)
        completion = boolean(row.get("completion_verified"))
        artifacts = boolean(row.get("artifacts_complete"))
        valid = (boolean(row.get("valid_run")) and input_valid and artifacts
                 and not boolean(row.get("technical_error"))
                 and row.get("status") == "ok" and (legacy or completion))
        row.update(platform=platform, scenario=scenario, arm=arm, episode_index=index,
                   model_id=model, batch_id=batch, runtime_revision=revision,
                   valid_run=valid, input_checks_ok=input_valid,
                   completion_verified=completion, artifacts_complete=artifacts,
                   P3=boolean(row.get("P3")) if valid else None,
                   P3_observed=boolean(row.get("P3_observed"), boolean(row.get("P3"))),
                   planned_episodes=int(manifest.get("episodes", 30)),
                   matrix_file=str(path), batch_root=str(path.parent), legacy=legacy)
        result.append(row)
    return result


def validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No matrix rows supplied")
    if len({row["model_id"] for row in rows}) != 1:
        raise ValueError("Do not combine different model IDs")
    sources = {}
    seen = set()
    snapshots = {}
    for row in rows:
        platform = row["platform"]
        source = (row["batch_id"], row["batch_root"])
        if platform in sources and sources[platform] != source:
            raise ValueError(f"More than one batch for platform {platform}")
        sources[platform] = source
        identity = (platform, row["scenario"], row["arm"], row["episode_index"])
        if identity in seen:
            raise ValueError(f"Duplicate episode identity: {identity}")
        seen.add(identity)
        paired_key = (platform, row["scenario"], row["episode_index"])
        digest = row.get("snapshot_sha256")
        if not digest:
            raise ValueError(f"Snapshot identity missing: {identity}")
        if paired_key in snapshots and snapshots[paired_key] != digest:
            raise ValueError(f"Paired snapshot mismatch: {paired_key}")
        snapshots[paired_key] = digest


def build_statistics(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    validate_rows(rows)
    cells = []
    for platform, scenario in SETTINGS:
        for arm in ARMS:
            attempted = [r for r in rows if (r["platform"], r["scenario"], r["arm"])
                         == (platform, scenario, arm)]
            if not attempted:
                continue
            valid = [r for r in attempted if r["valid_run"]]
            successes = sum(r["P3"] for r in valid)
            lo, hi = cp_bound(successes, len(valid), .05) if valid else ("", "")
            cells.append({"model_id": attempted[0]["model_id"],
                          "batch_id": attempted[0]["batch_id"],
                          "platform": platform, "scenario": scenario, "arm": arm,
                          "attempted": len(attempted), "valid": len(valid),
                          "invalid": len(attempted) - len(valid), "P3": successes,
                          "ci_lo": lo, "ci_hi": hi,
                          "export_calls": sum(int(r.get("export_calls") or 0) for r in valid),
                          "input_check_failures": sum(not r["input_checks_ok"] for r in attempted),
                          "completion_unknown": sum(r["legacy"] and not r["completion_verified"]
                                                    for r in valid),
                          "benign_proxy": sum(boolean(r.get("task_proxy")) for r in valid)})
    pairs = []
    for target, control in (("restore", "sham"), ("restore", "replay"), ("sham", "replay")):
        for platform, scenario in SETTINGS:
            lookup = lambda arm: {r["episode_index"]: r for r in rows
                                  if (r["platform"], r["scenario"], r["arm"])
                                  == (platform, scenario, arm) and r["valid_run"]}
            target_rows, control_rows = lookup(target), lookup(control)
            common = sorted(target_rows.keys() & control_rows.keys())
            if not common:
                continue
            b = sum(target_rows[i]["P3"] and not control_rows[i]["P3"] for i in common)
            c = sum(control_rows[i]["P3"] and not target_rows[i]["P3"] for i in common)
            lb, ub = cp_bound(b, len(common), .025)
            lc, uc = cp_bound(c, len(common), .025)
            pairs.append({"model_id": target_rows[common[0]]["model_id"],
                          "platform": platform, "scenario": scenario,
                          "contrast": f"{target}-{control}", "n": len(common),
                          "b": b, "c": c, "rd": (b-c)/len(common),
                          "ci_lo": lb-uc, "ci_hi": ub-lc,
                          "p": mcnemar_exact_p(b, c), "holm_p": "",
                          "family": ("E16_R-S_joint4" if control == "sham"
                                     else "exploratory_uncorrected"),
                          "family_status": "incomplete" if control == "sham" else "exploratory"})
    primary = [r for r in pairs if r["contrast"] == "restore-sham"]
    complete = {(r["platform"], r["scenario"]) for r in primary} == set(SETTINGS)
    planned = set()
    for platform, scenario in SETTINGS:
        for arm in ("restore", "sham"):
            selected = [r for r in rows if (r["platform"], r["scenario"], r["arm"])
                        == (platform, scenario, arm)]
            if not selected:
                complete = False
                continue
            counts = {r["planned_episodes"] for r in selected}
            planned.update(counts)
            if len(counts) != 1 or {r["episode_index"] for r in selected} != set(
                    range(1, next(iter(counts)) + 1)):
                complete = False
    complete = complete and len(planned) == 1
    if complete:
        for row, adjusted in zip(primary, holm([r["p"] for r in primary])):
            row["holm_p"] = adjusted
            row["family_status"] = "complete"
    return cells, pairs


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, "\ufeff" + stream.getvalue())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="E16 offline model-specific summary")
    parser.add_argument("--matrix", nargs="+", required=True)
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)
    validate_output_args(args)
    try:
        rows = [row for path in args.matrix for row in load_matrix(Path(path))]
        cells, pairs = build_statistics(rows)
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"model={rows[0]['model_id']} attempted={len(rows)} "
          f"valid={sum(r['valid_run'] for r in rows)}")
    for row in cells:
        print(f"{row['platform']}/{row['scenario']}/{row['arm']}: "
              f"P3={row['P3']}/{row['valid']} invalid={row['invalid']} "
              f"CP=[{row['ci_lo']},{row['ci_hi']}] "
              f"legacy_completion_unknown={row['completion_unknown']}")
    for row in pairs:
        print(f"{row['platform']}/{row['scenario']} {row['contrast']}: "
              f"n={row['n']} b/c={row['b']}/{row['c']} RD={row['rd']} "
              f"CI=[{row['ci_lo']},{row['ci_hi']}] p={row['p']} "
              f"Holm={row['holm_p']} family={row['family_status']}")
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_csv_atomic(out / "e16_cells.csv", cells)
        write_csv_atomic(out / "e16_pairs.csv", pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
