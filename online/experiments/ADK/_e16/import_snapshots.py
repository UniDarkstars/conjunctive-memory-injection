"""Import E16 snapshots from E13's SAVED natural-B0 retrieval records.

Rationale (snapshot_version e16-snap-e13import-1): the E13 natural runs
stored, per task, the REAL retrieval output with original metadata
(search_results.json: author/timestamp/content in recall order) AND the
actual injected request block (llm_requests.json). Reusing them as the
frozen E16 snapshots (a) removes the embedding-gateway dependency,
(b) preserves genuinely observed metadata - a fresh collection would
regenerate ADK timestamps and could reorder LangMem results via embedding
jitter. Only retrieval INPUT records are reused; no E13 model output,
tool trajectory or P3 enters E16 - the replay arm runs everything anew.

Read-only on the E13 tree. Output: snapshots/<scenario>/epi-XX.json with
the same schema as collect_snapshot.py plus import provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import RUNS, run_dir_for, validate_output_args, prepare_fresh_episode


for _p in [str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.append(_p)

import arms_core  # noqa: E402

SNAPSHOT_VERSION = "e16-snap-e13import-1"
E13_RUNS = RUNS / "ADK" / "e13"
BLOCK_RE = re.compile(r"<PAST_CONVERSATIONS>\n(.*?)\n</PAST_CONVERSATIONS>", re.DOTALL)


def actual_block_from_e13(run_dir: Path) -> str:
    llm = json.loads((run_dir / "llm_requests.json").read_text(encoding="utf-8"))
    for content in llm[0]["payload"]["contents"]:
        for part in content.get("parts", []):
            m = BLOCK_RE.search(part.get("text") or "")
            if m:
                return m.group(1)
    raise ValueError(f"no memory block in first request: {run_dir}")


def import_one(scenario: str, episode_index: int, snap_dir: Path,
               sm, verify_render: bool) -> dict:
    run_dir = E13_RUNS / scenario / "natural" / "B0" / f"epi-{episode_index:02d}"
    entries = json.loads((run_dir / "search_results.json").read_text(encoding="utf-8"))
    block = actual_block_from_e13(run_dir)

    if verify_render:
        rendered = "\n".join(
            (f"Time: {e['timestamp']}\n" if e.get("timestamp") else "")
            + (f"{e['author']}: {e['content']}" if e.get("author") else e["content"])
            for e in entries)
        if rendered != block:
            raise AssertionError(
                f"saved retrieval does not render to the actual block: {run_dir}")

    inst = sm.INSTANCES[episode_index - 1]
    canonical = arms_core.canonical_atoms(sm, inst["codes"], inst)
    analysis = arms_core.analyze_block(block, canonical)
    added_ts = max((e["timestamp"] or "") for e in entries) if entries else ""

    snapshot = {
        "experiment": "e16", "platform": "adk",
        "snapshot_version": SNAPSHOT_VERSION,
        "scenario": scenario, "episode_index": episode_index,
        "episode": f"epi-{episode_index:02d}", "codes": inst["codes"],
        "config": {"search_limit": sm.SEARCH_LIMIT, "n_competitors": sm.N_COMPETITORS},
        "agent_name": sm.AGENT_NAME,
        "system": sm.AGENT_INSTRUCTION,
        "user": inst["fire"],
        "tools": [t[0] for t in sm.BENIGN_TOOLS] + [sm.EXPORT_FUNCTION_NAME],
        "entries": entries,
        "block_preview": block,
        "block_preview_raw": block,
        "added_timestamp": added_ts,
        "canonical_atoms": canonical,
        "atoms_present": analysis["present"],
        "atoms_missing": analysis["missing"],
        "atom_evidence": analysis["evidence"],
        "collected_at": "imported from E13 natural run (2026-10-01)",
        "source_provenance": {
            "kind": "e13-import",
            "e13_run_dir": str(run_dir.relative_to(_RELEASE_ONLINE)),
            "e13_batch": "e13-rfull-1",
            "retrieval": "search_results.json (real InMemoryMemoryService output)",
            "block": "first-request <PAST_CONVERSATIONS> from llm_requests.json",
        },
    }
    snapshot["snapshot_sha256"] = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    out = snap_dir / scenario / f"epi-{episode_index:02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def main() -> None:
    global E13_RUNS
    parser = argparse.ArgumentParser(description="Import E16 snapshots from E13 (offline)")
    parser.add_argument("--e13-root", type=Path, default=E13_RUNS, help="Existing release-local E13 run root")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    parser.add_argument("--snap-dir", default=str(run_dir_for(__file__) / "new_snapshots"))
    parser.add_argument("--no-verify-render", action="store_true")
    args = parser.parse_args()
    validate_output_args(args)
    E13_RUNS = args.e13_root.resolve()
    from release_runtime import checked_output
    checked_output(args.snap_dir)

    snap_dir = Path(args.snap_dir)
    if snap_dir.resolve() == (Path(__file__).resolve().parent / "snapshots").resolve():
        existing = list(snap_dir.rglob("epi-*.json"))
        if existing:
            raise SystemExit(f"{len(existing)} snapshots already locked in {snap_dir}; "
                             "import refuses to overwrite (use another --snap-dir)")
    from collections import Counter
    dist = Counter()
    for scenario in args.scenarios:
        sm = importlib.import_module(f"study_materials_{scenario.lower()}")
        for i in range(1, args.episodes + 1):
            snap = import_one(scenario, i, snap_dir, sm,
                              verify_render=not args.no_verify_render)
            dist[(scenario, tuple(snap["atoms_missing"]))] += 1
    print(f"imported {sum(dist.values())} snapshots -> {snap_dir}")
    for (scen, missing), n in sorted(dist.items()):
        print(f"  {scen} D={{{','.join(missing)}}}: {n}")


if __name__ == "__main__":
    main()
