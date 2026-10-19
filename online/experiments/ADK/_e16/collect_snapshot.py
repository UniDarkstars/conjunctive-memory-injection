"""Snapshot collection for ADK E16 (offline: real write+retrieval path, no
model call, stops BEFORE the first main-model request).

Per e16_plan_v2: for every scenario x frozen task, seed the real
InMemoryMemoryService with the E13 B0 kit (attack atoms + M=12 competitors),
run the REAL keyword top-10 retrieval with the task's fire text, and freeze:
the natural entries with their original metadata (author/timestamp/content),
the trusted system instruction, the normal user task, the tool names, the
rendered natural block, the deterministic added-record timestamp, and the
canonical-atom analysis (A_i / D_i by full-text matching). Raw texts are
saved alongside the newline-normalised matching rule.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
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


VENDORED_ADK_SRC = str(Path(__file__).resolve().parents[3] / "agent-frameworks" / "adk-python" / "src")
for _p in [VENDORED_ADK_SRC, str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.append(_p)

import arms_core  # noqa: E402

SNAPSHOT_VERSION = "e16-snap-1"


def render_block(entries: list[dict]) -> str:
    """Replicates PreloadMemoryTool's <PAST_CONVERSATIONS> formatting."""
    lines = []
    for e in entries:
        if e.get("timestamp"):
            lines.append(f"Time: {e['timestamp']}")
        text = e["content"]
        lines.append(f"{e['author']}: {text}" if e.get("author") else text)
    return "\n".join(lines)


async def collect(scenario: str, episode_index: int, snap_dir: Path) -> Path:
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.events.event import Event
    from google.genai import types

    sm = importlib.import_module(f"study_materials_{scenario.lower()}")
    inst = sm.INSTANCES[episode_index - 1]
    kit = sm.iter_baselines(inst["codes"], inst)["B0"]

    service = InMemoryMemoryService()
    events = [
        Event(author=sm.AGENT_NAME,
              content=types.Content(role="assistant",
                                    parts=[types.Part(text=c["content"])]))
        for c in kit
    ]
    await service.add_events_to_memory(
        app_name="e16_collect", user_id="study_user", events=events)
    resp = await service.search_memory(
        app_name="e16_collect", user_id="study_user", query=inst["fire"])

    entries = [
        {"author": m.author, "timestamp": m.timestamp,
         "content": " ".join(p.text for p in m.content.parts if p.text)}
        for m in resp.memories
    ]
    block = render_block(entries)
    canonical = arms_core.canonical_atoms(sm, inst["codes"], inst)
    analysis = arms_core.analyze_block(block, canonical)
    # deterministic added-record timestamp, frozen per task and shared by R/S
    added_ts = max((e["timestamp"] or "") for e in entries) if entries else ""

    snapshot = {
        "experiment": "e16", "platform": "adk", "snapshot_version": SNAPSHOT_VERSION,
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
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    snapshot["snapshot_sha256"] = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    out = snap_dir / scenario / f"epi-{episode_index:02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


async def main_async(args) -> None:
    snap_dir = Path(args.snap_dir)
    for scenario in args.scenarios:
        for i in range(1, args.episodes + 1):
            out = await collect(scenario, i, snap_dir)
            if i == 1 or i == args.episodes:
                print(f"collected {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK E16 snapshot collection (offline)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    parser.add_argument("--snap-dir", default=str(run_dir_for(__file__) / "new_snapshots"))
    args = parser.parse_args()
    validate_output_args(args)
    from release_runtime import checked_output
    checked_output(args.snap_dir)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
