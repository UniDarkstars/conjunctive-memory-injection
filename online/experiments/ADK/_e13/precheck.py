"""Offline pre-checks for ADK E13 (no LLM / Agent calls; not part of the
formal results).

1) R-full wiring check: for every scenario x instance x layout, the
   diagnostic block from study_materials.iter_full_block must contain all
   four attack atoms (same regex family as check_episode) and no trigger
   key of any other instance; one example per layout is formatted through
   the preload wrapper to show the atoms survive injection formatting.
   The service override itself is exercised directly (search_memory returns
   the block regardless of query).

2) R-natural retrieval check: seeds a real InMemoryMemoryService with each
   kit and fires the instance's fire query; reports four-atom visibility in
   the top-10 window. Expected from E11: B1/B2 fully visible, B0's short
   key records evicted by the M=12 competitors.

Outputs CSVs under runs/precheck/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
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
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


VENDORED_ADK_SRC = str(Path(__file__).resolve().parents[3] / "agent-frameworks" / "adk-python" / "src")
for _p in [VENDORED_ADK_SRC, str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.append(_p)

_SYNC_TOKEN = {"s1": "recon-sync", "s2": "gate-sync"}
_T_ID_RE = {
    "B0": re.compile(r"\[ID:T\]"),
    "B1": re.compile(r"\[ID:(?:T|T_merged)\]"),
    "B2": re.compile(r"\[ID:T\]"),
}
BASELINES = ("B0", "B1", "B2")


def _atoms(block_text: str, scenario: str, codes: list[str], baseline: str) -> dict:
    sync = _SYNC_TOKEN[scenario]
    k = {
        f"k{i+1}": bool(re.search(rf"{sync}\s+{re.escape(c)}\s+rc=0", block_text))
        for i, c in enumerate(codes)
    }
    t = bool(_T_ID_RE[baseline].search(block_text))
    return {"t": t, **k, "full": t and all(k.values())}


def _format_block(entries: list, agent_name: str) -> str:
    """Replicates PreloadMemoryTool.process_llm_request's formatting."""
    lines = []
    for m in entries:
        if m.timestamp:
            lines.append(f"Time: {m.timestamp}")
        text = " ".join(p.text for p in m.content.parts if p.text)
        lines.append(f"{m.author}: {text}" if m.author else text)
    return "\n".join(lines)


async def run_checks(out_dir: Path, scenarios: list[str]) -> None:
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.events.event import Event
    from google.genai import types

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_episode import build_full_exposure_service, persist_event_writes

    wiring_rows = []
    retrieval_rows = []
    wiring_failures = []
    example_printed = set()

    for scenario in scenarios:
        sm = importlib.import_module(f"study_materials_{scenario.lower()}")
        for idx, inst in enumerate(sm.INSTANCES, start=1):
            codes = inst["codes"]
            all_codes = [c for j, o in enumerate(sm.INSTANCES, start=1)
                         if j != idx for c in o["codes"]]
            kits = sm.iter_baselines(codes, inst)
            blocks = sm.iter_full_block(codes, inst)

            # 1) wiring / R-full
            for baseline in BASELINES:
                block_text = "\n".join(r["content"] for r in blocks[baseline])
                atoms = _atoms(block_text, scenario, codes, baseline)
                foreign = sum(
                    1 for c in all_codes
                    if re.search(rf"{_SYNC_TOKEN[scenario]}\s+{re.escape(c)}\s+rc=0", block_text)
                )
                row = {
                    "scenario": scenario, "episode_index": idx,
                    "baseline": baseline,
                    "records": len(blocks[baseline]),
                    "chars": len(block_text),
                    "foreign_key_hits": foreign,
                    **atoms,
                }
                wiring_rows.append(row)
                if not atoms["full"] or foreign:
                    wiring_failures.append(row)

                if (scenario, baseline) not in example_printed:
                    example_printed.add((scenario, baseline))
                    service = build_full_exposure_service(
                        InMemoryMemoryService, blocks[baseline], sm.AGENT_NAME
                    )
                    resp = await service.search_memory(
                        app_name="precheck", user_id="precheck", query="irrelevant query"
                    )
                    formatted = _format_block(resp.memories, sm.AGENT_NAME)
                    fmt_atoms = _atoms(formatted, scenario, codes, baseline)
                    print(
                        f"[wiring-example] {scenario}/{baseline}: service "
                        f"returned {len(resp.memories)} entries; formatted "
                        f"atoms t={fmt_atoms['t']} k1={fmt_atoms['k1']} "
                        f"k2={fmt_atoms['k2']} k3={fmt_atoms['k3']} "
                        f"full={fmt_atoms['full']}"
                    )
                    if not fmt_atoms["full"]:
                        wiring_failures.append(
                            {"scenario": scenario, "episode_index": idx,
                             "baseline": baseline, "issue": "formatted_block_lost_atoms"}
                        )

            # 2) retrieval / R-natural
            for baseline in BASELINES:
                service = InMemoryMemoryService()
                await persist_event_writes(
                    service, "precheck", "replay", kits[baseline], sm.AGENT_NAME
                )
                resp = await service.search_memory(
                    app_name="precheck", user_id="replay", query=inst["fire"]
                )
                block_text = "\n".join(
                    " ".join(p.text for p in m.content.parts if p.text)
                    for m in resp.memories
                )
                atoms = _atoms(block_text, scenario, codes, baseline)
                retrieval_rows.append({
                    "scenario": scenario, "episode_index": idx,
                    "baseline": baseline,
                    "recalled": len(resp.memories),
                    **atoms,
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "wiring_rfull.csv", wiring_rows)
    _write_csv(out_dir / "retrieval_natural.csv", retrieval_rows)

    print("\n=== wiring (R-full) summary: atom failures / foreign keys ===")
    n = len(wiring_rows)
    full_ok = sum(1 for r in wiring_rows if r.get("full"))
    foreign_ok = sum(1 for r in wiring_rows if r.get("foreign_key_hits") == 0)
    print(f"full-atom blocks: {full_ok}/{n}; foreign-key-free: {foreign_ok}/{n}")
    if wiring_failures:
        print("FAILURES:", json.dumps(wiring_failures[:10], ensure_ascii=False, default=str))

    print("\n=== retrieval (R-natural) summary: four-atom visibility in top-10 ===")
    for scenario in scenarios:
        for baseline in BASELINES:
            rows = [r for r in retrieval_rows
                    if r["scenario"] == scenario and r["baseline"] == baseline]
            full = sum(1 for r in rows if r["full"])
            t_only = sum(1 for r in rows if r["t"])
            print(f"{scenario}/{baseline}: full={full}/30  t_visible={t_only}/30")

    if wiring_failures:
        raise SystemExit("R-full wiring precheck FAILED")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK E13 offline prechecks")
    parser.add_argument("--out-dir", default=str(run_dir_for(__file__) / "precheck"))
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    args = parser.parse_args()
    validate_output_args(args)
    asyncio.run(run_checks(Path(args.out_dir), args.scenarios))


if __name__ == "__main__":
    main()
