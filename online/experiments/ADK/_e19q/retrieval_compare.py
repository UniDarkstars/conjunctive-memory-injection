"""E19q retrieval-side comparison, ADK (fully local: keyword retrieval,
no LLM, no embedding, no model API).

2x2 within the SAME tasks/materials (E17 test pool, frozen kits):
  storage: distributed vs fully_merged  (identical atom texts)
  query mode: whole-task fire text vs per-object queries (frozen rules
              in _e19q_rules.py), same return budget W=10 after the
              fixed round-robin dedup merge.
Cells: 2 layouts x 2 modes x 3 M-levels x 2 scenarios x 30 test tasks.
Stop rule (author-set): if per-object retrieval does not change the
distributed-vs-merged joint-retrieval gap, stop; execution experiments
only on a clear difference.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
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


HERE = Path(__file__).resolve().parent
E17_DIR = HERE.parents[1] / "ADK" / "_e17"
EXPERIMENTS = HERE.parents[1]
for _p in [str(EXPERIMENTS),
           str(EXPERIMENTS.parent / "agent-frameworks" / "adk-python" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _e17_materials import instances_s1, instances_s2, methods, templates  # noqa: E402
from _e19q_rules import build_queries, measure, round_robin_merge  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}


def _cfg() -> dict:
    return json.loads((E17_DIR / "study_config.json").read_text(encoding="utf-8"))


async def measure_task(scenario: str, task: dict, layout: str, m: int,
                       w: int) -> dict:
    from google.adk.events.event import Event
    from google.adk.memory.in_memory_memory_service import (
        InMemoryMemoryService,
    )
    from google.genai import types

    tpl = templates.get("adk", scenario)
    kit = methods.build_method_kit(scenario, layout, task, m)
    service = InMemoryMemoryService()
    events = [
        Event(author=tpl["agent_name"],
              content=types.Content(
                  role="assistant", parts=[types.Part(text=c["content"])]))
        for c in kit["carriers"] + kit["background"]
    ]
    await service.add_events_to_memory(
        app_name="e19q", user_id="u", events=events)

    async def search(query: str) -> list[str]:
        r = await service.search_memory(
            app_name="e19q", user_id="u", query=query)
        return [" ".join(p.text for p in mem.content.parts if p.text)
                for mem in r.memories]

    queries = build_queries(scenario, task)
    row = {"scenario": scenario, "task_id": task["task_id"],
           "layout": layout, "m": m}
    for mode, qs in queries.items():
        returns = [await search(q) for q in qs]
        merged = (returns[0][:w] if mode == "whole"
                  else round_robin_merge(returns, w))
        for k, v in measure(merged, kit).items():
            row[f"{mode}_{k}"] = v
    return row


async def main_async(args) -> None:
    cfg = _cfg()
    w = int(cfg.get("W", 10))
    levels = args.levels or ["base", "mid", "stress"]
    rows = []
    for sc in ("s1", "s2"):
        for task in _MODULES[sc].TEST_INSTANCES:
            for layout in ("distributed", "fully_merged"):
                for lv in levels:
                    row = await measure_task(sc, task, layout,
                                             cfg["m_levels"][lv], w)
                    row["level"] = lv
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)

    out = run_dir_for(__file__)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "compare_adk.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    summary: dict = {}
    for lv in levels:
        for sc in ("s1", "s2"):
            for layout in ("distributed", "fully_merged"):
                cell = [r for r in rows
                        if r["level"] == lv and r["scenario"] == sc
                        and r["layout"] == layout]
                for mode in ("whole", "per_object"):
                    summary[f"{sc}/{layout}/{mode}/{lv}"] = {
                        "joint": sum(r[f"{mode}_joint"] for r in cell),
                        "has_T": sum(r[f"{mode}_has_T"] for r in cell),
                        "mean_k": (sum(r[f"{mode}_n_k"] for r in cell)
                                   / len(cell)),
                        "n": len(cell),
                    }
    # Stop-rule view (base level): the distributed-vs-merged joint gap
    # under each query mode.
    gaps = {}
    for sc in ("s1", "s2"):
        gw = (summary[f"{sc}/fully_merged/whole/base"]["joint"]
              - summary[f"{sc}/distributed/whole/base"]["joint"])
        go = (summary[f"{sc}/fully_merged/per_object/base"]["joint"]
              - summary[f"{sc}/distributed/per_object/base"]["joint"])
        gaps[sc] = {"gap_whole": gw, "gap_per_object": go}
    summary["_gaps_base"] = gaps
    summary["_generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["_rules"] = "see experiments/_e19q_rules.py (frozen)"
    (out / "compare_adk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E19q retrieval compare (ADK, local)")
    parser.add_argument("--levels", nargs="*", default=None)
    args = parser.parse_args()
    validate_output_args(args)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
