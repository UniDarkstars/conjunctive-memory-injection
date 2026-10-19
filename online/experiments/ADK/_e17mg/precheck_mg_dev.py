"""Dev-task retrieval precheck for e17mg, ADK side (fully local).

Keyword retrieval through the vendored ADK InMemoryMemoryService (top-10,
same preload semantics as the runner); no LLM, no embedding, no network.

Plan §5 step 2: ONLY the 20 dev tasks (10 per scenario) at the frozen
m_levels from the e17 study config; test tasks are never touched here, and
this precheck runs BEFORE any material freeze of the test split. Per task
it measures, over the method's frozen manifest components:
    per-record presence (full-text, whitespace-normalized containment),
    >=1 record (PRIMARY selection metric),
    all-three,
    count distribution 0/1/2/3.
Selection rule (plan §5 step 4, pre-registered): primary = >=1 rate;
ties broken by all-three rate, then mean retrieved count, then mapping
order (mg1/mg2/mg3).

Output: runs/precheck_mg/precheck_adk_mg_dev.csv + summary json.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
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
E17_DIR = HERE.parent / "_e17"
EXPERIMENTS = HERE.parents[1]
for _p in [str(EXPERIMENTS),
           str(EXPERIMENTS.parent / "agent-frameworks" / "adk-python" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _e17_materials import instances_s1, instances_s2, templates  # noqa: E402
from _e17mg_materials import mg_methods  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


async def _measure(scenario: str, task: dict, m: int) -> dict:
    from google.adk.events.event import Event
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.genai import types

    tpl = templates.get("adk", scenario)
    kit = mg_methods.build_mg_kit(scenario, "memorygraft", task, m)

    service = InMemoryMemoryService()
    events = [
        Event(
            author=tpl["agent_name"],
            content=types.Content(
                role="assistant", parts=[types.Part(text=c["content"])]
            ),
        )
        for c in kit["carriers"] + kit["background"]
    ]
    await service.add_events_to_memory(
        app_name="e17mg_precheck", user_id="study_user", events=events
    )
    resp = await service.search_memory(
        app_name="e17mg_precheck", user_id="study_user", query=task["fire"]
    )
    joined = "\n".join(" ".join(p.text for p in mem.content.parts if p.text)
                       for mem in resp.memories)
    ev = _norm(joined)
    presence = [
        _norm(c["match_text"]) in ev for c in kit["manifest"]["components"]
    ]
    n_present = sum(presence)
    return {
        "scenario": scenario,
        "task_id": task["task_id"],
        "split": "dev",
        "m": m,
        "returned": len(resp.memories),
        "n_present": n_present,
        "ge1": n_present >= 1,
        "all3": n_present == 3,
        "per_record": "|".join(str(int(p)) for p in presence),
    }


async def main_async(args) -> None:
    cfg = json.loads((E17_DIR / "study_config.json").read_text(encoding="utf-8"))
    levels = args.levels or ["base", "mid", "stress"]
    rows = []
    for sc in ("s1", "s2"):
        for task in _MODULES[sc].DEV_INSTANCES:
            for lv in levels:
                row = await _measure(sc, task, cfg["m_levels"][lv])
                row["level"] = lv
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    out_dir = run_dir_for(__file__) / "precheck_mg"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "precheck_adk_mg_dev.csv").open("w", newline="",
                                                    encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for lv in levels:
        sel = [r for r in rows if r["level"] == lv]
        for sc in ("s1", "s2"):
            cell = [r for r in sel if r["scenario"] == sc]
            summary[f"{sc}/{lv}"] = {
                "n": len(cell),
                "ge1_rate": sum(r["ge1"] for r in cell) / len(cell),
                "all3_rate": sum(r["all3"] for r in cell) / len(cell),
                "mean_present": sum(r["n_present"] for r in cell) / len(cell),
                "dist_0123": [
                    sum(1 for r in cell if r["n_present"] == k) for k in range(4)
                ],
            }
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["platform"] = "adk"
    summary["tasks"] = "dev only"
    (out_dir / "precheck_adk_mg_dev_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADK e17mg dev retrieval precheck (local keyword)")
    parser.add_argument("--levels", nargs="*", default=None,
                        help="subset of base/mid/stress (default: all three)")
    args = parser.parse_args()
    validate_output_args(args)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
