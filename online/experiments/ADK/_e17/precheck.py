"""Clean-criterion W/M precheck for ADK E17 (fully local, no LLM/embedding).

Sweeps the benign competition level M over the DEV tasks under the clean
group (pool slice only) and records the full retrieval table used to fix
M_base (experiment.md E17 §3: W/M chosen on clean-task normal recall,
criterion fixed before the run; this script only SAVES the table, the
selection and its rationale are recorded at freeze time).

ADK keyword retrieval width is the framework's preload default (top-10,
as in E13); M is the only swept axis here.

--method-check reruns the sweep for one attack method's kit (ours /
direct) at given M values - the dev-side pressure check (§5) - measuring
recall completeness of that method's components. External is skipped
until its payload is frozen.
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
_EXPERIMENTS = HERE.parents[1]
for _p in [str(_EXPERIMENTS),
           str(_EXPERIMENTS.parent / "agent-frameworks" / "adk-python" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _e17_materials import (  # noqa: E402
    benign_pools, instances_s1, instances_s2, methods, templates,
)

_MODULES = {"s1": instances_s1, "s2": instances_s2}
DEFAULT_M_LIST = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]


async def _measure(scenario: str, method: str, task: dict, m: int) -> dict:
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.events.event import Event
    from google.genai import types

    tpl = templates.get("adk", scenario)
    kit = methods.build_method_kit(
        scenario, method, task, m, allow_missing_external=True
    )
    if kit["manifest"].get("placeholder"):
        return {"skip": "external payload not frozen"}

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
        app_name="e17_precheck", user_id="dev", events=events
    )
    resp = await service.search_memory(
        app_name="e17_precheck", user_id="dev", query=task["fire"]
    )
    contents = [
        " ".join(p.text for p in m.content.parts if p.text)
        for m in resp.memories
        if m.content and m.content.parts
    ]
    joined = "\n".join(contents)

    import re

    comps = kit["manifest"]["components"]
    norm = lambda t: re.sub(r"\s+", " ", t or "").strip()
    ev = norm(joined)
    cover = [
        (norm(c["match_text"]) in ev) if "match_text" in c
        else bool(re.compile(c["match_regex"]).search(joined or ""))
        for c in comps
    ]
    code_hits = sum(1 for c in task["codes"] if c in joined)

    return {
        "n_returned": len(contents),
        "code_hits": code_hits,
        "any_record": bool(contents),
        "components": len(comps),
        "components_hit": sum(cover),
        "recall_complete": bool(cover) and all(cover),
    }


async def run_sweep(args) -> list[dict]:
    rows = []
    scenarios = args.scenarios or ["s1", "s2"]
    for scenario in scenarios:
        for method in ([args.method_check] if args.method_check else ["clean"]):
            for m in args.m_list:
                for task in _MODULES[scenario].DEV_INSTANCES:
                    r = await _measure(scenario, method, task, m)
                    rows.append({
                        "platform": "adk", "scenario": scenario,
                        "method": method, "m": m,
                        "task_id": task["task_id"], "family": task["family"],
                        "w": 10, **r, "ts": time.time(),
                    })
                    print(json.dumps(rows[-1], ensure_ascii=False))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("skip"):
            continue
        groups.setdefault((r["scenario"], r["method"], r["m"]), []).append(r)
    for (scenario, method, m), g in sorted(groups.items()):
        n = len(g)
        out.append({
            "scenario": scenario, "method": method, "m": m, "n_tasks": n,
            "all_nonempty": all(x["any_record"] for x in g),
            "mean_returned": sum(x["n_returned"] for x in g) / n,
            "mean_code_hits": sum(x["code_hits"] for x in g) / n,
            "recall_complete_tasks": sum(x["recall_complete"] for x in g),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK E17 clean-criterion precheck")
    parser.add_argument("--scenarios", nargs="*", default=None, choices=["s1", "s2"])
    parser.add_argument("--m-list", type=int, nargs="*", default=DEFAULT_M_LIST)
    parser.add_argument("--method-check", default=None,
                        choices=["clean", "direct", "distributed", "fully_merged"],
                        help="sweep this method instead of clean (pressure check)")
    parser.add_argument("--out-dir", default=str(run_dir_for(__file__) / "precheck"))
    args = parser.parse_args()
    validate_output_args(args)

    rows = asyncio.run(run_sweep(args))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.method_check or "clean"
    raw_path = out_dir / f"precheck_adk_{tag}.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    sum_path = out_dir / f"precheck_adk_{tag}_summary.csv"
    with sum_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"raw -> {raw_path}\nsummary -> {sum_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
