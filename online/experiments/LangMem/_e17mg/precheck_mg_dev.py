"""Dev-task retrieval precheck for e17mg, LangMem side (gateway embeddings).

Uses the same InMemoryStore + bge-m3 embedder wiring as the LangMem E17
precheck; requires environment variables NORMAL_BASE_URL/NORMAL_API_KEY and the gateway
embedding endpoint to be up - run only inside the gated dev window.

Plan §5 step 2: ONLY the 20 dev tasks at the frozen m_levels (W=6 from the
e17 study config); test tasks are never touched here. Metrics identical to
the ADK side: per-record presence, >=1 (PRIMARY), all-three, 0/1/2/3
distribution.

Output: runs/precheck_mg/precheck_langmem_mg_dev.csv + summary json.
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
for _p in [str(EXPERIMENTS), str(E17_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _e17_materials import instances_s1, instances_s2, templates  # noqa: E402
from _e17mg_materials import mg_methods  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}


def load_secrets_env() -> None:
    """Release credentials come only from process environment variables."""
    return


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def make_embedder():
    from langchain_openai import OpenAIEmbeddings
    import os

    return OpenAIEmbeddings(
        model="bge-m3",
        base_url=os.environ["NORMAL_BASE_URL"].strip(),
        api_key=os.environ["NORMAL_API_KEY"].strip(),
        check_embedding_ctx_length=False,
    )


async def _measure(scenario: str, task: dict, m: int, w: int) -> dict:
    from langgraph.store.memory import InMemoryStore

    tpl = templates.get("langmem", scenario)
    kit = mg_methods.build_mg_kit(scenario, "memorygraft", task, m)

    store = InMemoryStore(
        index={"dims": tpl["embedder_dims"], "embed": make_embedder()}
    )
    namespace = ("e17mg_precheck", scenario)
    for c in kit["carriers"] + kit["background"]:
        await store.aput(
            namespace, c["carrier_id"],
            {"content": c["content"],
             "content_marker": f"[ID:{c['carrier_id']}]",
             "cell": c["cell"]},
            index=["content"],
        )
    items = await store.asearch(namespace, query=task["fire"], limit=w)
    joined = "\n".join(
        (getattr(it, "value", {}) or {}).get("content", "") for it in items
    )
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
        "w": w,
        "returned": len(items),
        "n_present": n_present,
        "ge1": n_present >= 1,
        "all3": n_present == 3,
        "per_record": "|".join(str(int(p)) for p in presence),
    }


async def main_async(args) -> None:
    load_secrets_env()
    cfg = json.loads((E17_DIR / "study_config.json").read_text(encoding="utf-8"))
    w = int(args.w) if args.w else int(cfg.get("W", 6))
    levels = args.levels or ["base", "mid", "stress"]
    rows = []
    for sc in ("s1", "s2"):
        for task in _MODULES[sc].DEV_INSTANCES:
            for lv in levels:
                row = await _measure(sc, task, cfg["m_levels"][lv], w)
                row["level"] = lv
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    out_dir = run_dir_for(__file__) / "precheck_mg"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "precheck_langmem_mg_dev.csv").open("w", newline="",
                                                        encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

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
    summary["platform"] = "langmem"
    summary["tasks"] = "dev only"
    summary["w"] = w
    (out_dir / "precheck_langmem_mg_dev_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangMem e17mg dev retrieval precheck (gateway embeddings)")
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--w", type=int, default=None)
    args = parser.parse_args()
    validate_output_args(args)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
