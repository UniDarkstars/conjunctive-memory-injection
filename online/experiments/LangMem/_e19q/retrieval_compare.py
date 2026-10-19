"""E19q retrieval-side comparison, LangMem (semantic retrieval via the
gateway bge-m3 embedder; no chat model, no model API beyond embeddings).

Same 2x2 and frozen rules as the ADK side (_e19q_rules.py): E17 test
pool, distributed vs fully_merged kits, whole-task fire query vs
per-object queries, round-robin dedup merge capped at the frozen W=6.
Requires environment variables NORMAL_BASE_URL/NORMAL_API_KEY and the embedding
endpoint to be up - probe before long runs.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
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
E17_DIR = HERE.parents[1] / "LangMem" / "_e17"
EXPERIMENTS = HERE.parents[1]
for _p in [str(EXPERIMENTS), str(E17_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _e17_materials import instances_s1, instances_s2, methods, templates  # noqa: E402
from _e19q_rules import build_queries, measure, round_robin_merge  # noqa: E402

_MODULES = {"s1": instances_s1, "s2": instances_s2}


def load_secrets_env() -> None:
    """Release credentials come only from process environment variables."""
    return


def make_embedder():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="bge-m3",
        base_url=os.environ["NORMAL_BASE_URL"].strip(),
        api_key=os.environ["NORMAL_API_KEY"].strip(),
        check_embedding_ctx_length=False,
    )


def _cfg() -> dict:
    return json.loads((E17_DIR / "study_config.json").read_text(encoding="utf-8"))


async def measure_task(scenario: str, task: dict, layout: str, m: int,
                       w: int) -> dict:
    from langgraph.store.memory import InMemoryStore

    tpl = templates.get("langmem", scenario)
    kit = methods.build_method_kit(scenario, layout, task, m)
    store = InMemoryStore(
        index={"dims": tpl["embedder_dims"], "embed": make_embedder()}
    )
    namespace = ("e19q", scenario)
    for c in kit["carriers"] + kit["background"]:
        await store.aput(
            namespace, c["carrier_id"],
            {"content": c["content"],
             "content_marker": f"[ID:{c['carrier_id']}]",
             "cell": c["cell"]},
            index=["content"],
        )

    async def search(query: str) -> list[str]:
        items = await store.asearch(namespace, query=query, limit=w)
        return [(getattr(it, "value", {}) or {}).get("content", "")
                for it in items]

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
    load_secrets_env()
    cfg = _cfg()
    out = run_dir_for(__file__)
    out.mkdir(parents=True, exist_ok=True)

    if args.m is not None and args.w_list:
        # Sensitivity sweep (author-set 2026-10-16): fixed M (competition
        # retained), W in {6,8,10}; SAME config for both query modes and
        # both layouts; retrieval-only. Outputs are SEPARATE files - the
        # base W=6/M=6 results are kept untouched.
        combos = [(w, args.m) for w in args.w_list]
        tag = "sensitivity"
    else:
        w0 = int(cfg.get("W", 6))
        combos = [(w0, cfg["m_levels"][lv])
                  for lv in (args.levels or ["base", "mid", "stress"])]
        tag = "base" if len(combos) == 1 else "levels"

    rows = []
    for w, m in combos:
        for sc in ("s1", "s2"):
            for task in _MODULES[sc].TEST_INSTANCES:
                for layout in ("distributed", "fully_merged"):
                    row = await measure_task(sc, task, layout, m, w)
                    row["w"], row["m"], row["level"] = w, m, f"w{w}_m{m}"
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)

    with (out / f"compare_langmem_{tag}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    ws = sorted({w for w, _ in combos})
    summary: dict = {}
    for w in ws:
        for sc in ("s1", "s2"):
            for layout in ("distributed", "fully_merged"):
                cell = [r for r in rows if r["w"] == w
                        and r["scenario"] == sc and r["layout"] == layout]
                for mode in ("whole", "per_object"):
                    summary[f"{sc}/{layout}/{mode}/w{w}"] = {
                        "joint": sum(r[f"{mode}_joint"] for r in cell),
                        "has_T": sum(r[f"{mode}_has_T"] for r in cell),
                        "mean_k": (sum(r[f"{mode}_n_k"] for r in cell)
                                   / len(cell)),
                        "n": len(cell),
                    }
    gaps = {}
    for w in ws:
        gw = sum(summary[f"{sc}/fully_merged/whole/w{w}"]["joint"]
                 - summary[f"{sc}/distributed/whole/w{w}"]["joint"]
                 for sc in ("s1", "s2"))
        go = sum(summary[f"{sc}/fully_merged/per_object/w{w}"]["joint"]
                 - summary[f"{sc}/distributed/per_object/w{w}"]["joint"]
                 for sc in ("s1", "s2"))
        gaps[f"w{w}"] = {"gap_whole_combined": gw, "gap_per_object_combined": go}
    summary["_gaps"] = gaps
    summary["_generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["_rules"] = "see experiments/_e19q_rules.py (frozen)"
    (out / f"compare_langmem_{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E19q retrieval compare (LangMem, gateway embeddings)")
    parser.add_argument("--levels", nargs="*", default=None,
                        help="base path: frozen e17 levels (default all 3)")
    parser.add_argument("--m", type=int, default=None,
                        help="sensitivity mode: fixed M (with --w-list)")
    parser.add_argument("--w-list", nargs="*", type=int, default=None,
                        help="sensitivity mode: W values to sweep")
    args = parser.parse_args()
    validate_output_args(args)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
