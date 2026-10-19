"""Clean-criterion W/M precheck for LangMem E17 (gateway embeddings).

Sweeps W x M over the DEV tasks under the clean group (pool slice only)
and saves the full retrieval table for the M_base / W freeze decision
(experiment.md E17 §3). Embedding usage is counted per store operation
(1 embed per aput content, 1 embed per search query) and reported; runs
require the gateway env (NORMAL_BASE_URL / NORMAL_API_KEY).

--method-check reruns the sweep for one attack method at given M values
(dev-side pressure check); external is skipped until its payload is
frozen.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
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
_EXPERIMENTS = HERE.parents[1]
for _p in [str(_EXPERIMENTS), str(HERE),
           str(_EXPERIMENTS.parent / "agent-frameworks" / "langmem" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import compat  # noqa: F401,E402

from _e17_materials import (  # noqa: E402
    instances_s1, instances_s2, methods, templates,
)

_MODULES = {"s1": instances_s1, "s2": instances_s2}
DEFAULT_W_LIST = [6, 8, 10, 12]
DEFAULT_M_LIST = [0, 2, 4, 6, 8, 10, 12, 16, 20]


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


async def _measure(scenario: str, method: str, task: dict, m: int, w: int) -> dict:
    from langgraph.store.memory import InMemoryStore

    tpl = templates.get("langmem", scenario)
    kit = methods.build_method_kit(
        scenario, method, task, m, allow_missing_external=True
    )
    if kit["manifest"].get("placeholder"):
        return {"skip": "external payload not frozen"}

    store = InMemoryStore(index={"dims": tpl["embedder_dims"], "embed": make_embedder()})
    namespace = ("e17_precheck", scenario)
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

    comps = kit["manifest"]["components"]
    norm = lambda t: re.sub(r"\s+", " ", t or "").strip()
    ev = norm(joined)
    cover = [
        (norm(c["match_text"]) in ev) if "match_text" in c
        else bool(re.compile(c["match_regex"]).search(joined or ""))
        for c in comps
    ]
    code_hits = sum(1 for c in task["codes"] if c in joined)
    n_embed_ops = len(kit["carriers"]) + len(kit["background"]) + 1

    return {
        "n_returned": len(items),
        "code_hits": code_hits,
        "any_record": bool(items),
        "components": len(comps),
        "components_hit": sum(cover),
        "recall_complete": bool(cover) and all(cover),
        "embed_ops_est": n_embed_ops,
    }


async def run_sweep(args, sink=None) -> list[dict]:
    import openai

    rows = []
    scenarios = args.scenarios or ["s1", "s2"]
    for scenario in scenarios:
        for method in ([args.method_check] if args.method_check else ["clean"]):
            for w in args.w_list:
                for m in args.m_list:
                    for task in _MODULES[scenario].DEV_INSTANCES:
                        # The 600 req/min gateway bucket is account-shared, so
                        # even a paced sweep can catch it mid-saturation: wait
                        # out 429/5xx transients instead of losing the table.
                        for attempt in range(args.transient_retries + 1):
                            try:
                                r = await _measure(scenario, method, task, m, w)
                                break
                            except Exception as exc:
                                retriable = isinstance(
                                    exc, (openai.RateLimitError,
                                          openai.APIConnectionError)
                                ) or getattr(exc, "status_code", None) in (
                                    429, 500, 502, 503, 504)
                                if not retriable or attempt >= args.transient_retries:
                                    raise
                                wait = 30.0 * (attempt + 1)
                                print(f"transient {type(exc).__name__} at "
                                      f"{scenario}/{method}/w={w}/m={m}/"
                                      f"{task['task_id']}; retry {attempt + 1} "
                                      f"in {wait:.0f}s", file=sys.stderr,
                                      flush=True)
                                await asyncio.sleep(wait)
                        rows.append({
                            "platform": "langmem", "scenario": scenario,
                            "method": method, "m": m, "w": w,
                            "task_id": task["task_id"], "family": task["family"],
                            **r, "ts": time.time(),
                        })
                        print(json.dumps(rows[-1], ensure_ascii=False))
                        # The sweep re-embeds the whole kit per measure (M+3
                        # gateway requests); pacing keeps the account-wide
                        # 600 req/min limit from aborting the run mid-table.
                        if args.rpm:
                            ops = r.get("embed_ops_est", 1) if not r.get("skip") else 0
                            await asyncio.sleep(ops * 60.0 / args.rpm)
                        if sink is not None:
                            sink(rows[-1])
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("skip"):
            continue
        groups.setdefault((r["scenario"], r["method"], r["w"], r["m"]), []).append(r)
    for (scenario, method, w, m), g in sorted(groups.items()):
        n = len(g)
        out.append({
            "scenario": scenario, "method": method, "w": w, "m": m,
            "n_tasks": n,
            "all_nonempty": all(x["any_record"] for x in g),
            "mean_returned": sum(x["n_returned"] for x in g) / n,
            "mean_code_hits": sum(x["code_hits"] for x in g) / n,
            "recall_complete_tasks": sum(x["recall_complete"] for x in g),
            "embed_ops_est": sum(x["embed_ops_est"] for x in g),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E17 clean-criterion precheck")
    parser.add_argument("--scenarios", nargs="*", default=None, choices=["s1", "s2"])
    parser.add_argument("--w-list", type=int, nargs="*", default=DEFAULT_W_LIST)
    parser.add_argument("--m-list", type=int, nargs="*", default=DEFAULT_M_LIST)
    parser.add_argument("--method-check", default=None,
                        choices=["clean", "direct", "distributed", "fully_merged"])
    parser.add_argument("--out-dir", default=str(run_dir_for(__file__) / "precheck"))
    parser.add_argument("--rpm", type=float, default=360.0,
                        help="gateway requests/min cap the sweep paces itself "
                             "to (0 disables); the account bucket is shared")
    parser.add_argument("--transient-retries", type=int, default=6,
                        help="wait-and-retry cycles for 429/5xx embed "
                             "transients per measure (30s escalating wait)")
    args = parser.parse_args()
    validate_output_args(args)

    load_secrets_env()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.method_check or "clean"
    raw_path = out_dir / f"precheck_langmem_{tag}.csv"

    fh = raw_path.open("w", encoding="utf-8", newline="")
    state = {"writer": None}

    def sink(row):
        # incremental write: a long paced sweep must not lose its table
        # to a crash after the last measure
        if state["writer"] is None:
            state["writer"] = csv.DictWriter(fh, fieldnames=list(row.keys()))
            state["writer"].writeheader()
        state["writer"].writerow(row)
        fh.flush()

    try:
        rows = asyncio.run(run_sweep(args, sink))
    finally:
        fh.close()

    if not rows:
        raise SystemExit("sweep produced no rows")
    summary = summarize(rows)
    sum_path = out_dir / f"precheck_langmem_{tag}_summary.csv"
    with sum_path.open("w", encoding="utf-8", newline="") as sfh:
        writer = csv.DictWriter(sfh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"raw -> {raw_path}\nsummary -> {sum_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
