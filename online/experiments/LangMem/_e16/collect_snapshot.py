"""Snapshot collection for LangMem E16 (embedding calls only; no chat
model, stops BEFORE the first main-model request).

Per e16_plan_v2: for every scenario x frozen task, seed a fresh
InMemoryStore with the E13 B0 kit (attack atoms + M=5 competitors, W=8
frozen), run the REAL semantic retrieval with the task's fire text, and
freeze the natural result list (keys, contents, markers, scores, order),
the trusted system prompt, the normal user task, the tool names, the
rendered <memories> block, the frozen added-record keys, and the
canonical-atom analysis (A_i / D_i by full-text matching).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
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


_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
import compat  # noqa: F401,E402

VENDORED_LANGMEM_SRC = str(_THIS.parents[1] / "agent-frameworks" / "langmem" / "src")
if VENDORED_LANGMEM_SRC not in sys.path:
    pass  # langgraph/langchain dependencies are installed from requirements-online.txt

import arms_core  # noqa: E402

SNAPSHOT_VERSION = "e16-snap-1"


def load_secrets_env() -> None:
    """Release credentials come only from process environment variables."""
    return


def render_block(entries: list[dict]) -> str:
    """Replicates the runner's <memories> block: '[marker] content' lines."""
    return "\n".join(f"{e['content_marker']} {e['content']}" for e in entries)


async def collect(scenario: str, episode_index: int, snap_dir: Path) -> Path:
    from langchain_openai import OpenAIEmbeddings
    from langgraph.store.memory import InMemoryStore

    sm = importlib.import_module(f"study_materials_{scenario.lower()}")
    inst = sm.INSTANCES[episode_index - 1]
    kit = sm.iter_baselines(inst["codes"], inst)["B0"]

    base_url = os.environ["NORMAL_BASE_URL"].strip()
    api_key = os.environ["NORMAL_API_KEY"].strip()
    embedder = OpenAIEmbeddings(model="bge-m3", base_url=base_url, api_key=api_key,
                                check_embedding_ctx_length=False)
    store = InMemoryStore(index={"dims": sm.EMBEDDER_DIMS, "embed": embedder})
    namespace = ("e16", scenario.lower())
    for c in kit:
        await store.aput(namespace, c["store_key"],
                         {"content": c["content"],
                          "content_marker": f"[ID:{c['marker']}]",
                          "cell": c["cell"]},
                         index=["content"])
    items = await store.asearch(namespace, query=inst["fire"], limit=sm.SEARCH_LIMIT)

    entries = [
        {"key": it.key,
         "content": (it.value or {}).get("content", ""),
         "content_marker": (it.value or {}).get("content_marker", ""),
         "score": float(it.score) if it.score is not None else None}
        for it in items
    ]
    block = render_block(entries)
    canonical = arms_core.canonical_atoms(sm, inst["codes"], inst)
    analysis = arms_core.analyze_block(block, canonical)

    snapshot = {
        "experiment": "e16", "platform": "langmem",
        "snapshot_version": SNAPSHOT_VERSION,
        "scenario": scenario, "episode_index": episode_index,
        "episode": f"epi-{episode_index:02d}", "codes": inst["codes"],
        "config": {"search_limit": sm.SEARCH_LIMIT, "n_competitors": sm.N_COMPETITORS},
        "agent_name": sm.AGENT_NAME,
        "system": sm.AGENT_SYSTEM_PROMPT,
        "user": inst["fire"],
        "tools": [t[0] for t in sm.BENIGN_TOOL_SPECS] + [sm.EXPORT_FUNCTION_NAME],
        "entries": entries,
        "block_preview": block,
        "added_keys": {a: f"add/{a}" for a in arms_core.ATOM_ORDER},
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
    load_secrets_env()
    snap_dir = Path(args.snap_dir)
    for scenario in args.scenarios:
        for i in range(1, args.episodes + 1):
            out = await collect(scenario, i, snap_dir)
            if i == 1 or i == args.episodes:
                print(f"collected {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E16 snapshot collection (embeddings only)")
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
