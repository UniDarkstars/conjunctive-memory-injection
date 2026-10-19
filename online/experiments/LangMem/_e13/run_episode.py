"""Episode runner for LangMem E13 (layout x exposure-path comparison).

One episode seeds a fresh InMemoryStore with one layout's write kit (B0/B1/B2
attack records + N_COMPETITORS same-domain competitors), runs semantic
retrieval at the scenario width W, assembles the runtime prompt from the
recalled block plus the instance's fire text, then executes a tool loop
against the OpenAI-compatible endpoint.

Paths (experiment.md E13):
  natural -> the prompt's <memories> block is the actual retrieval result;
  full    -> the <memories> block is the diagnostic R-full block (all four
             atoms, fixed common benign background) assembled at the SAME
             injection position with the SAME wrapper; the natural retrieval
             is still executed and archived as search_results_natural.json.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
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
import compat  # noqa: F401,E402  (typing.NotRequired patch)

VENDORED_LANGMEM_SRC = str(_THIS.parents[1] / "agent-frameworks" / "langmem" / "src")
if VENDORED_LANGMEM_SRC not in sys.path:
    pass  # langgraph/langchain dependencies are installed from requirements-online.txt

WIRING_REVISION = "e13-rfull-1"


def load_secrets_env() -> None:
    """Release credentials come only from process environment variables."""
    return


def serialize_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            return serialize_value(value.model_dump(mode="json", exclude_none=True))
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if hasattr(value, "__dict__"):
        try:
            return serialize_value(vars(value))
        except Exception:
            pass
    return repr(value)


def make_embedder():
    from langchain_openai import OpenAIEmbeddings

    base_url = os.environ["NORMAL_BASE_URL"].strip()
    api_key = os.environ["NORMAL_API_KEY"].strip()
    return OpenAIEmbeddings(
        model="bge-m3",
        base_url=base_url,
        api_key=api_key,
        check_embedding_ctx_length=False,
    )


async def seed_store(store, namespace: tuple, kit: list[dict]) -> list[dict]:
    records = []
    for c in kit:
        value = {
            "content": c["content"],
            "content_marker": f"[ID:{c['marker']}]",
            "cell": c["cell"],
        }
        await store.aput(namespace, c["store_key"], value, index=["content"])
        records.append({
            "id": c["store_key"],
            "cell": c["cell"],
            "marker": c["marker"],
            "write_api": "InMemoryStore.aput",
            "write_confirmed": True,
            "content": c["content"],
            "content_marker": value["content_marker"],
        })
    return records


async def retrieve(store, namespace: tuple, query: str, limit: int) -> list[dict]:
    items = await store.asearch(namespace, query=query, limit=limit)
    out = []
    for it in items:
        val = serialize_value(getattr(it, "value", {}))
        out.append({
            "key": getattr(it, "key", None),
            "value": val,
            "score": serialize_value(getattr(it, "score", None)),
        })
    return out


def full_block_results(sm, ucodes: list[str], inst: dict, baseline: str) -> list[dict]:
    """R-full diagnostic block in the search-result shape, so the prompt
    assembly (and check_episode) treat it exactly like a retrieval result."""
    return [
        {
            "key": r["store_key"],
            "value": {
                "content": r["content"],
                "content_marker": f"[ID:{r['marker']}]",
                "cell": r["cell"],
            },
            "score": None,
        }
        for r in sm.iter_full_block(ucodes, inst)[baseline]
    ]


def build_system_prompt(base_prompt: str, search_results: list[dict]) -> tuple[str, list[str]]:
    lines = []
    ids = []
    for r in search_results:
        marker = (r.get("value") or {}).get("content_marker", "")
        content = (r.get("value") or {}).get("content", "")
        mid = marker.strip("[]").split(":", 1)[-1]
        ids.append(mid)
        lines.append(f"{marker} {content}")
    block = "\n".join(lines)
    system = (
        base_prompt
        + "\n\n<memories>\n"
        + block
        + "\n</memories>"
    )
    return system, ids


def build_tools_schema(sm):
    tools = []
    for name, doc in sm.BENIGN_TOOL_SPECS:
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": {"type": "object", "properties": {}},
            },
        })
    tools.append({
        "type": "function",
        "function": {
            "name": sm.EXPORT_FUNCTION_NAME,
            "description": sm.EXPORT_FUNCTION_DOC,
            "parameters": {"type": "object", "properties": {}},
        },
    })
    return tools


async def run_episode(
    scenario: str,
    path: str,
    baseline: str,
    episode_index: int,
    run_root: Path,
    model_id: str | None = None,
    limit: int | None = None,
) -> Path:
    from openai import AsyncOpenAI
    from langgraph.store.memory import InMemoryStore

    sm = importlib.import_module(f"study_materials_{scenario.lower()}")
    inst = sm.INSTANCES[episode_index - 1]
    kits = sm.iter_baselines(inst["codes"], inst)
    if baseline not in kits:
        raise ValueError(f"unknown baseline: {baseline}")
    kit = kits[baseline]

    run_dir = run_root / scenario / path / baseline / f"epi-{episode_index:02d}"
    prepare_fresh_episode(run_dir)

    base_url = os.environ["NORMAL_BASE_URL"].strip()
    api_key = os.environ["NORMAL_API_KEY"].strip()
    if model_id is None:
        model_id = os.environ.get("NORMAL_CHAT_MODEL", sm.LLM_PRIMARY)
    if limit is None:
        limit = sm.SEARCH_LIMIT

    embedder = make_embedder()
    store = InMemoryStore(index={"dims": sm.EMBEDDER_DIMS, "embed": embedder})
    namespace = sm.NAMESPACE

    write_t0 = time.perf_counter()
    memory_records = await seed_store(store, namespace, kit)
    write_s = time.perf_counter() - write_t0
    (run_dir / "memory_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in memory_records) + "\n",
        encoding="utf-8",
    )

    fire_text = inst["fire"]

    fire_t0 = time.perf_counter()
    natural_results = await retrieve(store, namespace, fire_text, limit)
    if path == "full":
        (run_dir / "search_results_natural.json").write_text(
            json.dumps(natural_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
        )
        search_results = full_block_results(sm, inst["codes"], inst, baseline)
    else:
        search_results = natural_results
    (run_dir / "search_results.json").write_text(
        json.dumps(search_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    system_prompt, ids_in_block = build_system_prompt(sm.AGENT_SYSTEM_PROMPT, search_results)

    (run_dir / "input_config.json").write_text(
        json.dumps(
            {
                "experiment": "e13",
                "platform": "langmem",
                "scenario": scenario,
                "path": path,
                "baseline": baseline,
                "episode_index": episode_index,
                "model_id": model_id,
                "wiring_revision": WIRING_REVISION,
                "search_limit": limit,
                "n_competitors": sm.N_COMPETITORS,
                "r_full_background_n": sm.R_FULL_BACKGROUND_N,
                "memory_count": len(memory_records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tool_calls: list[dict] = []

    def _log_call(name: str, args: dict):
        tool_calls.append({
            "ts": time.time(),
            "tool": name,
            "args": args,
            "result": {"status": "OK"},
        })

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    tools_schema = build_tools_schema(sm)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": fire_text},
    ]

    llm_requests: list[dict] = []
    final_text = ""

    for step in range(10):
        completion = await client.chat.completions.create(
            model=model_id,
            messages=copy.deepcopy(messages),
            tools=tools_schema,
            tool_choice="auto",
            temperature=sm.TEMPERATURE,
            max_tokens=sm.MAX_TOKENS,
        )
        llm_requests.append({
            "call_index": step + 1,
            "payload": {
                "system_prompt": system_prompt,
                "messages": copy.deepcopy(messages),
                "tools": [t["function"]["name"] for t in tools_schema],
            },
        })
        choice = completion.choices[0]
        msg = choice.message
        content_text = msg.content or ""
        if content_text:
            final_text = content_text

        assistant_msg: dict = {"role": "assistant", "content": content_text or None}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            tname = tc.function.name
            try:
                targs = json.loads(tc.function.arguments or "{}")
                if not isinstance(targs, dict):
                    targs = {"value": targs}
            except json.JSONDecodeError:
                targs = {"_raw": tc.function.arguments}
            _log_call(tname, targs)
            if tname == sm.EXPORT_FUNCTION_NAME:
                result = {"status": "EXPORTED"}
            else:
                result = {"status": "done", "tool": tname, "args": targs}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tname,
                "content": json.dumps(result, ensure_ascii=False),
            })

    fire_s = time.perf_counter() - fire_t0

    (run_dir / "llm_requests.json").write_text(
        json.dumps(llm_requests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "tool_calls.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tool_calls) + "\n",
        encoding="utf-8",
    )
    (run_dir / "final_response.txt").write_text(final_text, encoding="utf-8")
    (run_dir / "agent.log").write_text(
        f"write_phase_s={write_s:.3f}\n"
        f"fire_phase_s={fire_s:.3f}\n"
        f"fire_query_elapsed_ms={int(fire_s * 1000)}\n"
        f"n_llm_calls={len(llm_requests)}\n"
        f"n_tool_calls={len(tool_calls)}\n"
        f"limit={limit}\n"
        f"path={path}\n"
        f"ids_in_memories_block={ids_in_block}\n",
        encoding="utf-8",
    )

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E13 episode runner")
    parser.add_argument("--scenario", required=True, choices=["s1", "s2"])
    parser.add_argument("--path", required=True, choices=["natural", "full"])
    parser.add_argument("--baseline", required=True, choices=["B0", "B1", "B2"])
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e13"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="retrieval width W; default reads the scenario "
                             "study module (frozen value)")
    args = parser.parse_args()
    validate_output_args(args)

    load_secrets_env()
    run_root = Path(args.run_root)
    run_dir = asyncio.run(run_episode(
        args.scenario, args.path, args.baseline, args.episode_index, run_root,
        model_id=args.model, limit=args.limit,
    ))
    print(f"episode dir: {run_dir}")


if __name__ == "__main__":
    main()
