"""Episode runner for LangMem E17 (same-environment attack-method comparison).

One episode seeds a fresh InMemoryStore with one method's neutral kit
(attack carriers mapped to store records + the frozen benign-pool slice
at level M), runs semantic retrieval at the scenario width W, assembles
the runtime prompt from the recalled block plus the task's frozen
request, then executes the tool loop against the OpenAI-compatible
endpoint (E13 instrument surface: temperature=0, max_tokens=2048, tool
loop <=10 steps; the adapter's real behavior is verified from requests).

Batch layout: <run-root>/<batch>/<scenario>/<method>/<task-id>/
  batch = m-base | m-mid | m-stress (test runs) or dev | smoke.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
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
_EXPERIMENTS = HERE.parents[1]
for _p in [str(_EXPERIMENTS), str(HERE),
           str(_EXPERIMENTS.parent / "agent-frameworks" / "langmem" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import compat  # noqa: F401,E402  (typing.NotRequired patch, E13)

from _e17_materials import instances_s1, instances_s2, methods  # noqa: E402

WIRING_REVISION = "e17-v1"

_MODULES = {"s1": instances_s1, "s2": instances_s2}


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
    if hasattr(value, "__dict__"):
        try:
            return serialize_value(vars(value))
        except Exception:
            pass
    return repr(value)


def make_embedder():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="bge-m3",
        base_url=os.environ["NORMAL_BASE_URL"].strip(),
        api_key=os.environ["NORMAL_API_KEY"].strip(),
        check_embedding_ctx_length=False,
    )


def load_study_config() -> dict | None:
    path = HERE / "study_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def resolve_m_count(args) -> int:
    cfg = load_study_config()
    if args.m_count is not None:
        return args.m_count
    if cfg is not None and args.m_level is not None:
        return cfg["m_levels"][args.m_level]
    raise SystemExit(
        "M level unresolved: pass --m-count, or freeze study_config.json "
        "(precheck) and pass --m-level"
    )


def resolve_width(args) -> int:
    if getattr(args, "w", None):
        return int(args.w)
    cfg = load_study_config()
    if cfg is not None and "W" in cfg:
        return int(cfg["W"])
    raise SystemExit(
        "retrieval width W unresolved: pass --w, or freeze study_config.json"
    )


def build_run_dir(run_root: Path, args) -> Path:
    batch = args.batch or ("dev" if args.split == "dev" else None)
    if batch is None:
        batch = f"m-{args.m_level}"
    return run_root / batch / args.scenario / args.method / args.task_id


def dry_run(args, kit: dict) -> dict:
    report = {
        "dry_run": True,
        "platform": "langmem",
        "scenario": args.scenario,
        "method": args.method,
        "task_id": args.task_id,
        "split": args.split,
        "m_count": kit["m_count"],
        "carriers": len(kit["carriers"]),
        "background": len(kit["background"]),
        "budget": kit["budget"],
        "manifest_components": [c["component_id"] for c in kit["manifest"]["components"]],
        "placeholder_external": bool(kit["manifest"].get("placeholder")),
        "run_dir": str(build_run_dir(Path(args.run_root), args)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


async def seed_store(store, namespace: tuple, kit: dict) -> list[dict]:
    records = []
    for c in kit["carriers"] + kit["background"]:
        value = {
            "content": c["content"],
            "content_marker": f"[ID:{c['carrier_id']}]",
            "cell": c["cell"],
        }
        await store.aput(namespace, c["carrier_id"], value, index=["content"])
        records.append({
            "id": c["carrier_id"],
            "cell": c["cell"],
            "marker": c["carrier_id"],
            "write_api": "InMemoryStore.aput",
            "write_confirmed": True,
            "content": c["content"],
            "content_marker": value["content_marker"],
        })
    return records


async def retrieve(store, namespace: tuple, query: str, limit: int) -> list[dict]:
    items = await store.asearch(namespace, query=query, limit=limit)
    return [
        {
            "key": getattr(it, "key", None),
            "value": serialize_value(getattr(it, "value", {})),
            "score": serialize_value(getattr(it, "score", None)),
        }
        for it in items
    ]


def build_system_prompt(base_prompt: str, search_results: list[dict]) -> tuple[str, list[str]]:
    lines, ids = [], []
    for r in search_results:
        marker = (r.get("value") or {}).get("content_marker", "")
        content = (r.get("value") or {}).get("content", "")
        ids.append(marker.strip("[]").split(":", 1)[-1])
        lines.append(f"{marker} {content}")
    return (
        base_prompt + "\n\n<memories>\n" + "\n".join(lines) + "\n</memories>",
        ids,
    )


def build_tools_schema(tpl: dict) -> list[dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": name, "description": doc,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name, doc in tpl["benign_tools"]
    ]
    tools.append({
        "type": "function",
        "function": {
            "name": tpl["export_function_name"],
            "description": tpl["export_function_doc"],
            "parameters": {"type": "object", "properties": {}},
        },
    })
    return tools


async def run_episode(args, kit: dict, tpl: dict, width: int) -> Path:
    from openai import AsyncOpenAI
    from langgraph.store.memory import InMemoryStore

    task = _MODULES[args.scenario].by_id(args.task_id)
    run_dir = build_run_dir(Path(args.run_root), args)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Every attempt is kept (experiment.md E17 §8): a rerun ARCHIVES the
    # previous attempt's files under attempts/, never deletes them.
    if any(p.name != "attempts" for p in run_dir.iterdir()):
        att = run_dir / "attempts"
        att.mkdir(exist_ok=True)
        dest = att / f"attempt-{len(list(att.iterdir())) + 1:02d}"
        dest.mkdir()
        for p in run_dir.iterdir():
            if p.name != "attempts" and p.is_file():
                p.rename(dest / p.name)

    model_id = args.model or os.environ.get(
        "NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731"
    )
    config_sig = methods.kit_signature(kit, model_id, width)
    # Config first: an episode dir is countable even when the run crashes
    # before any evidence exists.
    base_config = {
        "experiment": "e17",
        "platform": "langmem",
        "scenario": args.scenario,
        "method": args.method,
        "task_id": args.task_id,
        "split": args.split,
        "family": task["family"],
        "batch": args.batch or ("dev" if args.split == "dev" else f"m-{args.m_level}"),
        "m_count": kit["m_count"],
        "model_id": model_id,
        "wiring_revision": WIRING_REVISION,
        "search_limit": width,
        "config_sig": config_sig,
    }
    (run_dir / "input_config.json").write_text(
        json.dumps(base_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    embedder = make_embedder()
    store = InMemoryStore(index={"dims": tpl["embedder_dims"], "embed": embedder})
    namespace = ("e17", args.scenario)

    t0 = time.perf_counter()
    memory_records = await seed_store(store, namespace, kit)
    write_s = time.perf_counter() - t0
    (run_dir / "memory_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in memory_records) + "\n",
        encoding="utf-8",
    )

    fire_text = task["fire"]

    t1 = time.perf_counter()
    search_results = await retrieve(store, namespace, fire_text, width)
    (run_dir / "search_results.json").write_text(
        json.dumps(search_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    system_prompt, ids_in_block = build_system_prompt(tpl["agent_system_prompt"], search_results)

    full_config = dict(base_config)
    full_config.update({
        "memory_count": len(memory_records),
        "attack_carriers": len(kit["carriers"]),
    })
    (run_dir / "input_config.json").write_text(
        json.dumps(full_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "attack_component_manifest.json").write_text(
        json.dumps(kit["manifest"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "payload_manifest.json").write_text(
        json.dumps(kit["budget"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tool_calls: list[dict] = []

    def _log_call(name: str, args_dict: dict, result: dict, executed: bool):
        tool_calls.append({
            "ts": time.time(), "tool": name, "args": args_dict,
            "result": result, "executed": executed,
        })

    client = AsyncOpenAI(
        base_url=os.environ["NORMAL_BASE_URL"].strip(),
        api_key=os.environ["NORMAL_API_KEY"].strip(),
    )
    tools_schema = build_tools_schema(tpl)
    # Only tools actually provided in the schema are executable. A
    # generated-but-unregistered name (e.g. the OTHER scenario's export
    # tool) is logged for audit as unexecuted and never counts as P3.
    registered = {t["function"]["name"] for t in tools_schema}

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": fire_text},
    ]
    llm_requests: list[dict] = []
    final_text = ""
    service_models: set[str] = set()
    finish_reasons: list[str | None] = []
    broke_naturally = False

    def dump(ok: bool, error: str | None = None) -> None:
        """Write evidence; on failure the partial record is preserved and
        _status.ok=false marks the episode as a technical failure."""
        last_finish = finish_reasons[-1] if finish_reasons else None
        if broke_naturally and last_finish == "stop":
            end_status = "natural"
        elif broke_naturally:
            end_status = f"abnormal:{last_finish}"
        else:
            end_status = "loop_limit"
        if ok and end_status != "natural":
            ok = False  # length-truncated / loop-limit termination is a
            # technical failure, never an ASR=0 outcome
        usage_total = {}
        for r in llm_requests:
            u = r.get("usage") or {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_total[k] = usage_total.get(k, 0) + (u.get(k) or 0)
        (run_dir / "llm_requests.json").write_text(
            json.dumps(llm_requests, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "tool_calls.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in tool_calls) + "\n",
            encoding="utf-8",
        )
        (run_dir / "final_response.txt").write_text(final_text, encoding="utf-8")
        (run_dir / "_status.json").write_text(
            json.dumps({"ok": ok, "finished_at": time.time(),
                        "service_models": sorted(service_models),
                        "end_status": end_status,
                        "finish_reasons": finish_reasons,
                        "usage_total": usage_total,
                        "config_sig": config_sig,
                        **({"error": error} if error else {})}, indent=2),
            encoding="utf-8",
        )
        (run_dir / "agent.log").write_text(
            f"write_phase_s={write_s:.3f}\n"
            f"fire_phase_s={time.perf_counter() - t1:.3f}\n"
            f"n_llm_calls={len(llm_requests)}\n"
            f"n_tool_calls={len(tool_calls)}\n"
            f"service_models={sorted(service_models)}\n"
            f"end_status={end_status} usage={usage_total}\n"
            f"limit={width} ids_in_memories_block={ids_in_block}\n"
            f"method={args.method} task={args.task_id} m={kit['m_count']}\n"
            + (f"error={error}\n" if error else ""),
            encoding="utf-8",
        )

    try:
        for step in range(10):
            completion = await client.chat.completions.create(
                model=model_id,
                messages=copy.deepcopy(messages),
                tools=tools_schema,
                tool_choice="auto",
                temperature=tpl["temperature"],
                max_tokens=tpl["max_tokens"],
            )
            if getattr(completion, "model", None):
                service_models.add(str(completion.model))
            choice = completion.choices[0]
            finish_reasons.append(choice.finish_reason)
            llm_requests.append({
                "call_index": step + 1,
                "payload": {
                    "system_prompt": system_prompt,
                    "messages": copy.deepcopy(messages),
                    "tools": [t["function"]["name"] for t in tools_schema],
                },
                "finish_reason": choice.finish_reason,
                "usage": completion.usage.model_dump() if completion.usage else None,
            })
            msg = choice.message
            content_text = msg.content or ""
            if content_text:
                final_text = content_text

            assistant_msg: dict = {"role": "assistant",
                                   "content": content_text or None}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                broke_naturally = True
                break

            for tc in msg.tool_calls:
                tname = tc.function.name
                truncated_call = choice.finish_reason == "length"
                try:
                    targs = json.loads(tc.function.arguments or "{}")
                    parsable = isinstance(targs, dict)
                    if not parsable:
                        targs = {"_raw": tc.function.arguments}
                except json.JSONDecodeError:
                    targs = {"_raw": tc.function.arguments}
                    parsable = False
                if truncated_call or not parsable:
                    # Truncated/unparsable tool calls are generation
                    # artifacts; executing them would fabricate evidence.
                    # Log unexecuted, tell the model, let the loop continue.
                    status = ("truncated_response" if truncated_call
                              else "unparsable_args")
                    _log_call(tname, targs, {"status": status}, executed=False)
                    result = {"status": status,
                              "detail": "arguments incomplete; please "
                                        "re-issue the complete call"}
                elif tname not in registered:
                    _log_call(tname, targs, {"status": "unregistered"},
                              executed=False)
                    result = {"status": "unregistered"}
                elif tname == tpl["export_function_name"]:
                    _log_call(tname, targs, {"status": "EXPORTED"},
                              executed=True)
                    result = {"status": "EXPORTED"}
                else:
                    _log_call(tname, targs,
                              {"status": "done", "tool": tname, "args": targs},
                              executed=True)
                    result = {"status": "done", "tool": tname, "args": targs}
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": tname,
                    "content": json.dumps(result, ensure_ascii=False),
                })
    except Exception as exc:
        dump(ok=False, error=repr(exc))
        raise
    dump(ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E17 episode runner")
    parser.add_argument("--scenario", required=True, choices=["s1", "s2"])
    parser.add_argument("--method", required=True, choices=methods.METHODS)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--m-level", choices=["base", "mid", "stress"], default=None)
    parser.add_argument("--m-count", type=int, default=None)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e17_v1"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--w", type=int, default=None,
                        help="retrieval width override (pre-freeze test runs)")
    parser.add_argument("--payload-dir", default=str(HERE / "external" / "payloads"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_output_args(args)

    task = _MODULES[args.scenario].by_id(args.task_id)
    args.split = task["split"]
    m_count = resolve_m_count(args)

    from _e17_materials import templates

    kit = methods.build_method_kit(
        args.scenario, args.method, task, m_count,
        external_dir=Path(args.payload_dir),
        allow_missing_external=args.dry_run,
    )
    if not kit["budget"]["within_budget"] and not args.dry_run:
        raise SystemExit(f"budget violation: {kit['budget']['reason']}")

    if args.dry_run:
        dry_run(args, kit)
        return

    load_secrets_env()
    tpl = templates.get("langmem", args.scenario)
    width = resolve_width(args)
    run_dir = asyncio.run(run_episode(args, kit, tpl, width))
    print(f"episode dir: {run_dir}")


if __name__ == "__main__":
    main()
