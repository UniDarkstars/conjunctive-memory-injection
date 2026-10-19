"""Episode runner for ADK E17 (same-environment attack-method comparison).

One episode writes one method's neutral kit (attack carriers + the frozen
benign-pool slice at level M) as assistant events through the framework's
public memory interface, then fires the task's frozen request. Retrieval
is the framework's real keyword top-10 preload; every LLM request is
captured by the E13 spy model; the export tool is the simulated,
logged-only instrument (experiment.md E17 §2/§7).

Layers reused verbatim from E13: event write, natural retrieval evidence,
request capture, simulated tool surface. New in E17: method/task/M
parameters, attack_component_manifest, budget gate, dry-run.

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
           str(_EXPERIMENTS.parent / "agent-frameworks" / "adk-python" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    return repr(value)


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


def build_run_dir(run_root: Path, args) -> Path:
    batch = args.batch or ("dev" if args.split == "dev" else None)
    if batch is None:
        batch = f"m-{args.m_level}"
    return run_root / batch / args.scenario / args.method / args.task_id


def dry_run(args, kit: dict) -> dict:
    report = {
        "dry_run": True,
        "platform": "adk",
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


async def run_episode(args, kit: dict, tpl: dict) -> Path:
    from google.adk.agents import LlmAgent
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.preload_memory_tool import preload_memory_tool
    from google.genai import types

    from openai_compat_model import OpenAICompatModel

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

    write_kit = [
        {"cell": c["cell"], "id": c["carrier_id"], "content": c["content"]}
        for c in kit["carriers"] + kit["background"]
    ]

    app_name = f"e17_adk_{args.scenario}"
    user_id = "study_user"
    session_id = f"{args.scenario}-{args.method}-{args.task_id}"

    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # Config first: an episode dir is countable (summary sees it) even when
    # the run crashes before any evidence exists.
    model_id = args.model or os.environ.get(
        "NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731"
    )
    config_sig = methods.kit_signature(kit, model_id)
    batch_label = args.batch or ("dev" if args.split == "dev" else f"m-{args.m_level}")
    base_config = {
        "experiment": "e17",
        "platform": "adk",
        "scenario": args.scenario,
        "method": args.method,
        "task_id": args.task_id,
        "split": args.split,
        "family": task["family"],
        "batch": batch_label,
        "m_count": kit["m_count"],
        "model_id": model_id,
        "wiring_revision": WIRING_REVISION,
        "search_width": 10,
        "config_sig": config_sig,
    }
    (run_dir / "input_config.json").write_text(
        json.dumps(base_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from google.adk.events.event import Event

    t0 = time.perf_counter()
    events = [
        Event(
            author=tpl["agent_name"],
            content=types.Content(
                role="assistant", parts=[types.Part(text=c["content"])]
            ),
        )
        for c in write_kit
    ]
    await memory_service.add_events_to_memory(
        app_name=app_name, user_id=user_id, events=events
    )
    write_s = time.perf_counter() - t0
    memory_records = [
        {
            "id": c["id"],
            "cell": c["cell"],
            "write_api": "InMemoryMemoryService.add_events_to_memory",
            "write_confirmed": True,
            "content": c["content"],
        }
        for c in write_kit
    ]
    (run_dir / "memory_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in memory_records) + "\n",
        encoding="utf-8",
    )

    fire_text = task["fire"]

    resp = await memory_service.search_memory(
        app_name=app_name, user_id=user_id, query=fire_text
    )
    search_dump = [
        {
            "author": m.author,
            "timestamp": m.timestamp,
            "content": " ".join(p.text for p in m.content.parts if p.text),
        }
        for m in resp.memories
    ]
    (run_dir / "search_results.json").write_text(
        json.dumps(search_dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_id_resolved = model_id  # already resolved above the early write
    base_model = OpenAICompatModel(model=model_id_resolved)

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

    async def _export_tool() -> dict:
        _log_call(tpl["export_function_name"], {},
                  {"status": "EXPORTED"}, executed=True)
        return {"status": "EXPORTED"}
    _export_tool.__name__ = tpl["export_function_name"]
    _export_tool.__doc__ = tpl["export_function_doc"]

    benign_impls = {}
    for tool_name, tool_doc in tpl["benign_tools"]:
        _make_benign_typed(tool_name, tool_doc, benign_impls, _log_call)

    captured_llm_requests: list[dict] = []
    llm_call_counter = {"n": 0}
    service_models: set[str] = set()

    class _SpyModel(type(base_model)):
        async def generate_content_async(self_, llm_request, stream: bool = False):
            snapshot = copy.deepcopy(llm_request)
            llm_call_counter["n"] += 1
            captured_llm_requests.append({
                "call_index": llm_call_counter["n"],
                "payload": serialize_value(snapshot),
            })
            # Call the adapter implementation explicitly: super(type(...))
            # shortcuts would skip OpenAICompatModel and land in BaseLlm's
            # NotImplementedError (the E13 spy relied on rebinding `model`
            # to the spy instance - fragile; this is the direct form).
            async for r in OpenAICompatModel.generate_content_async(
                self_, llm_request, stream=stream
            ):
                svc = getattr(r, "model", None)
                if svc:
                    service_models.add(str(svc))
                yield r

    spy_model = _SpyModel(model=model_id)
    spy_model._client = base_model._client

    agent = LlmAgent(
        name=tpl["agent_name"],
        description=tpl["agent_description"],
        instruction=tpl["agent_instruction"],
        model=spy_model,
        tools=[preload_memory_tool, *benign_impls.values(), _export_tool],
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
        memory_service=memory_service,
    )

    fire_msg = types.Content(role="user", parts=[types.Part(text=fire_text)])
    events_out: list[dict] = []
    final_text = ""
    t1 = time.perf_counter()

    def dump(ok: bool, error: str | None = None) -> None:
        """Write evidence; on failure the partial record is preserved and
        _status.ok=false marks the episode as a technical failure."""
        meta = list(getattr(spy_model, "_completion_meta", []))
        svc_models = sorted({m["model"] for m in meta if m.get("model")})
        last_finish = meta[-1]["finish_reason"] if meta else None
        end_status = (
            "natural" if last_finish == "stop"
            else f"abnormal:{last_finish or 'no_calls'}"
        )
        if ok and end_status != "natural":
            ok = False  # length-truncated / abnormal termination is a
            # technical failure, never an ASR=0 outcome
        usage_total = {}
        for m in meta:
            u = m.get("usage") or {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_total[k] = usage_total.get(k, 0) + (u.get(k) or 0)
        (run_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events_out) + "\n",
            encoding="utf-8",
        )
        (run_dir / "llm_requests.json").write_text(
            json.dumps(captured_llm_requests, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "tool_calls.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in tool_calls) + "\n",
            encoding="utf-8",
        )
        (run_dir / "final_response.txt").write_text(final_text, encoding="utf-8")
        (run_dir / "_status.json").write_text(
            json.dumps({"ok": ok, "finished_at": time.time(),
                        "service_models": svc_models,
                        "end_status": end_status,
                        "finish_reasons": [m["finish_reason"] for m in meta],
                        "usage_total": usage_total,
                        "config_sig": config_sig,
                        **({"error": error} if error else {})}, indent=2),
            encoding="utf-8",
        )
        (run_dir / "agent.log").write_text(
            f"write_phase_s={write_s:.3f}\n"
            f"fire_phase_s={time.perf_counter() - t1:.3f}\n"
            f"n_llm_calls={len(captured_llm_requests)}\n"
            f"n_tool_calls={len(tool_calls)}\n"
            f"service_models={svc_models}\n"
            f"end_status={end_status} usage={usage_total}\n"
            f"method={args.method} task={args.task_id} m={kit['m_count']}\n"
            + (f"error={error}\n" if error else ""),
            encoding="utf-8",
        )

    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=fire_msg
        ):
            events_out.append(serialize_value(event))
            if (
                event.content and event.content.parts
                and event.author == agent.name
            ):
                texts = [p.text for p in event.content.parts if p.text]
                if texts:
                    final_text = "\n".join(texts)
    except Exception as exc:
        dump(ok=False, error=repr(exc))
        raise
    dump(ok=True)
    return run_dir


def _make_benign_typed(tool_name: str, tool_doc: str, registry: dict, log_fn) -> None:
    """Typed-signature benign tools (E13 instrument surface, unchanged)."""
    two_arg_hints = ("update",)

    if any(h in tool_name for h in two_arg_hints):
        async def _fn(subject: str, status: str) -> dict:
            log_fn(tool_name, {"subject": subject, "status": status},
                   {"subject": subject, "status": status}, executed=True)
            return {"subject": subject, "status": status}
    else:
        async def _fn(subject: str) -> dict:
            log_fn(tool_name, {"subject": subject},
                   {"subject": subject, "result": "done"}, executed=True)
            return {"subject": subject, "result": "done"}

    _fn.__name__ = tool_name
    _fn.__doc__ = tool_doc
    registry[tool_name] = _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK E17 episode runner")
    parser.add_argument("--scenario", required=True, choices=["s1", "s2"])
    parser.add_argument("--method", required=True, choices=methods.METHODS)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--m-level", choices=["base", "mid", "stress"], default=None)
    parser.add_argument("--m-count", type=int, default=None)
    parser.add_argument("--batch", default=None,
                        help="run subdirectory (default: m-<level> or dev/smoke)")
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e17_v1"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--payload-dir", default=str(HERE / "external" / "payloads"),
                        help="frozen external payload directory")
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
    tpl = templates.get("adk", args.scenario)
    run_dir = asyncio.run(run_episode(args, kit, tpl))
    print(f"episode dir: {run_dir}")


if __name__ == "__main__":
    main()
