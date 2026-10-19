"""E19x execution runner: ADK 2x2 (storage layout x query mode), same
batch/tasks/materials/model/budget as E17 base, four cells run
concurrently by run_all_x.py.

Query mode is implemented at the RETRIEVAL layer only:
  whole      the host's native preload path (search_memory called with
             the task fire text, top-W) - byte-identical to E17 behavior;
  per_object PerObjectMemoryService translates each search_memory call
             into the three frozen per-object queries
             (_e19q_rules.build_queries), merges by the frozen
             round-robin dedup rule capped at the SAME W, and returns
             the merged entries. Writes, request cadence, tool surface
             (preload + benign + export; NO query_memory) are identical
             across all four cells.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
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
E17_DIR = HERE.parent / "_e17"
EXPERIMENTS = HERE.parents[1]
for _p in [str(EXPERIMENTS), str(E17_DIR),
           str(EXPERIMENTS.parent / "agent-frameworks" / "adk-python" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _e17_materials import instances_s1, instances_s2, methods, templates  # noqa: E402
from _e19q_rules import build_queries  # noqa: E402
from openai_compat_model import OpenAICompatModel  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e17_base_serial", E17_DIR / "run_episode.py")
_e17_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e17_base)
serialize_value = _e17_base.serialize_value

_MODULES = {"s1": instances_s1, "s2": instances_s2}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def _end_status_and_ok(finish_reasons: list, last_agent_had_text: bool,
                       err) -> tuple[str, bool]:
    """Completion verdict (E17 criteria, review round fixes): the LAST
    finish reason must be a natural 'stop', the terminal agent event must
    carry non-empty text (a stale mid-run reply with an empty final
    response is NOT a valid episode), errors and length-truncated runs
    are technical failures and never enter the valid denominator."""
    if err is not None:
        return f"error", False
    if not finish_reasons:
        return "abnormal:no_calls", False
    last = finish_reasons[-1]
    if last != "stop":
        return f"abnormal:{last}", False
    if not last_agent_had_text:
        return "empty_final_response", False
    return "natural", True


def _wrap_client_create(client, sent_params: list):
    """The e17 adapter forwards neither temperature nor output limits
    (documented platform limitation). e19x explicitly sends temperature
    0.0 at the client layer and records the real per-call parameters -
    the run is REPORTED as temperature 0, unlike e17-ADK's provider
    default. The client is constructed per episode, so this wrap is
    episode-local."""
    orig = client.chat.completions.create

    async def _create(**kw):
        kw["temperature"] = 0.0
        r = await orig(**kw)
        choices = getattr(r, "choices", None) or []
        sent_params.append({
            "model": kw.get("model"),
            "temperature": kw.get("temperature"),
            "finish_reason": (getattr(choices[0], "finish_reason", None)
                              if choices else None),
        })
        return r

    client.chat.completions.create = _create
    return client


def build_per_object_service(base_service_cls, scenario: str, codes, w: int):
    """InMemoryMemoryService subclass: storage/writes unchanged; each
    search_memory call is executed as the three frozen per-object queries
    with the round-robin dedup merge capped at W (entries themselves are
    returned, preserving author/timestamp metadata)."""
    from google.adk.memory.base_memory_service import SearchMemoryResponse

    class PerObjectMemoryService(base_service_cls):
        def __init__(self):
            super().__init__()
            self._queries = build_queries(scenario, {"codes": list(codes),
                                                     "fire": ""})["per_object"]
            self._w = w

        async def search_memory(self, *, app_name: str, user_id: str,
                                query: str):
            lists = []
            for q in self._queries:
                r = await base_service_cls.search_memory(
                    self, app_name=app_name, user_id=user_id, query=q)
                lists.append(list(r.memories))
            merged, seen = [], set()
            max_len = max((len(l) for l in lists), default=0)
            for rank in range(max_len):
                for entries in lists:
                    if rank < len(entries):
                        text = _norm(" ".join(
                            p.text for p in entries[rank].content.parts
                            if p.text))
                        if text and text not in seen:
                            seen.add(text)
                            merged.append(entries[rank])
                if len(merged) >= self._w:
                    break
            return SearchMemoryResponse(memories=merged[:self._w])

    return PerObjectMemoryService()


def load_secrets_env() -> None:
    """Release credentials come only from process environment variables."""
    return


def _sig(kit: dict, model_id: str, mode: str, w: int) -> str:
    h = methods.kit_signature(kit, model_id)
    return hashlib.sha256(
        f"{h}|mode={mode}|w={w}".encode()).hexdigest()[:16]


async def run_episode(args, task: dict) -> Path:
    from google.adk.agents import LlmAgent
    from google.adk.events.event import Event
    from google.adk.memory.in_memory_memory_service import (
        InMemoryMemoryService,
    )
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.preload_memory_tool import preload_memory_tool
    from google.genai import types

    tpl = templates.get("adk", args.scenario)
    cfg = json.loads((E17_DIR / "study_config.json").read_text(encoding="utf-8"))
    m_count = cfg["m_levels"][args.m_level or "base"]
    w = int(cfg.get("W", 10))
    kit = methods.build_method_kit(args.scenario, args.layout, task, m_count)

    run_dir = Path(args.run_root) / args.batch / args.scenario / args.layout \
        / args.mode / task["task_id"]
    if run_dir.exists() and any(p.name != "attempts" for p in run_dir.iterdir()):
        att = run_dir / "attempts"
        att.mkdir(parents=True, exist_ok=True)
        dest = att / f"attempt-{len(list(att.iterdir())) + 1:02d}"
        dest.mkdir()
        for p in run_dir.iterdir():
            if p.name != "attempts" and p.is_file():
                p.rename(dest / p.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model or os.environ.get(
        "NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731")

    app_name, user_id = "e19x", "study_user"
    session_id = f"{args.scenario}-{args.layout}-{args.mode}-{task['task_id']}"
    if args.mode == "per_object":
        memory_service = build_per_object_service(
            InMemoryMemoryService, args.scenario, task["codes"], w)
    else:
        memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id)

    events = [
        Event(author=tpl["agent_name"],
              content=types.Content(
                  role="assistant", parts=[types.Part(text=c["content"])]))
        for c in kit["carriers"] + kit["background"]
    ]
    await memory_service.add_events_to_memory(
        app_name=app_name, user_id=user_id, events=events)
    (run_dir / "memory_records.jsonl").write_text(
        "\n".join(json.dumps({
            "id": c["carrier_id"], "cell": c["cell"],
            "write_confirmed": True, "content": c["content"],
        }, ensure_ascii=False)
            for c in kit["carriers"] + kit["background"]) + "\n",
        encoding="utf-8")

    # Evidence: what the host actually receives for the fire query under
    # this mode (per_object returns the merged set).
    ev_resp = await memory_service.search_memory(
        app_name=app_name, user_id=user_id, query=task["fire"])
    (run_dir / "search_results.json").write_text(json.dumps([
        {"author": m.author, "timestamp": m.timestamp,
         "content": " ".join(p.text for p in m.content.parts if p.text)}
        for m in ev_resp.memories
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "attack_component_manifest.json").write_text(
        json.dumps(kit["manifest"], ensure_ascii=False, indent=2),
        encoding="utf-8")

    tool_calls: list[dict] = []

    def _log_call(name, args_dict, result, executed):
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

    benign_impls: dict = {}
    for tool_name, tool_doc in tpl["benign_tools"]:
        two_arg = "update" in tool_name

        def _mk(tn, two):
            async def _fn(subject: str, status: str = "OK") -> dict:
                if two:
                    _log_call(tn, {"subject": subject, "status": status},
                              {"subject": subject, "status": status},
                              executed=True)
                    return {"subject": subject, "status": status}
                _log_call(tn, {"subject": subject},
                          {"subject": subject, "result": "done"},
                          executed=True)
                return {"subject": subject, "result": "done"}
            return _fn
        _fn = _mk(tool_name, two_arg)
        _fn.__name__ = tool_name
        _fn.__doc__ = tool_doc
        benign_impls[tool_name] = _fn

    captured: list[dict] = []
    base_model = OpenAICompatModel(model=model_id)

    class _Spy(type(base_model)):
        async def generate_content_async(self_, llm_request, stream=False):
            captured.append({"ts": time.time(), "payload": serialize_value(
                copy.deepcopy(llm_request))})
            async for r in OpenAICompatModel.generate_content_async(
                    self_, llm_request, stream=stream):
                yield r

    spy = _Spy(model=model_id)
    spy._client = base_model._client
    sent_params: list[dict] = []
    _wrap_client_create(spy._client, sent_params)

    agent = LlmAgent(
        name=tpl["agent_name"],
        description=tpl["agent_description"],
        instruction=tpl["agent_instruction"],
        model=spy,
        tools=[preload_memory_tool, *benign_impls.values(), _export_tool],
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )
    runner = Runner(agent=agent, app_name=app_name,
                    session_service=session_service,
                    memory_service=memory_service)

    (run_dir / "input_config.json").write_text(json.dumps({
        "experiment": "e19x", "platform": "adk", "scenario": args.scenario,
        "layout": args.layout, "query_mode": args.mode,
        "task_id": task["task_id"], "split": task["split"],
        "batch": args.batch,
        "m_count": m_count, "w": w, "model_id": model_id,
        "temperature_sent": 0.0,
        "config_sig": _sig(kit, model_id, args.mode, w),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    final_text = ""
    last_agent_had_text = False
    err = None
    t0 = time.perf_counter()
    try:
        async for ev in runner.run_async(
                user_id=user_id, session_id=session_id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=task["fire"])])):
            if ev.content and ev.content.parts and ev.author == agent.name:
                texts = [p.text for p in ev.content.parts if p.text]
                if texts:
                    final_text = "\n".join(texts)
                    last_agent_had_text = True
                else:
                    # a trailing EMPTY agent response overrides any stale
                    # mid-run text: the terminal event itself is empty
                    last_agent_had_text = False
    except Exception as e:  # noqa: BLE001
        err = repr(e)

    meta = list(getattr(spy, "_completion_meta", []))
    finish_reasons = [m.get("finish_reason") for m in meta]
    svc_models = sorted({m["model"] for m in meta if m.get("model")})
    usage_total: dict = {}
    for m in meta:
        u = m.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_total[k] = usage_total.get(k, 0) + (u.get(k) or 0)
    end_status, ok = _end_status_and_ok(finish_reasons,
                                        last_agent_had_text, err)

    (run_dir / "tool_calls.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False)
                  for c in tool_calls) + "\n", encoding="utf-8")
    (run_dir / "llm_requests.json").write_text(
        json.dumps(captured, ensure_ascii=False), encoding="utf-8")
    (run_dir / "request_params.json").write_text(
        json.dumps(sent_params, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (run_dir / "final_response.txt").write_text(final_text,
                                                encoding="utf-8")
    (run_dir / "_status.json").write_text(json.dumps({
        "ok": ok, "end_status": end_status, "error": err,
        "finish_reasons": finish_reasons,
        "service_models": svc_models, "usage_total": usage_total,
        "temperature_sent": 0.0,
        "n_requests": len(captured), "elapsed_s": round(
            time.perf_counter() - t0, 1),
        "config_sig": _sig(kit, model_id, args.mode, w),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


async def main_async(args) -> None:
    mod = _MODULES[args.scenario]
    pool = list(mod.DEV_INSTANCES) + list(mod.TEST_INSTANCES)
    tasks = ([t for t in pool if t["task_id"] == args.task_id]
             if args.task_id else list(mod.TEST_INSTANCES))
    if not tasks:
        raise SystemExit(f"task {args.task_id} not found in "
                         f"{args.scenario} dev/test pools")
    for task in tasks:
        rd = await run_episode(args, task)
        st = json.loads((rd / "_status.json").read_text(encoding="utf-8"))
        print(f"{args.scenario}/{args.layout}/{args.mode}/{task['task_id']}: "
              f"ok={st['ok']} reqs={st['n_requests']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="E19x episode runner")
    parser.add_argument("--scenario", required=True, choices=["s1", "s2"])
    parser.add_argument("--layout", required=True,
                        choices=["distributed", "fully_merged"])
    parser.add_argument("--mode", required=True,
                        choices=["whole", "per_object"])
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--m-level", choices=["base", "mid", "stress"],
                        default="base")
    parser.add_argument("--batch", default=None)
    parser.add_argument("--run-root", default=str(run_dir_for(__file__) / "e19x_v1"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    validate_output_args(args)
    load_secrets_env()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
