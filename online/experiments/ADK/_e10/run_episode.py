"""Episode runner for ADK E10 (independent task instances & conjunctive attribution).

Each episode loads one of 30 frozen task instances for the scenario and
renders one of four E1-style variant kits (clean / T-only / keys-only /
both) via the study module's `iter_variants(ucodes, inst)` helper. The
memory-write layout is 9 cells in a fixed order; cell ids and contents
come from the frozen study materials (no on-the-fly string patching).

Write order (both):
  f1, f2, T, f3, k1, r1, k2, k3, f4
Write order (clean / T-only / keys-only): same length with A/R stand-ins.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import importlib
import json
import logging
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


VENDORED_ADK_SRC = str(Path(__file__).resolve().parents[3] / "agent-frameworks" / "adk-python" / "src")
for _p in [VENDORED_ADK_SRC, str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.append(_p)


def load_secrets_env() -> None:
    """Release credentials come only from process environment variables."""
    return


EPISODE_TIMING = {"write_phase_s": 0.0, "fire_phase_s": 0.0, "fire_query_elapsed_ms": 0}


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


async def persist_event_writes(
    memory_service,
    app_name: str,
    user_id: str,
    kit: list[dict],
    agent_name: str,
) -> list[dict]:
    """Seed the variant kit via the framework public write interface."""
    from google.adk.events.event import Event
    from google.genai import types

    events = [
        Event(
            author=agent_name,
            content=types.Content(
                role="assistant",
                parts=[types.Part(text=c["content"])],
            ),
        )
        for c in kit
    ]
    t0 = time.perf_counter()
    await memory_service.add_events_to_memory(
        app_name=app_name, user_id=user_id, events=events
    )
    EPISODE_TIMING["write_phase_s"] = time.perf_counter() - t0

    memory_records = [
        {
            "id": c["id"],
            "cell": c["cell"],
            "write_api": "InMemoryMemoryService.add_events_to_memory",
            "write_confirmed": True,
            "content": c["content"],
        }
        for c in kit
    ]
    return memory_records


async def run_episode(
    scenario: str,
    variant: str,
    episode_index: int,
    run_root: Path,
    model_id: str | None = None,
) -> Path:
    sm = importlib.import_module(f"study_materials_{scenario.lower()}")

    inst = sm.INSTANCES[episode_index - 1]
    kits = sm.iter_variants(inst["codes"], inst)
    if variant not in kits:
        raise ValueError(f"unknown variant: {variant}")
    kit = kits[variant]

    run_dir = run_root / scenario / variant / f"epi-{episode_index:02d}"
    prepare_fresh_episode(run_dir)

    from google.adk.agents import LlmAgent
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.preload_memory_tool import preload_memory_tool
    from google.genai import types

    from openai_compat_model import OpenAICompatModel

    app_name = f"nkey_adk_e10_{scenario.lower()}"
    user_id = "study_user"
    session_id = f"{scenario}-{variant}-{episode_index:02d}"

    logger = logging.getLogger(f"nkey.adk_e10.{scenario.lower()}")
    logger.setLevel(logging.INFO)

    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    memory_records = await persist_event_writes(
        memory_service, app_name, user_id, kit, sm.AGENT_NAME
    )
    (run_dir / "memory_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in memory_records) + "\n",
        encoding="utf-8",
    )

    if model_id is None:
        model_id = os.environ.get("NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731")
    model = OpenAICompatModel(model=model_id)

    tool_calls: list[dict] = []

    def _log_call(name: str, args: dict):
        tool_calls.append({
            "ts": time.time(),
            "tool": name,
            "args": args,
            "result": {"status": "OK"},
        })

    async def _export_tool() -> dict:
        _log_call(sm.EXPORT_FUNCTION_NAME, {})
        return {"status": "EXPORTED"}
    _export_tool.__name__ = sm.EXPORT_FUNCTION_NAME
    _export_tool.__doc__ = sm.EXPORT_FUNCTION_DOC

    benign_impls = {}
    for tool_name, tool_doc in sm.BENIGN_TOOLS:
        _make_benign_typed(tool_name, tool_doc, benign_impls, _log_call)

    agent_tools = [
        preload_memory_tool,
        *benign_impls.values(),
        _export_tool,
    ]

    captured_llm_requests: list[dict] = []
    llm_call_counter = {"n": 0}

    class _SpyModel(type(model)):
        async def generate_content_async(self_, llm_request, stream: bool = False):
            snapshot = copy.deepcopy(llm_request)
            llm_call_counter["n"] += 1
            captured_llm_requests.append({
                "call_index": llm_call_counter["n"],
                "payload": serialize_value(snapshot),
            })
            async for resp in super(type(model), self_).generate_content_async(llm_request, stream=stream):
                yield resp

    spy_model = _SpyModel(model=model_id)
    spy_model._client = model._client
    model = spy_model

    agent = LlmAgent(
        name=sm.AGENT_NAME,
        description=sm.AGENT_DESCRIPTION,
        instruction=sm.AGENT_INSTRUCTION,
        model=model,
        tools=agent_tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
        memory_service=memory_service,
    )

    fire_text = inst["fire"]
    fire_msg = types.Content(role="user", parts=[types.Part(text=fire_text)])

    events: list[dict] = []
    final_text = ""

    log_path = run_dir / "agent.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)
    root_logger.setLevel(logging.INFO)

    t0 = time.perf_counter()
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=fire_msg
        ):
            events.append(serialize_value(event))
            if (
                event.content
                and event.content.parts
                and event.author == agent.name
            ):
                texts = [p.text for p in event.content.parts if p.text]
                if texts:
                    final_text = "\n".join(texts)
    finally:
        EPISODE_TIMING["fire_phase_s"] = time.perf_counter() - t0
        EPISODE_TIMING["fire_query_elapsed_ms"] = int(EPISODE_TIMING["fire_phase_s"] * 1000)
        root_logger.removeHandler(fh)
        fh.close()

    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
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

    return run_dir


def _make_benign_typed(
    tool_name: str, tool_doc: str, registry: dict, log_fn
) -> None:
    """Create a typed-signature async function for ADK function declaration."""
    no_arg_prefixes = ("summarize",)
    two_arg_hints = ("update",)

    is_no_arg = any(tool_name.startswith(p) for p in no_arg_prefixes)
    is_two_arg = any(h in tool_name for h in two_arg_hints)

    if is_no_arg:
        async def _fn() -> dict:
            log_fn(tool_name, {})
            return {"result": "done"}
    elif is_two_arg:
        async def _fn(subject: str, status: str) -> dict:
            log_fn(tool_name, {"subject": subject, "status": status})
            return {"subject": subject, "status": status}
    else:
        async def _fn(subject: str) -> dict:
            log_fn(tool_name, {"subject": subject})
            return {"subject": subject, "result": "done"}

    _fn.__name__ = tool_name
    _fn.__doc__ = tool_doc
    registry[tool_name] = _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK E10 episode runner")
    parser.add_argument("--scenario", required=True, choices=["s1", "s2"])
    parser.add_argument("--variant", required=True,
                        choices=["clean", "T-only", "keys-only", "both"])
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--run-root", default=str(run_dir_for(__file__)))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    validate_output_args(args)

    load_secrets_env()
    run_root = Path(args.run_root)
    run_dir = asyncio.run(run_episode(
        args.scenario, args.variant, args.episode_index, run_root,
        model_id=args.model,
    ))
    print(f"episode dir: {run_dir}")
    print(f"write_phase_s={EPISODE_TIMING['write_phase_s']:.3f} "
          f"fire_phase_s={EPISODE_TIMING['fire_phase_s']:.3f} "
          f"fire_query_elapsed_ms={EPISODE_TIMING['fire_query_elapsed_ms']}")


if __name__ == "__main__":
    main()
