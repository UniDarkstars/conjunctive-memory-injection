"""Episode runner for ADK E16 (missing-atom restoration in natural requests).

One episode loads the task's FROZEN natural snapshot and runs ONE arm:

  replay  (N) the snapshot's memory block verbatim, every request;
  restore (R) snapshot entries + canonical texts of the missing atoms
              appended at the end (fixed T,k1,k2,k3 order), every request;
  sham    (S) snapshot entries + the same number of benign records at the
              same position, length-matched per record (chars + newlines).

A ReplayMemoryService returns the frozen MemoryEntry list regardless of the
query, so the original PreloadMemoryTool injects them at the native
transient-user position with the native wrapper. Original entry metadata
(author/timestamp) is replayed verbatim - unlike E13's
build_full_exposure_service, nothing is regenerated; added records carry the
per-task frozen shared timestamp. The list is never re-ranked or truncated
(restoring may exceed the natural top-10). No store writes happen here: the
real write+retrieval path was already exercised at snapshot collection.

Arms start from an empty session and identical initial tool state; the
model/tool loop then runs freely. Per-request input checks verify each
arm's invariants before trusting the run.
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


VENDORED_ADK_SRC = str(Path(__file__).resolve().parents[3] / "agent-frameworks" / "adk-python" / "src")
for _p in [VENDORED_ADK_SRC, str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.append(_p)

_EXPERIMENTS = str(Path(__file__).resolve().parents[2])
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)
from _e16_runtime import (  # noqa: E402
    RUNTIME_REVISION, IncompleteRun, atomic_write_json, atomic_write_text,
    prepare_episode_dir, runtime_metadata,
)

import arms_core  # noqa: E402

MEMORY_BLOCK_RE = re.compile(
    r"<PAST_CONVERSATIONS>\n(.*?)\n</PAST_CONVERSATIONS>", re.DOTALL)

ARMS = ("replay", "restore", "sham")
ARM_LETTER = {"replay": "N", "restore": "R", "sham": "S"}
WIRING_REVISION = "e16-replay-1"


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


def build_replay_service(base_service_cls, entries):
    """Returns frozen MemoryEntry list for ANY query; no rerank/truncate."""
    from google.adk.memory.base_memory_service import SearchMemoryResponse

    class ReplayMemoryService(base_service_cls):
        def __init__(self, frozen_entries):
            super().__init__()
            self._frozen = frozen_entries

        async def search_memory(self, *, app_name: str, user_id: str, query: str):
            return SearchMemoryResponse(memories=list(self._frozen))

    return ReplayMemoryService(entries)


def arm_entries_from_snapshot(sm, snapshot: dict, arm: str):
    """Materialize the arm's MemoryEntry list from the frozen snapshot."""
    from google.adk.memory.memory_entry import MemoryEntry
    from google.genai import types

    def entry(author, timestamp, content):
        return MemoryEntry(
            content=types.Content(role="assistant",
                                  parts=[types.Part(text=content)]),
            author=author, timestamp=timestamp)

    natural = [entry(e["author"], e["timestamp"], e["content"])
               for e in snapshot["entries"]]
    if arm == "replay":
        return natural, {}, None

    canonical = snapshot["canonical_atoms"]
    missing = snapshot["atoms_missing"]
    additions = arms_core.build_arm_additions(
        missing, canonical, sm, snapshot["codes"])
    texts = additions[{"restore": "R", "sham": "S"}[arm]]
    added = [entry(sm.AGENT_NAME, snapshot["added_timestamp"], texts[a])
             for a in additions["missing_ordered"]]
    return natural + added, additions, texts


def check_request_block(block_text: str, arm: str, sm, snapshot: dict,
                        additions: dict | None) -> dict:
    """Per-request invariants; hard failures are construction bugs."""
    norm = arms_core.normalize_text(block_text)
    snap_norm = arms_core.normalize_text(snapshot["block_preview"])
    canonical = snapshot["canonical_atoms"]
    out = {
        "snapshot_prefix_intact": norm.startswith(snap_norm),
        "arm_label_absent": all(lbl not in norm for lbl in ("replay", "restore", "sham")),
    }
    if arm == "replay":
        out["block_equals_snapshot"] = norm == snap_norm
        ok = out["block_equals_snapshot"] and out["arm_label_absent"]
    elif arm == "restore":
        atoms = arms_core.analyze_block(block_text, canonical)
        out["all_atoms_present"] = all(atoms["evidence"].values())
        out["atoms"] = atoms
        ok = out["all_atoms_present"] and out["snapshot_prefix_intact"] and out["arm_label_absent"]
    else:  # sham
        s_texts = additions["S"] if additions else {}
        out["sham_texts_present"] = all(
            arms_core.normalize_text(t) in norm for t in s_texts.values())
        restored_leak = [a for a, t in canonical.items()
                         if a in snapshot["atoms_missing"]
                         and arms_core.normalize_text(t) in norm]
        out["restored_atom_leak"] = restored_leak
        added_region = norm[len(snap_norm):]
        out["no_forbidden_in_additions"] = not any(
            bad in added_region for bad in arms_core.FORBIDDEN_SUBSTRINGS)
        ok = (out["sham_texts_present"] and not restored_leak
              and out["snapshot_prefix_intact"] and out["no_forbidden_in_additions"]
              and out["arm_label_absent"])
    out["ok"] = ok
    if not ok:
        raise AssertionError(f"input check failed for arm {arm}: {out}")
    return out


async def run_episode(scenario: str, episode_index: int, snapshot_path: Path,
                      arm: str, run_root: Path, model_id: str | None) -> Path:
    snapshot_path = Path(snapshot_path).resolve()
    run_root = Path(run_root).resolve()
    sm = importlib.import_module(f"study_materials_{scenario.lower()}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (snapshot["scenario"] != scenario
            or snapshot["episode_index"] != episode_index
            or snapshot["platform"] != "adk"):
        raise ValueError(f"snapshot identity mismatch: {snapshot_path}")
    canonical_now = arms_core.canonical_atoms(
        sm, snapshot["codes"], sm.INSTANCES[episode_index - 1])
    if canonical_now != snapshot["canonical_atoms"]:
        raise ValueError("materials drifted from the frozen snapshot")

    if model_id is None:
        model_id = os.environ.get("NORMAL_CHAT_MODEL", "DeepSeek-V4-Flash-0731")
    run_dir = prepare_episode_dir(run_root, scenario, arm, episode_index, "adk", model_id)
    missing_atoms = {"missing": snapshot["atoms_missing"],
                     "present": snapshot["atoms_present"],
                     "missing_count": len(snapshot["atoms_missing"]),
                     "evidence": snapshot["atom_evidence"],
                     "snapshot": snapshot_path.name,
                     "snapshot_sha256": snapshot["snapshot_sha256"]}
    input_config = {
            "experiment": "e16", "platform": "adk",
            "scenario": scenario, "episode_index": episode_index,
            "arm": arm, "arm_letter": ARM_LETTER[arm],
            "snapshot": str(snapshot_path),
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "snapshot_version": snapshot["snapshot_version"],
            "config": snapshot["config"],
            "model_id": model_id,
            "wiring_revision": WIRING_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "batch_id": run_root.name,
            "note_temperature": "adapter does not forward temperature (recorded fact)",
            "note_output_limit": "adapter does not forward an output-token limit",
            **runtime_metadata(),
    }
    arm_records = {"arm": arm, "natural_entries": snapshot["entries"],
                   "added_entries": [], "n_natural": len(snapshot["entries"])}
    additions = {}
    tool_calls: list[dict] = []
    captured_llm_requests: list[dict] = []
    responses: list[dict] = []
    checks: list[dict] = []
    events: list[dict] = []
    final_text = ""
    llm_call_counter = {"n": 0}

    def _log_call(name: str, args: dict):
        tool_calls.append({"ts": time.time(), "tool": name, "args": args,
                           "result": {"status": "OK"}})

    async def _export_tool() -> dict:
        _log_call(sm.EXPORT_FUNCTION_NAME, {})
        return {"status": "EXPORTED"}
    _export_tool.__name__ = sm.EXPORT_FUNCTION_NAME
    _export_tool.__doc__ = sm.EXPORT_FUNCTION_DOC

    log_path = run_dir / "agent.log"
    fh = None
    root_logger = logging.getLogger()
    previous_log_level = root_logger.level
    t0 = time.perf_counter()
    status = "incomplete"
    completed = False
    termination_reason = "no_final_response"
    failure = None
    final_event_seen = False
    try:
        from google.adk.agents import LlmAgent
        from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools.preload_memory_tool import preload_memory_tool
        from google.genai import types
        from openai_compat_model import OpenAICompatModel

        entries, additions, _texts = arm_entries_from_snapshot(sm, snapshot, arm)
        arm_records["added_entries"] = ([] if arm == "replay" else
            [{"atom": a, "author": sm.AGENT_NAME,
              "timestamp": snapshot["added_timestamp"],
              "content": (additions["R"] if arm == "restore" else additions["S"])[a]}
             for a in additions["missing_ordered"]])
        app_name = f"nkey_adk_e16_{scenario.lower()}"
        user_id = "study_user"
        session_id = f"{scenario}-{arm}-{episode_index:02d}"
        memory_service = build_replay_service(InMemoryMemoryService, entries)
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id)

        benign_impls = {}
        for tool_name, tool_doc in sm.BENIGN_TOOLS:
            _make_benign_typed(tool_name, tool_doc, benign_impls, _log_call)
        model = OpenAICompatModel(model=model_id)

        class _SpyModel(type(model)):
            async def generate_content_async(self_, llm_request, stream: bool = False):
                snapshot_req = copy.deepcopy(llm_request)
                llm_call_counter["n"] += 1
                captured_llm_requests.append(
                    {"call_index": llm_call_counter["n"],
                     "payload": serialize_value(snapshot_req)})
                block = ""
                for content in snapshot_req.contents or []:
                    for part in content.parts or []:
                        m = MEMORY_BLOCK_RE.search(part.text or "")
                        if m:
                            block = m.group(1)
                if not block:
                    raise AssertionError("memory block missing from request")
                checks.append(check_request_block(block, arm, sm, snapshot, additions))
                async for resp in super().generate_content_async(llm_request, stream=stream):
                    yield resp

        def record_request(payload):
            captured_llm_requests[-1]["api_request"] = payload

        def record_response(payload):
            nonlocal final_text
            responses.append({"call_index": llm_call_counter["n"], **payload})
            final_text = payload.get("content") or payload.get("refusal") or ""

        spy_model = _SpyModel(model=model_id)
        spy_model._client = model._client
        spy_model.set_observers(request=record_request, response=record_response)
        agent = LlmAgent(
            name=sm.AGENT_NAME, description=sm.AGENT_DESCRIPTION,
            instruction=snapshot["system"], model=spy_model,
            tools=[preload_memory_tool, *benign_impls.values(), _export_tool],
            generate_content_config=types.GenerateContentConfig(temperature=0.0),
        )
        runner = Runner(agent=agent, app_name=app_name,
                        session_service=session_service, memory_service=memory_service)
        fire_msg = types.Content(role="user", parts=[types.Part(text=snapshot["user"])])
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(fh)
        root_logger.setLevel(logging.INFO)
        async for event in runner.run_async(user_id=user_id, session_id=session_id,
                                            new_message=fire_msg):
            events.append(serialize_value(event))
            if getattr(event, "error_code", None):
                termination_reason = f"adk_error:{event.error_code}"
                raise RuntimeError(f"adk_error:{event.error_code}: {getattr(event, 'error_message', '')}")
            if getattr(event, "interrupted", False):
                raise IncompleteRun("adk_interrupted")
            if (event.content and event.content.parts
                    and event.author == agent.name):
                texts = [p.text for p in event.content.parts if p.text]
                if texts:
                    final_text = "\n".join(texts)
                if event.is_final_response():
                    final_event_seen = True
        if (not responses or len(responses) != len(captured_llm_requests)
                or responses[-1]["finish_reason"] != "stop"
                or responses[-1]["has_tool_calls"]
                or not responses[-1]["has_output"] or not final_event_seen):
            raise IncompleteRun("no_complete_final_response")
        status = "ok"
        completed = True
        termination_reason = "stop"
    except IncompleteRun as exc:
        termination_reason = exc.termination_reason
        failure = exc
    except asyncio.CancelledError as exc:
        termination_reason = "cancelled"
        failure = exc
    except Exception as exc:
        status = "technical_error"
        if not termination_reason.startswith("adk_error:"):
            termination_reason = f"exception:{type(exc).__name__}"
        failure = exc
    finally:
        elapsed = time.perf_counter() - t0
        if fh is not None:
            root_logger.removeHandler(fh)
            fh.close()
        root_logger.setLevel(previous_log_level)
        json_artifacts = {
            "input_config.json": input_config, "missing_atoms.json": missing_atoms,
            "arm_records.json": arm_records, "llm_requests.json": captured_llm_requests,
            "input_checks.json": checks, "llm_responses.json": responses,
        }
        if arm != "replay":
            json_artifacts["intervention_lengths.json"] = additions.get("lengths", {})
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        log_text += (f"status={status}\nn_llm_calls={len(captured_llm_requests)}\n"
                     f"n_tool_calls={len(tool_calls)}\nn_responses={len(responses)}\n")
        text_artifacts = {
            "events.jsonl": "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
            "tool_calls.jsonl": "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tool_calls),
            "final_response.txt": final_text, "agent.log": log_text,
        }
        artifact_errors = []
        for name, value in json_artifacts.items():
            try:
                atomic_write_json(run_dir / name, value)
            except Exception as exc:
                artifact_errors.append(f"{name}: {type(exc).__name__}: {exc}")
                failure = failure or exc
        for name, value in text_artifacts.items():
            try:
                atomic_write_text(run_dir / name, value)
            except Exception as exc:
                artifact_errors.append(f"{name}: {type(exc).__name__}: {exc}")
                failure = failure or exc
        if artifact_errors:
            status, completed, termination_reason = "technical_error", False, "artifact_write_error"
        status_record = {
            "runtime_revision": RUNTIME_REVISION, "status": status,
            "completed": completed, "termination_reason": termination_reason,
            "artifacts_complete": not artifact_errors,
            "counts": {"n_requests": len(captured_llm_requests),
                       "n_tool_calls": len(tool_calls), "n_responses": len(responses)},
            "service_model_ids": sorted({r["service_model"] for r in responses if r.get("service_model")}),
            "elapsed_s": round(elapsed, 3),
        }
        if failure is not None:
            status_record["error"] = f"{type(failure).__name__}: {failure}"
        if artifact_errors:
            status_record["artifact_errors"] = artifact_errors
        # A completed status is the final commit, after all observations.
        atomic_write_json(run_dir / "status.json", status_record)
    if failure is not None:
        raise failure
    return run_dir


def _make_benign_typed(tool_name, tool_doc, registry, log_fn) -> None:
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
    parser = argparse.ArgumentParser(description="ADK E16 episode runner")
    parser.add_argument("--scenario", required=True, choices=["s1", "s2"])
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--reference-snapshot", required=True, type=Path)
    parser.add_argument("--intervention", required=True, choices=list(ARMS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    validate_output_args(args)

    load_secrets_env()
    run_dir = asyncio.run(run_episode(
        args.scenario, args.episode_index, args.reference_snapshot,
        args.intervention, Path(args.run_root), args.model))
    print(f"episode dir: {run_dir}")


if __name__ == "__main__":
    main()
