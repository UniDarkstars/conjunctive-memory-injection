"""Episode runner for LangMem E16 (missing-atom restoration).

One episode loads the task's FROZEN natural snapshot and runs ONE arm
(replay/restore/sham). The arm's record list replaces the natural recall
at the ORIGINAL <memories> assembly position with the original wrapper;
the intervened block persists for the whole tool loop (no reversion to
natural retrieval). Trusted system and current user stay unchanged; the
model/tool loop then runs freely (temperature=0, max_tokens, tool_choice
and loop cap as E13). The service-returned model id is captured per call.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
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

sys.path.insert(0, str(_THIS.parents[1]))
from _e16_runtime import (  # noqa: E402
    RUNTIME_REVISION, IncompleteRun, atomic_write_json, atomic_write_text,
    prepare_episode_dir, runtime_metadata,
)

VENDORED_LANGMEM_SRC = str(_THIS.parents[1] / "agent-frameworks" / "langmem" / "src")
if VENDORED_LANGMEM_SRC not in sys.path:
    pass  # langgraph/langchain dependencies are installed from requirements-online.txt

import arms_core  # noqa: E402

ARMS = ("replay", "restore", "sham")
ARM_LETTER = {"replay": "N", "restore": "R", "sham": "S"}
WIRING_REVISION = "e16-replay-2"


def load_secrets_env() -> None:
    """Release credentials come only from process environment variables."""
    return


def build_system_prompt(base_prompt: str, records: list[dict]) -> tuple[str, list[str]]:
    lines, ids = [], []
    for r in records:
        marker = r.get("content_marker", "")
        content = r.get("content", "")
        ids.append(marker.strip("[]").split(":", 1)[-1])
        lines.append(f"{marker} {content}")
    system = base_prompt + "\n\n<memories>\n" + "\n".join(lines) + "\n</memories>"
    return system, ids


def arm_records_from_snapshot(snapshot: dict, arm: str) -> tuple[list[dict], dict | None]:
    natural = [dict(e) for e in snapshot["entries"]]
    if arm == "replay":
        return natural, None
    additions = arms_core.build_arm_additions(
        snapshot["atoms_missing"], snapshot["canonical_atoms"],
        _sm_for(snapshot["scenario"]), snapshot["codes"])
    texts = additions[{"restore": "R", "sham": "S"}[arm]]
    added = [{"key": snapshot["added_keys"][a],
              "content": texts[a],
              "content_marker": f"[ID:{a}]",
              "score": None}
             for a in additions["missing_ordered"]]
    return natural + added, additions


def _sm_for(scenario: str):
    return importlib.import_module(f"study_materials_{scenario.lower()}")


def check_request_block(block_text: str, arm: str, snapshot: dict) -> dict:
    norm = arms_core.normalize_text(block_text)
    snap_norm = arms_core.normalize_text(snapshot["block_preview"])
    canonical = snapshot["canonical_atoms"]
    missing = snapshot["atoms_missing"]
    out = {"snapshot_prefix_intact": norm.startswith(snap_norm),
           "arm_label_absent": all(lbl not in norm for lbl in ("replay", "restore", "sham"))}
    if arm == "replay":
        out["block_equals_snapshot"] = norm == snap_norm
        ok = out["block_equals_snapshot"] and out["arm_label_absent"]
    elif arm == "restore":
        atoms = arms_core.analyze_block(block_text, canonical)
        out["all_atoms_present"] = all(atoms["evidence"].values())
        ok = out["all_atoms_present"] and out["snapshot_prefix_intact"] and out["arm_label_absent"]
    else:
        restored_leak = [a for a, t in canonical.items()
                         if a in missing and arms_core.normalize_text(t) in norm]
        added_region = norm[len(snap_norm):]
        out["restored_atom_leak"] = restored_leak
        out["no_forbidden_in_additions"] = not any(
            bad in added_region for bad in arms_core.FORBIDDEN_SUBSTRINGS)
        ok = (not restored_leak and out["snapshot_prefix_intact"]
              and out["no_forbidden_in_additions"] and out["arm_label_absent"])
    out["ok"] = ok
    if not ok:
        raise AssertionError(f"input check failed for arm {arm}: {out}")
    return out


def extract_block(system_prompt: str) -> str:
    import re
    m = re.search(r"<memories>\n(.*?)\n</memories>", system_prompt, re.DOTALL)
    return m.group(1) if m else ""


def build_tools_schema(sm):
    tools = []
    for name, doc in sm.BENIGN_TOOL_SPECS:
        tools.append({"type": "function", "function": {
            "name": name, "description": doc,
            "parameters": {"type": "object", "properties": {}}}})
    tools.append({"type": "function", "function": {
        "name": sm.EXPORT_FUNCTION_NAME, "description": sm.EXPORT_FUNCTION_DOC,
        "parameters": {"type": "object", "properties": {}}}})
    return tools


async def run_episode(scenario: str, episode_index: int, snapshot_path: Path,
                      arm: str, run_root: Path, model_id: str | None) -> Path:
    from openai import AsyncOpenAI

    snapshot_path = Path(snapshot_path).resolve()
    run_root = Path(run_root).resolve()
    sm = _sm_for(scenario)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (snapshot["scenario"] != scenario or snapshot["episode_index"] != episode_index
            or snapshot["platform"] != "langmem"):
        raise ValueError(f"snapshot identity mismatch: {snapshot_path}")
    canonical_now = arms_core.canonical_atoms(
        sm, snapshot["codes"], sm.INSTANCES[episode_index - 1])
    if canonical_now != snapshot["canonical_atoms"]:
        raise ValueError("materials drifted from the frozen snapshot")

    if model_id is None:
        model_id = os.environ.get("NORMAL_CHAT_MODEL", sm.LLM_PRIMARY)
    run_dir = prepare_episode_dir(
        run_root, scenario, arm, episode_index, "langmem", model_id)

    artifacts = {
        "missing_atoms.json": {
            "missing": snapshot["atoms_missing"], "present": snapshot["atoms_present"],
            "missing_count": len(snapshot["atoms_missing"]),
            "evidence": snapshot["atom_evidence"], "snapshot": str(snapshot_path),
            "snapshot_sha256": snapshot["snapshot_sha256"],
        },
        "input_config.json": {
            "experiment": "e16", "platform": "langmem", "scenario": scenario,
            "episode_index": episode_index, "arm": arm, "arm_letter": ARM_LETTER[arm],
            "snapshot": str(snapshot_path), "snapshot_sha256": snapshot["snapshot_sha256"],
            "snapshot_version": snapshot["snapshot_version"], "config": snapshot["config"],
            "model_id": model_id, "temperature": sm.TEMPERATURE,
            "max_tokens": sm.MAX_TOKENS, "wiring_revision": WIRING_REVISION,
            "runtime_revision": RUNTIME_REVISION, "batch_id": run_root.name,
            **runtime_metadata(),
        },
        "arm_records.json": {
            "arm": arm, "natural_entries": snapshot["entries"], "added_entries": [],
            "n_natural": len(snapshot["entries"]),
        },
    }
    if arm != "replay":
        artifacts["intervention_lengths.json"] = {}

    tool_calls: list[dict] = []
    invalid_tool_calls: list[dict] = []
    llm_requests: list[dict] = []
    llm_responses: list[dict] = []
    checks: list[dict] = []
    final_text = ""
    service_models: list[str] = []
    ids_in_block: list[str] = []

    def _log_call(name, args):
        tool_calls.append({"ts": time.time(), "tool": name, "args": args,
                           "result": {"status": "OK"}})

    t0 = time.perf_counter()
    status = "technical_error"
    completed = False
    termination_reason = "initialization_error"
    caught_error: BaseException | None = None
    try:
        records, additions = arm_records_from_snapshot(snapshot, arm)
        system_prompt, ids_in_block = build_system_prompt(snapshot["system"], records)
        if additions:
            artifacts["arm_records.json"]["added_entries"] = [
                {"atom": a, "key": snapshot["added_keys"][a],
                 "content_marker": f"[ID:{a}]",
                 "content": additions["R" if arm == "restore" else "S"][a]}
                for a in additions["missing_ordered"]]
            artifacts["intervention_lengths.json"] = additions["lengths"]
        for name, value in artifacts.items():
            atomic_write_json(run_dir / name, value)
        # Empty logs are explicit artifacts, rather than indistinguishable
        # from a failed write or a previous run's observations.
        atomic_write_text(run_dir / "tool_calls.jsonl", "")
        client = AsyncOpenAI(base_url=os.environ["NORMAL_BASE_URL"].strip(),
                             api_key=os.environ["NORMAL_API_KEY"].strip())
        tools_schema = build_tools_schema(sm)
        declared_tools = {t["function"]["name"] for t in tools_schema}
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": snapshot["user"]}]
        for step in range(10):
            block = extract_block(system_prompt)
            if not block:
                raise AssertionError("memories block missing from system prompt")
            checks.append(check_request_block(block, arm, snapshot))
            request = {
                "model": model_id, "messages": copy.deepcopy(messages),
                "tools": tools_schema, "tool_choice": "auto",
                "temperature": sm.TEMPERATURE, "max_tokens": sm.MAX_TOKENS,
            }
            llm_requests.append({
                "call_index": step + 1,
                "payload": {"system_prompt": system_prompt,
                            "messages": copy.deepcopy(messages),
                            "tools": [t["function"]["name"] for t in tools_schema]},
                "api_request": copy.deepcopy(request),
            })
            atomic_write_json(run_dir / "input_checks.json", checks)
            atomic_write_json(run_dir / "llm_requests.json", llm_requests)
            completion = await client.chat.completions.create(**request)
            service_model = getattr(completion, "model", None)
            if service_model:
                service_models.append(service_model)
            choice = completion.choices[0] if completion.choices else None
            msg = choice.message if choice else None
            finish_reason = choice.finish_reason if choice else None
            refusal = getattr(msg, "refusal", None) if msg else None
            raw_response = completion.model_dump(mode="json")
            llm_responses.append({
                "call_index": step + 1, "finish_reason": finish_reason,
                "refusal": refusal, "service_model_id": service_model,
                "usage": raw_response.get("usage"), "payload": raw_response,
            })
            atomic_write_json(run_dir / "llm_responses.json", llm_responses)
            if msg is None:
                raise IncompleteRun("empty_response")
            content_text = msg.content or ""
            final_text = content_text or refusal or ""
            if finish_reason in {"length", "content_filter"}:
                raise IncompleteRun(finish_reason)
            if finish_reason not in {"stop", "tool_calls"}:
                raise IncompleteRun("unexpected_finish_reason")
            assistant_msg = {"role": "assistant", "content": content_text or None}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls]
            messages.append(assistant_msg)
            if not msg.tool_calls:
                if finish_reason != "stop":
                    raise IncompleteRun("tool_calls_missing")
                if not content_text and not refusal:
                    raise IncompleteRun("empty_response")
                status, completed = "ok", True
                termination_reason = "refusal" if refusal else "stop"
                break
            for tc in msg.tool_calls:
                tname = tc.function.name
                tool_error = None
                try:
                    targs = json.loads(tc.function.arguments)
                    if not isinstance(targs, dict):
                        tool_error = "Tool arguments must be a JSON object."
                except json.JSONDecodeError:
                    tool_error = "Tool arguments are not valid JSON."
                if tname not in declared_tools:
                    tool_error = f"Unknown tool: {tname}"
                if tool_error:
                    invalid_tool_calls.append({
                        "call_index": step + 1, "tool_call_id": tc.id, "tool": tname,
                        "arguments": tc.function.arguments, "error": tool_error,
                    })
                    result = {"error": tool_error}
                else:
                    _log_call(tname, targs)
                    atomic_write_text(run_dir / "tool_calls.jsonl", "".join(
                        json.dumps(t, ensure_ascii=False) + "\n" for t in tool_calls))
                    result = ({"status": "EXPORTED"}
                              if tname == sm.EXPORT_FUNCTION_NAME
                              else {"status": "done", "tool": tname, "args": targs})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "name": tname,
                                 "content": json.dumps(result, ensure_ascii=False)})
        else:
            raise IncompleteRun("loop_limit")
    except IncompleteRun as exc:
        status = "incomplete"
        termination_reason = exc.termination_reason
        caught_error = exc
    except BaseException as exc:  # persist partial observations, then re-raise
        status = "technical_error"
        completed = False
        termination_reason = "technical_error"
        caught_error = exc

    elapsed = time.perf_counter() - t0
    artifacts.update({
        "llm_requests.json": llm_requests, "llm_responses.json": llm_responses,
        "input_checks.json": checks,
        "tool_calls.jsonl": "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tool_calls),
        "final_response.txt": final_text,
    })
    write_errors = {}
    for name, value in artifacts.items():
        try:
            if isinstance(value, str):
                atomic_write_text(run_dir / name, value)
            else:
                atomic_write_json(run_dir / name, value)
        except Exception as exc:
            write_errors[name] = f"{type(exc).__name__}: {exc}"
    if write_errors:
        status, completed, termination_reason = "technical_error", False, "artifact_write_error"
        if caught_error is None:
            caught_error = RuntimeError(f"Artifact writes failed: {', '.join(write_errors)}")
    try:
        atomic_write_text(run_dir / "agent.log", (
            f"status={status}\narm={arm}\nn_llm_calls={len(llm_requests)}\n"
            f"n_tool_calls={len(tool_calls)}\nn_responses={len(llm_responses)}\n"
            f"termination_reason={termination_reason}\n"
            f"ids_in_memories_block={ids_in_block}\n"
            f"service_model_ids={sorted(set(service_models))}\n"
        ))
    except Exception as exc:
        write_errors["agent.log"] = f"{type(exc).__name__}: {exc}"
        status, completed, termination_reason = "technical_error", False, "artifact_write_error"
        if caught_error is None:
            caught_error = RuntimeError("Artifact write failed: agent.log")
    final_status = {
        "runtime_revision": RUNTIME_REVISION, "status": status,
        "completed": completed, "termination_reason": termination_reason,
        "artifacts_complete": not write_errors, "elapsed_s": round(elapsed, 3),
        "counts": {"n_requests": len(llm_requests), "n_tool_calls": len(tool_calls),
                   "n_responses": len(llm_responses)},
        "service_model_ids": sorted(set(service_models)),
        "invalid_tool_calls": invalid_tool_calls,
    }
    if caught_error is not None:
        final_status["error"] = f"{type(caught_error).__name__}: {caught_error}"
    if write_errors:
        final_status["artifact_write_errors"] = write_errors
    # This commit marker is written only after every observation artifact.
    atomic_write_json(run_dir / "status.json", final_status)
    if caught_error is not None:
        raise caught_error
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E16 episode runner")
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
