"""OpenAI-compatible BaseLlm adapter for the ADK E16 study.

Same as E1 adapter, with _convert_tools fixed to read tool declarations
via _get_declaration() (the callable) rather than the nonexistent
function_declaration attribute on ADK FunctionTool instances.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode

import sys
import uuid
from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

_EXPERIMENTS = str(Path(__file__).resolve().parents[2])
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)
from _e16_runtime import IncompleteRun  # noqa: E402


def _json_value(value):
    """Keep the service response, including fields unknown to this adapter."""
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return str(value)


def _content_to_parts(content: types.Content) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    tool_parts: list[dict[str, Any]] = []
    for part in content.parts or []:
        if part.text:
            text_parts.append(part.text)
        elif part.function_call is not None:
            fc = part.function_call
            tool_parts.append({
                "kind": "call",
                "id": fc.id or "",
                "name": fc.name or "",
                "args": dict(fc.args or {}),
            })
        elif part.function_response is not None:
            fr = part.function_response
            tool_parts.append({
                "kind": "response",
                "id": fr.id or "",
                "name": fr.name or "",
                "response": dict(fr.response or {}),
            })
    return "\n".join(p for p in text_parts if p), tool_parts


def _extract_model_text(candidate_content: types.Content | None) -> tuple[str, list[types.Part]]:
    if candidate_content is None or not candidate_content.parts:
        return "", []
    texts: list[str] = []
    fn_parts: list[types.Part] = []
    for part in candidate_content.parts:
        if part.text:
            texts.append(part.text)
        elif part.function_call is not None:
            fn_parts.append(part)
    return "\n".join(texts), fn_parts


class OpenAICompatModel(BaseLlm):
    """Minimal ADK model that talks to an OpenAI-compatible chat endpoint."""

    model: str

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        base_url = os.environ.get("NORMAL_BASE_URL", "").strip()
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("NORMAL_API_KEY") or ""
        if not base_url or not api_key:
            raise RuntimeError(
                "NORMAL_BASE_URL and an API key (NORMAL_API_KEY or GOOGLE_API_KEY) must be set."
            )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._request_observer = None
        self._response_observer = None

    def set_observers(self, *, request=None, response=None) -> None:
        self._request_observer = request
        self._response_observer = response

    @classmethod
    def supported_models(cls) -> list[str]:
        return [".*"]

    def _build_message_for_content(
        self, content: types.Content
    ) -> dict[str, Any] | list[dict[str, Any]]:
        role = content.role or "user"
        text, tool_parts = _content_to_parts(content)

        if tool_parts and not text and all(p["kind"] == "response" for p in tool_parts):
            messages = []
            for p in tool_parts:
                messages.append({
                    "role": "tool",
                    "tool_call_id": p["id"] or f"call_{uuid.uuid4().hex[:24]}",
                    "name": p["name"],
                    "content": json.dumps(p["response"], ensure_ascii=False),
                })
            return messages

        msg: dict[str, Any] = {
            "role": "assistant" if role == "model" else role,
            "content": text or "",
        }
        calls = [p for p in tool_parts if p["kind"] == "call"]
        if calls:
            msg["tool_calls"] = [
                {
                    "id": c["id"] or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(c["args"], ensure_ascii=False),
                    },
                }
                for c in calls
            ]
            if not msg["content"]:
                msg["content"] = None
        return msg

    def _convert_tools(self, llm_request: LlmRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name, tool_obj in llm_request.tools_dict.items():
            if name in seen or name == "preload_memory":
                continue
            seen.add(name)
            # ADK FunctionTool exposes the declaration via _get_declaration()
            # (a callable), not via a .function_declaration attribute.
            declaration = None
            if hasattr(tool_obj, "_get_declaration") and callable(
                getattr(tool_obj, "_get_declaration")
            ):
                declaration = tool_obj._get_declaration()
            if declaration is None:
                description = getattr(tool_obj, "description", "")
                parameters = {"type": "object", "properties": {}}
            else:
                description = getattr(declaration, "description", None) or getattr(
                    tool_obj, "description", ""
                )
                # Try parameters (Google genai Schema), then parameters_json_schema
                # (set when JSON_SCHEMA_FOR_FUNC_DECL is enabled).
                params = getattr(declaration, "parameters", None)
                if params is not None and hasattr(params, "model_dump"):
                    parameters = params.model_dump(exclude_none=True, by_alias=True)
                elif isinstance(params, dict):
                    parameters = dict(params)
                else:
                    json_schema = getattr(declaration, "parameters_json_schema", None)
                    if json_schema and isinstance(json_schema, dict):
                        parameters = json_schema
                    else:
                        parameters = {"type": "object", "properties": {}}
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        return tools

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if stream:
            raise NotImplementedError("Streaming is not used by the E16 harness.")

        messages: list[dict[str, Any]] = []
        if llm_request.config and llm_request.config.system_instruction:
            si = llm_request.config.system_instruction
            if isinstance(si, str):
                text = si
            else:
                text, _ = _content_to_parts(si)
            if text:
                messages.append({"role": "system", "content": text})

        for content in llm_request.contents or []:
            built = self._build_message_for_content(content)
            if isinstance(built, list):
                messages.extend(built)
            else:
                messages.append(built)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        tools = self._convert_tools(llm_request)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Preserve E13's actual parameter behavior: do not forward config
        # temperature or token limits. Record exactly what is being sent.
        if self._request_observer is not None:
            self._request_observer(_json_value(kwargs))
        completion = await self._client.chat.completions.create(**kwargs)
        choices = completion.choices or []
        choice = choices[0] if choices else None
        message = choice.message if choice is not None else None
        finish_reason = getattr(choice, "finish_reason", None)
        tool_calls = getattr(message, "tool_calls", None) or []
        text = getattr(message, "content", None) or ""
        refusal = getattr(message, "refusal", None) or ""
        response_record = {
            "finish_reason": finish_reason,
            "content": text,
            "service_model": getattr(completion, "model", None),
            "usage": _json_value(getattr(completion, "usage", None)),
            "raw_response": _json_value(completion),
            "has_tool_calls": bool(tool_calls),
            "has_output": bool(text.strip() or refusal.strip()),
            "refusal": refusal or None,
        }
        if self._response_observer is not None:
            self._response_observer(response_record)
        if choice is None:
            raise IncompleteRun("empty_response")
        if finish_reason in ("length", "content_filter"):
            raise IncompleteRun(finish_reason)
        if finish_reason not in ("stop", "tool_calls"):
            raise IncompleteRun("unexpected_finish_reason")
        if finish_reason == "tool_calls" and not tool_calls:
            raise IncompleteRun("tool_calls_missing")
        if not response_record["has_output"] and not tool_calls:
            raise IncompleteRun("empty_response")

        parts: list[types.Part] = []
        if text.strip() or refusal:
            parts.append(types.Part.from_text(text=text if text.strip() else refusal))
        # Validate the whole response before yielding any executable calls.
        for tool_call in tool_calls:
            args_raw = tool_call.function.arguments
            try:
                args = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid_tool_arguments:{tool_call.function.name}") from exc
            if not isinstance(args, dict):
                raise ValueError(f"non_object_tool_arguments:{tool_call.function.name}")
            parts.append(
                types.Part.from_function_call(
                    name=tool_call.function.name, args=args
                )
            )
            parts[-1].function_call.id = tool_call.id

        content = types.Content(role="model", parts=parts)
        yield LlmResponse(
            content=content,
            turn_complete=True,
            finish_reason="STOP",
            model_version=getattr(completion, "model", None),
            custom_metadata={"openai_finish_reason": finish_reason,
                             "openai_usage": response_record["usage"]},
        )
