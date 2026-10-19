"""B-v4 frozen atom templates for E17, loaded from the E11 materials by path.

E17 binds the SAME authorization boundary, T template, export tools and
benign tool surface as E10/E11/E13 (B-v4), only to NEW task identifiers
(instances_s1/s2). Loading by file path (the E13 pattern) keeps the atom
texts from drifting. The agent prompt is platform-specific (ADK
AGENT_INSTRUCTION vs LangMem AGENT_SYSTEM_PROMPT); atoms are identical.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


_EXPERIMENTS = Path(__file__).resolve().parents[1]

SYNC_TOKEN = {"s1": "recon-sync", "s2": "gate-sync"}
EXPORT_TOOL = {"s1": "export_settlement_detail", "s2": "export_source_bundle"}

_cache: dict[tuple[str, str], object] = {}


def _load(platform: str, scenario: str):
    key = (platform, scenario)
    if key in _cache:
        return _cache[key]
    import sys

    src = (
        _EXPERIMENTS / {"adk": "ADK", "langmem": "LangMem"}.get(platform, platform) / "_e11" / f"study_materials_{scenario}.py"
    )
    # The E11 module does `from instances_<scenario> import INSTANCES`, so
    # its directory must be importable for the duration of the exec ONLY.
    # Leaving it on sys.path shadows later same-name imports with the E11
    # copies (bit the ADK smoke: run_episode then imported _e11's
    # openai_compat_model instead of _e17's).
    path = str(src.parent)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        spec = importlib.util.spec_from_file_location(
            f"e11_{platform}_{scenario}_e17", src
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if inserted:
            sys.path.remove(path)
    _cache[key] = mod
    return mod


def get(platform: str, scenario: str) -> dict:
    """Platform-facing B-v4 constants for one scenario."""
    m = _load(platform, scenario)
    out = {
        "scenario": scenario,
        "sync_token": SYNC_TOKEN[scenario],
        "export_function_name": m.EXPORT_FUNCTION_NAME,
        "export_function_doc": m.EXPORT_FUNCTION_DOC,
        "t_template": m.T_TEMPLATE,
    }
    if platform == "adk":
        out["agent_instruction"] = m.AGENT_INSTRUCTION
        out["agent_name"] = m.AGENT_NAME
        out["agent_description"] = m.AGENT_DESCRIPTION
        out["benign_tools"] = list(m.BENIGN_TOOLS)
    else:
        out["agent_system_prompt"] = m.AGENT_SYSTEM_PROMPT
        out["namespace"] = ("e17", scenario)
        out["embedder_dims"] = m.EMBEDDER_DIMS
        out["benign_tools"] = list(m.BENIGN_TOOL_SPECS)
        out["temperature"] = m.TEMPERATURE
        out["max_tokens"] = m.MAX_TOKENS
    return out


def t_content(scenario: str, codes: list[str]) -> str:
    """Bind the frozen T template to a task's three public codes."""
    tpl = _load("adk", scenario).T_TEMPLATE  # atoms identical across platforms
    return tpl.format(c1=codes[0], c2=codes[1], c3=codes[2])


def key_contents(scenario: str, codes: list[str], inst: dict) -> dict[str, str]:
    """The three condition records, bound to the task's codes and blurb."""
    sync = SYNC_TOKEN[scenario]
    rec = inst["records"][0]
    return {
        f"k{i+1}": f"[ID:k{i+1}] {sync} {code} rc=0. {rec}"
        for i, code in enumerate(codes)
    }
