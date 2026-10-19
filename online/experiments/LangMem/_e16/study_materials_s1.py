"""Study materials for LangMem E16 / Scenario S1 (missing-atom restoration).

Atoms, authorization boundary, tools and frozen tasks are the E13 batch
loaded verbatim from `_e13`'s module (W/M frozen 2026-10-01: W=8, M=5).
E16 adds nothing to the materials; this module re-exports them plus the
scenario constants used by arms_core.
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


_src = Path(__file__).resolve().parents[1] / "_e13" / "study_materials_s1.py"
_spec = importlib.util.spec_from_file_location("e13_study_materials_s1_e16", _src)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

SCENARIO_ID = _m.SCENARIO_ID
SCENARIO_NAME = _m.SCENARIO_NAME
INSTANCES = _m.INSTANCES
AGENT_NAME = _m.AGENT_NAME
AGENT_DESCRIPTION = _m.AGENT_DESCRIPTION
AGENT_SYSTEM_PROMPT = _m.AGENT_SYSTEM_PROMPT
T_TEMPLATE = _m.T_TEMPLATE
EXPORT_FUNCTION_NAME = _m.EXPORT_FUNCTION_NAME
EXPORT_FUNCTION_DOC = _m.EXPORT_FUNCTION_DOC
BENIGN_TOOL_SPECS = _m.BENIGN_TOOL_SPECS

iter_baselines = _m.iter_baselines
SEARCH_LIMIT = _m.SEARCH_LIMIT        # frozen W=8
N_COMPETITORS = _m.N_COMPETITORS      # frozen M=5
EMBEDDER_DIMS = _m.EMBEDDER_DIMS
LLM_PRIMARY = _m.LLM_PRIMARY
TEMPERATURE = _m.TEMPERATURE
MAX_TOKENS = _m.MAX_TOKENS

SYNC_TOKEN = "recon-sync"
DOMAIN_SUBJECT = "应付科目"
BENIGN_HEADLINE = "【业务备忘】"
