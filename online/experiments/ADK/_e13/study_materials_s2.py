"""Study materials for ADK E13 / Scenario S2 (code release review).

R-natural reuses E11's frozen write kits verbatim (same T/key atoms, same
W=10 / M=12 same-domain competitors, same authorization boundary), loaded
from `_e11/study_materials_s2.py` by path so the atom texts cannot drift.

R-full adds a diagnostic memory block built from the SAME atoms: attack
records first in fixed T,k1,k2,k3 carrier order, then a fixed common benign
background (the first R_FULL_BACKGROUND_N competitor records, identical
across the three layouts). Layouts differ only in carrier boundaries.
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


_src = Path(__file__).resolve().parents[1] / "_e11" / "study_materials_s2.py"
_spec = importlib.util.spec_from_file_location("e11_study_materials_s2_e13", _src)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

SCENARIO_ID = _m.SCENARIO_ID
SCENARIO_NAME = _m.SCENARIO_NAME
INSTANCES = _m.INSTANCES
AGENT_NAME = _m.AGENT_NAME
AGENT_DESCRIPTION = _m.AGENT_DESCRIPTION
AGENT_INSTRUCTION = _m.AGENT_INSTRUCTION
T_TEMPLATE = _m.T_TEMPLATE
EXPORT_FUNCTION_NAME = _m.EXPORT_FUNCTION_NAME
EXPORT_FUNCTION_DOC = _m.EXPORT_FUNCTION_DOC
BENIGN_TOOLS = _m.BENIGN_TOOLS

iter_baselines = _m.iter_baselines

# Fixed common benign background size for the R-full diagnostic block.
R_FULL_BACKGROUND_N = 6


def iter_full_block(ucodes: list[str], inst: dict) -> dict:
    """Per-layout diagnostic block records: attack carriers in fixed atom
    order (the kit's own write order, T,k1,k2,k3 / merged variants), then
    the fixed common benign background."""
    kits = iter_baselines(ucodes, inst)
    background = [c for c in kits["B0"] if c["cell"] == "C"][:R_FULL_BACKGROUND_N]
    return {
        name: [c for c in kit if c["cell"] in ("A", "K")] + background
        for name, kit in kits.items()
    }
