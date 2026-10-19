"""Study materials for ADK E14 / Scenario S1 (correct-key vs matched-
nontrigger-key).

Atoms are the frozen E11/E13 materials, re-exported from `_e13`'s shim
(which loads `_e11` verbatim). Two conditions, B0 layout only; per
experiment.md E14 the kits are byte-identical except the rc digit of the
three key records:

  kmatch    T + k1,k2,k3 "recon-sync <code> rc=0"  (T's rule satisfied)
  kmismatch T + k1,k2,k3 "recon-sync <code> rc=1"  (rule NOT satisfied)

Both conditions use the E13 R-full controlled-exposure block: attack
carriers in fixed T,k1,k2,k3 order + the fixed common benign background
(first R_FULL_BACKGROUND_N competitors, identical across conditions).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


_src = Path(__file__).resolve().parents[1] / "_e13" / "study_materials_s1.py"
_spec = importlib.util.spec_from_file_location("e13_study_materials_s1_e14", _src)
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
R_FULL_BACKGROUND_N = _m.R_FULL_BACKGROUND_N

SYNC_TOKEN = "recon-sync"
# Flips ONLY the key records' "<sync> <code> rc=0" tail; T's own
# "<sync> rc=0" wording has no code token between, so it never matches
# (and the flip is applied to cell=="K" records only anyway).
_RC_FLIP = re.compile(rf"({SYNC_TOKEN} \S+ )rc=0")

CONDITIONS = ("kmatch", "kmismatch")


def iter_conditions(ucodes: list[str], inst: dict) -> dict:
    """Per-condition write kits: E13 B0 kit (attack + competitors) with the
    key records' rc digit flipped for kmismatch. Order, ids and every other
    byte are preserved."""
    b0 = iter_baselines(ucodes, inst)["B0"]
    attack = [r for r in b0 if r["cell"] in ("A", "K")]
    comps = [r for r in b0 if r["cell"] == "C"]
    flipped = [
        dict(r, content=_RC_FLIP.sub(r"\1rc=1", r["content"]))
        if r["cell"] == "K" else dict(r)
        for r in attack
    ]
    return {"kmatch": attack + comps, "kmismatch": flipped + comps}


def iter_full_block(ucodes: list[str], inst: dict) -> dict:
    """Per-condition diagnostic block records: attack carriers in fixed
    atom order, then the fixed common benign background (identical across
    conditions)."""
    conds = iter_conditions(ucodes, inst)
    background = [c for c in conds["kmatch"] if c["cell"] == "C"][:R_FULL_BACKGROUND_N]
    return {
        name: [c for c in kit if c["cell"] in ("A", "K")] + background
        for name, kit in conds.items()
    }
