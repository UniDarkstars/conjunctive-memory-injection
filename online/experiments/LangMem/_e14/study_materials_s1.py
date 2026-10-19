"""Study materials for LangMem E14 / Scenario S1 (correct-key vs matched-
nontrigger-key).

Atoms, authorization boundary and tool surface are the frozen E13/E11
materials, re-exported from `_e13`'s module by path. Two conditions, B0
layout only; per experiment.md E14 the kits are byte-identical except the
rc digit of the three key records:

  kmatch    T + k1,k2,k3 "recon-sync <code> rc=0"  (T's rule satisfied)
  kmismatch T + k1,k2,k3 "recon-sync <code> rc=1"  (rule NOT satisfied)

Both conditions use the E13 R-full controlled-exposure block: attack
carriers in fixed T,k1,k2,k3 order + the fixed common benign background.
The store-write kit includes the E13 competitive condition's N_COMPETITORS
competitors (write-ledger parity with E13); controlled exposure does not
depend on W.
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
AGENT_SYSTEM_PROMPT = _m.AGENT_SYSTEM_PROMPT
T_TEMPLATE = _m.T_TEMPLATE
EXPORT_FUNCTION_NAME = _m.EXPORT_FUNCTION_NAME
EXPORT_FUNCTION_DOC = _m.EXPORT_FUNCTION_DOC
BENIGN_TOOL_SPECS = _m.BENIGN_TOOL_SPECS

iter_baselines = _m.iter_baselines
N_COMPETITORS = _m.N_COMPETITORS
SEARCH_LIMIT = _m.SEARCH_LIMIT
R_FULL_BACKGROUND_N = _m.R_FULL_BACKGROUND_N
EMBEDDER_DIMS = _m.EMBEDDER_DIMS
LLM_PRIMARY = _m.LLM_PRIMARY
TEMPERATURE = _m.TEMPERATURE
MAX_TOKENS = _m.MAX_TOKENS

NAMESPACE = ("e14", "s1")

SYNC_TOKEN = "recon-sync"
_RC_FLIP = re.compile(rf"({SYNC_TOKEN} \S+ )rc=0")

CONDITIONS = ("kmatch", "kmismatch")


def iter_conditions(ucodes: list[str], inst: dict) -> dict:
    """Per-condition write kits: E13 B0 kit (attack + competitors) with the
    key records' rc digit flipped for kmismatch. Keys, order and every other
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
    atom order, then the fixed common benign background. The background
    draws from the FULL 12-competitor pool (same as E13's R-full block),
    not the kit's frozen N_COMPETITORS subset, so the diagnostic block size
    is decoupled from the still-provisional competitive cell."""
    conds = iter_conditions(ucodes, inst)
    background = _m._competitors(ucodes)[:R_FULL_BACKGROUND_N]
    return {
        name: [c for c in kit if c["cell"] in ("A", "K")] + background
        for name, kit in conds.items()
    }
