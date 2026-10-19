"""Re-export the 30 frozen E10 S2 instances for E13 (same pairing as E10/E11).

Loaded from E10's frozen table by path to avoid same-name module collisions.
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


_path = Path(__file__).resolve().parents[1] / "_e10" / "instances_s2.py"
_spec = importlib.util.spec_from_file_location("e10_instances_s2_e13", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

INSTANCES = _mod.INSTANCES
