"""Offline construction pre-checks for LangMem E16 (no API calls).

LangMem natural retrieval needs embeddings, so the natural-D_i part is
covered at snapshot collection (and by ADK's offline simulation of the
same rules). Here: canonical atoms over all tasks, sham construction on
synthetic missing-sets {k1,k2,k3}/{T}/{T,k3}/{} for every task, and the
<memories> block render round-trip on synthetic entries.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

import arms_core  # noqa: E402
from collect_snapshot import render_block  # noqa: E402

MISSING_SETS = [("k1", "k2", "k3"), ("T",), ("T", "k3"), ()]


def main() -> None:
    for scenario in ("s1", "s2"):
        sm = importlib.import_module(f"study_materials_{scenario.lower()}")
        for i, inst in enumerate(sm.INSTANCES, 1):
            canon = arms_core.canonical_atoms(sm, inst["codes"], inst)
            for atom, text in canon.items():
                assert text and f"[ID:{atom}]" in text, (scenario, i, atom)
            assert len(set(canon.values())) == 4
            for missing in MISSING_SETS:
                arms_core.build_arm_additions(list(missing), canon, sm, inst["codes"])
        print(f"[{scenario}] canonical + sham construction OK "
              f"({len(sm.INSTANCES)} tasks x {len(MISSING_SETS)} missing-sets)")

        inst = sm.INSTANCES[0]
        canon = arms_core.canonical_atoms(sm, inst["codes"], inst)
        entries = [{"content_marker": f"[ID:{a}]", "content": canon[a]}
                   for a in arms_core.ATOM_ORDER]
        block = render_block(entries)
        analysis = arms_core.analyze_block(block, canon)
        assert all(analysis["evidence"].values()), "round-trip block must contain all atoms"
        print(f"[{scenario}] block render round-trip OK")
    print("LangMem E16 PRECHECK PASSED "
          "(natural D_i validated at snapshot collection; ADK precheck covers the offline rules)")


if __name__ == "__main__":
    main()
