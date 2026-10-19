"""Write freeze_mg.json for the e17mg_v1 batch.

Freezes everything this batch depends on BEYOND the e17_v1 receipt:
the new _e17mg_materials sources, this directory's runner/driver/precheck/
checker, plus drift-detection hashes of the reused frozen inputs (the
e17_v1 freeze.json itself, study_config.json and run_episode.py it
delegates to) and the scheduling seed.

Review gap fill (2026-10-14): the e17_v1 freeze.json hash proves only that
the RECEIPT file is unchanged - it does not re-verify the sources that
receipt lists. Two runtime-touched input classes are now hashed directly:
  shared_e17_materials - experiments/_e17_materials/*.py (task suite,
      benign pools, budget, method constructors, templates loader),
      imported by run_episode_mg/run_all_mg/precheck/check_episode and by
      the reused _e17/run_episode.py itself;
  e11_scene_templates - the E11 scene modules actually exec'd via
      _e17_materials.templates at runtime on this platform
      (study_materials_s1/s2) plus their transitive same-dir imports
      (instances_s1/s2).

Run at MATERIAL FREEZE time (plan §5 step 5), after the dev selection has
finalized the six record texts; --out overrides the receipt path (draft
use), --seed overrides the scheduling seed (default 20261013).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
ROOT = EXPERIMENTS.parent
MAT = EXPERIMENTS / "_e17mg_materials"
SHARED_MAT = EXPERIMENTS / "_e17_materials"
E17_DIR = HERE.parent / "_e17"
E11_DIR = HERE.parent / "_e11"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--out", default=str(HERE / "freeze_mg.json"))
    args = parser.parse_args()

    new_files = [HERE / "run_episode_mg.py", HERE / "run_all_mg.py",
                 HERE / "precheck_mg_dev.py", HERE / "check_episode_mg.py"]
    new_files += sorted(MAT.glob("*.py"))
    reused_refs = [E17_DIR / "freeze.json", E17_DIR / "study_config.json",
                   E17_DIR / "run_episode.py"]
    shared_materials = sorted(SHARED_MAT.glob("*.py"))
    e11_templates = [E11_DIR / "study_materials_s1.py",
                     E11_DIR / "study_materials_s2.py",
                     E11_DIR / "instances_s1.py",
                     E11_DIR / "instances_s2.py"]

    missing = [str(f) for f in (new_files + reused_refs + shared_materials
                                + e11_templates) if not f.exists()]
    if missing:
        raise SystemExit(f"freeze blocked, missing: {missing}")

    freeze = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "receipt_kind": "release-copy-20261019; not the historical experiment freeze",
        "seed": args.seed,
        "batch": "e17mg_v1",
        # Keys are REPO-ROOT-RELATIVE ("experiments/ADK/_e17mg/..."),
        # one base directory for every section (review fix 1).
        "new_files": {p.relative_to(ROOT).as_posix(): sha(p)
                      for p in new_files},
        "reused_from_e17_v1": {p.relative_to(ROOT).as_posix(): sha(p)
                               for p in reused_refs},
        "shared_e17_materials": {p.relative_to(ROOT).as_posix(): sha(p)
                                 for p in shared_materials},
        "e11_scene_templates": {p.relative_to(ROOT).as_posix(): sha(p)
                                for p in e11_templates},
        "note": "e17_v1 files are NOT modified by e17mg; reused hashes are "
                "recorded for drift detection only",
    }
    out = Path(args.out).resolve()
    if _RELEASE_ONLINE not in out.parents:
        raise ValueError("Release receipts must remain inside online/")
    out.write_bytes(json.dumps(freeze, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"wrote {out} ({len(new_files)} new files, "
          f"{len(reused_refs)} reused refs, "
          f"{len(shared_materials)} shared e17 materials, "
          f"{len(e11_templates)} e11 scene templates)")


if __name__ == "__main__":
    main()
