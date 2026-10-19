"""Write freeze_mg.json for the e17mg_v1 batch.

Freezes everything this batch depends on BEYOND the e17_v1 receipt:
the new _e17mg_materials sources, this directory's runner/driver/precheck/
checker, plus drift-detection hashes of the reused frozen inputs (the
e17_v1 freeze.json itself, study_config.json and run_episode.py it
delegates to) and the scheduling seed.

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
E17_DIR = HERE.parent / "_e17"


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

    missing = [str(f) for f in new_files + reused_refs if not f.exists()]
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
        "note": "e17_v1 files are NOT modified by e17mg; reused hashes are "
                "recorded for drift detection only",
    }
    out = Path(args.out).resolve()
    if _RELEASE_ONLINE not in out.parents:
        raise ValueError("Release receipts must remain inside online/")
    out.write_bytes(json.dumps(freeze, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"wrote {out} ({len(new_files)} new files, "
          f"{len(reused_refs)} reused refs)")


if __name__ == "__main__":
    main()
