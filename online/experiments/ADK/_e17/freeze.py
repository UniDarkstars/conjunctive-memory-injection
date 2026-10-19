"""Write freeze.json: hashes of everything a formal batch depends on.

Freezes the study config, frozen external payloads and the shared
material sources (task suite, pools, constructors, budget) plus the
scheduling seed, so a batch's exact conditions are reproducible from one
receipt file (experiment.md E17 §9 step 4).
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
_EXPERIMENTS = HERE.parents[1]
MAT = _EXPERIMENTS / "_e17_materials"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()

    files = [HERE / "study_config.json"]
    files += sorted((HERE / "external" / "payloads").glob("frozen_payload_*.json"))
    files += sorted(MAT.glob("*.py"))
    files += [Path(__file__).resolve().parent / "run_episode.py"]

    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"freeze blocked, missing: {missing}")

    freeze = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "receipt_kind": "release-copy-20261019; not the historical experiment freeze",
        "seed": args.seed,
        # All paths relative to the REPO experiments/ dir (single base).
        "files": {
            f.resolve().relative_to(_EXPERIMENTS).as_posix(): sha(f)
            for f in files
        },
    }
    out = HERE / "freeze.json"
    out.write_bytes(json.dumps(freeze, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"freeze.json written ({len(freeze['files'])} files, seed {args.seed})")


if __name__ == "__main__":
    main()
