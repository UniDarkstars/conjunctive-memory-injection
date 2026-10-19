"""Small shared file/identity guards for new E16 runs (no API calls)."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode

import sys
import tempfile

RUNTIME_REVISION = "e16-runtime-2"


class IncompleteRun(RuntimeError):
    def __init__(self, termination_reason: str):
        self.termination_reason = termination_reason
        super().__init__(termination_reason)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


@contextmanager
def exclusive_lock(path: Path):
    """Fail on an existing lock; never steal a lock from another run."""
    path = Path(path)
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Run lock already exists: {path}") from exc
    try:
        stream.write(json.dumps({"pid": os.getpid()}))
        stream.close()
        yield
    finally:
        stream.close()
        path.unlink(missing_ok=True)


def ensure_batch_manifest(run_root: Path, expected: dict) -> dict:
    """Keep a model/platform/config identity fixed for the whole directory."""
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "batch_manifest.json"
    with exclusive_lock(run_root / ".manifest.lock"):
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            differences = [key for key, value in expected.items()
                           if saved.get(key) != value]
            if differences:
                raise ValueError(f"Batch identity/config mismatch ({', '.join(differences)}): {run_root}")
            return saved
        existing = [p for p in run_root.iterdir()
                    if p.name not in {".manifest.lock", ".batch.lock"}]
        if existing:
            raise FileExistsError(f"Existing unversioned batch; use a new run root: {run_root}")
        atomic_write_json(path, expected)
        return dict(expected)


def prepare_episode_dir(run_root: Path, scenario: str, arm: str,
                        episode_index: int, platform: str, model_id: str) -> Path:
    run_root = Path(run_root).resolve()
    ensure_batch_manifest(run_root, {
        "experiment": "e16", "platform": platform, "model_id": model_id,
        "runtime_revision": RUNTIME_REVISION, "batch_id": run_root.name,
    })
    run_dir = run_root / scenario / arm / f"epi-{episode_index:02d}"
    # mkdir is the exclusive claim for this episode. Existing observations
    # are never erased; a failed attempt needs its own new batch directory.
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "status.json", {
        "runtime_revision": RUNTIME_REVISION, "status": "running",
        "completed": False, "artifacts_complete": False,
        "termination_reason": "running",
    })
    return run_dir


def runtime_metadata() -> dict:
    try:
        sdk_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = None
    return {"python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "openai_version": sdk_version}
