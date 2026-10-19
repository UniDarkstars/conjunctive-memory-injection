"""Local output paths and safe defaults shared by release copies only."""
from __future__ import annotations

import os
from pathlib import Path

ONLINE = Path(__file__).resolve().parent
REPOSITORY = ONLINE.parent
RUNS = ONLINE / 'runs'
OUTPUTS = REPOSITORY / 'outputs' / 'online'

# Do not force a credential-bearing account-specific Hugging Face directory.
os.environ.setdefault('HF_HOME', str(ONLINE / 'models_cache'))


def run_dir_for(filename):
    if os.environ.get('CMI_RUN_ROOT'):
        return checked_output(os.environ['CMI_RUN_ROOT'])
    relative = Path(filename).resolve().relative_to(ONLINE)
    parts = relative.parts
    if parts[0] == 'experiments' and len(parts) > 2:
        return RUNS / parts[1] / parts[2].lstrip('_')
    return RUNS / 'misc'


def checked_output(path):
    path = Path(path).resolve()
    if not any(path == base or base in path.parents for base in (RUNS, OUTPUTS)):
        raise ValueError(f'Output must stay inside {RUNS} or {OUTPUTS}: {path}')
    return path


def validate_output_args(args):
    for name in ('run_root', 'out_dir', 'csv', 'output', 'out', 'root', 'run_dir'):
        value = getattr(args, name, None)
        if value:
            checked_output(value)
    # Episode checkers write verdicts into each supplied run directory.
    for value in getattr(args, 'run_dirs', []) or []:
        checked_output(value)


def prepare_fresh_episode(path):
    path = checked_output(path)
    # A rerun must use a new directory; it cannot mix old and new evidence.
    path.mkdir(parents=True, exist_ok=False)
    (path / 'tool_calls.jsonl').write_text('', encoding='utf8')
    return path
