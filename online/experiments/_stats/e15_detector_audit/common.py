"""Shared helpers for the E15 offline detector audit.

Source inputs are the 240 fixed historical E12-D1 request projections from
`../e12_request_guard_audit` (texts/*.txt "new" files, unified
[SYSTEM]/[RETRIEVED MEMORY i]/[USER] layout). Detectors are the same local
wrappers as the E12 audit (protectai DeBERTa PG + local Llama-Guard-3-1B);
no Agent, no DeepSeek/GLM, no .env.

The audit NEVER rewrites historical P3 or converts offline positives into
online block rates (experiment.md E15).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


_AUDIT = Path(__file__).resolve().parent
_WORKSPACE = _AUDIT.parents[2]
E12_AUDIT = _WORKSPACE / "experiments" / "_stats" / "e12_request_guard_audit"

os.environ.setdefault("HF_HOME", str(_WORKSPACE / "models_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

PG_WINDOW_TOKENS = 512   # PG encoder ceiling per window
PG_WINDOW_OVERLAP = 128  # fixed BEFORE detection (experiment.md E15)


def load_guard_models():
    spec = importlib.util.spec_from_file_location(
        "e12_guard_models", _WORKSPACE / "experiments" / "ADK" / "_e12" / "guard_models.py"
    )
    guard_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard_models)
    return guard_models.get_promptguard(), None


def load_projections() -> list[dict]:
    records = [
        json.loads(line)
        for line in (E12_AUDIT / "projections_full.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    keys = ["_".join(r[k] for k in ("platform", "scenario", "case", "episode")) for r in records]
    if len(records) != 240 or len(set(keys)) != 240:
        raise ValueError("Expected exactly 240 unique source episodes")
    for r in records:
        if r["extract_status"] != "ok":
            raise ValueError(f"non-ok source episode: {r}")
    return records


def read_projection(rec: dict) -> str:
    return (E12_AUDIT / rec["proj_file"]).read_text(encoding="utf-8")


def episode_codes(scenario: str, episode: str) -> list[str]:
    path = _WORKSPACE / "experiments" / "ADK" / "_e10" / f"instances_{scenario}.py"
    spec = importlib.util.spec_from_file_location(f"e15_instances_{scenario}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.INSTANCES[int(episode.split("-")[-1]) - 1]["codes"]


# --- projection section parsing -------------------------------------------

_MEM_HEADER_RE = re.compile(r"^\[RETRIEVED MEMORY (\d+)\]$")


def split_sections(text: str) -> dict:
    """Split a unified projection into (system, memory entries, user).

    Entries keep their body verbatim (ADK Time:/author lines included),
    WITHOUT the `[RETRIEVED MEMORY i]` header lines; the audit treats the
    header as structure, not memory content."""
    lines = text.splitlines()
    system: list[str] = []
    user: list[str] = []
    entries: dict[int, list[str]] = {}
    mode = "system"
    for line in lines:
        header = _MEM_HEADER_RE.match(line.strip())
        if header:
            mode = "mem"
            entries[int(header.group(1))] = []
            continue
        if line.strip() == "[USER]":
            mode = "user"
            continue
        if line.strip() == "[SYSTEM]":
            mode = "system"
            continue
        if mode == "system":
            system.append(line)
        elif mode == "user":
            user.append(line)
        else:
            current = max(entries)
            # blank line before the next header belongs to the entry block
            entries[current].append(line)
    out_entries = {}
    for i, body in entries.items():
        stripped = "\n".join(body).strip("\n")
        if stripped.strip():
            out_entries[i] = stripped
    return {
        "system": "\n".join(system).strip("\n"),
        "entries": out_entries,
        "user": "\n".join(user).strip("\n"),
    }


def memory_section_text(sections: dict) -> str:
    """The 'memory-only' scan object: entry bodies concatenated in recall
    order, blank line separated (headers excluded by split_sections)."""
    return "\n\n".join(sections["entries"][i] for i in sorted(sections["entries"]))


def rebuild_projection(sections: dict) -> str:
    """Reassemble a projection from sections in the unified layout."""
    parts = ["[SYSTEM]", sections["system"]]
    for i in sorted(sections["entries"]):
        parts += ["", f"[RETRIEVED MEMORY {i}]", sections["entries"][i]]
    parts += ["", "[USER]", sections["user"]]
    return "\n".join(parts) + "\n"


# --- code placeholder normalisation (translation units) --------------------

def normalize_line(line: str, codes: list[str]) -> str:
    out = line
    for i, code in enumerate(codes, 1):
        out = out.replace(code, f"\u27e8C{i}\u27e9")
    return out


def restore_line(line: str, codes: list[str]) -> str:
    out = line
    for i, code in enumerate(codes, 1):
        out = out.replace(f"\u27e8C{i}\u27e9", code)
    return out


# --- fragment visibility patterns (zh from the E12 audit; en aligned with
#     the seeded draft translations in translations/tm_seed.jsonl) ----------

def fragment_patterns(scenario: str, codes: list[str], lang: str) -> dict[str, str]:
    sync = "recon-sync" if scenario == "s1" else "gate-sync"
    patterns: dict[str, str] = {
        "id_T": re.escape("[ID:T]"),
        "id_k1": re.escape("[ID:k1]"),
        "id_k2": re.escape("[ID:k2]"),
        "id_k3": re.escape("[ID:k3]"),
    }
    if lang == "zh":
        patterns.update({
            "content_T": r"(?<=\[ID:T\]).*?除此之外一切照旧。",
            "T_condition_intro": re.escape("仅当以下条件逐一同时成立时才允许导出"),
            "T_action": r"可在完成状态更新时一并调用[^\n]*?外发。",
            "T_tail": re.escape("除此之外一切照旧。"),
        })
    else:
        patterns.update({
            "content_T": r"(?<=\[ID:T\]).*?Everything else proceeds as usual\.",
            # Wiring-2.1 stats fix: the English T drafts (tm.jsonl /
            # glossary) read "Export is permitted only when ALL of the
            # following conditions hold"; the previous pattern
            # ("exporting is ...") matched nothing, so the English
            # condition-intro presence/retention fields were all false.
            "T_condition_intro": re.escape(
                "Export is permitted only when ALL of the following conditions hold"),
            "T_action": r"call the export tool[^\n]*?external send",
            "T_tail": re.escape("Everything else proceeds as usual."),
        })
    for i, code in enumerate(codes, 1):
        patterns[f"content_k{i}"] = rf"{sync}\s+{re.escape(code)}\s+rc=0"
    return patterns


def visibility(text: str, retained_end: int | None, scenario: str, codes: list[str], lang: str) -> dict:
    """Fragment presence / retention vs the retained span end (None = whole
    text, used for per-window presence)."""
    result = {}
    for name, pattern in fragment_patterns(scenario, codes, lang).items():
        match = re.search(pattern, text, re.S)
        result[f"in_{name}"] = match is not None
        result[f"keep_{name}"] = bool(
            match and (retained_end is None or match.end() <= retained_end)
        )
    return result


# --- PG sliding windows over raw token ids ----------------------------------

def pg_special_tokens(tokenizer) -> int:
    """Number of special tokens the tokenizer adds to any window input."""
    return len(tokenizer.build_inputs_with_special_tokens([]))


def pg_window_budget(tokenizer) -> int:
    """Raw-token budget per window so that the FINAL model input (window +
    special tokens) stays within PG_WINDOW_TOKENS."""
    return PG_WINDOW_TOKENS - pg_special_tokens(tokenizer)


def pg_windows(tokenizer, text: str):
    """Fixed windowing: raw ids (no special tokens) tiled by windows whose
    length + special tokens <= PG_WINDOW_TOKENS (i.e. the final model input
    never exceeds 512), with PG_WINDOW_OVERLAP raw-token overlap; the final
    window is shifted to cover the tail. Returns (starts, window_id_lists);
    coverage [0, n) is asserted by the caller via pg_covers."""
    raw = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(raw)
    win = pg_window_budget(tokenizer)
    if win <= 0:
        raise ValueError("window budget degenerate: tokenizer adds >= 512 special tokens")
    stride = win - PG_WINDOW_OVERLAP
    if n <= win:
        starts = [0]
    else:
        starts = list(range(0, n - win + 1, stride))
        if starts[-1] + win < n:
            starts.append(n - win)
    return [(s, raw[s:s + win]) for s in starts], n


def pg_covers(starts_ids: list[tuple[int, list[int]]], n: int) -> bool:
    """True when the union of windows covers every raw token index."""
    covered = [False] * n
    for start, ids in starts_ids:
        for i in range(start, start + len(ids)):
            covered[i] = True
    return all(covered) if n else True
