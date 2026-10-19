"""Run the E15 offline detector audit over the 240 historical D1 request
projections (no Agent, no chat model, local GPU/CPU only).

Scan objects (experiment.md E15), fixed BEFORE detection:
  PG (protectai DeBERTa, argmax, no thresholds)
    pg_full_prefix : whole projection, PG's own 512-token prefix truncation
    pg_mem_prefix  : memory entries only (recall order), same prefix truncation
    pg_mem_windows : per memory entry, raw-token windows whose length plus
                     special tokens stays <= 512 (128 raw-token overlap,
                     final window shifted to cover the tail), every window
                     classified; entry positive = any window INJECTION,
                     episode positive = any entry. Coverage asserted.

Languages: zh = the archived projection verbatim; en = line-level
translation memory (codes restored verbatim). For EVERY language the scan
texts and the memory sections are re-derived from THAT language's text, so
the English memory scans actually scan English. The en arm runs only when
every unit used by all 240 projections is marked reviewed in
translations/tm.jsonl (or with --allow-partial-en, which then reports
coverage and runs covered episodes only; --force-en ignores review status
for debugging and stamps the run as unreviewed).

Outputs are per language and never overwrite each other or prior runs:
  results_episodes[_smoke]_{lang}.jsonl/.csv
  results_windows[_smoke]_{lang}.jsonl
Offline positives never rewrite historical P3 and are not online block rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


import common

_TIME_RE = re.compile(r"^Time: .+$")


def load_tm() -> dict[str, dict]:
    tm_path = common._AUDIT / "translations" / "tm.jsonl"
    tm = {}
    for line in tm_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            tm[row["zh"]] = row
    return tm


def is_structural(line: str) -> bool:
    s = line.strip()
    return (
        not s
        or s in ("[SYSTEM]", "[USER]")
        or bool(common._MEM_HEADER_RE.match(s))
        or bool(_TIME_RE.match(s))
    )


def assemble_en(text_zh: str, codes: list[str], tm: dict[str, dict]) -> tuple[str, list[str]]:
    """Line-level TM restore. Returns (en_text, missing_unit_list)."""
    out, missing = [], []
    for line in text_zh.splitlines():
        if is_structural(line):
            out.append(line)
            continue
        unit = common.normalize_line(line, codes)
        entry = tm.get(unit)
        if entry and entry.get("en"):
            out.append(common.restore_line(entry["en"], codes))
        else:
            missing.append(unit)
    return "\n".join(out), missing


def en_ready(records: list[dict], tm: dict[str, dict]) -> tuple[bool, dict]:
    """Every unit used by every projection must have a reviewed en entry."""
    per_episode: dict[str, list[str]] = {}
    unreviewed_used: set[str] = set()
    for rec in records:
        codes = common.episode_codes(rec["scenario"], rec["episode"])
        text = common.read_projection(rec)
        _, missing = assemble_en(text, codes, tm)
        key = "_".join(rec[k] for k in ("platform", "scenario", "case", "episode"))
        if missing:
            per_episode[key] = missing
        for line in text.splitlines():
            if is_structural(line):
                continue
            entry = tm.get(common.normalize_line(line, codes))
            if entry and entry.get("en") and not entry.get("reviewed"):
                unreviewed_used.add(common.normalize_line(line, codes))
    ok = not per_episode and not unreviewed_used
    return ok, {"episodes_missing": per_episode, "unreviewed_used": sorted(unreviewed_used)[:20]}


def texts_for_lang(lang: str, text_zh: str, codes: list[str], tm: dict[str, dict]) -> tuple[str, dict, list[str]]:
    """Per-language scan inputs: the scan text AND its memory sections are
    both re-derived from the same language's text."""
    if lang == "zh":
        return text_zh, common.split_sections(text_zh), []
    text_en, missing = assemble_en(text_zh, codes, tm)
    return text_en, common.split_sections(text_en), missing


def pg_classify_ids(pg, tokenizer, window_ids: list[int]) -> dict:
    """Classify one raw-token window; special tokens are re-added per window
    and the FINAL model input is asserted to stay within 512 tokens."""
    import torch

    enc = tokenizer.prepare_for_model(window_ids)
    batch = {
        k: torch.tensor([v]).to(pg._device)
        for k, v in enc.items() if k in tokenizer.model_input_names
    }
    assert len(batch["input_ids"][0]) <= common.PG_WINDOW_TOKENS, (
        f"window input {len(batch['input_ids'][0])} tokens exceeds "
        f"{common.PG_WINDOW_TOKENS} after special tokens"
    )
    with pg._lock, torch.no_grad():
        logits = pg._model(**batch).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = int(logits.argmax(dim=-1).item())
    return {
        "label_id": pred,
        "p_injection": float(probs[1].item()) if len(probs) > 1 else None,
    }


def run_config_set(text: str, sections: dict, lang: str, rec: dict, codes: list[str], pg, lg,
                   win_stream, key: str) -> dict:
    scenario = rec["scenario"]
    out: dict = {}

    def prefix_stats(scan_text: str, label: str):
        full = pg._tok(scan_text)
        kept = pg._tok(scan_text, truncation=True, max_length=512, return_offsets_mapping=True)
        end = max((b for a, b in kept["offset_mapping"] if b > a), default=0)
        verdict = pg.classify(scan_text)
        vis = common.visibility(scan_text, end, scenario, codes, lang)
        out[f"pg_{label}_tokens_full"] = len(full["input_ids"])
        out[f"pg_{label}_tokens_retained"] = len(kept["input_ids"])
        out[f"pg_{label}_truncated"] = len(full["input_ids"]) > len(kept["input_ids"])
        out[f"pg_{label}_retained_end"] = end
        out[f"pg_{label}_detected"] = verdict["blocked"]
        out[f"pg_{label}_p_injection"] = verdict["p_injection"]
        out.update({f"pg_{label}_{k}": v for k, v in vis.items()})

    # PG full projection prefix
    prefix_stats(text, "full_prefix")

    # PG memory-only prefix (sections are per-language)
    mem_text = common.memory_section_text(sections)
    if mem_text:
        prefix_stats(mem_text, "mem_prefix")
    else:
        out["pg_mem_prefix_no_memory"] = True

    # PG memory-only full-coverage windows
    if mem_text:
        n_entries = 0
        entry_positives = 0
        for idx in sorted(sections["entries"]):
            entry_text = sections["entries"][idx]
            wins, n_raw = common.pg_windows(pg._tok, entry_text)
            if not common.pg_covers(wins, n_raw):
                raise AssertionError(f"window coverage gap: {key} entry {idx}")
            entry_hit = False
            for w_start, w_ids in wins:
                verdict = pg_classify_ids(pg, pg._tok, w_ids)
                decoded = pg._tok.decode(w_ids, skip_special_tokens=True)
                vis = common.visibility(decoded, None, scenario, codes, lang)
                win_stream.write(json.dumps({
                    "episode_key": key, "lang": lang, "entry": idx,
                    "start": w_start, "tokens": len(w_ids),
                    "label_id": verdict["label_id"],
                    "p_injection": verdict["p_injection"],
                    "text": decoded, **vis,
                }, ensure_ascii=False) + "\n")
                if verdict["label_id"] != 0:
                    entry_hit = True
            n_entries += 1
            entry_positives += int(entry_hit)
        out["pg_mem_windows_entries"] = n_entries
        out["pg_mem_windows_entry_positives"] = entry_positives
        out["pg_mem_windows_detected"] = entry_positives > 0
    else:
        out["pg_mem_windows_no_memory"] = True

    # MiTA reports PG only; no Llama Guard weights are required.
    return out


def output_paths(lang: str, smoke: bool) -> tuple[Path, Path]:
    tag = "_smoke" if smoke else ""
    return (run_dir_for(__file__) / f"results_episodes{tag}_{lang}.jsonl",
            run_dir_for(__file__) / f"results_windows{tag}_{lang}.jsonl")


def row_complete(row: dict, lang: str) -> bool:
    """Completion for the langs actually run this invocation: zh needs the
    zh PG results; en either has full results or a recorded missing count."""
    if f"{lang}_pg_full_prefix_detected" in row:
        return True
    return lang == "en" and "en_missing_units" in row


def main() -> int:
    parser = argparse.ArgumentParser(description="E15 offline detector audit")
    parser.add_argument("--only", help="episode key prefix for a smoke run")
    parser.add_argument("--langs", default="zh,en", help="comma list; en skipped unless TM complete")
    parser.add_argument("--allow-partial-en", action="store_true")
    parser.add_argument("--force-en", action="store_true",
                        help="run en even with unreviewed TM entries (EXPLORATORY, stamped in output)")
    args = parser.parse_args()
    validate_output_args(args)

    records = common.load_projections()
    tm = load_tm()
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    if "en" in langs and not args.force_en:
        ok, diag = en_ready(records, tm)
        if not ok:
            if not args.allow_partial_en:
                print("en arm SKIPPED: translation memory incomplete/unreviewed.")
                print(f"  episodes with missing units: {len(diag['episodes_missing'])}")
                print(f"  used-but-unreviewed units (first 20 shown): {len(diag['unreviewed_used'])}")
                for u in diag["unreviewed_used"][:5]:
                    print(f"    {u[:70]}")
                langs = [l for l in langs if l != "en"]

    pg, lg = common.load_guard_models()
    smoke = bool(args.only)

    out_paths = {lang: output_paths(lang, smoke) for lang in langs}
    for ep_path, _ in out_paths.values():
        ep_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_lang: dict[str, list[dict]] = {lang: [] for lang in langs}
    started = time.monotonic()
    streams = {lang: ep_path.open("w", encoding="utf-8") for lang, (ep_path, _) in out_paths.items()}
    win_streams = {lang: w_path.open("w", encoding="utf-8") for lang, (_, w_path) in out_paths.items()}
    try:
        for rec in records:
            key = "_".join(rec[k] for k in ("platform", "scenario", "case", "episode"))
            if args.only and not key.startswith(args.only):
                continue
            codes = common.episode_codes(rec["scenario"], rec["episode"])
            text_zh = common.read_projection(rec)
            for lang in langs:
                text, sections, missing = texts_for_lang(lang, text_zh, codes, tm)
                row = {k: rec[k] for k in ("platform", "scenario", "case", "episode")}
                row["input_chars"] = len(text)
                row["n_mem_entries"] = len(sections["entries"])
                if missing:
                    row["en_missing_units"] = len(missing)
                else:
                    res = run_config_set(text, sections, lang, rec, codes, pg, lg,
                                         win_streams[lang], key)
                    row.update({f"{lang}_{k}": v for k, v in res.items()})
                    if lang == "en" and args.force_en and not tm_reviewed_all(rec, codes, tm):
                        row["en_unreviewed_drafts"] = True
                rows_by_lang[lang].append(row)
                streams[lang].write(json.dumps(row, ensure_ascii=False) + "\n")
                streams[lang].flush()
            done = sum(len(v) for v in rows_by_lang.values())
            if done % 20 == 0 or args.only:
                print(f"{done} episode-lang rows; {time.monotonic()-started:.1f}s", flush=True)
    finally:
        for s in list(streams.values()) + list(win_streams.values()):
            s.close()

    total_errors = 0
    for lang in langs:
        rows = rows_by_lang[lang]
        if not rows:
            continue
        ep_path, _ = out_paths[lang]
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with ep_path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        errors = sum(1 for r in rows if not row_complete(r, lang))
        total_errors += errors
        print(f"[{lang}] completed {len(rows)} episodes; {errors} incomplete rows; {ep_path.name}")

    return 1 if total_errors else 0


def tm_reviewed_all(rec: dict, codes: list[str], tm: dict[str, dict]) -> bool:
    text = common.read_projection(rec)
    for line in text.splitlines():
        if is_structural(line):
            continue
        entry = tm.get(common.normalize_line(line, codes))
        if not entry or not entry.get("en") or not entry.get("reviewed"):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
