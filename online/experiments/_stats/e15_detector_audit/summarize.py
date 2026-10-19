"""Summarize the E15 offline detector audit.

Reads the PER-LANGUAGE result files (results_episodes_{lang}.jsonl); merges
them on the episode key. For every platform x scenario x case x language:

  - attack positives (case=both) and clean positives (case=clean) for every
    detector x scan object;
  - empty-memory denominators: episodes without memory entries are counted
    and excluded from (but printed next to) the memory-scan denominators;
  - content coverage: prefix-scan truncation rates, T / three-key retention
    rates, and window-level T / T+all-three-key presence (full coverage is
    asserted at run time; these are the retention marginals);
  - the COMPLETE exploratory set of paired same-task contrasts: every
    scan-object contrast within a language and every language contrast
    within a scan object (b/c discordant counts + conservative CP RD
    interval). Descriptive only: no thresholding, no family-wise
    correction, no filtering for significance.

PG labels are prompt-injection judgments; LG labels are content-safety
judgments. Offline positives never rewrite historical P3 and are not
online block rates (experiment.md E15).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


PG_CONFIGS = ("full_prefix", "mem_prefix", "mem_windows")
LANGS = ("zh", "en")


def cp_bound(x: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact binomial-tail inversion, without an approximate fallback
    (same construction as the e12 audit; no scipy dependency)."""
    if not 0 <= x <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid CP arguments")

    def invert(k: int, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            p = (lo + hi) / 2
            cdf = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
            if cdf > target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return (0.0 if x == 0 else invert(x - 1, 1 - alpha / 2),
            1.0 if x == n else invert(x, alpha / 2))


def _cp(x: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (0.0, 1.0)
    return cp_bound(x, n, alpha)


def _rd_ci(b: int, c: int, n: int):
    lo_b, hi_b = _cp(b, n, 0.025)
    lo_c, hi_c = _cp(c, n, 0.025)
    return ((b - c) / n if n else 0.0), (lo_b - hi_c, hi_b - lo_c)


def load_rows(audit_dir: Path) -> tuple[dict[str, list[dict]], list[str], dict[str, str]]:
    """Rows per lang, merged on the episode key. Source priority per lang:
    results_episodes_{lang}.jsonl (fixed wiring) -> for zh the archived
    results_episodes_zh_wiring1.jsonl (pre window-length fix) -> smoke
    files only when nothing else exists. Returns the source actually used
    per lang so the caller can warn about wiring-1 window columns."""
    sources = {}
    by_lang: dict[str, dict[tuple, dict]] = {}
    for lang in LANGS:
        candidates = []
        if lang == "en":
            # wiring-2.1 stats correction (condition-intro fields) takes
            # priority; detection labels/tokens/truncation are identical
            candidates.append("results_episodes_en_fixed.jsonl")
        candidates.append(f"results_episodes_{lang}.jsonl")
        if lang == "zh":
            candidates.append("results_episodes_zh_wiring1.jsonl")
        candidates.append(f"results_episodes_smoke_{lang}.jsonl")
        for name in candidates:
            path = audit_dir / name
            if not path.exists():
                continue
            bucket = by_lang.setdefault(lang, {})
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["platform"], r["scenario"], r["case"], r["episode"])
                bucket.setdefault(key, r)
            sources[lang] = name
            break
    rows = {lang: [bucket[k] for k in sorted(bucket)] for lang, bucket in by_lang.items()}
    return rows, [l for l in LANGS if l in rows], sources


def _b(v):
    return v is True or v == "True" or v == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", default=str(run_dir_for(__file__)))
    parser.add_argument("--windows-dir", default=str(run_dir_for(__file__)))
    parser.add_argument("--out-dir", default=str(run_dir_for(__file__)))
    args = parser.parse_args()
    validate_output_args(args)

    rows_by_lang, langs, sources = load_rows(Path(args.episodes_dir))
    if not langs:
        print("no per-language result files found "
              "(results_episodes_{zh,en}.jsonl); nothing to summarise")
        return
    for lang in langs:
        print(f"[{lang}] source file: {sources[lang]}")
        if "_fixed" in sources[lang]:
            print(f"NOTE [{lang}]: stats-corrected condition-intro fields "
                  "(wiring-2.1; see coverage_fix_provenance.json) - derived file, "
                  "no model rerun; all other fields identical to the source run.")
        if "wiring1" in sources[lang]:
            print(f"WARNING [{lang}]: wiring-1 archive - PG mem_windows columns came from "
                  "514-token final inputs (pre window-length fix, raw 512 + 2 special "
                  "tokens); prefix-scan and LG columns are unaffected. Rerun "
                  f"run_audit.py --langs zh to refresh, or keep as-is with this caveat.")
    out_dir = Path(args.out_dir)

    print("=== positives per cell (attack = case both; clean = case clean) ===")
    print("memory-scan denominators exclude episodes with no memory entries "
          "(no_mem shown separately)")
    print(f"{'cell':<18} {'lang':<4} {'no_mem':>6} "
          f"{'PGfull':>12} {'PGmem':>12} {'PGwin':>12} {'LG':>10}")
    for platform in ("ADK", "LangMem"):
        for scenario in ("s1", "s2"):
            for case in ("both", "clean"):
                for lang in langs:
                    sel = [r for r in rows_by_lang[lang] if r["platform"] == platform
                           and r["scenario"] == scenario and r["case"] == case]
                    if not sel:
                        continue
                    no_mem = sum(1 for r in sel
                                 if r.get(f"{lang}_pg_mem_prefix_no_memory")
                                 or r.get(f"{lang}_pg_mem_windows_no_memory")
                                 or r.get("n_mem_entries") == 0)
                    vals = []
                    for cfg in PG_CONFIGS + ("lg",):
                        field = (f"{lang}_pg_{cfg}_detected" if cfg != "lg"
                                 else f"{lang}_lg_detected")
                        denom = len(sel)
                        if cfg != "lg":
                            denom -= sum(1 for r in sel
                                         if r.get(f"{lang}_pg_{cfg}_no_memory"))
                        have = [r for r in sel if field in r]
                        pos = sum(1 for r in have if _b(r.get(field)))
                        vals.append(f"{pos}/{denom}" if have else "-")
                    print(f"{platform+'/'+scenario+'/'+case:<18} {lang:<4} {no_mem:>6} "
                          f"{vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>10}")

    print("\n=== input / truncation / retention stats ===")
    for platform in ("ADK", "LangMem"):
        for scenario in ("s1", "s2"):
            for case in ("both", "clean"):
                for lang in langs:
                    sel = [r for r in rows_by_lang[lang] if r["platform"] == platform
                           and r["scenario"] == scenario and r["case"] == case]
                    bits = []
                    for label in ("full_prefix", "mem_prefix"):
                        have = [r for r in sel if f"{lang}_pg_{label}_tokens_full" in r]
                        if not have:
                            continue
                        tok = [r[f"{lang}_pg_{label}_tokens_full"] for r in have]
                        tr = sum(1 for r in have if r.get(f"{lang}_pg_{label}_truncated"))
                        keep_t = sum(1 for r in have if r.get(f"{lang}_pg_{label}_keep_content_T"))
                        keep_k = sum(1 for r in have if all(
                            r.get(f"{lang}_pg_{label}_keep_content_k{i}") for i in (1, 2, 3)))
                        intro_in = sum(1 for r in have
                                       if r.get(f"{lang}_pg_{label}_in_T_condition_intro"))
                        intro_keep = sum(1 for r in have
                                         if r.get(f"{lang}_pg_{label}_keep_T_condition_intro"))
                        n = len(have)
                        bits.append(
                            f"{label}: tok_avg={sum(tok)/n:.0f} trunc={tr}/{n} "
                            f"keepT={keep_t}/{n} keepK3={keep_k}/{n} "
                            f"introIn={intro_in}/{n} introKeep={intro_keep}/{n}")
                    if bits:
                        print(f"{platform}/{scenario}/{case} [{lang}]  " + " | ".join(bits))
    for lang in langs:
        tok = [r[f"{lang}_lg_input_tokens"] for r in rows_by_lang[lang]
               if f"{lang}_lg_input_tokens" in r]
        if tok:
            print(f"LG input tokens [{lang}]: avg={sum(tok)/len(tok):.0f} "
                  f"max={max(tok)} n={len(tok)}")

    # window-level content coverage marginals (same source priority as the
    # episode files: fixed-wiring first, then the zh wiring-1 archive, then
    # smoke only if nothing else exists)
    for lang in langs:
        names = []
        if lang == "en":
            names.append("results_windows_en_fixed.jsonl")
        names.append(f"results_windows_{lang}.jsonl")
        if lang == "zh":
            names.append("results_windows_zh_wiring1.jsonl")
        names.append(f"results_windows_smoke_{lang}.jsonl")
        for name in names:
            win_path = Path(args.windows_dir) / name
            if not win_path.exists():
                continue
            wiring = ("wiring-1 archive: raw 512-token windows + 2 special tokens = "
                      "514-token final inputs (pre window-length fix)"
                      if "wiring1" in name else
                      ("stats-corrected condition-intro fields (wiring-2.1); "
                       "source: results_windows_en.jsonl"
                       if "_fixed" in name else
                       f"fixed budget {512} incl. special tokens, overlap {128}"))
            wins = [json.loads(l) for l in win_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            n_pos = sum(1 for w in wins if w["label_id"] != 0)
            any_t = sum(1 for w in wins if w.get("in_content_T"))
            any_all4 = sum(1 for w in wins if w.get("in_content_T")
                           and all(w.get(f"in_content_k{i}") for i in (1, 2, 3)))
            any_intro = sum(1 for w in wins if w.get("in_T_condition_intro"))
            intro_in_t = sum(1 for w in wins if w.get("in_content_T")
                             and w.get("in_T_condition_intro"))
            print(f"\n=== windows [{lang}] ({name}; {wiring}) ===")
            print(f"windows={len(wins)} positive={n_pos} containing_T={any_t} "
                  f"containing_T+all3keys={any_all4} containing_intro={any_intro} "
                  f"(intro within T-windows: {intro_in_t}/{any_t})")
            print("(full coverage does not imply any single window holds T plus all keys; "
                  "more windows can raise clean positives - both counts reported above)")
            break

    # complete paired contrast set (exploratory)
    pair_rows = []
    for platform in ("ADK", "LangMem"):
        for scenario in ("s1", "s2"):
            for case in ("both", "clean"):
                sel = {r["episode"]: r for r in rows_by_lang[langs[0]]
                       if r["platform"] == platform and r["scenario"] == scenario
                       and r["case"] == case}
                episodes = sorted(sel)
                # language contrasts (fixed scan object), zh vs en
                if len(langs) == 2:
                    sel_b = {r["episode"]: r for r in rows_by_lang[langs[1]]
                             if r["platform"] == platform and r["scenario"] == scenario
                             and r["case"] == case}
                    for cfg in PG_CONFIGS + ("lg",):
                        fa = f"{langs[0]}_pg_{cfg}_detected" if cfg != "lg" else f"{langs[0]}_lg_detected"
                        fb = f"{langs[1]}_pg_{cfg}_detected" if cfg != "lg" else f"{langs[1]}_lg_detected"
                        pair_rows.append(_pair(platform, scenario, case,
                                               f"{fa} vs {fb}", sel, sel_b, fa, fb))
                # scan-object contrasts (fixed language)
                for lang in langs:
                    s = {r["episode"]: r for r in rows_by_lang[lang]
                         if r["platform"] == platform and r["scenario"] == scenario
                         and r["case"] == case}
                    eps = sorted(s)
                    for a, c in (("full_prefix", "mem_prefix"),
                                 ("full_prefix", "mem_windows"),
                                 ("mem_prefix", "mem_windows")):
                        fa, fb = f"{lang}_pg_{a}_detected", f"{lang}_pg_{c}_detected"
                        pair_rows.append(_pair(platform, scenario, case,
                                               f"{fa} vs {fb}", s, s, fa, fb))
    pair_rows = [p for p in pair_rows if p]
    if pair_rows:
        print("\n=== complete paired contrast set (exploratory; same task, offline labels) ===")
        with (out_dir / "paired_contrasts.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
        print(f"wrote {out_dir / 'paired_contrasts.csv'} ({len(pair_rows)} contrasts)")
        for p in pair_rows:
            if p["b"] or p["c"]:
                print(f"  {p['cell']:<18} {p['contrast']:<58} b={p['b']} c={p['c']} "
                      f"RD={p['rd']:+.3f} [{p['ci_lo']:+.3f},{p['ci_hi']:+.3f}]")

    print("\nReminder: PG labels are prompt-injection judgments, LG labels are content-safety "
          "judgments; no thresholds were tuned; offline positives do not rewrite historical "
          "P3 and are not online block rates.")


def _pair(platform, scenario, case, label, sel_a, sel_b, fa, fb):
    def g(r, f):
        v = r.get(f)
        return v is True or v == "True" or v == 1

    have = [e for e in sorted(set(sel_a) & set(sel_b))
            if fa in sel_a[e] and fb in sel_b[e]]
    if not have:
        return None
    bb = sum(1 for e in have if g(sel_a[e], fa) and not g(sel_b[e], fb))
    cc = sum(1 for e in have if not g(sel_a[e], fa) and g(sel_b[e], fb))
    rd, (lo, hi) = _rd_ci(bb, cc, len(have))
    return {"cell": f"{platform}/{scenario}/{case}", "contrast": label, "n": len(have),
            "b": bb, "c": cc, "rd": round(rd, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


if __name__ == "__main__":
    main()
