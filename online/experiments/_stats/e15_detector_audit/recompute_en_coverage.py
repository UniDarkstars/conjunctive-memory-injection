"""Recompute the ENGLISH condition-intro coverage fields from existing
logs only (wiring-2.1 stats correction; NO model, Agent, PG or LG rerun).

What was wrong: common.fragment_patterns(lang="en") matched
"exporting is permitted ..." while the reviewed English material reads
"Export is permitted only when ALL of the following conditions hold", so
every English in/keep_T_condition_intro field was recorded false. Detection
labels, scores, token counts and truncation positions are unaffected.

Recomputation inputs (nothing is re-detected):
  full_prefix : English input restored from the source projection + the
                CURRENT translation memory (unchanged since the run); the
                SAVED en_pg_full_prefix_retained_end gives the truncation.
  mem_prefix  : the same restored English text, memory section rebuilt with
                the original run's exact order/format (split_sections ->
                memory_section_text); SAVED retained_end.
  mem_windows : the window texts SAVED in results_windows_en.jsonl
                (presence recomputed on the saved text).

Only in_T_condition_intro / keep_T_condition_intro change. Every other
field of every row is copied verbatim, and restoration fidelity is proven
by recomputing ALL other visibility fields and asserting they equal the
saved values (for both prefix configs; per window). Empty-memory requests
keep their original missing-field semantics.

Outputs (originals untouched):
  results_episodes_en_fixed.jsonl / .csv
  results_windows_en_fixed.jsonl
  coverage_fix_provenance.json
"""

from __future__ import annotations

import csv
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


import common
import run_audit

_FIELDS = ("id_T", "content_T", "content_k1", "content_k2", "content_k3",
           "T_action", "T_tail")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    audit = run_dir_for(__file__)
    src_ep = audit / "results_episodes_en.jsonl"
    src_win = audit / "results_windows_en.jsonl"
    records = {(r["platform"], r["scenario"], r["case"], r["episode"]): r
               for r in common.load_projections()}
    tm = run_audit.load_tm()

    src_rows = [json.loads(l) for l in src_ep.read_text(encoding="utf-8").splitlines() if l.strip()]
    src_wins = [json.loads(l) for l in src_win.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(src_rows) != 240:
        raise SystemExit(f"expected 240 source episode rows, got {len(src_rows)}")

    fixed_rows = []
    counts = {"full_intro_in": 0, "full_intro_keep": 0,
              "mem_intro_in": 0, "mem_intro_keep": 0,
              "full_denom": 0, "mem_denom": 0}
    mismatch = []

    for row in src_rows:
        rec = records[(row["platform"], row["scenario"], row["case"], row["episode"])]
        codes = common.episode_codes(rec["scenario"], rec["episode"])
        text_en, _, missing = run_audit.texts_for_lang(
            "en", common.read_projection(rec), codes, tm)
        if missing:
            raise SystemExit(f"TM no longer covers {rec['proj_file']} - cannot restore input")
        fixed = dict(row)

        sections = common.split_sections(text_en)

        for config, scan_text in (("full_prefix", text_en),):
            prefix = f"en_pg_{config}_"
            end = row[f"{prefix}retained_end"]
            vis = common.visibility(scan_text, end, rec["scenario"], codes, "en")
            for f in _FIELDS:  # restoration fidelity: other fields must reproduce
                for kind in ("in", "keep"):
                    a, b_ = row[f"{prefix}{kind}_{f}"], vis[f"{kind}_{f}"]
                    if bool(a) != bool(b_):
                        mismatch.append((row["episode"], config, f"{kind}_{f}", a, b_))
            fixed[f"{prefix}in_T_condition_intro"] = vis["in_T_condition_intro"]
            fixed[f"{prefix}keep_T_condition_intro"] = vis["keep_T_condition_intro"]
            counts["full_denom"] += 1
            counts["full_intro_in"] += int(vis["in_T_condition_intro"])
            counts["full_intro_keep"] += int(vis["keep_T_condition_intro"])

        if "en_pg_mem_prefix_retained_end" in row:  # memory present
            mem_text = common.memory_section_text(sections)
            end = row["en_pg_mem_prefix_retained_end"]
            vis = common.visibility(mem_text, end, rec["scenario"], codes, "en")
            for f in _FIELDS:
                for kind in ("in", "keep"):
                    a = row[f"en_pg_mem_prefix_{kind}_{f}"]
                    if bool(a) != bool(vis[f"{kind}_{f}"]):
                        mismatch.append((row["episode"], "mem_prefix", f"{kind}_{f}", a, vis[f"{kind}_{f}"]))
            fixed["en_pg_mem_prefix_in_T_condition_intro"] = vis["in_T_condition_intro"]
            fixed["en_pg_mem_prefix_keep_T_condition_intro"] = vis["keep_T_condition_intro"]
            counts["mem_denom"] += 1
            counts["mem_intro_in"] += int(vis["in_T_condition_intro"])
            counts["mem_intro_keep"] += int(vis["keep_T_condition_intro"])
        # empty-memory rows: mem fields stay exactly as saved (absent)

        fixed_rows.append(fixed)

    # whole-row invariance: apart from the 4 intro fields nothing may differ
    intro_keys = {f"en_pg_{c}_{k}_T_condition_intro"
                  for c in ("full_prefix", "mem_prefix") for k in ("in", "keep")}
    for orig, fixed in zip(src_rows, fixed_rows):
        assert set(orig) == set(fixed), f"field set changed for {orig['episode']}"
        for k, v in orig.items():
            if k not in intro_keys:
                assert fixed[k] == v, f"non-intro field changed: {orig['episode']} {k}"

    # windows: recompute presence on the SAVED window texts
    fixed_wins = []
    win_intro = 0
    win_intro_in_t_windows = 0
    t_windows = 0
    for w in src_wins:
        vis = common.visibility(w["text"], None, w["episode_key"].split("_")[1],
                                _codes_for(w["episode_key"]), "en")
        for f in _FIELDS:
            if bool(w.get(f"in_{f}")) != bool(vis[f"in_{f}"]):
                mismatch.append((w["episode_key"], "window", f"in_{f}", w.get(f"in_{f}"), vis[f"in_{f}"]))
        fw = dict(w)
        fw["in_T_condition_intro"] = vis["in_T_condition_intro"]
        fw["keep_T_condition_intro"] = vis["keep_T_condition_intro"]
        if vis["in_T_condition_intro"]:
            win_intro += 1
        if w.get("in_content_T"):
            t_windows += 1
            win_intro_in_t_windows += int(vis["in_T_condition_intro"])
        fixed_wins.append(fw)

    win_intro_keys = {"in_T_condition_intro", "keep_T_condition_intro"}
    for orig, fixed in zip(src_wins, fixed_wins):
        assert set(orig) == set(fixed)
        for k, v in orig.items():
            if k not in win_intro_keys:
                assert fixed[k] == v, f"non-intro window field changed: {orig['episode_key']} {k}"

    if mismatch:
        for m in mismatch[:10]:
            print("MISMATCH:", m)
        raise SystemExit("input restoration or pattern boundary check failed - "
                         "inspect before writing derived files")

    ep_out = audit / "results_episodes_en_fixed.jsonl"
    ep_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fixed_rows) + "\n",
                      encoding="utf-8")
    fields = list(dict.fromkeys(k for r in fixed_rows for k in r))
    with ep_out.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(fixed_rows)
    win_out = audit / "results_windows_en_fixed.jsonl"
    win_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fixed_wins) + "\n",
                       encoding="utf-8")

    counts.update({
        "episodes": len(fixed_rows), "windows": len(fixed_wins),
        "win_intro_present": win_intro, "t_windows": t_windows,
        "win_intro_in_t_windows": win_intro_in_t_windows,
    })
    (audit / "coverage_fix_provenance.json").write_text(
        json.dumps({
            "fix": "wiring-2.1 stats correction of the English T_condition_intro rule",
            "note": "统计修正，非模型重跑；未改译文/审核标记/检测标签/分数/token/截断位置",
            "rule_before": "exporting is permitted only when ALL of the following conditions hold",
            "rule_after": "Export is permitted only when ALL of the following conditions hold",
            "source_episodes": {"file": src_ep.name, "sha256": _sha(src_ep)},
            "source_windows": {"file": src_win.name, "sha256": _sha(src_win)},
            "derived_episodes": ep_out.name,
            "derived_windows": win_out.name,
            "fixed_fields": sorted(intro_keys | win_intro_keys),
            "restoration_check": "all other visibility fields recomputed from the restored "
                                 "English inputs equal the saved values; empty-memory rows "
                                 "keep their missing-field semantics",
            "counts": counts,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("episodes fixed:", len(fixed_rows), "| windows fixed:", len(fixed_wins))
    print(f"full_prefix  : intro in {counts['full_intro_in']}/{counts['full_denom']}, "
          f"retained {counts['full_intro_keep']}/{counts['full_denom']}")
    print(f"mem_prefix   : intro in {counts['mem_intro_in']}/{counts['mem_denom']}, "
          f"retained {counts['mem_intro_keep']}/{counts['mem_denom']}")
    print(f"windows      : intro present {win_intro}/{len(fixed_wins)}; "
          f"within the {t_windows} full-T windows: {win_intro_in_t_windows}/{t_windows}")
    print("wrote", ep_out.name, "+ csv,", win_out.name, ", coverage_fix_provenance.json")
    return 0


def _codes_for(episode_key: str) -> list[str]:
    # episode_key = <Platform>_<scenario>_<case>_epi-NN
    parts = episode_key.split("_")
    scenario, episode = parts[1], parts[3]
    return common.episode_codes(scenario, episode)


if __name__ == "__main__":
    raise SystemExit(main())
