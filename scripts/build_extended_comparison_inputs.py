"""Maintainer extraction: read explicit workspace; write sanitized release inputs only.

This script never imports experiment runners. Expected results are projected from
archived statistical outputs and manuscript statements, independently of src/.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E17 = "experiments/_stats/e17_summary/e17_integrated_20261010/"
MG = "experiments/_stats/e17mg_summary/e17mg_integrated_20261014/"
E15 = "experiments/_stats/e15_detector_audit/"
SUMMARY = "experiments/_stats/e14_e15_summary/"
Q = "experiments/LangMem/_e19q/runs/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    workspace = parser.parse_args().workspace.resolve()
    sources = {}

    def read(name):
        raw = (workspace / name).read_bytes()
        sources[name] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
        return raw.decode("utf-8-sig")

    def csvread(name):
        return list(csv.DictReader(io.StringIO(read(name))))

    def flag(value):
        if value not in (True, False, "True", "False"):
            raise ValueError(f"Invalid saved flag: {value!r}")
        return int(value is True or value == "True")

    def write(name, rows):
        assert rows
        path = ROOT / "data/processed" / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    results = {}
    def archived(key, name, fields=None, predicate=lambda r: True):
        # Keep source decimal precision; the checker accepts numeric strings.
        rows = [r for r in csvread(name) if predicate(r)]
        results[key] = [{k: r[k] for k in (fields or r)} for r in rows]

    query_rows = []
    qref = []
    for suffix in ("", "_sensitivity"):
        for row in csvread(Q + f"compare_langmem{suffix}.csv"):
            for mode in ("whole", "per_object"):
                r = {k: row[k] for k in ("scenario", "task_id", "layout", "m", "level")}
                r.update(w=row.get("w", "6"), query_mode=mode,
                         n_records=row[f"{mode}_n_records"], has_t=flag(row[f"{mode}_has_T"]),
                         n_k=row[f"{mode}_n_k"], joint=flag(row[f"{mode}_joint"]))
                r.update({f"k{i+1}": v for i, v in enumerate(row[f"{mode}_per_k"].split("|"))})
                query_rows.append(r)
        saved = json.loads(read(Q + f"compare_langmem{suffix}_summary.json"))
        for key, value in saved.items():
            if key.startswith("_"):
                continue
            scenario, layout, mode, level = key.split("/")
            if suffix:
                level += "_m12"
            qref.append(dict(scenario=scenario, layout=layout, query_mode=mode,
                             level=level, **value))
    write("e19q_tasks.csv", query_rows)
    results["e19q_cells"] = sorted(qref, key=lambda r: tuple(r[k] for k in ("scenario", "layout", "query_mode", "level")))

    e17_rows = []
    for r in csvread(E17 + "episodes.csv"):
        assert r["technical_failure"] == "False" and r["model_id"] == "DeepSeek-V4-Flash-0731"
        assert r["service_models"] == "dsv4"
        assert r["m_count_cfg"] == r["actual_m"]
        out = {k: r[k] for k in ("platform", "scenario", "method", "task_id", "batch", "model_id", "search_limit", "m_count_cfg")}
        out.update({k: flag(r[k]) for k in ("p3", "recall_complete", "first_full", "all_full", "benign_proxy")})
        out["export_calls"] = r["export_calls"]
        assert out["p3"] == int(int(out["export_calls"]) > 0)
        e17_rows.append(out)
    write("e17_episodes.csv", e17_rows)
    archived("e17_cells", E17 + "cells.csv", ["platform", "scenario", "method", "batch", "n", "p3", "p3_rate", "cp_low", "cp_high", "recall_complete", "first_full", "all_full", "benign_proxy"])
    archived("e17_pairs", E17 + "pairs.csv")
    archived("e17_stability", E17 + "stability_pairs.csv")
    archived("e17_did", E17 + "did.csv")

    mg_rows = []
    metrics = ("p3", "recall_ge1", "recall_all", "first_ge1", "first_all", "every_request_ge1", "every_request_all", "benign_proxy")
    for r in csvread(MG + "episodes.csv"):
        assert r["classification"] == "valid" and r["split"] == "test" and r["model_id"] == "DeepSeek-V4-Flash-0731"
        assert r["service_models"] == "dsv4"
        out = {k: r[k] for k in ("platform", "batch", "scenario", "method", "task_id", "m_count", "search_width", "model_id", "target_executions")}
        out.update({k: flag(r[k]) for k in metrics})
        assert out["p3"] == int(int(out["target_executions"]) > 0)
        mg_rows.append(out)
    write("e17mg_episodes.csv", mg_rows)
    fields = ["platform", "batch", "scenario", "method", "n", "m_count", "search_width"]
    fields += [f"{metric}_{suffix}" for metric in metrics for suffix in ("n", "rate", "ci_low", "ci_high")]
    archived("e17mg_cells", MG + "cells.csv", fields)
    pairfields = ["platform", "scenario", "lhs", "rhs", "a_both", "d_neither", "lhs_success", "rhs_success", "b", "c", "n_pair", "rd", "ci_low", "ci_high", "mcnemar_p"]
    archived("e17mg_pairs", MG + "pairs_primary.csv", ["family", *pairfields, "holm_family_size", "holm_p"])
    archived("e17mg_historical_pairs", MG + "pairs_exploratory.csv", ["family", *pairfields, "cross_batch"])

    fields = ["tokens_full", "truncated", "in_content_T", "keep_content_T", "in_T_action", "keep_T_action", "in_T_condition_intro", "keep_T_condition_intro", "keep_content_k1", "keep_content_k2", "keep_content_k3", "detected"]
    coverage, labels, windows = [], [], []
    for lang, suffix in (("zh", "zh"), ("en", "en_fixed")):
        for line in read(E15 + f"results_episodes_{suffix}.jsonl").splitlines():
            r = json.loads(line)
            identity = dict(platform=r["platform"], scenario=r["scenario"], case=r["case"], lang=lang, task_id=r["episode"])
            for scan in ("full_prefix", "mem_prefix", "mem_windows"):
                prefix = f"{lang}_pg_{scan}_"
                labels.append(dict(identity, config=scan, n_memory_records=r["n_mem_entries"],
                                   detected=flag(r[prefix + "detected"]) if prefix + "detected" in r else ""))
                if scan == "mem_windows" or prefix + "tokens_full" not in r:
                    continue
                coverage.append(dict(identity, config=scan, **{k: int(r[prefix+k]) if k == "tokens_full" else flag(r.get(prefix+k, False)) for k in fields}))
        for line in read(E15 + f"results_windows_{suffix}.jsonl").splitlines():
            r = json.loads(line)
            windows.append(dict(lang=lang, source_id=r["episode_key"], record_index=r["entry"], start=r["start"], tokens=r["tokens"], positive=r["label_id"],
                                **{k: flag(r[k]) for k in ("in_content_T", "in_T_action", "in_content_k1", "in_content_k2", "in_content_k3")}))
    write("e15_coverage_episodes.csv", coverage)
    write("e15_coverage_labels.csv", labels)
    write("e15_coverage_windows.csv", windows)
    archived("e15_coverage", SUMMARY + "e15_coverage.csv")
    archived("e15_windows", SUMMARY + "e15_windows.csv")
    archived("e15_pairs", SUMMARY + "paired_contrasts.csv", predicate=lambda r: "_lg_" not in r["contrast"])
    # These totals are independent manuscript claims, not calculations from the exported CSV.
    manuscript = read("MiTA2027_20261017/english/sections/discussion.tex")
    detected, n = map(int, re.search(r"detected only (\d+)/(\d+) attacks", manuscript).groups())
    record_detected, record_n = map(int, re.search(r"Per-record scanning increased detection to (\d+)/(\d+)", manuscript).groups())
    assert "English memory-prefix scanning retained the template and all three records" in manuscript
    results["e15_coverage_joint"] = [{"lang": "en", "config": "mem_prefix", "n": n, "complete": n, "positive": detected, "complete_missed": n-detected}]
    results["e15_detection_totals"] = [{"lang": "en", "config": "mem_windows", "n": record_n, "positive": record_detected}]

    for name in ("experiments/_stats/e17_summary/did_permutation.py", "experiments/_stats/e17_summary/e17_summary.py", "experiments/_stats/e17_summary/stats_core.py", "experiments/_stats/e15_detector_audit/summarize.py"):
        read(name)
    # Stable independent source order; all result lists use an explicit sort shared by the API.
    sortkeys = {"e17_cells": ("platform", "scenario", "method", "batch"), "e17_pairs": ("platform", "scenario", "target", "control"), "e17_stability": ("platform", "scenario", "method"), "e17_did": ("platform", "scenario"), "e17mg_cells": ("platform", "batch", "scenario", "method"), "e17mg_pairs": ("platform", "scenario"), "e17mg_historical_pairs": ("platform", "scenario"), "e15_coverage": ("platform", "scenario", "case", "lang", "config"), "e15_windows": ("lang",), "e15_pairs": ("cell", "contrast")}
    for name, keys in sortkeys.items():
        results[name].sort(key=lambda r: tuple(str(r[k]) for k in keys))
    # CSV numbers must be typed for strict verification. Only known statistic
    # fields are converted; identities, labels and missing clean values remain
    # strings. This schema is independent of release calculation output.
    integer_fields = {
        "n", "b", "c", "n_pair", "p3", "recall_complete", "first_full", "all_full", "benign_proxy",
        "cond_dropped", "m_base", "m_stress", "n_tasks", "n_mc", "seed", "m_count", "search_width",
        "a_both", "d_neither", "lhs_success", "rhs_success", "holm_family_size", "truncated",
        "full_T_present", "keep_T", "action_present", "keep_action", "keep_all_keys", "intro_present",
        "keep_intro", "max_tokens", "entries", "positive", "containing_T", "containing_action",
        "containing_T_allkeys", "min_raw_tokens", "max_raw_tokens", "max_windows_per_entry",
        "joint", "has_T", "complete", "complete_missed",
    }
    float_fields = {
        "p3_rate", "cp_low", "cp_high", "rd", "ci_low", "ci_high", "mcnemar_p", "holm_p",
        "rd_fm", "rd_ext", "t_obs", "p_perm", "avg_tokens", "ci_lo", "ci_hi", "mean_k",
    }
    integer_fields.update(metric + "_n" for metric in metrics)
    float_fields.update(metric + "_" + suffix for metric in metrics for suffix in ("rate", "ci_low", "ci_high"))
    for output_rows in results.values():
        for row in output_rows:
            for field, value in row.items():
                if value == "":
                    continue
                if field in integer_fields:
                    row[field] = int(value)
                elif field in float_fields:
                    row[field] = float(value)
    reference = {"sources": list(sources.values()), "tolerances": {"default": 1e-12, "e15_pairs": 5.1e-5}, "results": results}
    (ROOT / "reference/extended_comparisons.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    provenance = {"description": "Read-only projection of formal final-valid observations; no prompt, service log, account, token score or local absolute path exported.", "sources": list(sources.values()), "selection": {"e19q": "LangMem baseline W6/M6 and same-task W6/8/10 M12 sensitivity, retrieval only", "e17": "all three competition levels, 1800 final-valid episodes", "e17mg": "480 final-valid episodes: Fully merged baseline only, MemoryGraft at three levels", "e15": "Chinese wiring-2 / corrected English wiring-2.1, PG only; paired identities; empty memory labels remain nullable"}, "permutation": {"numpy_version_used_original_environment": "2.2.6", "generator": "numpy.random.default_rng", "seed": 20261008, "n_mc": 100000, "unit": "one within-task swap of both base/high outcomes"}, "reference_policy": "Archived aggregate statistics and current manuscript claims supply expected results; no src calculation is imported here."}
    (ROOT / "docs/provenance/extended_comparisons.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Extracted extended observations and independent archived/manuscript references")


if __name__ == "__main__":
    main()
