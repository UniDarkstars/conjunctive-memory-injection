"""Offline core-study statistics and supplementary Tables S1/S2.

Only copied, task-level observations are read. Independent reference statistics
are deliberately not imported by this module.
"""
from collections import defaultdict
import csv
from itertools import product
from pathlib import Path

from .statistics_exact import clopper_pearson, holm, mcnemar_exact_p, paired_risk_difference
from .tables_1_3 import build_tables as validate_main_inputs


PLATFORMS = ("ADK", "LangMem")
SCENARIOS = ("s1", "s2")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _groups(rows, fields):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[f] for f in fields)
        _require(row["task_id"] not in groups[key], "Duplicate paired task")
        groups[key][row["task_id"]] = row
    return groups


def _cell(group, **identity):
    n = len(group)
    x = sum(int(row["executed"]) for row in group.values())
    lo, hi = clopper_pearson(x, n)
    return dict(identity, n=n, execution_count=x, rate=x / n,
                ci_lo=lo, ci_hi=hi)


def _pair(first, second, **identity):
    _require(set(first) == set(second), "Unpaired condition task identities")
    _require(len(first) == 30, "A core comparison requires 30 paired tasks")
    a = {task: int(row["executed"]) for task, row in first.items()}
    z = {task: int(row["executed"]) for task, row in second.items()}
    b = sum(a[i] == 1 and z[i] == 0 for i in a)
    c = sum(a[i] == 0 and z[i] == 1 for i in a)
    rd, lo, hi = paired_risk_difference(b, c, len(a))
    return dict(identity, n=len(a), b=b, c=c, rd=rd, ci_lo=lo, ci_hi=hi,
                p_exact=mcnemar_exact_p(b, c))


def _adjust(rows, family):
    for row, adjusted in zip(rows, holm([r["p_exact"] for r in rows])):
        row["p_holm"] = adjusted
        row["family"] = family
    return rows


def _supplement_row(row, table):
    result = dict(row)
    result.update(display_rd=f"{row['rd']:+.3f}",
                  display_ci_lo=f"{row['ci_lo']:.3f}",
                  display_ci_hi=f"{row['ci_hi']:.3f}")
    for field in ("p_exact", "p_holm"):
        value = row[field]
        result[f"display_{field}"] = (f"{value:.4f}" if table == 2 else
                                         f"{value:.6g}" if value < .001 else f"{value:.6f}")
    return result


def build_results(data_dir: Path) -> dict[str, list[dict]]:
    """Return ten output datasets, keyed by intended CSV stem.

    E13 adjusts all sixteen natural/full contrasts before selecting S2.
    E19x keeps its four primary and two exploratory comparisons separate.
    """
    data_dir = Path(data_dir)
    # Reuse the six-table package's batch, exposure, outcome and task checks.
    validate_main_inputs(data_dir)
    components = _groups(_read(data_dir / "table1_components.csv"),
                         ("platform", "scenario", "condition"))
    e10_cells = [_cell(components[p, s, condition], experiment="E10", platform=p,
                       scenario=s, condition=condition)
                 for p, s, condition in product(PLATFORMS, SCENARIOS,
                                                ("clean", "T-only", "keys-only", "both"))]
    e10_pairs = _adjust([
        _pair(components[p, s, "both"], components[p, s, "T-only"],
              experiment="E10", platform=p, scenario=s, contrast="both-T-only")
        for p, s in product(PLATFORMS, SCENARIOS)
    ], "E10_DeepSeek_both_vs_T")

    layouts = _groups(_read(data_dir / "table2_layouts.csv"),
                      ("platform", "scenario", "path", "layout"))
    e13_cells = []
    for p, s, path, layout in product(PLATFORMS, SCENARIOS,
                                      ("natural", "full"), ("B0", "B1", "B2")):
        group = layouts[p, s, path, layout]
        row = _cell(group, experiment="E13", platform=p, scenario=s,
                    path=path, layout=layout)
        row["first_complete_count"] = sum(int(r["first_full"]) for r in group.values())
        row["all_complete_count"] = sum(int(r["all_requests_full"]) for r in group.values())
        e13_cells.append(row)
    e13_pairs = _adjust([
        _pair(layouts[p, s, path, "B0"], layouts[p, s, path, comparison],
              experiment="E13", platform=p, scenario=s, path=path,
              contrast=f"B0-{comparison}")
        for p, s, path, comparison in product(PLATFORMS, SCENARIOS,
                                              ("natural", "full"), ("B1", "B2"))
    ], "E13_joint16")

    queries = _groups(_read(data_dir / "table3_queries.csv"),
                      ("scenario", "layout", "query_mode"))
    e19x_cells = []
    for s, layout, query in product(SCENARIOS, ("distributed", "fully_merged"),
                                    ("whole", "per_object")):
        group = queries[s, layout, query]
        row = _cell(group, experiment="E19x", platform="ADK", scenario=s,
                    layout=layout, query_mode=query)
        row["joint_retrieval_count"] = sum(int(r["joint_retrieval"]) for r in group.values())
        row["first_complete_count"] = sum(int(r["first_full"]) for r in group.values())
        e19x_cells.append(row)
    primary = _adjust([
        _pair(queries[s, "distributed", query], queries[s, "fully_merged", query],
              experiment="E19x", platform="ADK", scenario=s, query_mode=query,
              contrast="Distributed-Fully merged")
        for query, s in product(("whole", "per_object"), SCENARIOS)
    ], "E19x_primary_joint4")
    exploratory = _adjust([
        _pair(queries[s, "distributed", "per_object"], queries[s, "distributed", "whole"],
              experiment="E19x", platform="ADK", scenario=s, query_mode="Distributed",
              contrast="Per object-Whole task")
        for s in SCENARIOS
    ], "E19x_exploratory_joint2")
    e19x_pairs = primary + exploratory

    matched = _read(data_dir / "e14_episodes.csv")
    expected = set(product(PLATFORMS, SCENARIOS, ("kmatch", "kmismatch")))
    for row in matched:
        _require(row["batch"] == "e14" and row["wiring_revision"] == "e14-rfull-1",
                 "E14: wrong batch or wiring revision")
        _require(row["model"] == "DeepSeek-V4-Flash-0731", "E14: wrong model")
        _require(row["condition"] in ("kmatch", "kmismatch"), "E14: invalid condition")
        _require(row["condition_rc"] == ("0" if row["condition"] == "kmatch" else "1"),
                 "E14: condition return code mismatch")
        _require(row["executed"] in ("0", "1"), "E14: nonbinary outcome")
        calls = int(row["export_calls"])
        _require(calls >= 0 and int(row["executed"]) == int(calls > 0),
                 "E14: outcome/count disagreement")
        _require(all(row[field] == "1" for field in
                     ("first_full", "all_requests_full", "final_response_present")),
                 "E14: missing completion or condition-aware full exposure")
        _require(int(row["request_count"]) > 0, "E14: missing actual request")
        _require(row["task_id"] == f"epi-{int(row['task_index']):02d}",
                 "E14: task identity/index mismatch")
    matching = _groups(matched, ("platform", "scenario", "condition"))
    _require(set(matching) == expected, "E14: missing/unexpected condition groups")
    tasks = {f"epi-{i:02d}" for i in range(1, 31)}
    _require(all(set(group) == tasks for group in matching.values()),
             "E14: incomplete fixed task set")
    e14_cells = []
    for p, s, condition in product(PLATFORMS, SCENARIOS, ("kmatch", "kmismatch")):
        group = matching[p, s, condition]
        row = _cell(group, experiment="E14", platform=p, scenario=s, condition=condition)
        row["all_complete_count"] = sum(int(r["all_requests_full"]) for r in group.values())
        e14_cells.append(row)
    e14_pairs = _adjust([
        _pair(matching[p, s, "kmatch"], matching[p, s, "kmismatch"],
              experiment="E14", platform=p, scenario=s, contrast="kmatch-kmismatch")
        for p, s in product(PLATFORMS, SCENARIOS)
    ], "E14_joint4")
    return {
        "e10_cells": e10_cells, "e10_pairs": e10_pairs,
        "e13_cells": e13_cells, "e13_pairs": e13_pairs,
        "e19x_cells": e19x_cells, "e19x_pairs": e19x_pairs,
        "e14_cells": e14_cells, "e14_pairs": e14_pairs,
        "supplement_s1": [_supplement_row(r, 1) for r in e19x_pairs],
        "supplement_s2": [_supplement_row(r, 2) for r in e13_pairs if r["path"] == "full"],
    }
