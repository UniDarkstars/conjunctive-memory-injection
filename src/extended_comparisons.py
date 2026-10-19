"""Offline query, method-competition and detector-coverage analyses.

All results are recalculated from sanitized observations. NumPy is used solely
to preserve E17's frozen random generator and Monte Carlo permutation protocol.
"""
from collections import defaultdict
from itertools import product
from pathlib import Path
import csv

from .statistics_exact import clopper_pearson, paired_risk_difference, mcnemar_exact_p, holm

METHODS = ("clean", "direct", "distributed", "fully_merged", "external")
MG_METRICS = ("p3", "recall_ge1", "recall_all", "first_ge1", "first_all", "every_request_ge1", "every_request_all", "benign_proxy")
PLATFORMS = ("adk", "langmem")
SCENARIOS = ("s1", "s2")
SCANS = ("full_prefix", "mem_prefix", "mem_windows")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _read(data, filename):
    with (Path(data) / filename).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(rows and all(None not in r and None not in r.values() for r in rows), f"Empty or malformed input: {filename}")
    return rows


def _int(value):
    _require(value.isdigit(), f"Invalid nonnegative integer: {value!r}")
    return int(value)


def _flag(value):
    _require(value in ("0", "1"), f"Invalid boolean: {value!r}")
    return int(value)


def _group(rows, keys, expected=None, n=30):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in keys)
        _require(row["task_id"] not in groups[key], f"Duplicate paired task: {key}, {row['task_id']}")
        groups[key][row["task_id"]] = row
    if expected is not None:
        _require(set(groups) == expected, f"Missing or unexpected groups: {keys}")
    if n is not None:
        for key, group in groups.items():
            _require(len(group) == n, f"Expected {n} tasks: {key}")
            prefix = "epi" if next(iter(group)).startswith("epi-") else "test"
            _require(set(group) == {f"{prefix}-{i:02d}" for i in range(1, n+1)}, f"Unexpected task identities: {key}")
    return groups


def _pair(a, b, outcome="p3"):
    _require(set(a) == set(b) and a, "Unpaired or empty task identities")
    positive = sum(_flag(a[k][outcome]) and not _flag(b[k][outcome]) for k in a)
    negative = sum(_flag(b[k][outcome]) and not _flag(a[k][outcome]) for k in a)
    rd, lo, hi = paired_risk_difference(positive, negative, len(a))
    return dict(b=positive, c=negative, n_pair=len(a), rd=rd, ci_low=lo, ci_high=hi,
                mcnemar_p=mcnemar_exact_p(positive, negative))


def _holm(rows, field="mcnemar_p"):
    for row, adjusted in zip(rows, holm([r[field] for r in rows])):
        row["holm_p"] = adjusted


def _sort(rows, keys):
    return sorted(rows, key=lambda r: tuple(str(r[k]) for k in keys))


def did_permutation(rows, n_mc=100000, seed=20261008):
    """Original task swap, RNG stream, tolerance and add-one p-value.

    Chunked draws have the same C-order sequence as one rng.random(n) per
    replicate in the original loop. Never substitute Python random or an exact
    sign test; those would not reproduce the archived Monte Carlo p-values.
    """
    import numpy as np
    _require(bool(rows) and n_mc > 0, "Permutation requires tasks and replicates")
    arrays = [np.array([r[k] for r in rows], dtype=np.int8) for k in
              ("y_fm_base", "y_fm_high", "y_ext_base", "y_ext_high")]
    _require(all(np.isin(a, [0, 1]).all() for a in arrays), "Nonbinary permutation outcomes")
    fb, fh, eb, eh = arrays
    rd_fm, rd_ext = float((fh-fb).mean()), float((eh-eb).mean())
    observed = abs(rd_fm-rd_ext)
    rng = np.random.default_rng(seed)
    n_ge = 0
    for start in range(0, n_mc, 10000):
        swaps = rng.random((min(10000, n_mc-start), len(rows))) < .5
        fbase, fhigh = np.where(swaps, eb, fb), np.where(swaps, eh, fh)
        ebase, ehigh = np.where(swaps, fb, eb), np.where(swaps, fh, eh)
        statistics = abs((fhigh-fbase).mean(axis=1)-(ehigh-ebase).mean(axis=1))
        n_ge += int((statistics >= observed-1e-12).sum())
    return dict(n_tasks=len(rows), rd_fm=rd_fm, rd_ext=rd_ext, t_obs=observed,
                p_perm=(n_ge+1)/(n_mc+1), n_mc=n_mc, seed=seed)


def _queries(data):
    rows = _read(data, "e19q_tasks.csv")
    levels = ("base", "w6_m12", "w8_m12", "w10_m12")
    groups = _group(rows, ("scenario", "layout", "query_mode", "level"),
                    set(product(SCENARIOS, ("distributed", "fully_merged"), ("whole", "per_object"), levels)))
    output = []
    for key, group in groups.items():
        for row in group.values():
            w, m = (6, 6) if row["level"] == "base" else (int(row["level"].split("_")[0][1:]), 12)
            _require((_int(row["w"]), _int(row["m"])) == (w, m), "E19q unexpected retrieval configuration")
            _require(_int(row["n_records"]) <= w, "E19q exceeded return width")
            nk = sum(_flag(row[f"k{i}"]) for i in (1, 2, 3))
            _require(nk == _int(row["n_k"]), "E19q key counts disagree")
            _require(_flag(row["joint"]) == int(_flag(row["has_t"]) and nk == 3), "E19q joint flag disagrees")
        output.append(dict(zip(("scenario", "layout", "query_mode", "level"), key),
                           joint=sum(_flag(r["joint"]) for r in group.values()),
                           has_T=sum(_flag(r["has_t"]) for r in group.values()),
                           mean_k=sum(_int(r["n_k"]) for r in group.values())/len(group), n=len(group)))
    return {"e19q_cells": _sort(output, ("scenario", "layout", "query_mode", "level"))}


def _methods(data):
    rows = _read(data, "e17_episodes.csv")
    groups = _group(rows, ("platform", "scenario", "method", "batch"),
                    set(product(PLATFORMS, SCENARIOS, METHODS, ("m-base", "m-mid", "m-stress"))))
    cells, pairs, stability, dids = [], [], [], []
    for key, group in groups.items():
        platform, scenario, method, batch = key
        w, base = (10, 8) if platform == "adk" else (6, 6)
        m = base * {"m-base": 1, "m-mid": 2, "m-stress": 4}[batch]
        for row in group.values():
            _require(row["model_id"] == "DeepSeek-V4-Flash-0731" and (_int(row["search_limit"]), _int(row["m_count_cfg"])) == (w, m), "E17 model/configuration mismatch")
            _require(_flag(row["p3"]) == int(_int(row["export_calls"]) > 0), "E17 execution disagrees with tool count")
        x = sum(_flag(r["p3"]) for r in group.values())
        lo, hi = clopper_pearson(x, len(group))
        cell = dict(platform=platform, scenario=scenario, method=method, batch=batch,
                    n=len(group), p3=x, p3_rate=x/len(group), cp_low=lo, cp_high=hi)
        for field in ("recall_complete", "first_full", "all_full", "benign_proxy"):
            cell[field] = "" if method == "clean" and field != "benign_proxy" else sum(_flag(r[field]) for r in group.values())
        cells.append(cell)
    for platform, scenario in product(PLATFORMS, SCENARIOS):
        for control in ("direct", "external", "distributed"):
            pairs.append(dict(platform=platform, scenario=scenario, target="fully_merged", control=control,
                              **_pair(groups[platform, scenario, "fully_merged", "m-base"], groups[platform, scenario, control, "m-base"])))
        for method in METHODS:
            low, high = groups[platform, scenario, method, "m-base"], groups[platform, scenario, method, "m-stress"]
            stability.append(dict(platform=platform, scenario=scenario, method=method, **_pair(high, low),
                                  m_base=_int(next(iter(low.values()))["m_count_cfg"]),
                                  m_stress=_int(next(iter(high.values()))["m_count_cfg"]), cond_dropped=0))
        keys = sorted(groups[platform, scenario, "fully_merged", "m-base"])
        quad = [{"task_id": task, **{field: _flag(groups[platform, scenario, method, batch][task]["p3"])
                 for field, method, batch in (("y_fm_base", "fully_merged", "m-base"), ("y_fm_high", "fully_merged", "m-stress"), ("y_ext_base", "external", "m-base"), ("y_ext_high", "external", "m-stress"))}} for task in keys]
        base = 8 if platform == "adk" else 6
        dids.append(dict(platform=platform, scenario=scenario, cond_dropped=0, m_base=base, m_stress=4*base, **did_permutation(quad)))
    _require(len(pairs) == 12 and len(dids) == 4, "Incomplete frozen E17 comparison families")
    _holm(pairs)
    _holm(dids, "p_perm")

    mg = _read(data, "e17mg_episodes.csv")
    expected = {(p, batch, s, method) for p, s in product(PLATFORMS, SCENARIOS)
                for batch, method in [("mg-base", "fully_merged"), ("mg-base", "memorygraft"), ("mg-mid", "memorygraft"), ("mg-stress", "memorygraft")]}
    mgroups = _group(mg, ("platform", "batch", "scenario", "method"), expected)
    mcells, mpairs, historical = [], [], []
    for key, group in mgroups.items():
        platform, batch, scenario, method = key
        w, base = (10, 8) if platform == "adk" else (6, 6)
        m = base * {"mg-base": 1, "mg-mid": 2, "mg-stress": 4}[batch]
        cell = dict(platform=platform, batch=batch, scenario=scenario, method=method, n=len(group), m_count=m, search_width=w)
        for row in group.values():
            _require(row["model_id"] == "DeepSeek-V4-Flash-0731" and (_int(row["m_count"]), _int(row["search_width"])) == (m, w), "MemoryGraft model/configuration mismatch")
            _require(_flag(row["p3"]) == int(_int(row["target_executions"]) > 0), "MemoryGraft tool count disagreement")
        for metric in MG_METRICS:
            count = sum(_flag(r[metric]) for r in group.values())
            lo, hi = clopper_pearson(count, len(group))
            cell.update({metric+"_n": count, metric+"_rate": count/len(group), metric+"_ci_low": lo, metric+"_ci_high": hi})
        mcells.append(cell)
    for platform, scenario in product(PLATFORMS, SCENARIOS):
        for cross in (False, True):
            a = mgroups[platform, "mg-base", scenario, "memorygraft" if cross else "fully_merged"]
            b = groups[platform, scenario, "direct", "m-base"] if cross else mgroups[platform, "mg-base", scenario, "memorygraft"]
            stats = _pair(a, b)
            paired = dict(family="E17mg_exploratory4" if cross else "E17mg_primary_joint4", platform=platform, scenario=scenario,
                          lhs="memorygraft" if cross else "fully_merged", rhs="direct_historical" if cross else "memorygraft",
                          a_both=sum(_flag(a[k]["p3"]) and _flag(b[k]["p3"]) for k in a),
                          d_neither=sum(not _flag(a[k]["p3"]) and not _flag(b[k]["p3"]) for k in a),
                          lhs_success=sum(_flag(r["p3"]) for r in a.values()), rhs_success=sum(_flag(r["p3"]) for r in b.values()), **stats)
            if cross:
                paired["cross_batch"] = "True"
                historical.append(paired)
            else:
                paired["holm_family_size"] = 4
                mpairs.append(paired)
    _holm(mpairs)
    return {"e17_cells": _sort(cells, ("platform", "scenario", "method", "batch")),
            "e17_pairs": _sort(pairs, ("platform", "scenario", "target", "control")),
            "e17_stability": _sort(stability, ("platform", "scenario", "method")),
            "e17_did": _sort(dids, ("platform", "scenario")),
            "e17mg_cells": _sort(mcells, ("platform", "batch", "scenario", "method")),
            "e17mg_pairs": _sort(mpairs, ("platform", "scenario")),
            "e17mg_historical_pairs": _sort(historical, ("platform", "scenario"))}


def _detector(data):
    coverage = _read(data, "e15_coverage_episodes.csv")
    groups = _group(coverage, ("platform", "scenario", "case", "lang", "config"),
                    set(product(("ADK", "LangMem"), SCENARIOS, ("both", "clean"), ("zh", "en"), ("full_prefix", "mem_prefix"))), n=None)
    cells = []
    for key, group in groups.items():
        records = list(group.values())
        cell = dict(zip(("platform", "scenario", "case", "lang", "config"), key), n=len(group))
        for output, field in (("truncated", "truncated"), ("full_T_present", "in_content_T"), ("keep_T", "keep_content_T"), ("action_present", "in_T_action"), ("keep_action", "keep_T_action"), ("intro_present", "in_T_condition_intro"), ("keep_intro", "keep_T_condition_intro")):
            cell[output] = sum(_flag(r[field]) for r in records)
        cell["keep_all_keys"] = sum(all(_flag(r[f"keep_content_k{i}"]) for i in (1, 2, 3)) for r in records)
        tokens = [_int(r["tokens_full"]) for r in records]
        cell.update(avg_tokens=sum(tokens)/len(tokens), max_tokens=max(tokens))
        cells.append(cell)
    rows = _read(data, "e15_coverage_labels.csv")
    label_groups = _group(rows, ("platform", "scenario", "case", "lang", "config"),
                          set(product(("ADK", "LangMem"), SCENARIOS, ("both", "clean"), ("zh", "en"), SCANS)))
    for key, group in label_groups.items():
        for r in group.values():
            empty = _int(r["n_memory_records"]) == 0 and r["config"] != "full_prefix"
            if empty:
                _require(r["case"] == "clean" and r["detected"] == "", "Empty-memory detector label must be missing")
            else:
                _flag(r["detected"])
        if key[-1] in ("full_prefix", "mem_prefix"):
            visible = {k for k, r in group.items() if r["detected"] != ""}
            _require(set(groups[key]) == visible, "Coverage inventory differs from scan-label inventory")
            for task in visible:
                _require(groups[key][task]["detected"] == group[task]["detected"], "Coverage/label outcome disagreement")
    pairs = []
    for platform, scenario, case in product(("ADK", "LangMem"), SCENARIOS, ("both", "clean")):
        contrasts = [("zh", scan, "en", scan) for scan in SCANS]
        contrasts += [(lang, a, lang, b) for lang in ("zh", "en") for a, b in (("full_prefix", "mem_prefix"), ("full_prefix", "mem_windows"), ("mem_prefix", "mem_windows"))]
        for lang_a, scan_a, lang_b, scan_b in contrasts:
            a, b = label_groups[platform, scenario, case, lang_a, scan_a], label_groups[platform, scenario, case, lang_b, scan_b]
            _require(set(a) == set(b), "Unpaired detector source identities")
            tasks = [k for k in a if a[k]["detected"] != "" and b[k]["detected"] != ""]
            stats = _pair({k: a[k] for k in tasks}, {k: b[k] for k in tasks}, "detected")
            pairs.append(dict(cell=f"{platform}/{scenario}/{case}", contrast=f"{lang_a}_pg_{scan_a}_detected vs {lang_b}_pg_{scan_b}_detected", n=stats["n_pair"], b=stats["b"], c=stats["c"], rd=round(stats["rd"], 4), ci_lo=round(stats["ci_low"], 4), ci_hi=round(stats["ci_high"], 4)))
    windows = _read(data, "e15_coverage_windows.csv")
    identities = set()
    window_cells = []
    for row in windows:
        identity = row["lang"], row["source_id"], _int(row["record_index"]), _int(row["start"])
        _require(identity not in identities, "Duplicate detector coverage window")
        _require(_int(row["start"]) == 0 and _int(row["tokens"])+2 <= 512, "Unexpected detector window configuration")
        identities.add(identity)
    for lang in ("zh", "en"):
        chosen = [r for r in windows if r["lang"] == lang]
        toks = [_int(r["tokens"]) for r in chosen]
        entries = {(r["source_id"], r["record_index"]) for r in chosen}
        window_cells.append(dict(lang=lang, n=len(chosen), entries=len(entries), positive=sum(_flag(r["positive"]) for r in chosen),
                                 containing_T=sum(_flag(r["in_content_T"]) for r in chosen), containing_action=sum(_flag(r["in_T_action"]) for r in chosen),
                                 containing_T_allkeys=sum(all(_flag(r[f"in_content_{a}"]) for a in ("T", "k1", "k2", "k3")) for r in chosen),
                                 min_raw_tokens=min(toks), max_raw_tokens=max(toks), max_windows_per_entry=max(sum((r["source_id"], r["record_index"]) == e for r in chosen) for e in entries)))
    # Check source/record inventory against labels: a missing window is not a negative.
    for (platform, scenario, case, lang, scan), group in label_groups.items():
        if scan != "mem_windows":
            continue
        for task, r in group.items():
            source = f"{platform}_{scenario}_{case}_{task}"
            selected = [w for w in windows if w["lang"] == lang and w["source_id"] == source]
            _require({_int(w["record_index"]) for w in selected} == set(range(1, _int(r["n_memory_records"])+1)), "Detector window/source inventory disagreement")
            if selected:
                _require(int(any(_flag(w["positive"]) for w in selected)) == _flag(r["detected"]), "Detector window/source labels disagree")
    selected = [r for r in coverage if r["lang"] == "en" and r["config"] == "mem_prefix" and r["case"] == "both"]
    full = lambda r: all(_flag(r[f"keep_content_{atom}"]) for atom in ("T", "k1", "k2", "k3"))
    joint = dict(lang="en", config="mem_prefix", n=len(selected), complete=sum(full(r) for r in selected), positive=sum(_flag(r["detected"]) for r in selected), complete_missed=sum(full(r) and not _flag(r["detected"]) for r in selected))
    record_attack = [r for r in rows if r["lang"] == "en" and r["config"] == "mem_windows" and r["case"] == "both"]
    return {"e15_coverage": _sort(cells, ("platform", "scenario", "case", "lang", "config")),
            "e15_pairs": _sort(pairs, ("cell", "contrast")), "e15_windows": _sort(window_cells, ("lang",)),
            "e15_coverage_joint": [joint],
            "e15_detection_totals": [dict(lang="en", config="mem_windows", n=len(record_attack), positive=sum(_flag(r["detected"]) for r in record_attack))]}


def build_results(data_dir):
    """Return named CSV results without reading references or writing outputs."""
    return {**_queries(data_dir), **_methods(data_dir), **_detector(data_dir)}
