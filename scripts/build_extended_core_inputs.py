"""Maintainer-only, read-only extraction of core inputs and independent references.

Requires --workspace. Writes only the named files inside this release, never
imports the original experiment modules and never contacts a model service.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from itertools import product
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ("ADK", "LangMem")
SCENARIOS = ("s1", "s2")
EXPORT = {"s1": "export_settlement_detail", "s2": "export_source_bundle"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flag(value):
    require(value in ("True", "False"), "Unexpected source Boolean")
    return int(value == "True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    workspace = parser.parse_args().workspace.resolve()
    sources = {}

    def read(relative, role):
        path = workspace / relative
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        name = path.relative_to(workspace).as_posix()
        require(name not in sources or sources[name]["sha256"] == digest,
                "Source changed during extraction")
        sources[name] = dict(path=name, sha256=digest, role=role)
        return data.decode("utf-8-sig")

    def rows(relative, role):
        return list(csv.DictReader(io.StringIO(read(relative, role))))

    def js(relative, role):
        return json.loads(read(relative, role))

    def jl(relative, role):
        return [json.loads(line) for line in read(relative, role).splitlines() if line.strip()]

    codes = {}
    for s in SCENARIOS:
        tree = ast.parse(read(f"experiments/ADK/_e10/instances_{s}.py", "Frozen E14 task identifiers"))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "INSTANCES" for t in node.targets))
        codes[s] = [list(ast.literal_eval(call.args[3])) for call in assignment.value.elts]
        require(len(codes[s]) == 30, "Unexpected task count")

    e14 = []
    request_count = 0
    for p in PLATFORMS:
        base = f"experiments/{p}/_e14/runs/e14"
        matrix = rows(f"{base}/matrix.csv", "E14 official final episode matrix")
        expected = set(product(SCENARIOS, ("kmatch", "kmismatch"), range(1, 31)))
        observed = [(r["scenario"], r["condition"], int(r["episode_index"])) for r in matrix]
        require(len(observed) == len(expected) and set(observed) == expected,
                "Missing or duplicate E14 task")
        for source in matrix:
            s, condition = source["scenario"], source["condition"]
            i = int(source["episode_index"])
            task = f"epi-{i:02d}"
            require(source["episode"] == task, "E14 task identity mismatch")
            run = f"{base}/{s}/{condition}/{task}"
            cfg = js(f"{run}/input_config.json", "E14 condition and run identity, local validation only")
            rc = "0" if condition == "kmatch" else "1"
            require(cfg["experiment"] == "e14" and cfg["platform"] == p.lower()
                    and cfg["scenario"] == s and cfg["episode_index"] == i
                    and cfg["condition"] == condition and cfg["condition_rc"] == rc,
                    "E14 source configuration identity mismatch")
            require(cfg["wiring_revision"] == "e14-rfull-1"
                    and cfg["model_id"] == "DeepSeek-V4-Flash-0731"
                    and cfg["r_full_background_n"] == 6,
                    "Wrong E14 model, revision or common background")
            calls = jl(f"{run}/tool_calls.jsonl", "E14 actual export-event count validation; text not exported")
            n_calls = sum(c.get("tool") == EXPORT[s] for c in calls)
            require(n_calls == int(source["P3_export_calls"])
                    and int(n_calls > 0) == flag(source["P3_verdict"]),
                    "E14 raw tool outcome disagrees with matrix")
            requests = js(f"{run}/llm_requests.json", "E14 condition-aware actual-request exposure validation; text not exported")
            blocks = []
            for request in requests:
                payload = request.get("payload", {})
                if p == "ADK":
                    found = []
                    for content in payload.get("contents", []) or []:
                        for part in content.get("parts", []) or []:
                            found.extend(re.findall(r"<PAST_CONVERSATIONS>\n(.*?)\n</PAST_CONVERSATIONS>",
                                                    part.get("text") or "", re.S))
                else:
                    found = re.findall(r"<memories>\n(.*?)\n</memories>",
                                       payload.get("system_prompt", ""), re.S)
                require(len(found) == 1, "E14 request lacks a unique memory block")
                blocks.extend(found)
            require(bool(blocks) and len(blocks) == int(source["P2_n_llm_requests"]),
                    "E14 request count mismatch")
            sync = "recon-sync" if s == "s1" else "gate-sync"
            patterns = [r"\[ID:T\]"] + [rf"{sync}\s+{re.escape(c)}\s+rc={rc}" for c in codes[s][i - 1]]
            all_full = int(all(all(re.search(pattern, block) for pattern in patterns) for block in blocks))
            require(all_full == 1 and flag(source["P2_full_exposure_all_requests"]) == 1
                    and flag(source["P2_full_exposure"]) == 1,
                    "E14 condition-aware full exposure failed")
            final = bool(read(f"{run}/final_response.txt", "E14 successful completion validation; text not exported").strip())
            require(final and flag(source["final_response_present"]) == 1,
                    "E14 final response missing")
            request_count += len(requests)
            e14.append(dict(platform=p, scenario=s, condition=condition, task_id=task,
                            task_index=i, batch="e14", model=cfg["model_id"],
                            wiring_revision=cfg["wiring_revision"], condition_rc=rc,
                            executed=int(n_calls > 0), export_calls=n_calls,
                            first_full=1, all_requests_full=all_full,
                            request_count=len(requests), final_response_present=int(final)))
    e14.sort(key=lambda r: (r["platform"], r["scenario"], r["condition"], r["task_id"]))
    target = ROOT / "data/processed/e14_episodes.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(e14[0]))
        writer.writeheader()
        writer.writerows(e14)

    # All expected statistics below come from previously archived official
    # summaries. No release calculation is called to produce a reference.
    reference = {}
    legacy_cells = rows("experiments/_stats/e10_e12_reanalysis_cells.csv", "Independent archived E10 proportion statistics (six-decimal precision)")
    legacy_pairs = rows("experiments/_stats/e10_e12_reanalysis_pairs.csv", "Independent archived E10 paired statistics (RD/CI six-decimal precision)")
    e10cells = {(r["platform"], r["scenario"], r["condition"]): r for r in legacy_cells
                if r["experiment"] == "e10" and r["model"] == "DeepSeek"}
    reference["e10_cells"] = []
    for p, s, condition in product(PLATFORMS, SCENARIOS, ("clean", "T-only", "keys-only", "both")):
        r = e10cells[p, s, condition]
        n, x = int(r["n"]), int(r["P3_x"])
        reference["e10_cells"].append(dict(experiment="E10", platform=p, scenario=s, condition=condition,
                                           n=n, execution_count=x, rate=x / n,
                                           ci_lo=float(r["P3_lo"]), ci_hi=float(r["P3_hi"])))
    e10pairs = {(r["platform"], r["scenario"]): r for r in legacy_pairs
                if r["experiment"] == "e10" and r["model"] == "DeepSeek"
                and r["target"] == "both" and r["control"] == "T-only"}
    reference["e10_pairs"] = []
    for p, s in product(PLATFORMS, SCENARIOS):
        r = e10pairs[p, s]
        reference["e10_pairs"].append(dict(experiment="E10", platform=p, scenario=s,
            contrast="both-T-only", n=int(r["n"]), b=int(r["target_only_b"]), c=int(r["control_only_c"]),
            rd=float(r["RD"]), ci_lo=float(r["RD_lo"]), ci_hi=float(r["RD_hi"]),
            p_exact=float(r["exact_mcnemar_p"]), p_holm=float(r["holm_p"]), family=r["holm_family"]))

    e13cells = {(r["platform"], r["scenario"], r["path"], r["baseline"]): r
                for r in rows("experiments/_stats/e13_summary/cells.csv", "Independent E13 cells")}
    reference["e13_cells"] = []
    for p, s, path, layout in product(PLATFORMS, SCENARIOS, ("natural", "full"), ("B0", "B1", "B2")):
        r = e13cells[p.lower(), s, path, layout]
        n, x = int(r["n"]), int(r["P3"])
        reference["e13_cells"].append(dict(experiment="E13", platform=p, scenario=s, path=path, layout=layout,
            n=n, execution_count=x, rate=x / n, ci_lo=float(r["P3_ci_lo"]), ci_hi=float(r["P3_ci_hi"]),
            first_complete_count=int(r["P2_full"]), all_complete_count=int(r["P2_full_all_requests"])))
    e13pairs = {(r["platform"], r["scenario"], r["path"], r["contrast"]): r
                for r in rows("experiments/_stats/e13_summary/pairs.csv", "Independent E13 sixteen-comparison joint family")}
    reference["e13_pairs"] = []
    for p, s, path, other in product(PLATFORMS, SCENARIOS, ("natural", "full"), ("B1", "B2")):
        r = e13pairs[p.lower(), s, path, f"B0-{other}"]
        reference["e13_pairs"].append(dict(experiment="E13", platform=p, scenario=s, path=path,
            contrast=r["contrast"], n=int(r["n"]), b=int(r["b"]), c=int(r["c"]), rd=float(r["rd"]),
            ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]), p_exact=float(r["p_exact"]),
            p_holm=float(r["p_holm_joint16"]), family="E13_joint16"))

    x = js("experiments/ADK/_e19x/runs/e19x_v1/analysis_x-main.json", "Independent contemporaneous E19x analysis")
    reference["e19x_cells"] = []
    for s, layout, query in product(SCENARIOS, ("distributed", "fully_merged"), ("whole", "per_object")):
        r = x["cells"][f"{s}/{layout}/{query}"]
        reference["e19x_cells"].append(dict(experiment="E19x", platform="ADK", scenario=s,
            layout=layout, query_mode=query, n=r["n"], execution_count=r["exec"], rate=r["exec"] / r["n"],
            ci_lo=r["exec_ci95"][0], ci_hi=r["exec_ci95"][1], joint_retrieval_count=r["joint_ret"],
            first_complete_count=r["first_full"]))
    reference["e19x_pairs"] = []
    for family, key in (("E19x_primary_joint4", "primary_family"), ("E19x_exploratory_joint2", "exploratory_family")):
        for r in x[key]:
            primary = key == "primary_family"
            query = r["label"].split(" | ")[1] if primary else "Distributed"
            reference["e19x_pairs"].append(dict(experiment="E19x", platform="ADK", scenario=r["label"][:2],
                query_mode=query, contrast="Distributed-Fully merged" if primary else "Per object-Whole task",
                n=r["n_pairs"], b=r["b_a_only"], c=r["c_b_only"], rd=r["rd"],
                ci_lo=r["rd_ci_conservative"][0], ci_hi=r["rd_ci_conservative"][1],
                p_exact=r["p_raw"], p_holm=r["p_holm"], family=family))

    e14cells = {(r["platform"], r["scenario"], r["condition"]): r
                for r in rows("experiments/_stats/e14_e15_summary/e14_cells.csv", "Independent E14 cells")}
    reference["e14_cells"] = []
    for p, s, condition in product(PLATFORMS, SCENARIOS, ("kmatch", "kmismatch")):
        r = e14cells[p, s, condition]
        n, count = int(r["n"]), int(r["P3"])
        reference["e14_cells"].append(dict(experiment="E14", platform=p, scenario=s, condition=condition,
            n=n, execution_count=count, rate=count / n, ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]),
            all_complete_count=int(r["full_all"])))
    e14pairs = {(r["platform"], r["scenario"]): r
                for r in rows("experiments/_stats/e14_e15_summary/e14_pairs.csv", "Independent E14 four-comparison family")}
    reference["e14_pairs"] = []
    for p, s in product(PLATFORMS, SCENARIOS):
        r = e14pairs[p, s]
        reference["e14_pairs"].append(dict(experiment="E14", platform=p, scenario=s, contrast="kmatch-kmismatch",
            n=int(r["n"]), b=int(r["b"]), c=int(r["c"]), rd=float(r["rd"]),
            ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]), p_exact=float(r["p_exact"]),
            p_holm=float(r["p_holm"]), family="E14_joint4"))

    # Supplement display references come directly from the current TeX, not
    # by formatting the release's calculation or its independent CSV reference.
    tex = read("MiTA2027_20261017/english/statistical_tables.tex", "Current supplementary Tables S1/S2 displayed values")
    s1_rows, s2_rows = [], []
    for line in tex.splitlines():
        if " & " not in line or r"\rd{" not in line:
            continue
        cells = [cell.strip() for cell in line.rstrip("\\").split(" & ")]
        table = 1 if cells[0] in ("Supplier", "Release") else 2
        offset = 2 if table == 1 else 3
        rd_match = re.fullmatch(r"\\rd\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}", cells[offset + 2])
        require(rd_match is not None, "Unexpected supplementary RD syntax")
        b, c = map(int, cells[offset + 1].split("/"))
        row = dict(n=int(cells[offset]), b=b, c=c,
                   display_rd=rd_match[1], display_ci_lo=rd_match[2], display_ci_hi=rd_match[3])
        if table == 1:
            row.update(scenario={"Supplier": "s1", "Release": "s2"}[cells[0]],
                       query_mode={"Whole task": "whole", "Per object": "per_object",
                                   "Distributed": "Distributed"}[cells[1]],
                       contrast="Per object-Whole task" if cells[1] == "Distributed"
                       else "Distributed-Fully merged")
        else:
            row.update(platform=cells[0], scenario={"Supplier": "s1", "Release": "s2"}[cells[1]],
                       path="full", contrast={"D$-$F": "B0-B1", "D$-$C": "B0-B2"}[cells[2]])
        for key, text in zip(("p_exact", "p_holm"), cells[offset + 3:offset + 5]):
            cleaned = text.replace("$", "").replace(r"\!", "")
            scientific = re.fullmatch(r"([0-9.]+)\\times10\{([+-]?[0-9]+)\}", cleaned.replace("^", ""))
            if scientific:
                row[f"display_{key}"] = f"{float(scientific[1]) * 10**int(scientific[2]):.6g}"
            else:
                row[f"display_{key}"] = cleaned
        (s1_rows if table == 1 else s2_rows).append(row)
    require(len(s1_rows) == 6 and len(s2_rows) == 8, "Unexpected supplementary table size")
    reference["supplement_s1"] = [dict(r, **display) for r, display in zip(reference["e19x_pairs"], s1_rows)]
    full = [r for r in reference["e13_pairs"] if r["path"] == "full"]
    reference["supplement_s2"] = [dict(r, **display) for r, display in zip(full, s2_rows)]

    reference_sources = [s for s in sources.values() if s["path"].startswith("experiments/_stats/")
                         or s["path"].endswith(("analysis_x-main.json", "statistical_tables.tex"))]
    payload = dict(description="Independent archived official statistics and current supplementary TeX; never used as calculation inputs.",
                   tolerances={"default": 1e-12, "e10_cells": 5.1e-7, "e10_pairs": 5.1e-7},
                   sources=reference_sources, results=reference)
    (ROOT / "reference/extended_core.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    require(all(hashlib.sha256((workspace / s["path"]).read_bytes()).hexdigest() == s["sha256"]
                for s in sources.values()), "An original source changed during extraction")
    provenance = dict(extracted_on="2026-10-19", sources=list(sources.values()),
        source_integrity="Sources were read only and SHA-256 recorded from parsed bytes; no original modules were executed.",
        source_files_rechecked_unchanged=len(sources),
        extracted_file="data/processed/e14_episodes.csv", extracted_rows=len(e14),
        source_selection="Both formal _e14/runs/e14 batches only; 2 platforms x 2 scenarios x 2 conditions x 30 tasks. Smoke and precheck excluded.",
        raw_checks=dict(tool_logs=len(e14), completion_files=len(e14), actual_requests=request_count,
                        condition_aware_all_request_exposure=True),
        field_mapping={"task_id/task_index": "matrix episode/episode_index with fixed epi-01..30 validation",
                       "platform/scenario/condition/batch/model/wiring_revision/condition_rc": "source path and input_config cross-check",
                       "export_calls/executed": "scenario target-tool events, matches matrix P3_export_calls/P3_verdict",
                       "first_full/all_requests_full": "independent regex checks of T marker and task-specific rc=0 or rc=1 records in every actual request memory block",
                       "request_count": "len(llm_requests) checked against matrix P2_n_llm_requests",
                       "final_response_present": "nonempty saved final response checked against matrix"},
        omissions="Raw prompts, response text, tool arguments/results, service traces, account configurations, and absolute workspace paths are not exported.",
        reference_policy="Only archived official cells/pairs and current supplementary TeX populate expected values; release results are not used to make the reference.",
        independent_reference_rows={k: len(v) for k, v in reference.items()})
    (ROOT / "docs/provenance/extended_core.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(e14)} E14 episodes; checked {request_count} actual requests; froze independent references for {len(reference)} datasets.")


if __name__ == "__main__":
    main()
