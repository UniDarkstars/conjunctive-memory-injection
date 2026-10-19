"""Recompute Table IV from final valid, paired episode outcomes.

This module reads only the release's processed CSV. It never imports the
original experiment runners, opens service traces, or calls a model API.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

from .statistics_exact import holm, mcnemar_exact_p, paired_risk_difference


MODELS = ("DeepSeek-V4-Flash-0731", "Kimi-K3")
ARMS = {MODELS[0]: ("replay", "restore", "sham"),
        MODELS[1]: ("restore", "sham")}
BATCHES = {MODELS[0]: "e16_v1", MODELS[1]: "e16_kimik3"}
PLATFORMS = {"adk": "ADK", "langmem": "LangMem"}
SCENARIOS = {"s1": "Supplier", "s2": "Release"}


def build_tables(data_dir: Path) -> list[dict]:
    """Return Table IV plus its unrounded paired-comparison statistics."""
    with (data_dir / "table4_episodes.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    groups = defaultdict(dict)
    for row in records:
        model, platform, scenario, arm = (row[k] for k in
                                          ("model", "platform", "scenario", "arm"))
        if (model not in MODELS or platform not in PLATFORMS
                or scenario not in SCENARIOS or arm not in ARMS[model]):
            raise ValueError("Table IV: unexpected model/platform/scenario/arm")
        if row["batch"] != BATCHES[model]:
            raise ValueError("Table IV: source batch does not match model")
        if (row["valid_run"] != "1" or row["status"] != "ok"
                or row["technical_error"] != "0" or row["input_checks_ok"] != "1"):
            raise ValueError("Table IV: invalid episode in final-valid analysis")
        index = int(row["task_index"])
        calls, executed = int(row["target_tool_calls"]), int(row["executed"])
        if not 1 <= index <= 30 or calls < 0 or executed not in (0, 1):
            raise ValueError("Table IV: invalid episode index or outcome")
        if executed != int(calls >= 1):
            raise ValueError("Table IV: binary outcome disagrees with tool count")
        if not re.fullmatch(r"[0-9a-f]{64}", row["snapshot_sha256"]):
            raise ValueError("Table IV: missing or invalid frozen-input identity")
        key = model, platform, scenario, arm
        if index in groups[key]:
            raise ValueError("Table IV: duplicate task in a condition")
        groups[key][index] = (executed, row["snapshot_sha256"])

    rows, statistics = [], []
    expected_tasks = set(range(1, 31))
    for model in MODELS:
        family = []
        for platform, platform_name in PLATFORMS.items():
            for scenario, scenario_name in SCENARIOS.items():
                arms = {arm: groups[model, platform, scenario, arm]
                        for arm in ARMS[model]}
                if any(set(values) != expected_tasks for values in arms.values()):
                    raise ValueError("Table IV: each condition must contain the same 30 tasks")
                for task in expected_tasks:
                    if len({values[task][1] for values in arms.values()}) != 1:
                        raise ValueError("Table IV: paired task frozen inputs differ")
                restore, benign = arms["restore"], arms["sham"]
                n = len(restore)
                b = sum(restore[i][0] == 1 and benign[i][0] == 0 for i in expected_tasks)
                c = sum(restore[i][0] == 0 and benign[i][0] == 1 for i in expected_tasks)
                rd, low, high = paired_risk_difference(b, c, n)
                counts = {arm: sum(outcome for outcome, _ in values.values())
                          for arm, values in arms.items()}
                replay = f"{counts['replay']}/{n}" if "replay" in counts else "--"
                row = ["DeepSeek" if model == MODELS[0] else model,
                       platform_name, scenario_name, replay,
                       f"{counts['restore']}/{n}", f"{counts['sham']}/{n}",
                       f"{100 * rd:.1f} [{100 * low:.1f}, {100 * high:.1f}]", ""]
                detail = {"model": model, "platform": platform, "scenario": scenario,
                          "contrast": "restore-sham", "n": n, "b": b, "c": c,
                          "replay_executed": counts.get("replay"),
                          "restore_executed": counts["restore"],
                          "benign_executed": counts["sham"],
                          "rd": rd, "ci_lo": low, "ci_hi": high,
                          "p": mcnemar_exact_p(b, c),
                          "family": f"{model}:restore-sham:four-platform-scenario-settings"}
                rows.append(row)
                family.append(detail)
        adjusted = holm([detail["p"] for detail in family])
        for row, detail, p in zip(rows[-4:], family, adjusted):
            detail["holm_p"] = p
            row[-1] = "<0.001" if p < 0.001 else f"{p:.4f}"
        statistics.extend(family)

    return [{"number": 4, "slug": "restoration",
             "caption": "Restoring omitted attack text versus length-matched benign additions.",
             "columns": ["Model", "Platform", "Scenario", "Replay", "Restore", "Benign",
                         "Restore - benign [95% CI] (pp)", "Holm p"],
             "rows": rows,
             "notes": ("Execution counts refer to tasks with at least one unauthorized export; "
                       "--: not run. Differences are Restore minus Benign in percentage points (pp); "
                       "95% CIs are conservative, individual and unadjusted. Exact McNemar p values "
                       "use four-comparison Holm adjustment within each model. Kimi final valid "
                       "episodes include two same-configuration retries; attempts do not increase n."),
             "statistics": statistics}]
