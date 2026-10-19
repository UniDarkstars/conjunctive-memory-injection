"""Summarize LangMem E10 results from matrix.csv.

Per scenario/variant reports n, P1, P2_full, P2_partial, P3 count + CP 95%
interval, and normal-task completion rate. Paired risk differences
(both - clean, both - T-only, both - keys-only) use the instance-paired
episodes (episode_index aligns across variants) with Newcombe-Wilson
score intervals for the difference of paired proportions.

Requires statsmodels in the interpreter environment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode



def _cp_interval(x: int, n: int) -> tuple[float, float]:
    from statsmodels.stats.proportion import proportion_confint

    if n == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(x, n, alpha=0.05, method="beta")
    return (float(lo), float(hi))


def _paired_rd_ci(diff_means: float, positives: int, negatives: int, n: int):
    """Paired risk-difference 95% CI via Tango-style matched-pair score CI.

    diff_means is RD = mean(b_i - c_i); positives/negatives are the
    discordant-pair counts P(1,0), P(0,1). Returns (lo, hi).
    """
    if n == 0:
        return (0.0, 0.0)
    import math
    rd = diff_means
    total_disc = positives + negatives
    if total_disc == 0:
        p = (rd + 1.0) / 2.0
        z = 1.959963984540054
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        lo2 = (centre - margin) / denom
        hi2 = (centre + margin) / denom
        return (2 * lo2 - 1, 2 * hi2 - 1)

    p12 = positives / n
    p21 = negatives / n
    rd_hat = p12 - p21
    z = 1.959963984540054
    z2 = z * z

    def score_stat(rd_val: float) -> float:
        s = p12 + p21
        var = (s - rd_val ** 2 * s) / n
        if var <= 0:
            return 0.0
        return ((rd_hat - rd_val) ** 2) / var

    def _search(lo, hi, want_upper):
        for _ in range(80):
            mid = (lo + hi) / 2
            if (score_stat(mid) < z2) == want_upper:
                lo = mid if want_upper else lo
                hi = hi if want_upper else mid
            else:
                hi = mid if want_upper else hi
                lo = lo if want_upper else mid
        return (lo + hi) / 2

    lo = _search(-1.0, rd_hat, want_upper=True)
    hi = _search(rd_hat, 1.0, want_upper=False)
    return (lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E10 summarizer")
    parser.add_argument("--matrix", default=str(
        run_dir_for(__file__) / "e10" / "matrix.csv"
    ))
    args = parser.parse_args()
    validate_output_args(args)

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        print(f"matrix not found: {matrix_path}", file=sys.stderr)
        raise SystemExit(1)

    rows = []
    with matrix_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    def b(v: str) -> int:
        return 1 if v == "True" else 0

    by_cell: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["scenario"], row["variant"])
        by_cell[key][int(row["episode_index"])] = row

    def _fmt_ci(ci):
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    for scenario in ["s1", "s2"]:
        print(f"\n--- Scenario {scenario.upper()} ---")
        header = (
            f"{'variant':<10} {'n':>4} {'P1':>4} {'P2_full':>7} {'P2_part':>8} "
            f"{'P3(x/n)':>22} {'task_done%':>11}"
        )
        print(header)
        print("-" * len(header))
        for variant in ["clean", "T-only", "keys-only", "both"]:
            insts = by_cell.get((scenario, variant), {})
            n = len(insts)
            if n == 0:
                continue
            p1 = sum(b(r["P1_pass"]) for r in insts.values())
            p2_full = sum(b(r["P2_full"]) for r in insts.values())
            p2_part = sum(b(r["P2_partial"]) for r in insts.values())
            p3 = sum(b(r["P3_verdict"]) for r in insts.values())
            task = sum(b(r["task_complete"]) for r in insts.values()) / n
            ci = _cp_interval(p3, n)
            p3_txt = f"{p3}/{n} {_fmt_ci(ci)}"
            print(
                f"{variant:<10} {n:>4} {p1:>4} {p2_full:>7} {p2_part:>8} "
                f"{p3_txt:>22} {task:>10.0%}"
            )

        # Paired risk differences vs both.
        both = by_cell.get((scenario, "both"), {})
        for ctrl in ["clean", "T-only", "keys-only"]:
            grp = by_cell.get((scenario, ctrl), {})
            paired = sorted(set(both) & set(grp))
            if not paired:
                continue
            diffs = [
                b(both[i]["P3_verdict"]) - b(grp[i]["P3_verdict"])
                for i in paired
            ]
            pos = sum(1 for d in diffs if d > 0)
            neg = sum(1 for d in diffs if d < 0)
            rd = sum(diffs) / len(diffs)
            ci = _paired_rd_ci(rd, pos, neg, len(diffs))
            print(
                f"  RD(both - {ctrl}) = {rd:+.3f}  "
                f"95% CI {_fmt_ci(ci)}  (n={len(diffs)} paired)"
            )


if __name__ == "__main__":
    main()

