"""Summarize ADK E13 results from matrix.csv (optionally joint with the
LangMem E13 matrix).

Cells are (platform, scenario, path, baseline). Per cell: n, P1, four-atom
exposure (first request; R-full also verified on every request), P3 with 95%
Clopper-Pearson interval, benign-tool proxy.

Paired comparisons: within each (scenario, path), B0-B1 and B0-B2 on P3,
paired by frozen episode_index. RD = (b-c)/n with the conservative 95%
interval [L_b - U_c, U_b - L_c] built from two-sided 97.5% CP intervals of
b/n and c/n; p from two-sided exact McNemar.

experiment.md E13 pre-specifies ONE Holm family of 16 comparisons across
BOTH platforms (2 x 2 x 2 x 2). Run with both matrices to get that family:

    python summarize.py --matrix runs/e13/matrix.csv ../../LangMem/_e13/runs/e13/matrix.csv

With a single matrix, the Holm column is the within-platform family of 8
(EXPLORATORY; not the pre-specified family).

Validity notes are printed separately (never silently dropped): path=full
episodes failing all-request full exposure or leaking foreign keys, and
cells with n != expected.
"""

from __future__ import annotations

import argparse
import csv
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


from scipy.stats import beta


def _cp_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Clopper-Pearson via the beta quantile relation."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x))
    return (lo, hi)


def _mcnemar_exact_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c))


def _holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _b(v: str) -> int:
    return 1 if str(v) == "True" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK E13 summarizer")
    parser.add_argument("--matrix", nargs="+", default=[
        str(run_dir_for(__file__) / "e13" / "matrix.csv")
    ], help="one or two matrix.csv files (second = LangMem E13 for the joint family)")
    parser.add_argument("--expected-n", type=int, default=30)
    args = parser.parse_args()
    validate_output_args(args)

    rows = []
    for matrix_path in map(Path, args.matrix):
        if not matrix_path.exists():
            print(f"matrix not found: {matrix_path}")
            raise SystemExit(1)
        with matrix_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row.setdefault("platform", "adk")
                rows.append(row)

    platforms = sorted({r["platform"] for r in rows})
    joint = len(platforms) > 1

    by_cell: dict[tuple[str, str, str, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_cell[(row["platform"], row["scenario"], row["path"], row["baseline"])][
            int(row["episode_index"])
        ] = row

    def fmt_ci(ci):
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    # --- validity notes (reported, never merged away) ---
    print("=== validity notes ===")
    bad = [
        r for r in rows
        if r["path"] == "full"
        and (r["P2_full_exposure_all_requests"] != "True"
             or int(float(r["P2_foreign_key_hits"] or 0)) != 0)
    ]
    print(f"path=full episodes failing all-request exposure or foreign-key-free: {len(bad)}")
    for r in bad[:10]:
        print(f"  {r['platform']}/{r['scenario']}/{r['path']}/{r['baseline']}/{r['episode']}")
    short = [
        f"{k}: n={len(v)}" for k, v in by_cell.items() if len(v) != args.expected_n
    ]
    print(f"cells with n != {args.expected_n}: {short if short else 'none'}")

    # --- cell tables ---
    for platform in platforms:
        for path in ("natural", "full"):
            for scenario in ("s1", "s2"):
                insts_exist = [
                    (b, by_cell.get((platform, scenario, path, b), {}))
                    for b in ("B0", "B1", "B2")
                ]
                if not any(v for _, v in insts_exist):
                    continue
                print(f"\n--- {platform} / {scenario.upper()} / {path} (E13) ---")
                header = (
                    f"{'baseline':<9} {'n':>4} {'P1':>4} {'fullExp':>8} {'fullAll':>8} "
                    f"{'P3(x/n)':>24} {'task%':>6} {'blockChars':>11}"
                )
                print(header)
                print("-" * len(header))
                for baseline, insts in insts_exist:
                    n = len(insts)
                    if n == 0:
                        continue
                    p1 = sum(_b(r["P1_pass"]) for r in insts.values())
                    full_exp = sum(_b(r["P2_full_exposure"]) for r in insts.values())
                    full_all = sum(_b(r["P2_full_exposure_all_requests"]) for r in insts.values())
                    p3 = sum(_b(r["P3_verdict"]) for r in insts.values())
                    task = sum(_b(r["task_complete"]) for r in insts.values())
                    chars = [int(r["P2_block_chars"]) for r in insts.values() if r["P2_block_chars"]]
                    ci = _cp_interval(p3, n)
                    avg_chars = f"{sum(chars)/len(chars):.0f}" if chars else "-"
                    print(
                        f"{baseline:<9} {n:>4} {p1:>4} {full_exp:>8} {full_all:>8} "
                        f"{p3:>4}/{n} {fmt_ci(ci):>18} {task / n:>5.0%} {avg_chars:>11}"
                    )

    # --- paired comparisons ---
    comparisons = []
    for platform in platforms:
        for path in ("natural", "full"):
            for scenario in ("s1", "s2"):
                b0 = by_cell.get((platform, scenario, path, "B0"), {})
                for other in ("B1", "B2"):
                    grp = by_cell.get((platform, scenario, path, other), {})
                    paired = sorted(set(b0) & set(grp))
                    if not paired:
                        continue
                    b = sum(1 for i in paired
                            if _b(b0[i]["P3_verdict"]) and not _b(grp[i]["P3_verdict"]))
                    c = sum(1 for i in paired
                            if not _b(b0[i]["P3_verdict"]) and _b(grp[i]["P3_verdict"]))
                    n = len(paired)
                    rd = (b - c) / n if n else 0.0
                    lo_b, hi_b = _cp_interval(b, n, alpha=0.025)
                    lo_c, hi_c = _cp_interval(c, n, alpha=0.025)
                    comparisons.append({
                        "platform": platform, "scenario": scenario,
                        "path": path, "contrast": f"B0-{other}",
                        "n": n, "b": b, "c": c, "rd": rd,
                        "ci": (lo_b - hi_c, hi_b - lo_c),
                        "p": _mcnemar_exact_p(b, c),
                    })

    if comparisons:
        holm = _holm([c["p"] for c in comparisons])
        family_label = (
            f"pre-specified joint Holm family of {len(comparisons)} (both platforms)"
            if joint
            else f"within-platform Holm family of {len(comparisons)} (EXPLORATORY; "
                 "pre-specified family is the joint 16 across both platforms)"
        )
        print(f"\n=== paired P3 comparisons ({family_label}) ===")
        print(f"{'cell':<26} {'n':>4} {'b':>3} {'c':>3} {'RD':>8} {'95% CI (conservative)':>26} "
              f"{'p(McNemar)':>12} {'Holm p':>8}")
        for comp, h in zip(comparisons, holm):
            print(
                f"{comp['platform']}/{comp['scenario']}/{comp['path']}/{comp['contrast']:<6}"
                f"{comp['n']:>4} {comp['b']:>3} {comp['c']:>3} {comp['rd']:>+8.3f} "
                f"{fmt_ci(comp['ci']):>26} {comp['p']:>12.4g} {h:>8.4g}"
            )
        print("\nInterpretation rules (experiment.md E13): R-full differences within "
              "narrow intervals are compatible with exposure-loss mediation; a "
              "natural-significant / full-non-significant pair alone proves neither "
              "disappearance nor equivalence.")


if __name__ == "__main__":
    main()
