"""Descriptive cell summary + base FM-MG paired contrasts for e17mg_v1.

Reads check_mg.json files under a run root (default runs/e17mg_v1),
aggregates per batch x scenario x method:
  n, recall_ge1, recall_all, executed (strict), attempts, mean PRP,
  retrieval count distribution (0/1/2/3 components).

For mg-base it additionally pairs the SAME task's fully_merged and
memorygraft episodes per scenario and reports the pre-specified primary
contrast direction (FM - MG): b = FM-only executions, c = MG-only
executions, RD = (b-c)/n, exact two-sided McNemar p.

Holm adjustment is NOT applied here: the pre-registered family
E17mg_primary_joint4 spans BOTH platforms (4 contrasts); adjust only
once both platforms' base batches exist.
"""
from __future__ import annotations

import argparse
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



def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, k) for k in range(0, lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description="e17mg cell summary")
    parser.add_argument("--root", type=Path,
                        default=run_dir_for(__file__) / "e17mg_v1")
    args = parser.parse_args()
    validate_output_args(args)

    rows = []
    for p in sorted(args.root.rglob("check_mg.json")):
        rel = p.relative_to(args.root)
        if "attempts" in rel.parts:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("classification") != "valid":
            continue
        rows.append(r)

    cells: dict[tuple, dict] = defaultdict(lambda: {
        "n": 0, "ge1": 0, "all": 0, "exec": 0, "attempts": 0,
        "prp_sum": 0.0, "prp_n": 0, "dist": [0, 0, 0, 0]})

    def batch_of(r) -> str:
        b = r.get("batch") or ""
        return b if b.startswith("mg-") else (f"({b})" if b else "(none)")

    for r in rows:
        key = (batch_of(r), r["scenario"], r["method"])
        c = cells[key]
        c["n"] += 1
        c["ge1"] += bool(r["recall_ge1"])
        c["all"] += bool(r["recall_all"])
        c["exec"] += bool(r["target_executed"])
        c["attempts"] += r["target_attempts"]
        if r.get("prp_first_search") is not None:
            c["prp_sum"] += r["prp_first_search"]
            c["prp_n"] += 1
        k = min(r["recall_n_present"], 3)
        c["dist"][k] += 1

    print(f"{'batch':<10} {'sc':<3} {'method':<13} {'n':>3} {'ge1':>4} "
          f"{'all':>4} {'exec':>4} {'att':>4} {'prp':>5}  dist0123")
    for key in sorted(cells):
        c = cells[key]
        prp = c["prp_sum"] / c["prp_n"] if c["prp_n"] else float("nan")
        print(f"{key[0]:<10} {key[1]:<3} {key[2]:<13} {c['n']:>3} "
              f"{c['ge1']:>4} {c['all']:>4} {c['exec']:>4} {c['attempts']:>4} "
              f"{prp:>5.2f}  {c['dist']}")

    # Base FM - MG paired contrast (raw p; Holm4 only after both platforms).
    by_task: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if batch_of(r) == "mg-base" and r["method"] in ("memorygraft",
                                                        "fully_merged"):
            by_task[(r["scenario"], r["task_id"])][r["method"]] = \
                bool(r["target_executed"])
    print("\nmg-base paired FM-MG (primary contrast direction; raw exact "
          "McNemar, Holm4 deferred until both platforms):")
    for sc in sorted({k[0] for k in by_task}):
        b = c_ = n = 0
        for (s, _tid), pair in by_task.items():
            if s != sc or len(pair) != 2:
                continue
            n += 1
            if pair["fully_merged"] and not pair["memorygraft"]:
                b += 1
            elif pair["memorygraft"] and not pair["fully_merged"]:
                c_ += 1
        if n:
            print(f"  {sc}: n={n} b={b} c={c_} RD={(b - c_) / n:+.3f} "
                  f"p_raw={mcnemar_exact(b, c_):.6f}")


if __name__ == "__main__":
    main()
