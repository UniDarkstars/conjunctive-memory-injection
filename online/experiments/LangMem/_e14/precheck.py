"""Offline pre-checks for LangMem E14 (no API calls of any kind; not part
of the formal results).

1) Parity check (experiment.md E14 "只改 rc 字段"): for every scenario x
   instance, the kmatch and kmismatch kits have identical record counts,
   store keys, order, T bytes and competitor bytes; each key record differs
   ONLY by the rc digit flip and has identical length; the diagnostic
   blocks have identical lengths across conditions.

2) R-full wiring check: every condition's block contains T plus the three
   keys under the condition's rc digit, and no other instance's key (either
   rc digit); one example per condition is pushed through
   build_system_prompt to show the atoms survive the actual <memories>
   wrapper.

Outputs CSVs under runs/precheck/.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
import compat  # noqa: F401,E402

from run_episode import build_system_prompt, full_block_results  # noqa: E402

_SYNC_TOKEN = {"s1": "recon-sync", "s2": "gate-sync"}
_T_ID_RE = re.compile(r"\[ID:T\]")
CONDITIONS = ("kmatch", "kmismatch")
_CONDITION_RC = {"kmatch": "0", "kmismatch": "1"}


def _atoms(block_text: str, scenario: str, codes: list[str], condition: str) -> dict:
    sync = _SYNC_TOKEN[scenario]
    rc = _CONDITION_RC[condition]
    k = {
        f"k{i+1}": bool(re.search(rf"{sync}\s+{re.escape(c)}\s+rc={rc}", block_text))
        for i, c in enumerate(codes)
    }
    t = bool(_T_ID_RE.search(block_text))
    return {"t": t, **k, "full": t and all(k.values())}


def run_checks(out_dir: Path, scenarios: list[str]) -> None:
    wiring_rows = []
    failures = []
    example_printed = set()

    for scenario in scenarios:
        sm = importlib.import_module(f"study_materials_{scenario.lower()}")
        flip_re = re.compile(rf"({_SYNC_TOKEN[scenario]} \S+ )rc=0")

        for idx, inst in enumerate(sm.INSTANCES, start=1):
            codes = inst["codes"]
            all_codes = [c for j, o in enumerate(sm.INSTANCES, start=1)
                         if j != idx for c in o["codes"]]
            kits = sm.iter_conditions(codes, inst)
            blocks = sm.iter_full_block(codes, inst)

            # 1) parity between conditions
            ka, kb = kits["kmatch"], kits["kmismatch"]
            if len(ka) != len(kb):
                failures.append({"scenario": scenario, "episode_index": idx,
                                 "issue": "kit_length_mismatch"})
            for ra, rb in zip(ka, kb):
                if ra["store_key"] != rb["store_key"] or ra["cell"] != rb["cell"]:
                    failures.append({"scenario": scenario, "episode_index": idx,
                                     "issue": f"key_or_cell_differs:{ra['store_key']}"})
                    continue
                if ra["cell"] != "K" and ra["content"] != rb["content"]:
                    failures.append({"scenario": scenario, "episode_index": idx,
                                     "issue": f"non_key_byte_diff:{ra['store_key']}"})
                if ra["cell"] == "K":
                    expected = flip_re.sub(r"\g<1>rc=1", ra["content"])
                    if rb["content"] != expected:
                        failures.append({"scenario": scenario, "episode_index": idx,
                                         "issue": f"key_not_pure_rc_flip:{ra['store_key']}"})
                    if len(ra["content"]) != len(rb["content"]):
                        failures.append({"scenario": scenario, "episode_index": idx,
                                         "issue": f"key_length_differs:{ra['store_key']}"})

            len_a = sum(len(r["content"]) for r in blocks["kmatch"])
            len_b = sum(len(r["content"]) for r in blocks["kmismatch"])
            if len_a != len_b:
                failures.append({"scenario": scenario, "episode_index": idx,
                                 "issue": "block_length_differs"})

            # 2) wiring / R-full (through the actual prompt assembly)
            for condition in CONDITIONS:
                block_text = "\n".join(r["content"] for r in blocks[condition])
                atoms = _atoms(block_text, scenario, codes, condition)
                foreign = sum(
                    1 for c in all_codes
                    if re.search(rf"{_SYNC_TOKEN[scenario]}\s+{re.escape(c)}\s+rc=[01]", block_text)
                )
                row = {
                    "scenario": scenario, "episode_index": idx,
                    "condition": condition,
                    "records": len(blocks[condition]),
                    "chars": len(block_text),
                    "foreign_key_hits": foreign,
                    **atoms,
                }
                wiring_rows.append(row)
                if not atoms["full"] or foreign:
                    failures.append(row)

                if (scenario, condition) not in example_printed:
                    example_printed.add((scenario, condition))
                    results = full_block_results(sm, codes, inst, condition)
                    system_prompt, _ = build_system_prompt(sm.AGENT_SYSTEM_PROMPT, results)
                    fmt_atoms = _atoms(system_prompt, scenario, codes, condition)
                    print(
                        f"[wiring-example] {scenario}/{condition}: prompt block "
                        f"{len(results)} records; atoms t={fmt_atoms['t']} "
                        f"k1={fmt_atoms['k1']} k2={fmt_atoms['k2']} "
                        f"k3={fmt_atoms['k3']} full={fmt_atoms['full']}"
                    )
                    if not fmt_atoms["full"]:
                        failures.append({"scenario": scenario, "episode_index": idx,
                                         "condition": condition,
                                         "issue": "prompt_lost_atoms"})

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "wiring_rfull.csv", wiring_rows)

    print("\n=== wiring (R-full) summary ===")
    n = len(wiring_rows)
    full_ok = sum(1 for r in wiring_rows if r.get("full"))
    foreign_ok = sum(1 for r in wiring_rows if r.get("foreign_key_hits") == 0)
    chars_by_key = {
        (r["scenario"], r["episode_index"], r["condition"]): r["chars"]
        for r in wiring_rows
    }
    pairs = [k for k in chars_by_key if k[2] == "kmatch"]
    chars_equal = all(
        chars_by_key[k] == chars_by_key[(k[0], k[1], "kmismatch")] for k in pairs
    )
    print(f"full-atom blocks: {full_ok}/{n}; foreign-key-free: {foreign_ok}/{n}; "
          f"block chars equal across conditions: {chars_equal}")
    if failures:
        print("FAILURES:", json.dumps(failures[:10], ensure_ascii=False, default=str))
        raise SystemExit("E14 precheck FAILED")
    print("parity + wiring precheck PASSED")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E14 offline prechecks")
    parser.add_argument("--out-dir", default=str(run_dir_for(__file__) / "precheck"))
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    args = parser.parse_args()
    validate_output_args(args)
    run_checks(Path(args.out_dir), args.scenarios)


if __name__ == "__main__":
    main()
