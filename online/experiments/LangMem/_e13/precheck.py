"""Offline pre-checks for LangMem E13 (no Agent/LLM chat calls; not part of
the formal results).

1) R-full wiring check (offline): every scenario x instance x layout block
   from iter_full_block contains all four atoms and no foreign trigger key;
   one example per layout is pushed through build_system_prompt to show the
   atoms survive the actual <memories> wrapper.

2) W/M retrieval sweep (embedding calls only, never the chat model): seeds
   a real InMemoryStore per layout (competitor pool = 12, trimmed down per
   M via adelete), retrieves at each W, and reports four-atom visibility.
   The recommended competitive cell: M > 0, B1 and B2 fully recalled, B0 full
   exposure < 30 (merged survives, distributed keys evicted). Freeze the
   winning W/M into study_materials before the formal run; this precheck
   does NOT tune on P3.

Outputs CSVs under runs/precheck/.
"""

from __future__ import annotations

import argparse
import asyncio
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
from langchain_core.embeddings import Embeddings  # noqa: E402

VENDORED_LANGMEM_SRC = str(_THIS.parents[1] / "agent-frameworks" / "langmem" / "src")
if VENDORED_LANGMEM_SRC not in sys.path:
    pass  # langgraph/langchain dependencies are installed from requirements-online.txt

from run_episode import (  # noqa: E402
    build_system_prompt, full_block_results, load_secrets_env, make_embedder,
)

_SYNC_TOKEN = {"s1": "recon-sync", "s2": "gate-sync"}
_T_ID_RE = {
    "B0": re.compile(r"\[ID:T\]"),
    "B1": re.compile(r"\[ID:(?:T|T_merged)\]"),
    "B2": re.compile(r"\[ID:T\]"),
}
BASELINES = ("B0", "B1", "B2")
SWEEP_WS = [3, 4, 5, 6, 8, 10]
SWEEP_MS = [0, 5, 8, 12]


def _atoms(block_text: str, scenario: str, codes: list[str], baseline: str) -> dict:
    sync = _SYNC_TOKEN[scenario]
    k = {
        f"k{i+1}": bool(re.search(rf"{sync}\s+{re.escape(c)}\s+rc=0", block_text))
        for i, c in enumerate(codes)
    }
    t = bool(_T_ID_RE[baseline].search(block_text))
    return {"t": t, **k, "full": t and all(k.values())}


class CachedEmbedder(Embeddings):
    """Memoising wrapper: T/key/competitor texts repeat across layouts, so
    each unique text is embedded once per precheck process."""

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[str, list[float]] = {}

    def embed_documents(self, texts):
        out = [None] * len(texts)
        missing = []
        for i, t in enumerate(texts):
            if t in self.cache:
                out[i] = self.cache[t]
            else:
                missing.append((i, t))
        if missing:
            vecs = self.inner.embed_documents([t for _, t in missing])
            for (i, t), v in zip(missing, vecs):
                self.cache[t] = v
                out[i] = v
        return out

    def embed_query(self, text):
        if text not in self.cache:
            self.cache[text] = self.inner.embed_query(text)
        return self.cache[text]


def wiring_check(out_dir: Path, scenarios: list[str]) -> list[dict]:
    rows, failures, printed = [], [], set()
    for scenario in scenarios:
        sm = importlib.import_module(f"study_materials_{scenario.lower()}")
        for idx, inst in enumerate(sm.INSTANCES, start=1):
            codes = inst["codes"]
            all_codes = [c for j, o in enumerate(sm.INSTANCES, start=1)
                         if j != idx for c in o["codes"]]
            blocks = sm.iter_full_block(codes, inst)
            for baseline in BASELINES:
                block_text = "\n".join(r["content"] for r in blocks[baseline])
                atoms = _atoms(block_text, scenario, codes, baseline)
                foreign = sum(
                    1 for c in all_codes
                    if re.search(rf"{_SYNC_TOKEN[scenario]}\s+{re.escape(c)}\s+rc=0", block_text)
                )
                rows.append({
                    "scenario": scenario, "episode_index": idx,
                    "baseline": baseline, "records": len(blocks[baseline]),
                    "chars": len(block_text), "foreign_key_hits": foreign, **atoms,
                })
                if not atoms["full"] or foreign:
                    failures.append(rows[-1])

                if (scenario, baseline) not in printed:
                    printed.add((scenario, baseline))
                    results = full_block_results(sm, codes, inst, baseline)
                    system_prompt, _ = build_system_prompt(sm.AGENT_SYSTEM_PROMPT, results)
                    fmt_atoms = _atoms(system_prompt, scenario, codes, baseline)
                    print(
                        f"[wiring-example] {scenario}/{baseline}: prompt block "
                        f"{len(results)} records; atoms t={fmt_atoms['t']} "
                        f"k1={fmt_atoms['k1']} k2={fmt_atoms['k2']} "
                        f"k3={fmt_atoms['k3']} full={fmt_atoms['full']}"
                    )
                    if not fmt_atoms["full"]:
                        failures.append({"scenario": scenario, "episode_index": idx,
                                         "baseline": baseline,
                                         "issue": "prompt_lost_atoms"})

    _write_csv(out_dir / "wiring_rfull.csv", rows)
    full_ok = sum(1 for r in rows if r.get("full"))
    foreign_ok = sum(1 for r in rows if r.get("foreign_key_hits") == 0)
    print(f"\nwiring (R-full): full-atom blocks {full_ok}/{len(rows)}; "
          f"foreign-key-free {foreign_ok}/{len(rows)}")
    if failures:
        print("FAILURES:", json.dumps(failures[:10], ensure_ascii=False, default=str))
        raise SystemExit("R-full wiring precheck FAILED")
    return rows


async def sweep(out_dir: Path, scenarios: list[str]) -> None:
    from langgraph.store.memory import InMemoryStore

    load_secrets_env()
    embedder = CachedEmbedder(make_embedder())

    for scenario in scenarios:
        sweep_rows = []
        sm = importlib.import_module(f"study_materials_{scenario.lower()}")
        # results[(m, w, baseline)] = list of full-atom bools per instance
        results: dict[tuple[int, int, str], list[bool]] = {}

        for idx, inst in enumerate(sm.INSTANCES, start=1):
            codes = inst["codes"]
            kits = sm.iter_baselines(codes, inst)
            full_pool = sm._competitors(codes)  # all 12, store_key comp/01..12

            for baseline in BASELINES:
                store = InMemoryStore(index={"dims": sm.EMBEDDER_DIMS, "embed": embedder})
                ns = sm.NAMESPACE
                # seed attack kit + the full 12-competitor pool (embeddings
                # memoised, so re-aput across M cells is nearly free)
                for rec in kits[baseline] + full_pool:
                    await store.aput(ns, rec["store_key"], {
                        "content": rec["content"],
                        "content_marker": f"[ID:{rec['marker']}]",
                        "cell": rec["cell"],
                    }, index=["content"])

                for m in SWEEP_MS:
                    # trim the store down to exactly M competitors
                    keep = {c["store_key"] for c in full_pool[:m]}
                    for c in full_pool:
                        if c["store_key"] in keep:
                            await store.aput(ns, c["store_key"], {
                                "content": c["content"],
                                "content_marker": f"[ID:{c['marker']}]",
                                "cell": c["cell"],
                            }, index=["content"])
                        else:
                            await store.adelete(ns, c["store_key"])

                    for w in SWEEP_WS:
                        items = await store.asearch(ns, query=inst["fire"], limit=w)
                        block_text = "\n".join(
                            str((getattr(it, "value", {}) or {}).get("content", ""))
                            for it in items
                        )
                        atoms = _atoms(block_text, scenario, codes, baseline)
                        results.setdefault((m, w, baseline), []).append(atoms["full"])

        for (m, w, baseline), flags in sorted(results.items()):
            sweep_rows.append({
                "scenario": scenario, "M": m, "W": w, "baseline": baseline,
                "full": sum(flags), "n": len(flags),
            })
        _write_csv(out_dir / f"sweep_{scenario}.csv", sweep_rows)

        # recommendation: smallest M, then smallest W, where merged layouts
        # survive fully but the distributed layout loses full exposure
        candidates = []
        for m in SWEEP_MS:
            if m == 0:  # retain the no-competition control in CSV, not recommendations
                continue
            for w in SWEEP_WS:
                def rate(b):
                    rows_b = [r for r in sweep_rows
                              if r["scenario"] == scenario and r["M"] == m
                              and r["W"] == w and r["baseline"] == b]
                    return (rows_b[0]["full"], rows_b[0]["n"]) if rows_b else (0, 0)
                b1, n1 = rate("B1")
                b2, n2 = rate("B2")
                b0, n0 = rate("B0")
                if n0 == 30 and b1 == 30 and b2 == 30 and b0 < 30:
                    candidates.append((m, w, b0))
        print(f"\n[{scenario}] competitive cells (M>0, B1=30/30, B2=30/30, B0<30/30), "
              f"ordered by (M asc, W asc):")
        for m, w, b0 in candidates:
            print(f"  M={m} W={w}: B0 full={b0}/30")
        if not candidates:
            print("  none in the swept grid - widen SWEEP_WS/SWEEP_MS")

    print("\nFreeze the chosen cell into study_materials_sX.py "
          "(SEARCH_LIMIT / N_COMPETITORS) before the formal run.")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangMem E13 offline prechecks")
    parser.add_argument("--out-dir", default=str(run_dir_for(__file__) / "precheck"))
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2"])
    parser.add_argument("--skip-sweep", action="store_true",
                        help="run only the offline wiring check (no embeddings)")
    args = parser.parse_args()
    validate_output_args(args)

    out_dir = Path(args.out_dir)
    wiring_check(out_dir, args.scenarios)
    if not args.skip_sweep:
        asyncio.run(sweep(out_dir, args.scenarios))


if __name__ == "__main__":
    main()
