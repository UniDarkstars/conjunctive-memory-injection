"""Offline construction pre-checks for ADK E16 (no model/API calls).

1) Canonical atoms: every task's four canonical texts are non-empty, carry
   their codes/markers, and differ between atoms.
2) Sham construction: for synthetic missing-sets {k1,k2,k3}/{T}/{T,k3}/{}
   on every task: exact per-record char + newline-position matching, marker
   equality, code occurrence order, forbidden substrings absent, sham never
   equals canonical.
3) Natural-block simulation: the real offline keyword service reproduces
   the E13 pattern (B0 under W10/M12: T recalled, three keys evicted) and
   the real D_i drives the same R/S validations.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

# Release-local paths; no original-workspace reads or writes.
import sys
_RELEASE_ONLINE = next(p for p in Path(__file__).resolve().parents
                       if (p / "release_runtime.py").is_file())
if str(_RELEASE_ONLINE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ONLINE))
from release_runtime import run_dir_for, validate_output_args, prepare_fresh_episode


VENDORED_ADK_SRC = str(Path(__file__).resolve().parents[3] / "agent-frameworks" / "adk-python" / "src")
for _p in [VENDORED_ADK_SRC, str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.append(_p)

import arms_core  # noqa: E402
from collect_snapshot import render_block  # noqa: E402

MISSING_SETS = [("k1", "k2", "k3"), ("T",), ("T", "k3"), ()]


def check_canonical(sm, scenario: str) -> None:
    for i, inst in enumerate(sm.INSTANCES, 1):
        canon = arms_core.canonical_atoms(sm, inst["codes"], inst)
        for atom, text in canon.items():
            assert text and f"[ID:{atom}]" in text, (scenario, i, atom)
        assert len(set(canon.values())) == 4, (scenario, i, "atoms not distinct")
        t_codes = [c for c in inst["codes"] if c in canon["T"]]
        assert t_codes, (scenario, i, "T carries no business codes")
    print(f"[{scenario}] canonical atoms OK over {len(sm.INSTANCES)} tasks")


def check_sham_construction(sm, scenario: str) -> None:
    for i, inst in enumerate(sm.INSTANCES, 1):
        canon = arms_core.canonical_atoms(sm, inst["codes"], inst)
        for missing in MISSING_SETS:
            arms_core.build_arm_additions(list(missing), canon, sm, inst["codes"])
    print(f"[{scenario}] sham construction OK "
          f"({len(sm.INSTANCES)} tasks x {len(MISSING_SETS)} missing-sets)")


async def check_natural_simulation(sm, scenario: str) -> dict:
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.events.event import Event
    from google.genai import types

    counts = {"T": 0, "k1": 0, "k2": 0, "k3": 0, "none": 0, "full": 0}
    d_patterns: dict[str, int] = {}
    for i, inst in enumerate(sm.INSTANCES, 1):
        kit = sm.iter_baselines(inst["codes"], inst)["B0"]
        service = InMemoryMemoryService()
        events = [Event(author=sm.AGENT_NAME,
                        content=types.Content(role="assistant",
                                              parts=[types.Part(text=c["content"])]))
                  for c in kit]
        await service.add_events_to_memory(
            app_name="precheck", user_id="u", events=events)
        resp = await service.search_memory(
            app_name="precheck", user_id="u", query=inst["fire"])
        entries = [{"author": m.author, "timestamp": m.timestamp,
                    "content": " ".join(p.text for p in m.content.parts if p.text)}
                   for m in resp.memories]
        block = render_block(entries)
        canonical = arms_core.canonical_atoms(sm, inst["codes"], inst)
        analysis = arms_core.analyze_block(block, canonical)
        for a in analysis["present"]:
            counts[a] += 1
        counts["full"] += int(all(analysis["evidence"].values()))
        counts["none"] += int(not analysis["present"])
        key = ",".join(analysis["missing"]) or "(none)"
        d_patterns[key] = d_patterns.get(key, 0) + 1
        # real-D_i R/S validation
        arms_core.build_arm_additions(analysis["missing"], canonical, sm, inst["codes"])
    print(f"[{scenario}] natural simulation: atom presence {counts}")
    print(f"[{scenario}] D_i patterns: {d_patterns}")
    return {"counts": counts, "d_patterns": d_patterns}


async def main_async() -> None:
    expected = {"T": 30}  # E13 frozen pattern: B0 natural keeps T, evicts keys
    for scenario in ("s1", "s2"):
        sm = importlib.import_module(f"study_materials_{scenario.lower()}")
        check_canonical(sm, scenario)
        check_sham_construction(sm, scenario)
        result = await check_natural_simulation(sm, scenario)
        for atom, exp in expected.items():
            got = result["counts"][atom]
            assert got == exp, f"{scenario}/{atom}: expected {exp} from E13, got {got}"
    print("ADK E16 PRECHECK PASSED")


if __name__ == "__main__":
    asyncio.run(main_async())
