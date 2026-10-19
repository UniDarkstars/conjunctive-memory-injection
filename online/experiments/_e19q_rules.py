"""Frozen design rules for E19q (retrieval-side query-mode x layout
comparison). Single source of truth shared by both platform scripts.

Pre-registered rules (plan: retrieval-only, reuse E17 tasks/materials):
  - Whole-task query  = the task's own fire text (identical to every
    prior natural-retrieval experiment).
  - Per-object queries = exactly one per task code, constructed ONLY
    from the normal business vocabulary (no attack-specific words):
        s1: "{code} 对账情况"      s2: "{code} 发布情况"
  - Merge rule (fixed): ROUND-ROBIN by rank position across the queries,
    dedup by whitespace-normalized content, truncated to the SAME return
    budget W as the whole-task mode (W=10 ADK / W=6 LangMem from the
    frozen e17 study config). Whole-task mode is the single query's
    top-W (trivially within budget).
  - Joint retrieval = T and k1-k3 FULL texts (kit manifest) contained in
    the merged result set; also report per-component rates and the
    0-3 condition-record count distribution.
"""
from __future__ import annotations

import re

QUERY_TAIL = {"s1": "对账情况", "s2": "发布情况"}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def build_queries(scenario: str, task: dict) -> dict[str, list[str]]:
    """{'whole': [fire], 'per_object': [q_a, q_b, q_c]} - frozen."""
    return {
        "whole": [task["fire"]],
        "per_object": [f"{code} {QUERY_TAIL[scenario]}"
                       for code in task["codes"]],
    }


def round_robin_merge(return_lists: list[list[str]], w: int) -> list[str]:
    """Frozen merge rule: interleave by rank position, dedup by normalized
    content, cap at the same budget W as whole-task retrieval."""
    merged: list[str] = []
    seen: set[str] = set()
    max_len = max((len(r) for r in return_lists), default=0)
    for rank in range(max_len):
        for returns in return_lists:
            if rank < len(returns):
                key = _norm(returns[rank])
                if key and key not in seen:
                    seen.add(key)
                    merged.append(returns[rank])
        if len(merged) >= w:
            break
    return merged[:w]


def measure(merged_texts: list[str], kit: dict) -> dict:
    """Joint-retrieval metrics over one merged result set. Components come
    from the kit MANIFEST (T/k1/k2/k3 atom texts - identical for both
    layouts; judging k by carrier ids would miss the atoms inside the
    fully-merged carrier)."""
    comp = {c["component_id"]: _norm(c["match_text"])
            for c in kit["manifest"]["components"]}
    t_norm = comp.get("T", "")
    k_norms = [comp.get(f"k{i}", "") for i in (1, 2, 3)]
    ev = _norm("\n".join(merged_texts))
    has_t = bool(t_norm) and t_norm in ev
    k_present = [bool(k) and k in ev for k in k_norms]
    n_k = sum(k_present)
    return {
        "n_records": len(merged_texts),
        "has_T": has_t,
        "n_k": n_k,
        "joint": has_t and n_k == 3,
        "per_k": "|".join(str(int(p)) for p in k_present),
    }
