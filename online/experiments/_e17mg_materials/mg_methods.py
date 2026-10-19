"""Kit construction for the e17mg_v1 batch (plan §2/§3/§6).

Methods in this batch:
  memorygraft   3 mapped experience records (cell G), no conjunctive gate;
  fully_merged  DELEGATED to the frozen e17_v1 constructor - the concurrent
                base arm uses the byte-identical Fully merged materials
                (plan §6: "沿用 e17_v1 的 Fully merged 冻结构造器"), so the
                only difference from e17_v1 runs is batch/schedule context.

Background pool, budget accounting and kit signature are reused from
`_e17_materials` unchanged.
"""
from __future__ import annotations

from _e17_materials import benign_pools, budget, methods
from _e17mg_materials import mg_records

MG_METHODS = ["memorygraft", "fully_merged"]

MG_MANIFEST_NOTE = (
    "delivery metrics: primary = at-least-one record retrieved (each record "
    "independently contains the full target action); also report all-three "
    "and 0/1/2/3 count distribution - plan §7"
)


def build_mg_kit(scenario: str, method: str, task: dict, m_count: int) -> dict:
    """Build one method's neutral kit at competition level M (plan §6)."""
    if method == "fully_merged":
        kit = methods.build_method_kit(
            scenario, "fully_merged", task, m_count,
            allow_missing_external=True,
        )
        kit["batch"] = "e17mg_v1"
        return kit
    if method != "memorygraft":
        raise ValueError(f"unknown e17mg method: {method}")

    carriers = mg_records.mg_carriers(scenario, task)
    mg_records.assert_judgments(carriers, task, scenario)
    background = benign_pools.pool_records(scenario, task, m_count)
    manifest = {
        "method": "memorygraft",
        "delivery_metrics": MG_MANIFEST_NOTE,
        "source_seeds": mg_records.SOURCE_SEEDS,
        "source_commit": mg_records.SOURCE_COMMIT,
        "components": [
            {"component_id": c["carrier_id"], "match_text": c["content"]}
            for c in carriers
        ],
    }
    budget_report = budget.check_kit("memorygraft", carriers)
    kit = {
        "method": method,
        "batch": "e17mg_v1",
        "scenario": scenario,
        "task_id": task["task_id"],
        "m_count": m_count,
        "carriers": carriers,
        "background": background,
        "manifest": manifest,
        "budget": budget_report,
    }
    if not budget_report["within_budget"]:
        raise AssertionError(
            f"budget violation for memorygraft/{task['task_id']}: "
            f"{budget_report['reason']}"
        )
    return kit


def kit_signature(kit: dict, model_id: str, width: int | None = None) -> str:
    """Same resume-integrity contract as e17 (model, M, width, texts)."""
    return methods.kit_signature(kit, model_id, width)
