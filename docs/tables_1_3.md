# Tables I–III: task-level inputs

These tables are recomputed from the released CSV files, not from published cell totals. The runtime does not read the original workspace, import the experiment runners, query an API, or re-run a model.

| Paper table | Released input | Original study and selection | Rows |
| --- | --- | --- | ---: |
| I, component controls | `data/processed/table1_components.csv` | E10 B-v4, main DeepSeek `runs/e10` batch, both platforms, two scenarios, four conditions | 480 |
| II, layouts | `data/processed/table2_layouts.csv` | E13 `e13-rfull-1`, both platforms, two scenarios, natural/full paths, three layouts | 720 |
| III, query strategy | `data/processed/table3_queries.csv` | ADK `e19x_v1/x-main`, two scenarios, two layouts, whole/per-object queries | 240 |

The `task_id` is the frozen task's episode identifier, not a new randomly assigned ID. E10/E13 use `epi-01`–`epi-30`; E19x uses `test-01`–`test-30` from its separate task suite. Scenario is part of task identity. Conditions within a platform and scenario are checked against the same complete task-ID set, with no duplicates and exactly 30 observations each.

`executed` is an episode-level binary outcome. A task with multiple exports contributes one successful task to the displayed numerator; `export_calls` preserves the count. E10/E13 use recorded scenario-specific export-tool events, matching their official checker. E19x additionally requires `executed=true` and `result.status=EXPORTED`, matching its official checker. These batch-specific definitions are preserved rather than silently replaced.

For Table II, `first_full` means all four components were present in the first actual model-request memory block. `all_requests_full` separately checks every captured request; it must be true for all supplied-content runs. First-request delivery is not relabeled as a retrieval-layer observation. `first_t`, `first_k1`, `first_k2`, and `first_k3` preserve the component checks used to calculate complete delivery.

For Table III, `joint_retrieval` comes from full, whitespace-normalized component-text coverage in the saved retrieval results. `first_full` is independently checked against the first actual request's memory block. Only joint retrieval and execution are shown in the paper table; both stages remain in the released CSV. All four conditions in each scenario come from the same `x-main` batch at W=10, M=8 and temperature 0. No E17 executions or E19q retrieval-only results are substituted.

The importer checks all 1,440 source episodes against their raw tool logs. It also checks E13 first/all-request exposure against actual request captures and derives E19x coverage directly from the original component manifests, retrieval results and requests. Only identifiers and numerical/Boolean outcomes are exported. Raw prompt text, model response text, account settings and service traces are not part of this release.

`docs/provenance/tables_1_3.json` records the source-file hashes, export field mappings, filter rules and batch rationale. `scripts/build_tables_1_3_inputs.py` is a maintainer-only extraction helper requiring `--workspace`; it is not called by offline reproduction and writes only into this release.

These inputs reproduce the six-table release's Tables I–III. They do not independently recreate retrieval from original memories or re-run model execution. Later write-order and merge-strategy sensitivity audits are separate studies and are not substituted for the manuscript's frozen results.
