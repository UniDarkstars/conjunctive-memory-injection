# Tables V and VI: inputs and computation

This release recomputes the reported table cells from processed observations. It does
not rerun an agent, contact a model service, rerun the detector, or re-extract raw
request and tool traces. The upstream integration versions and original field
mappings are recorded in `provenance/tables_5_6.json`.

## Table V: attack methods at baseline competition

`data/processed/table5_episodes.csv` contains 840 selected valid task observations:
600 from the original five-method baseline and 240 from the separate concurrent
Fully merged / MemoryGraft baseline. Higher competition levels are outside this
main table and are not included in this release CSV. Failed archived attempts are
not silently encoded as unsuccessful tasks: the upstream final-valid episode
selection is retained.

Each cell contains the same 30 task identities within its scenario. Baseline
settings are ADK `W=10, M=8` and LangMem `W=6, M=6`.

- Delivery is measured in the **first model request**, not merely retrieval or
  exposure in a later request.
- Original methods use the saved method-specific complete-delivery indicator.
- Concurrent Fully merged requires all four attack components.
- MemoryGraft requires **at least one complete experience record**, not all three.
- Clean delivery is undefined and displayed as `--`.
- Execution counts tasks with `target_executions > 0`; repeated calls count once.
- The two Fully merged batches remain separate. They must not be pooled or
  substituted for one another.

The CSV retains only identity, baseline configuration, a nullable delivery flag,
and target-tool execution count. Account settings, service traces, run directories,
and attack text are not copied.

## Table VI: PG memory scans

`data/processed/table6_episodes.csv` contains 480 language observations from 240
source requests. Chinese and English refer to **paired text versions of the same
source requests**, rather than 480 independent agent tasks. Every
language/platform/scenario/condition group has 30 source identities.

`data/processed/table6_windows.csv` contains 2,964 saved window labels: 1,482 per
language. It contains numeric record/window indices and binary labels, not text.
The calculation marks a record positive if any window is positive, then marks a
source positive if any of its records is positive. The supplied observations have
one complete window per record. That property and the token limit are checked.
The result is also checked against the saved source-level aggregate; the aggregate
is a validation field, not the input used to count per-record positives.

Empty memory retrievals remain in the episode CSV for a complete inventory, but
are excluded from both memory-scan denominators. This produces 26 eligible clean
sources for ADK Supplier and 28 for ADK Release in each language; both LangMem clean
groups retain 30. Attack groups retain all 30. No denominator is hardcoded in the
calculation. Request-prefix labels are included only to verify the table note.

The source versions are formal Chinese wiring-2 and English `en_fixed`
(wiring-2.1). The latter corrects coverage metadata while retaining the detector
labels. Smoke runs, wiring-1, and pre-correction coverage files are excluded.

## Validation boundaries

The offline module rejects duplicate source identities, unexpected settings,
missing task groups, missing paired identities, inconsistent memory inventories,
invalid booleans, malformed numeric fields, and duplicate window indices. It
recomputes all numerical table cells without loading a manuscript reference or
hardcoding observed counts. The unified release runner performs the separate
comparison against the manuscript reference.

Provenance paths identify workspace sources for the author; they are not runtime
dependencies. Runtime file access is limited to the three release CSVs above.
