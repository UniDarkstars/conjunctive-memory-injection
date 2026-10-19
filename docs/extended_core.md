# Core-study statistics and supplementary Tables S1/S2

`src.extended_core.build_results(data_dir)` reads copied task-level observations
and returns ten named datasets. Each key is an intended output CSV stem.
It uses only the Python standard library, makes no network calls, reads no
original experiment directories, and does not read its expected-value file.

| Output | Rows | Input and scope |
| --- | ---: | --- |
| `e10_cells` | 16 | E10 B-v4, main DeepSeek batch, four component conditions |
| `e10_pairs` | 4 | Combined minus template-only; original exploratory four-comparison Holm family |
| `e13_cells` | 24 | Natural/full exposure × three layouts × both platforms/scenarios |
| `e13_pairs` | 16 | Distributed minus each merged layout; the original **joint 16-comparison family** |
| `e19x_cells` | 8 | Concurrent ADK `e19x_v1/x-main` query × layout comparisons |
| `e19x_pairs` | 6 | Four layout contrasts (primary family) and two query contrasts (separate exploratory family) |
| `e14_cells` | 8 | Matching/mismatching under condition-aware full exposure |
| `e14_pairs` | 4 | Matching minus mismatching; four-comparison Holm family |
| `supplement_s1` | 6 | E19x paired statistics and the displayed precision of the current statistical attachment |
| `supplement_s2` | 8 | E13 full-exposure subset, **after** joint 16-comparison adjustment |

The module reuses `table1_components.csv`, `table2_layouts.csv`, and
`table3_queries.csv`. The new `e14_episodes.csv` has 240 rows: two platforms,
two scenarios, two conditions and 30 fixed paired tasks. It preserves the final
valid task identity, condition return code, export-call count, binary execution,
first/all-request exposure, request count and completion indicator.

## Outcome and uncertainty

An executed task has at least one logged scenario-specific export call. Multiple
calls do not increase the numerator. E10/E13/E14 retain their original logged
target-tool criterion; E19x retains its original stricter executed/status
criterion, as documented in `tables_1_3.md`.

Cell intervals are two-sided 95% Clopper–Pearson intervals. This includes a
nonzero upper bound for zero-export cells; zero observed exports do not imply a
zero population rate. Paired differences are `(b-c)/n`, with the original
conservative individual 95% intervals formed from two 97.5% Clopper–Pearson
intervals. These intervals are unadjusted. Two-sided exact McNemar p-values are
Holm-adjusted within the declared families. The confidence interval and adjusted
test are different constructions and are not forced to agree.

S2 contains only eight displayed rows, but its p-values retain the natural and
full-exposure family of sixteen comparisons. S1's primary four comparisons and
exploratory two comparisons are adjusted separately. No results from E19q or
earlier method-comparison executions replace the concurrent E19x observations.

## E14 source verification

The maintenance extractor reads only the formal `e14` batches, excluding smoke
and prechecks. For all 240 tasks it checks source configuration identity, the
model/revision, six shared background records in configuration, final response,
and actual export-tool events. It checks all **843 captured model requests**
for a memory block containing T and the task's three condition records.

Exposure is condition-aware: Matching uses `rc=0`; Mismatching uses `rc=1`.
Thus a mismatching record may be fully exposed without satisfying the template's
condition. The extractor never classifies mismatching records as missing merely
because their return code is 1. All exported outcome and exposure values agree
with the formal episode matrices.

The raw requests, responses, tool arguments/results and account settings are
read locally for verification but are not copied into these processed files.
Provenance contains only workspace-relative source names, hashes and roles.
All 972 read source files were rehashed unchanged after extraction.

## Independent reference and validation

`reference/extended_core.json` is populated from pre-existing official cell/pair
tables and `analysis_x-main.json`, not from the new implementation. For S1/S2,
displayed RD bounds and p-values, scenario identities and contrast labels are
also extracted directly from the current English `statistical_tables.tex`.
Expected values are used only after calculation, for validation.

The E10 historical reference rounded its rates/intervals to six decimals;
comparison therefore allows absolute error `5.1e-7` for its two datasets.
Other numerical comparisons use `1e-12`. All 100 output rows and **1,286 fields**
matched the independent references, including S1/S2's displayed values.

Run the focused tests from the repository root:

```text
python -m unittest discover -s tests -p test_extended_core.py -v
```

Six tests passed on 2026-10-19. Besides the independent-reference comparison,
they reject an incorrect return code, incomplete condition exposure and a
missing paired task. A paired-outcome perturbation keeps each condition's
execution total unchanged but changes discordant pairs and the exact p-value,
demonstrating that the analysis actually uses task-level pairing.

Maintainer-only regeneration, not needed by readers:

```text
python scripts/build_extended_core_inputs.py --workspace <original-workspace>
```

This command writes only the release's E14 processed CSV, independent reference
JSON and provenance JSON. It never imports or runs the original experiment
code and never rewrites the original sources. Ordinary reproduction needs none
of the original raw files.
