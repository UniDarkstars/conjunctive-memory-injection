# Table IV: restoring omitted attack content

The calculation uses `data/processed/table4_episodes.csv`, a projection of 600
final episode outcomes: 360 DeepSeek episodes and 240 Kimi episodes. Each row
represents one task under one condition. No model API or original workspace is
needed to recompute the table.

## Conditions and pairing

- `replay`: retain the original natural retrieval result.
- `restore`: retain that result and add the omitted attack content.
- `sham` (displayed as **Benign**): retain that result and add length-matched
  benign text.

Kimi ran only `restore` and `sham`. Its replay cells are **--**, meaning not run,
not zero executions. The Kimi dataset contains the final valid observations for
two tasks whose first attempts were truncated. Those retries replace the failed
attempts; they do not add tasks or increase the denominator.

The analysis pairs records within model, platform, and scenario by `task_index`.
It requires exactly the same 30 indices and the same `snapshot_sha256` for each
task across its conditions. The digest identifies the frozen natural-retrieval
input; the underlying prompt text is not included in this release projection.

## Outcome and statistics

`target_tool_calls` counts actual calls to the scenario's target export tool.
`executed` is `1` when that count is at least one, otherwise `0`. Repeated calls
within an episode count as one positive task. The per-condition denominator
includes all 30 valid tasks.

For Restore versus Benign, `b` counts tasks executing only under Restore and
`c` counts tasks executing only under Benign. The paired risk difference is
`(b-c)/n`. The conservative individual 95% interval is `[L_b-U_c, U_b-L_c]`,
using separate two-sided 97.5% Clopper-Pearson intervals for `b/n` and `c/n`.
This interval is not obtained by inverting the McNemar test.

The test is two-sided exact McNemar. The four platform/scenario comparisons
form one Holm family within DeepSeek and a separate four-comparison family
within Kimi. Models are not pooled. Displayed differences and interval bounds
are percentage points rounded to one decimal; adjusted p-values below 0.001
display as `<0.001`, otherwise to four decimals. The calculation also returns
unrounded estimates and paired counts in the table's `statistics` field.

## Source checks and release boundary

Extraction checked every exported outcome against its saved tool-call log and
final status artifact. All 600 target-call counts agreed. The eight recomputed
paired results also agree with the archived statistical tables to an absolute
tolerance of `1e-13`. Source paths, SHA-256 digests, selection rules, and field
mapping are recorded in `docs/provenance/table_4.json`.

Only the processed projection is needed at runtime. Raw requests, model
responses, tool arguments/results, account configuration, and absolute local
paths are omitted. The `valid_run` column records successful extraction checks;
the DeepSeek legacy batch and Kimi runtime batch have different original
validity fields, documented in the provenance file.
