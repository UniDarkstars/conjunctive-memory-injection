# Query diagnostics, competition and detector coverage

`src/extended_comparisons.py:build_results(data_dir)` reads the projected CSVs in
`data/processed/` and returns named lists of result rows. It performs no network
access, opens no original workspace files, and writes no files. The shared
release entry writes its returned results into the release's output directory.
Expected numbers are independently extracted from the archived statistical
files and the current English manuscript into
`reference/extended_comparisons.json`; they are never imported by this module.

## Included observations

| Inputs | Scope | Output |
| --- | --- | --- |
| `e19q_tasks.csv` | 960 retrieval observations: 480 task/layout/configuration combinations, each under whole-task and per-object queries | 32 LangMem cells: baseline W=6/M=6 and W=6,8,10/M=12 sensitivity, both scenarios/layouts |
| `e17_episodes.csv` | 1,800 final-valid episodes; five methods, two platforms/scenarios, three competition levels | 60 cells; 12 baseline paired contrasts; 20 within-method stress-minus-base contrasts; four competition-change permutation contrasts |
| `e17mg_episodes.csv` | 480 final-valid episodes | 16 cells; four concurrent baseline contrasts; four explicitly historical exploratory contrasts |
| `e15_coverage_episodes.csv` | Request-prefix and memory-prefix retention fields and paired detector labels | 32 coverage cells and the complete-coverage/missed-attack joint count |
| `e15_coverage_labels.csv` | All three PG scanning strategies for 240 source requests in paired Chinese/English versions | 72 previously archived descriptive paired contrasts; English per-record 91/120 total |
| `e15_coverage_windows.csv` | 2,964 PG record-window labels with numeric indices and content-retention flags | Two window summaries and per-source inventory/label validation |

E19q is a retrieval-only diagnostic. A source task's repeated configuration and
query observations are not independent trials, and a zero joint retrieval count
is not a tool-execution result. These files preserve the original formal
insertion order and fusion policy. They do not claim robustness to later
alternative ordering/fusion diagnostics.

E17 retains the original baseline `(W,M)=(10,8)` for ADK and `(6,6)` for LangMem,
with 2x/4x competing records. E17mg ran Fully merged only at baseline in its
concurrent batch; MemoryGraft ran all three levels. No nonexistent concurrent
Fully merged stress runs are synthesized. The requested model and the returned
service model are checked at extraction; generation/selection conditions remain
the original formal batch conditions. This release reconstructs saved outcomes,
not model-service behavior.

## Statistical protocol

Each proportion uses its actual observation denominator and a two-sided 95%
Clopper–Pearson interval. Paired risk differences use matching task IDs and the
project's conservative interval: `[L_b-U_c,U_b-L_c]` from two separate 97.5% CP
intervals. These individual intervals are not multiplicity adjusted. Exact
two-sided McNemar p-values use the original families:

- E17 baseline: Fully merged against Direct, AgentPoison adaptation, and
  Distributed, across both platforms/scenarios, one family of 12.
- E17 stress-minus-base per-method contrasts: descriptive, unadjusted, as in
  `stability_pairs.csv`.
- E17 competition-change comparison: Fully merged versus AgentPoison
  adaptation, one family of four permutation p-values.
- E17mg baseline: concurrent Fully merged versus MemoryGraft, one family of
  four. The separate MemoryGraft versus historical Direct contrasts retain their
  exploratory, unadjusted, cross-batch status.
- E15 descriptive language/scan contrasts: original signed risk differences
  and conservative intervals rounded to four decimals; no new p-values,
  multiplicity families or significance claims are introduced.

The E17 permutation calculation requires **NumPy**. It preserves
`numpy.random.default_rng(seed=20261008)`, 100,000 replicates, sorted task order,
one independent within-task swap of the methods' complete `(base, high)` outcome
pairs, the absolute difference in mean changes, comparison tolerance `1e-12`,
and the add-one p-value `(n_ge+1)/(n_mc+1)`. Random draws are processed in chunks
without changing their sequence. The original local environment used NumPy
2.2.6. All four archived Monte Carlo p-values match exactly. This is not replaced
with a different random generator, an exact sign test, or a changed test statistic.

## Detector coverage and what it means

Source identities are paired across languages and scans. A memory scan excludes
zero-memory sources; request-prefix scans retain them. Within each source,
per-record positives are recomputed by any-positive aggregation of its windows.
All records in these data fit into one complete window, but no individual window
contains the template together with all three condition records. Window coverage
and source-level aggregation therefore have distinct meanings.

The complete-coverage/missed-detection statement is checked **jointly per source**:
an English memory prefix must retain the full template plus k1, k2 and k3, and its
own detector label must be negative. This yields 120 complete prefixes, 31
positives and **89 complete-but-missed inputs**, rather than inferring that joint
count solely from unrelated aggregate margins. Per-record English positives are
91/120. Coverage metadata use the corrected English wiring-2.1 files and formal
Chinese wiring-2 files; old wiring-1 and smoke results are excluded.

## Provenance and release limits

The maintainer-only command
`python scripts/build_extended_comparison_inputs.py --workspace <original-workspace>`
exports these fields and archived references. Ordinary reproduction never needs
the original workspace. Source paths are relative and hashed in
`docs/provenance/extended_comparisons.json`. The export contains no prompts,
window texts, raw model/service responses, account settings or credentials.

This module includes the reported competition and detector evidence. It does not
claim to rerun the online agents, learned embedding retrieval, attack adaptation
optimization, classifier inference, or later exploratory protocol variants.
Other modules provide the six main tables and core paired experiments. Full
historical first-attempt sensitivity and internal cost/budget bookkeeping are not
silently treated as equivalent to this final-valid analysis.
