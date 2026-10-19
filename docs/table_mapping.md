# Manuscript tables and source outcomes

Target: the six numbered tables in `MiTA2027_20261017/english/sections/results.tex` and `discussion.tex`, with corresponding Chinese tables. The release reference contains their exact displayed cells and source-file hashes.

All source paths below are relative to the original research workspace, for provenance only. Normal reproduction never resolves or opens them.

| Table / LaTeX label | Source study | Input and selection | Important distinction |
| --- | --- | --- | --- |
| I / `tab:components` | E10 | Both platform component-control batches; selected valid episodes | Clean / template-only / records-only / combined, 30 tasks per cell |
| II / `tab:layout` | E13 | Both platform natural-retrieval and supplied-content batches | First-request delivery is not execution; natural vs supplied-content outcomes remain separate |
| III / `tab:query` | E19x | `e19x_v1`, `x-main`, eight concurrent ADK cells | Do not substitute earlier retrieval-only E19q or cross-batch E17 outcomes |
| IV / `tab:restore` | E16 | DeepSeek final episode summary and Kimi final valid matrices | Replay was not run for Kimi; pair restore/benign by source task and snapshot |
| V / `tab:methods` | E17 + E17mg | `e17_integrated_20261010`, `m-base`; `e17mg_integrated_20261014`, `mg-base` | Original Fully merged and concurrent Fully merged are separate rows; only baseline competition |
| VI / `tab:detection` | E15 detector audit | Formal Chinese and corrected English run outputs, including record-window labels | Aggregate any-positive within source identity; exclude empty clean retrievals, preserve language pairing |

Detailed per-file source hashes, transformations, selected fields and counts are in:

- [Tables I-III](provenance/tables_1_3.json)
- [Table IV](provenance/table_4.json)
- [Tables V-VI](provenance/tables_5_6.json)

These six-table CSVs are projections of saved observations, not synthetic fixtures and not expanded versions of table totals. The complete release adds separate extended inputs/statistics and online materials; see [full paper coverage](paper_coverage.md) and [online experiments](online_experiments.md). The table-specific provenance records above retain the scope and fingerprints of the original six-table extraction.
