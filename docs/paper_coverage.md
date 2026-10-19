# Coverage of the current MiTA manuscript

The target is the current English/Chinese MiTA paper and its two supplementary statistical tables, not every historical study in the workspace. Experiment IDs below preserve internal provenance; manuscript-facing names remain descriptive.

| Paper evidence | Released inputs | Outputs from `scripts/reproduce_all.py` |
| --- | --- | --- |
| Component controls, Table I; three of four exploratory execution contrasts significant | `table1_components.csv` | Table I, `e10_cells.csv`, `e10_pairs.csv` |
| Natural layout comparison, Table II; all eight paired contrasts significant | `table2_layouts.csv` | Table II, `e13_cells.csv`, `e13_pairs.csv` |
| Full content supply, Table II and supplementary S2 | Same per-task layout input, retaining natural/full paths | `supplement_s2.csv`, with the original joint 16-comparison Holm adjustment |
| ADK query × layout comparison, Table III and S1 | `table3_queries.csv` | Table III, `e19x_cells.csv`, `e19x_pairs.csv`, `supplement_s1.csv` |
| LangMem query baseline and W=6/8/10 sensitivity at M=12 | Released E19q task observations | `e19q_cells.csv` |
| DeepSeek and Kimi missing-content restoration, Table IV | `table4_episodes.csv` | Table IV, `table_04_statistics.csv`; Kimi has no replay arm |
| Joint matching versus mismatching, four platform/scenario contrasts | `e14_episodes.csv` | `e14_cells.csv`, `e14_pairs.csv` |
| Original method baseline, Table V | Original E17 observations, all levels retained | Table V, `e17_cells.csv`, `e17_pairs.csv` |
| Method behavior under increased competition | Same E17 observations | `e17_stability.csv`, `e17_did.csv`; original 100,000-draw task-level permutation |
| MemoryGraft and concurrent Fully merged, Table V and higher competition | Independent E17mg observations | Table V, `e17mg_cells.csv`, `e17mg_pairs.csv` |
| Detector positives/clean alerts, Table VI | `table6_episodes.csv`, `table6_windows.csv` | Table VI; empty clean retrievals excluded from memory-scan denominators |
| Complete content coverage despite missed detection | Released E15 coverage, labels and window observations | `e15_coverage.csv`, `e15_windows.csv`, `e15_coverage_joint.csv`, `e15_detection_totals.csv` |

The corresponding per-task inputs and exact field mappings are described in [core statistics](extended_core.md), [method/query/detector statistics](extended_comparisons.md), and `provenance/`. Original absolute machine paths are not required for ordinary reproduction.

## What is independently checked

Six main tables are checked against values extracted from the current TeX. Extended numerical results are checked against independently archived formal statistics; supplementary displays are additionally checked against current S1/S2 TeX cells. E10 archived values are rounded to six decimals and use an explicit 5.1e-7 tolerance; other reference tolerances are recorded per result. The reference is opened after calculation.

The full entry verifies 50 main-table rows and 360 extended rows. Some extended rows are additional diagnostics retained from the formal analyses: historical MemoryGraft comparisons and detector paired comparisons. Their inclusion does not convert the paper's descriptive detector discussion into new significance claims.

## Online implementation coverage

`online/run.py --list` lists ten entry names spanning the seven study groups: E10, E13, E14, E16 DeepSeek/Kimi, E17/E17mg, E19q/E19x, and PG. The material/runtime dependencies include earlier-study helpers when needed; that does not add historical E1–E9, E12 or E18 results to the paper's evidence.

The package includes reviewed task text, payloads, restoration snapshots and detector input projections needed to inspect or rerun the experiments. It excludes real credentials, service account configuration, model weights, virtual environments and original request/response logs. The classifier requires separately installed weights; chat experiments require a configured service. New model runs cannot be guaranteed to recreate historical outputs exactly.

## Historical observation versus corrected release implementation

Processed observations retain their original values. The release implementation corrects stale-tool-log and missing-memory-block validation defects and relocates runtime paths. It is not represented as the byte-identical historical implementation. `provenance/online_sources.json` maps source hashes to release hashes, and the two defects were checked against the existing stored runs before being changed in the copy. See [release changes](release_changes.md).
