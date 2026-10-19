# Conjunctive memory injection: experiments and reproducibility

Research code and selected observations for **Conjunctive Prompt Injection in Memory-Augmented Agents: Joint Retrieval and Unauthorized Tool Execution**, by Junwen Tan and Yong Ding, Guilin University of Electronic Technology.

[中文使用说明](README_zh.md)

This repository contains the seven experiment groups reported in the MiTA manuscript: component controls, storage layouts and supplied content, query strategies, missing-content restoration, condition matching, method/competition comparisons, and prompt-injection detection. It provides two separate routes:

- **Offline numerical reproduction:** recompute the six main tables, supplementary tables S1–S2, and additional reported statistics from selected task/window observations.
- **Online experiment implementation:** inspect the tasks, attack materials, retrieval rules, simulated tools, runners and checkers under `online/`, or run new experiments after configuring dependencies and services.

Online runs were **not** performed during release preparation. Saved numerical reproduction does not guarantee identical new model outputs. Historical model revisions and API availability limit exact online replication. All export tools are local simulations, not connections to real business systems.

## Reproduce the reported numbers

Use Python **3.10–3.12** for the pinned numerical environment:

```console
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
```

The full entry needs NumPy 1.26.4 to retain the original Monte Carlo random stream. Installation may require network access; the reproduction command itself does not. It reads only this repository and writes only `outputs/`. It does not read an original experiment workspace, use an API key, download a model, or contact a service.

To calculate and check without writing results:

```console
python scripts/reproduce_all.py --verify-only
```

The six main tables alone still require only Python's standard library:

```console
python scripts/reproduce_tables.py
```

Key outputs:

| Output | Contents |
| --- | --- |
| `outputs/tables.md` | Six main-text tables together |
| `outputs/table_01_components.csv` through `table_06_detection.csv` | Main tables I–VI |
| `outputs/table_04_statistics.csv` | Restoration counts, RD, CI, exact and adjusted p values |
| `outputs/supplement_s1.csv`, `supplement_s2.csv` | Query/layout and full-content paired comparisons |
| Other `outputs/e*.csv` files | Component, layout, query, matching, competition and detector diagnostics |
| `outputs/all_results.json` | Extended results with unrounded values |
| `outputs/all_verification.json` | Independent-reference comparisons, versions and input fingerprints |

Counts are computed from observation rows. Reference files are consulted **after** calculation; they are not substitutes for data. A mismatch causes failure. Do not regenerate reference files merely to make a failed comparison pass.

## Coverage and interpretation

| Study ID | Reported comparison | Reproduction route |
| --- | --- | --- |
| E10 | Clean, template only, records only, combined | Table I and four exploratory paired comparisons |
| E13 | Three storage layouts under natural retrieval and supplied content | Table II, 16-comparison Holm family and S2 |
| E19 | Whole-task / per-object queries | ADK Table III and S1; LangMem baseline and return-width diagnostics |
| E16 | Replay, restore, benign additions | Table IV; separate DeepSeek and Kimi families; Kimi replay was not run |
| E14 | Matching / mismatching return codes with complete exposure | Task-level observations, cell counts and four paired comparisons |
| E17 / E17mg | Attack constructions and competition | Table V; all reported competition levels, original/concurrent paired families and task-level permutation test |
| E15 | Detector input coverage, positives and clean alerts | Table VI; source/window coverage and complete-coverage false negatives |

See [full evidence map](docs/paper_coverage.md), [core statistics](docs/extended_core.md), [competition and detection](docs/extended_comparisons.md), and [source provenance](docs/provenance/).

Task IDs preserve pairing; repeated conditions are not new independent tasks. Retrieval, first-request exposure and tool execution remain distinct. Confidence intervals are conservative individual intervals, while Holm-adjusted p values use the documented comparison families. Original Fully merged and MemoryGraft-concurrent Fully merged are separate batches. Detector counts aggregate saved classifications; offline reproduction does not rerun the classifier. Extra archived detector comparisons are diagnostic outputs, not new inferential claims in the manuscript.

## Online code and materials

Start with:

```console
python online/run.py --list
```

The launcher has an explicit dry-run mode that lists the selected command without contacting a model. Follow [the online guide](docs/online_experiments.md) for the experiment-specific commands, dependency installation, environment variables and model locations. Credentials are supplied through the process environment; real `.env` and account configuration files are excluded. Prepared payloads and restoration snapshots are research materials, not service credentials or raw API traces.

New runs must use new output directories inside the release. These outputs are ignored by Git. Existing released observations in `data/processed/` are not overwritten by online runners.

The release fixes two defects in historical early runners/checkers: zero-tool-call runs now write an empty tool log, and E13 evaluates memory coverage for every request, including missing-block requests. The historical results remain unchanged. Source-to-release hashes and modifications are recorded in `docs/provenance/online_sources.json`; see [release changes](docs/release_changes.md).

The method-comparison implementations are **adaptations**: AgentPoison omits the original target-action generation constraint, and MemoryGraft uses pre-seeded experiences instead of the original document-ingestion pipeline. ADK's lexical tie-breaking and attack-first insertion order are part of the released configuration. These implementations do not establish position-independent attack success or complete reproduction of the external papers.

## Layout and tests

```text
data/processed/     selected task/window observations, without raw service logs
src/               offline calculations
scripts/           reproduction entries and maintainer extraction tools
reference/         independent archived/manuscript values for verification
online/            task materials, online runners and required local runtime code
docs/              protocols, coverage, provenance and release validation
tests/             statistics, damaged-input, isolation and regression checks
outputs/           generated results, ignored by Git
```

```console
python -m unittest discover -s tests -v
```

Maintainer `build_*_inputs.py` and manuscript-reference tools need an explicitly supplied original workspace; normal reproduction never invokes them. Original research files are outside this standalone repository.

## Citation and licensing

Citation metadata is in [CITATION.cff](CITATION.cff). The authors have not yet selected a distribution license for their research code/data. Third-party code retains its own notices and licenses. See [license notes](LICENSE_NOTES.md).
