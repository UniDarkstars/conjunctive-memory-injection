# Running the experimental protocols

The `online/` tree contains the task generators, attack materials, simulated business tools, retrieval implementations, model adapters, runners, and episode checkers for the manuscript experiments. It is independent of the original author workspace. Historical processed outcomes used for the paper remain separate from newly generated runs.

## Environment and first commands

Use Python 3.11 or 3.12 in a new virtual environment. From the repository root:

```console
python -m pip install -r online/requirements-online.txt
python online/run.py --list
python online/run.py --study e13 --platform adk --dry-run
python -B -m unittest discover -s tests -p test_online_release.py -v
```

The listed dependency versions were read from the local experiment environment. Installing the dependency file in a new environment and rerunning the model experiments have **not** been performed as part of release preparation. The bundled ADK source has its own dependency declarations and Apache-2.0 license. `online/experiments/ADK/_e17/external/vendor_AgentPoison/LICENSE` retains the upstream license for the AgentPoison reference; the adapters are research-specific implementations rather than the complete original repositories.

Set `NORMAL_BASE_URL` and `NORMAL_API_KEY` in the process environment. `online/environment.example` documents the variables and contains no credentials; scripts do not load `.env` files. The gateway must support OpenAI-compatible chat completions, and for LangMem also embeddings under the name `bge-m3`. Do not publish keys or copy account configuration into this tree.

The historical chat-model names are `DeepSeek-V4-Flash-0731` and `Kimi-K3`. Availability depends on the gateway. `--model YOUR_MODEL_NAME` selects another model; such a run is a new replication condition, not a recovery of the original model snapshot. Some legacy ADK adapters also accept a process-level `GOOGLE_API_KEY`; use a clean environment without that variable when supplying `NORMAL_API_KEY`.

The launcher only prints a plan unless `--execute` is present. Planning imports no model framework, creates no run directories, and performs no downloads or API calls. For an actual run:

```console
python online/run.py --study e13 --platform adk --run-id replication-01 --execute
```

This command can incur model-service costs. Export tools are local simulated functions that return an `EXPORTED` status; they do not send business data to an external destination.

## Protocol map

| Study | Comparisons | Main source directory | Launcher arguments |
| --- | --- | --- | --- |
| E10 | Clean, template only, condition records only, template plus records | `online/experiments/{ADK,LangMem}/_e10` | `--study e10 --platform adk` or `langmem` |
| E13 | Distributed, fully merged, condition-record merged; natural retrieval and controlled full exposure | `online/experiments/{ADK,LangMem}/_e13` | `--study e13 --platform adk` or `langmem` |
| E19q | Whole-task versus per-object queries, crossed with distributed versus merged storage | `online/experiments/{ADK,LangMem}/_e19q` | `--study e19q --platform adk` or `langmem` |
| E19x | Actual ADK execution under the same four query/layout conditions | `online/experiments/ADK/_e19x` | `--study e19x --platform adk` |
| E16 | DeepSeek replay, attack-content restoration, and length-matched benign addition | `online/experiments/{ADK,LangMem}/_e16` | `--study e16 --platform adk` or `langmem` |
| E16 model replication | Kimi restoration versus benign addition; no Kimi replay arm | Same E16 directories | `--study e16-kimi --platform adk` or `langmem` |
| E14 | Matching versus mismatching conditions under full content exposure | `online/experiments/{ADK,LangMem}/_e14` | `--study e14 --platform adk` or `langmem` |
| E17 | Clean, direct injection, distributed, fully merged, and AgentPoison adaptation at three competition levels | `online/experiments/{ADK,LangMem}/_e17` | `--study e17 --platform adk --competition all` |
| E17 MemoryGraft extension | MemoryGraft adaptation and concurrent fully merged controls, as specified by its batch driver | `online/experiments/{ADK,LangMem}/_e17mg` | `--study e17mg --platform adk --competition all` |
| PG extension | Full-input prefix, memory-only prefix, and memory-entry windows, paired Chinese/English inputs | `online/experiments/_stats/e15_detector_audit` | `--study pg` |

Append `--dry-run` to inspect commands or `--execute` to run them. E10/E13/E14/E16 accept `--scenario s1` or `s2` to run one scenario. E17, MemoryGraft, and E19 schedules run both scenarios; the launcher does not filter their task-block schedules. `s1` is supplier reconciliation and `s2` is release checks. PG always scans the supplied projections for both platforms; the launcher's platform argument does not filter that audit.

ADK E19q is local keyword retrieval after the reference tokenizer is available. LangMem E19q calls the embedding service but not a chat model; the launcher includes the base comparison and the M=12, W=6/8/10 sensitivity grid. E19x execution is ADK only. The reviewed evidence does not include a LangMem E19x execution batch.

## Inspecting the implementation

- `study_materials_s1.py` and `study_materials_s2.py` define trusted instructions, task content, record construction, and tool specifications. The E11 files are included only because later constructors import these frozen templates; they do not add another scheduled study.
- `run_episode.py` constructs the memory, captures model requests and responses, executes local tool functions, and writes the run artifacts. E19x uses `run_exec.py`; MemoryGraft uses `run_episode_mg.py`, which delegates the episode body to the release's E17 runner.
- `check_episode.py`, `analyze_x.py`, and `check_episode_mg.py` evaluate saved evidence. Joint retrieval, content exposure in the request, and actual target-tool execution remain separate outcomes. A target-tool name appearing in a prompt or model explanation is not counted as an executed export.
- `online/experiments/_e19q_rules.py` defines the per-object queries and the deduplicating round-robin merge under the same final carrier budget W. The release preserves the evaluated write order, keyword scores, and stable tie ordering. Attack-first placement is part of this experimental condition; these results do not establish placement-independent robustness.
- `online/experiments/_e17_materials` and `_e17mg_materials` define the independent development/test tasks, competition pools, attack constructors, and method-specific completeness criteria.

The AgentPoison adaptation retains retrieval-oriented trigger construction and fixed target-operation payloads. It does not implement the original paper's target-action generation constraint and should not be described as a complete faithful replication. MemoryGraft is evaluated through pre-seeded, complete experience records in this controlled tool setting. Results compare these specified adaptations.

E16 ships 60 reviewed input snapshots per platform. They contain task inputs and frozen retrieved memories, not model outputs or target-tool outcomes. Their internal snapshot hashes are checked before use. New snapshots can be collected with `collect_snapshot.py`; `import_snapshots.py --e13-root ...` can instead read a newly generated E13 run. New snapshot outputs go under ignored run directories, leaving the supplied snapshots unchanged.

PG ships 240 reviewed request-input projections and the reviewed translation memory. Their index contains only identities, extraction counts, text hashes, and relative paths. Raw service traces and account configurations are excluded. PG uses `protectai/deberta-v3-base-prompt-injection-v2`; classifier weights are not bundled. Actual execution may download them to `online/models_cache`, or use a caller-provided Hugging Face cache. No separate Llama Guard model is required for this release's PG command. The historical PG source did not pin a model-weight revision, which limits exact future weight recovery.

E17, MemoryGraft, and E19 constructors use the `BAAI/bge-m3` reference tokenizer for attack-budget accounting; a tokenizer cache or its initial download is required even for the ADK retrieval-only run. Optional LangMem trigger optimization additionally uses local `BAAI/bge-m3` model weights with revision `5617a9f61b028005a4858fdac845db406aefb181`; these weights are also not bundled. The formal method runs use the included frozen payloads and do not rerun optimization. Optional candidate staging, development ledgers, payload construction and consistency checks write below ignored run directories; they do not replace the included formal payloads. MemoryGraft development accounting uses a common ledger across platforms and keeps its combined per-scenario cap.

## Outputs, fixes, and provenance

Normal launcher outputs go to `online/runs/<platform>/<study>/<run-id>/`. Direct scripts also default to `online/runs/`; explicit output arguments are limited to that tree or `outputs/online/`. Use a distinct run ID for a new experiment. These directories and model caches are ignored by Git. Inspect each driver's `--help` for its resume or archived-retry options; the first three studies refuse to reuse an existing episode directory.

Two defects were fixed only in the release copy:

1. E10/E13/E14 create a fresh episode directory and an empty tool-call log before running, and always save the current tool-call list. A zero-tool run cannot inherit an older successful tool-call log.
2. E13 full-exposure checks include every saved model request. Missing or duplicate memory blocks make all-request full exposure false rather than silently dropping the request.

The targeted historical-data review found no evidence that these two defects affected the reported results. The original code and historical result files were not modified or rerun.

`docs/provenance/online_sources.json` maps each copied file to its original relative source, original SHA-256, release SHA-256, and modification status. The E17 and MemoryGraft `freeze*.json` files are newly generated **release-copy receipts**, explicitly marked as distinct from the historical freezes. Their purpose is to verify this portable copy before new runs. The corresponding `freeze*.py` scripts intentionally write maintenance receipts beside source files; they are not experimental run outputs.

Release checks cover fresh-directory behavior, missing/duplicate memory blocks, 120 snapshot hashes, 240 projection hashes, all launcher plans, path confinement, and absence of implicit `.env` loading. Local framework imports and entrypoint help were checked using the existing experiment environment. One full-constructor smoke check reached a Hugging Face tokenizer metadata lookup, which the network proxy rejected; it was stopped and pure framework/template imports were then verified with offline mode enforced. No chat or embedding service call was made. These checks do not claim that model services were called, fresh dependencies were installed, or the historical stochastic results were reproduced.
