# Licenses and scope

Copyright (c) 2026 Junwen Tan and Yong Ding. The authors' contributions are
licensed as follows. These grants do not replace third-party licenses.

## Software: MIT

The authors' original code and software documentation are licensed under the
[MIT License](LICENSE). This includes `src/`, `scripts/`, `tests/`, the release
launcher and helpers in `online/`, and the authors' experiment implementations
in `online/experiments/`. Executable task and material definitions in Python
files are part of the software and are covered by MIT. Software configuration,
dependency lists, and documentation are also covered by MIT unless identified
below as research data or third-party content.

## Research materials and processed data: CC BY 4.0

The authors' separately distributed research materials and processed data are
licensed under [Creative Commons Attribution 4.0 International](LICENSE-DATA)
([official license](https://creativecommons.org/licenses/by/4.0/)). This covers:

- All released observations in `data/processed/`.
- The archived comparison values in `reference/`.
- The authors' JSON, JSONL, CSV, and TXT research materials under
  `online/experiments/`, including prepared attack payloads, task/configuration
  records, reviewed restoration snapshots, detector projections, and translations.
  Software dependency lists and third-party content are excluded from this item.

For attribution, identify Junwen Tan and Yong Ding, the project title
*Conjunctive Prompt Injection in Memory-Augmented Agents: Joint Retrieval and
Unauthorized Tool Execution*, and the
[source repository](https://github.com/UniDarkstars/conjunctive-memory-injection).
Retain the license notice and indicate changes as required by CC BY 4.0.
The accompanying manuscript metadata is available in [CITATION.cff](CITATION.cff).

## Third-party content and excluded files

- The bundled ADK source in `online/agent-frameworks/adk-python/` retains its
  [Apache-2.0 license](online/agent-frameworks/adk-python/LICENSE) and existing notices.
- The AgentPoison reference under
  `online/experiments/ADK/_e17/external/vendor_AgentPoison/` retains its
  [upstream MIT license](online/experiments/ADK/_e17/external/vendor_AgentPoison/LICENSE)
  and copyright notice. Other third-party notices remain applicable wherever present.
- Installed Python packages and externally obtained models remain subject to
  their own licenses. Model weights are not included.

These grants apply to the materials actually distributed in this repository.
Raw service logs, real `.env` files, account configurations, and the separately
maintained manuscript, figures, and IEEE template files are not included in
these grants. The repository does not relicense those files.
