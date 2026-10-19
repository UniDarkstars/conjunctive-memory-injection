# Changes made only in the standalone release

This repository was assembled from a previously verified six-table offline package. Original experiment code, frozen copies, data and paper sources were not edited. No online experiments were run and no GitHub upload was performed.

## Numerical coverage

- Added E10, E13 and E19x paired statistics with their original adjustment families; added supplementary S1 and S2.
- Added E14 task observations, matching/mismatching counts and four paired statistics.
- Added LangMem E19q baseline and return-width sensitivity, all reported E17/E17mg competition levels and method comparisons.
- Preserved E17's NumPy generator, seed 20261008 and 100,000 Monte Carlo draws, including the original add-one p-value estimator.
- Added E15 source/window content coverage and joint coverage/label checks. Complete coverage and missed detection are intersected by source identity, rather than inferred from separate marginal totals.
- Added a unified full-analysis entry while preserving the dependency-free six-table entry. E10's validation error text was corrected to say batch/material labels; it does not independently establish an absent historical LangMem model field.

## Corrected historical defects

1. **Empty tool logs and reused directories.** Early E10/E13/E14 runners wrote `tool_calls.jsonl` only when a call occurred. A later zero-call run in the same directory could leave old evidence. Release runners require a fresh episode directory and write an empty log for zero calls.
2. **E13 missing blocks.** Historical all-request exposure checks collected only existing memory blocks. The corrected checks keep one result per request and reject missing or duplicate blocks. Tests include a complete request followed by a block-free request, which the old implementation could wrongly accept.

The pre-release audit checked 1,440 saved E10/E13/E14 runs for tool-record consistency and 2,647 E13 requests across 720 runs for memory blocks. No evidence was found that these two defects changed the currently reported results; every saved E13 request contained exactly one block. This is evidence about saved runs, not a guarantee about every historical filesystem operation.

## Portability and credentials

Online code uses repository-relative material/runtime locations and guarded local output directories. Credentials are obtained from the process environment, and automatic reads of account `.env` files are removed. Reviewed restoration snapshots, detector projections and fixed attack payloads are included separately from raw service logs. Third-party runtime licenses remain with the copied source.

Release freeze receipts describe the corrected local copy. Historical source hashes are recorded separately in `provenance/online_sources.json`; a regenerated release receipt is not presented as the original experiment freeze. Online text is normalized to LF before these receipts are made, and Git attributes preserve that convention. Offline numerical CSV bytes are retained.

## Verification boundary

Offline reconstruction, independent-reference comparisons, isolation tests and targeted regression tests are performed locally. Online list/dry-run and argument/resource checks do not contact chat services or run model inference. Dependency installation in a new environment and an actual online rerun are separate operations; the release does not claim they occurred during preparation.
