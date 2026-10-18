# Proposed research release notes — NOT PUBLISHED

Candidate source baseline: `7cc4dbbe9c47355d1afa46f6f46f502394e9e999` plus the
reviewed M16 qualification PR. No version tag is assigned or created.

- Consolidates M6–M15 stacked work in dependency order, preserving M0–M15 evidence
  and original commits. Adds clean-checkout qualification and bounded inference examples.
- Includes frozen ESM-2, mapped residue graphs, edge-aware GNNs, frozen ProteinMPNN,
  bidirectional attention, native stability and function heads, masked unpaired
  multitask learning, assay metadata, typed confidence, explicit epistasis,
  uncertainty diagnostics and final-block LoRA as documented research workflows.
- Preserves original splits, assay units, sign conventions and validation-only selection.
  Reports negative transfer, failed controls and held-out-family limitations.
- CI covers offline Python 3.11–3.14 and the complete CPU neural suite; formatting is gated.
- New README links verified results, protocols and source/license provenance.

Breaking changes: none to existing model/data APIs. No old config, score or prediction
is replaced. `scripts/infer_m16.py` deliberately supports only known source-matched
benchmark variants with available pinned weights. It does not provide a universally
validated unknown-protein predictor or transferable uncertainty intervals.

Release holds: project source-code license, final exact-head qualification review,
explicit acceptance of incomplete network cold-start reproduction and the scientific
limits in the [qualification matrix](../benchmarks/m16/QUALIFICATION.md).
Do not publish a tag/release until these gates are reviewed successfully.
