# SOD1 blinding incident disposition — M20 Phase 0

Decision: **SOD1 may be considered for a NEW prospective study with a disclosed blinding deviation; pristine blinded replication is disallowed.** This is conditional scientific disposition of the incident, not dataset admission, run authorization, permission clearance, or independent validation of predictive performance.

## Reviewer and scope

The reviewer is a separate context-isolated AI agent that did not select the SOD1 candidates. It is not an external human independent auditor. It inspected only the eight committed sources listed below and their relevant history, including the three specified commits. No cache, raw data, aggregate distribution values, phenotype tables, score endpoints, outcome workbook cells, or external outcomes were read. No models, tuning, acquisition, or contacts were performed. This review cannot self-certify actual unseen historical memory, human exposure, or unrecorded process behavior.

## Incident and evidentiary strength

**Self-reported process history:** BLINDING_DEVIATION.md says the initial abundance metadata GET included a nonempty scoreDistribution object and two whole-metadata Python JSON inspections decoded it into memory before exclusion was recognized. A later shape-only inspection reported a nonempty dictionary. The record says values were not printed, queried, used for filtering or model selection, committed, or supplied to a model; no variant-level outcomes were accessed. qualification.json repeats the failure and those assertions. These are contemporaneous committed disclosures, not execution traces proving non-consumption. The inspected history does not contain the two initial inspection commands or their complete runtime state. The initial audit's categorical no-score-access wording must be read with the later deviation disclosure, not used to erase it.

**Code-supported guarantees, bounded to inspected paths:** project_metadata receives the entire raw metadata text in memory; it scans excluded value spans lexically but decodes values with json.loads only for allowed top-level keys. scoreDistribution is absent from the default allowlist and is not returned. Thus subsequent use of this default projection excludes that top-level value from semantic JSON decoding; it does not mean the response bytes never entered memory. Metadata key decoding and whitelisted nested metadata decoding still occur. Caller-supplied allowlists and other readers lie outside this guarantee. The source contains no inspected path consuming scoreDistribution for ranking, filtering, model selection, calibration, or predictions. Genotype acquisition restricts the abundance URN and namespace, checks an explicit header allowlist before application-level body parsing, and rejects malformed row widths. This is a parsing boundary, not a guarantee that network buffers never contain body bytes or that a server cannot mislabel content.

**Tests and history:** d4a0ef2 adds projection and genotype-header safeguards; the synthetic tests exercise excluded nested/escaped content and a score-column header rejected before body reads. f6ec52e records the incident/protocol/qualification and adds overwrite protection; 56be251b changes the reported Data S2 lineage gate while retaining the incident failure. Tests were inspected, not executed in this review. They support intended present control flow, not retrospective reconstruction of the earlier session or every possible input. The extended bounded review includes the visible candidate-selection implementation in scripts/audit_sod1.py and its configuration; imported helper implementations remain outside this review.

**Consumption finding:** mechanical decoding is itself consumption of outcome-derived aggregate metadata by the earlier process and violates strict pre-decode exclusion. It is not, by itself, evidence that a decision algorithm used distribution values. The supplied evidence is consistent with the reported absence of decision-logic consumption and contains no affirmative evidence of such use, but does not prove historical non-use. Past human/process exposure beyond the disclosed record is unknowable here. Reported variant-label blinding remains an assertion consistent with the inspected controls, not an independently reconstructed fact.

## Extended candidate-selection control-flow review

The committed scripts/audit_sod1.py reads the projected metadata only to verify assay reference sequence and URN. It reads the genotype export, verifies assay/RefSeq HGVS agreement, decodes canonical non-WT single substitutions, excludes the unobserved initiator Met, rejects duplicate mutant sequences, and asserts structural feature validity. For each eligible mutant, it computes SHA256 of the literal `m18:external:SOD1` concatenated with the canonical mutant sequence, sorts ascending by that digest, and takes the first 96 rows. The genotype accession is carried as the selected ID; it is not the ranking input. No visible scoreDistribution lookup or distribution-dependent branch feeds this selection path.

configs/m185.json pins the reference, coordinate checksum, positions 2–154 and abundance URN, and sets target_read_permission=false. Its selection_prefix and max_candidates agree with the source's literal salt and 96-row slice, but those two configuration keys are not actually read by this source. The primary_target_column name is declarative; the inspected audit does not fetch that outcome column. scripts/sod1_sources.py supplies the projected metadata and header-validated genotype rows described above.

Thus the visible selection algorithm depends on salt, reference/mutant sequences, genotype eligibility and structural validation, not aggregate distribution values. This is stronger code-path evidence of outcome-independent selection than the incident narrative alone. It does not establish which commands were actually run, whether an earlier transient process used outcomes, or whether uninspected imported helpers or supplied inputs differed. No candidate output, genotype data, coordinates, distributions or outcomes were read or recomputed for this review; the actual selected 96 IDs were not validated against execution results. Historical non-consumption and unknown past memory remain unproved.

## Disposition and remaining blockers

A new prospective study may consider SOD1 only with this incident disclosed before any further selection, prediction, or outcome opening and with a newly frozen protocol that explicitly accepts the aggregate-metadata deviation. Do not describe the old process as pristine, untouched, or a clean replication; a new preregistration cannot restore past blinding. The responsible scientific owner must explicitly accept or reject this residual uncertainty and the adequacy of this AI review. If policy requires an external human independent reviewer, that requirement remains unsatisfied and must be fulfilled before proceeding. This document does not substitute an AI opinion for such a requirement.

All separate admission and rights gates remain active. In the inspected qualification, CspA/CI2 exact-item permissions remain unresolved; SOD1 cannot clear them. Other gate PASS entries were read as reported status and not independently re-audited here. No final admission, external run, outcome opening, or pristine-blinding claim follows from this disposition.

Preserve original M17 H1–H5, four families, original >=3/4 directional agreement, source folds, budgets, seeds, controls, native assay semantics, assay-local evaluation and equal-family macro. SOD1 remains a proposed GFP replacement, not a fifth family. No threshold weakening, outcome-driven replacement, or retrospective reclassification is authorized.

## Evidence identity

Review baseline: `402bd8815dcd8857d416f47b1b41fdb922cf471d`. Git history inspected at `d4a0ef2ef3df895e926d733abd242e81190ceeeb`, `f6ec52e59aca24b25428b6c26ee90195b8d2e0d3`, `56be251b938b9cb8e9ef454893931eecaa2538c4`. SHA-256 hashes identify committed file bytes, not uninspected raw data. The companion JSON also records Git blob IDs.

| Committed source | SHA-256 |
| --- | --- |
| `benchmarks/m185/BLINDING_DEVIATION.md` | `696fbeb9202cc42a2f8ae20efd6c354149d334277975a2d2bbf507a409750c21` |
| `benchmarks/m185/ADMISSION_PROTOCOL.md` | `58f6e52736cc162fc97a81740300872805d3e08ccfcd0299c58bfab540aa0f38` |
| `benchmarks/m185/qualification.json` | `e094343ca32aa14be3a66cfc8e0512deb0c1b072420aca87415bdc393d5afaab` |
| `scripts/sod1_sources.py` | `8b623525da617496b345c6207be2b250ccb3228885419172b1cfb4bdcd56078a` |
| `tests/test_sod1_sources.py` | `473ac244f23088ff6a5a7ff9d593eb1a10e09e8775d0d2d727bbce31698ab746` |
| `docs/SOD1_BLINDED_ADMISSION.md` | `ae9f468004619878f48d78537279635ffafc78a3e89c51ec1d8908b1b373cead` |
| `scripts/audit_sod1.py` | `694c6cd8c7b2cc4803d793886b96001f3064321c812d72bbad49d83a677d0edf` |
| `configs/m185.json` | `927656a24744fa03cfbb1e1ecc95c80da963c6de87b2ad5f65d959efc6f091eb` |
