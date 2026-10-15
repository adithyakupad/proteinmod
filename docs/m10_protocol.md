# M10: prespecified structure-confidence input and learned reliability fusion

Issue #19 tests incremental mutation-effect ranking signal from **reference-geometry quality**, not predictive uncertainty. Main decision: compare confidence-input versus neutral-input encoder, and confidence-gated versus zero-confidence capacity control; combine both and retain all failures. No benefit assumed, no M11 explicit epistasis, M13 uncertainty or M15 LoRA work.

## Dependency and change boundary

Stack on verified M9 head9610fe4 (PR #18), then #16/16a53a1 → #14/dd9e8bc → #12/1753cbb. All40 dependency push/PR checks succeeded; all are drafts. Preserve main/ad2e564. Existing APIs, M0–M9 artifacts and configurations are immutable. Add typed reliability descriptors, confidence masking/preparation, frozen-encoder ablations, fusion heads, tests/CI, reporting and replay. No new dependencies, downloads, datasets or structure replacements.

## Immutable cohort and roles

Primary cohort is exactly M9's1,080 observations:180 paired abundance/activity genotypes each for PTEN, CYP2C9 and VKOR1. Assert every sequence, outcome, mutation, assay and role ID equals archived M9 records. G1 keeps756/162/162 training/validation/test rows, genotypes grouped across endpoints. G2 rotates whole proteins,360 per role, with both endpoint families seen in training. All models score identical test IDs. Reuse M9 provenance/licensing/source checksums and M2 graphs. Keep assay identity and train-assay rank semantics unchanged; no new assay-conditioned learning.

All primary structures are AlphaFold DB v6. **Experimental-versus-AlphaFold performance is unidentifiable here**. B-factor and resolution handling is unit-tested with synthetic inputs; no experimental-source performance claim. No optional M4 contrast in this bounded first experiment. All540 genotypes are single substitutions: multiple-mutation invariance is engineering-tested, not empirically validated performance. Keep mapping exclusions zero, fail preparation if the matched cohort changes.

## Typed reliability and normalization

Local pLDDT is AlphaFold predicted geometry confidence in[0,100], not confidence in functional effects or mutant geometry. B-factors describe experimental displacement/disorder in Angstrom squared; never convert them into pLDDT. Experimental resolution is in Angstrom and distinct from either. Store source semantics and M2 manifest provenance. Reject source-incompatible values, out-of-range pLDDT, negative B and invalid mapping fractions.

Descriptor fields: source AF/experimental/unknown; mean mutated-site pLDDT and 8A spatial-neighborhood pLDDT; mean B-factor in those same regions; four observed fractions for confidence missingness; coordinate coverage, missing-coordinate fraction, mutation-site coverage, alignment-mismatch fraction; experimental resolution and availability; structure availability. Confidence masks move with their values. Empty/unmapped mutation sets yield explicit zero observation/coverage, never invented local quality. Geometry of unmapped sites is absent. Neighborhood uses union of8A balls including mutated sites; mutation-set aggregation is permutation invariant.

Use fixed physical scaling pLDDT/100, B-factor/(100 A²), resolution/(5 A), bounded mapping/missingness fractions. No per-protein fitted scales or label-derived normalization. M9's within-protein StandardScaler amplified genuine global-representation shifts; M10 declares raw LayerNorm-based frozen representations for **every** new comparison before fitting. This is a new matched baseline, not an alleged drop-in improvement over historical M9 heads.

## Encoder input A and controls

Reuse exact frozen ESM-2 t12 35M revision6fbf070e65b0b7291e7bbcd451118c216cff79d8, M2 graphs and M7 function-only pretrained GNN/bidirectional encoder. That encoder trained on old amylase/validated on NucB; M9 similarity audits remain applicable. No M6 stability head or M8 scalar is called. PLM pretraining exposure remains unknown.

M3/M7 **already have** typed B-factor,pLDDT and missingness in the GNN input. A reuses these existing pretrained input weights; the real profile enables those channels **before message passing**, not just appended to a prediction head. Neutral input sets values0 and missing bits1 for every residue, masking legacy confidence too. This isolates confidence input with exactly the same frozen parameters and same new head. No newly initialized confidence adapter or end-to-end GNN training. Real-input frozen representations must replay M9's100-dimensional arrays exactly before fitting.

Shuffled profile permutes whole confidence value/mask records among residues **within each reference protein**, preserving source and marginal distribution, once per seed0,1,2. Coordinates, sequence, mutation flags, graph edges and labels stay fixed. Recompute neighborhood/site summaries after shuffling. Apply the same reference permutation across roles: confidence is label-free reference data shared by variants, not outcome/variant information crossing splits. Shuffling destroys local confidence-to-geometry association but preserves protein-level confidence marginals; it is not a null distribution over many datasets.

## Fusion C, modes and capacity

Frozen bidirectional encoder returns48 sequence-query pooled coordinates and48 structure-query pooled coordinates plus count, sequence/structure availability and mapped-mutation coverage. Query ordering is immaterial. Both query branches already include cross-attention: a gate weight on the structural-query expert **does not measure all structural influence** on the other branch. Never interpret gate/attention as causal biology or calibrated uncertainty.

Two compact experts map each48-vector+mutation count through hidden16 to a sigmoid ranking score. Reliability gate takes both48-vectors+18 reliability descriptors through hidden16 and sigmoid, then mixes expert predictions. A learned scalar mix is the plain weighted late-fusion baseline on the same two cross-attended experts. Concatenation head is100→16→1 sigmoid. All components are instantiated in every mode to match registered budgets; active capacity differs between concatenation and mixture. Gate-zero and shuffled comparisons, rather than registered counts alone, isolate confidence signal.

Fixed modes:
- unconditioned: neutral GNN confidence, concatenation head;
- confidence_input: real GNN confidence, identical concatenation head;
- weighted_late: neutral GNN confidence, learned scalar expert mix;
- gate: neutral GNN confidence, learned reliability gate;
- combined: real GNN confidence and learned reliability gate;
- gate_zero: neutral GNN confidence, same gate but reliability vector zero;
- input_shuffled: shuffled GNN confidence, concatenation head;
- combined_shuffled: shuffled GNN confidence and shuffled reliability gate inputs.

Structure absence recomputes encoder features with all structural attention keys/query masks disabled and a sequence-only path; structural pooled features/coverage and gate weight are exactly zero. Missing structure is distinct from low/unknown confidence. Mixture uses its trained sequence expert; concatenation uses its same trained head with missing geometry. It must never select an untrained random fallback head. All-missing geometry is evaluated as an inference stress without validation reselection, **not calibrated missing-modality performance** (training contains complete structures).

## Fitting, selection and reporting

Seeds0,1,2; CPU4threads deterministic, hidden16, AdamW.003/decay.01, clipping5, batch64, max60epochs/patience10. Only new heads/gates/scalar train; ESM, GNN and attention remain frozen. No test-led model selection. Fit MSE to training-assay midrank percentiles; earliest optimum validation macro Spearman selects epoch. All modes/seeds retained. No common native scale/probability/held-out-quantile claim from sigmoid scores.

Evaluate native-measurement Spearman per assay, then equal endpoint mean per protein and equal protein macro. Null constant rho contributes zero without dropping cases. Report per-assay/protein/source, paired seed changes, low(<70)/high(>=70) site-pLDDT slices, missingness, mutation burden, generalization gaps, gate distributions/correlation with pLDDT and costs. Three proteins/three seeds are not population confidence intervals. All sources are AF and burdens single-mutant; explicitly report absent slice identifiability. All-missing fallback test scores are stress diagnostics only.

Commit protocol/config/tests before fitting. Run full repeat changing only output; compare predictions and all selections/metrics except runtime and checkpoint serialization hashes. Reload every saved head and replay predictions/gates/fallback, recompute metrics and matched IDs, compare all historical hashes. Binary tokens/features/weights remain ignored. Record revisions, devices, cache-hit preparation and head runtimes separately, RSS, versions and fingerprints. No M8 OOD stability estimates as reliability/thermodynamic inputs.

## Reproduce

Use pinned Python3.12 requirements-m3.txt and M9 preparation from docs/m9_protocol.md. M10 uses existing pinned residue caches and M7 encoder; no new raw downloads.

```
python -m proteinmod.reliability_experiments prepare --config configs/m10.json
python -m proteinmod.reliability_experiments run --config configs/m10.json
python scripts/verify_m10.py --input runs/m10
MPLCONFIGDIR=/tmp/proteinmod-m10-matplotlib python scripts/report_m10.py --input runs/m10 --output runs/m10
ruff check .
ruff format --check .
pytest -q
```

Report-only reproduction takes --input benchmarks/m10 --output runs/m10-report and requires no large arrays or weights. A complete repeat uses a config copy changing only output to runs/m10_repeat.

M9's historical-file enumerator is bounded to M0–M8 so adding future milestone artifacts does not make the archived M9 verifier falsely report a historical change. Its three callers (runner, verifier, test) were checked; a future-artifact regression test covers this compatibility fix. No M9 data/config/results or fitting logic changes.
