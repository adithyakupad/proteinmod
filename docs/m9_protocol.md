# M9 prespecified assay-aware protocol

Issue #17 asks whether observable assay metadata improves mutation-effect ranking on held-out proteins. The primary comparison is `structured` versus the same-capacity `zero` head; `shuffled_train` is the negative control. No M10 work, PLM tuning, thermodynamic conditioning, or new joint pretraining.

## Dependencies and scope

M9 is stacked on draft PR #16 (`16a53a1`), itself stacked on draft PR #14 (`dd9e8bc`) and #12 (`1753cbb`). All 30 dependency push/PR CI checks passed on those exact heads. Drafts remain reviewable; main stays unchanged. Preserve all 199 M0–M8 benchmark files and existing configurations. Rebase only after dependency review/merges; the M9 PR base is `codex/m8-stability-informed-function`.

The change reaches new dataset/metadata adapters, cached feature preparation, experiment heads/runner, config, tests, CI, reporting/replay scripts and documentation. Existing APIs and experiments are unchanged. Source data, dense features and weights remain ignored. Existing files with mismatching checksums fail closed.

## Collection and identifiability

The historical M4 collection has three proteins and three unique assays, one assay per protein: assay-ID effects cannot be separated from protein identity. Retain that history without reinterpretation. The new bounded collection crosses **PTEN, CYP2C9 and VKOR1 with abundance and activity endpoints**. Each protein has both measurements of exactly the same selected genotypes. This identifies a shared endpoint contrast beyond a protein-only label; it does not make individual assay IDs independent of protein. No direct numerical comparison with the old different-protein M4/M8 cohorts is valid.

Pin six ProteinGym v1.3 substitution CSV members, source SHA256s, reference-catalog commit, sequences, AlphaFold DB v6 structures and metadata provenance in configs/m9.json. Download only the selected ZIP members via validated HTTP ranges/CRC (about 0.51 MB compressed; whole archive43 MB). ProteinGym curated data license MIT; AlphaFold structures CC-BY-4.0; retain original study citations. DMS_score is the processed higher-is-better experimental outcome, never a physical energy unit. Raw phenotype names/directions are retained for provenance, not reapplied to already processed scores.

For each protein intersect the two measured genotype sets, order by SHA256(salt + protein + sequence), select180 without reading labels. This gives540 single-mutant genotypes and1,080 real experimental observations. Report raw counts, intersection, unpaired exclusions and cap exclusions. Sequence/mutation/reference equality and duplicate/nonfinite labels are checked. The assay adapter accepts multiple mutations; this collection does not validate multi-mutant performance.

G1 groups both endpoint observations of each genotype into the same70/15/15 role (126/27/27 genotypes per protein;756/162/162 observations). G2 rotates test protein, next validation protein and remaining training protein:360 rows per role. All proteins have both endpoint families, so test families are known from training. Complete mutant sequences and protein identities cannot cross G2 roles. All methods evaluate identical IDs. Preserve original M0–M8 partitions; these are new declared M9 partitions, not resplits of old benchmarks.

## Observable metadata

Schema: family, biological endpoint, technology, biological context, substrate/reporting proxy, processed-score unit/direction, assay ID, temperature in Celsius, pH, provenance. Numeric conditions carry distinct observed/unavailable/not_applicable states and zero-filled standardized values with explicit masks. No default37C or pH7.4. The curated source snapshot does not provide numeric pH/temperature, so all six records mark them unavailable; no numerical-condition effect can be estimated here. This describes curation availability, not a claim that original papers never reported conditions. Activity measurements are heterogeneous proxies (PTEN yeast growth, CYP fluorescent probe, VKOR carboxylation reporter); do not assert a common native scale or identical biochemistry.

Vocabulary and numeric scaling fit **training metadata only**, UNK embedding0 for unseen values, explicit missing tokens/statuses. If a condition has no observed training values, unseen observed values are zeroed rather than extrapolated from an untrained scaler. Unknown metadata fields (including targets or outcome statistics) are rejected. Labels, test label distributions, patient variables and protein IDs are forbidden primary metadata inputs. Protein ID is an explicit diagnostic control only. G2 assay IDs are unseen and mapped to UNK. Family/endpoint categories are known; technology/substrate can be unseen. Record counts.

## Features and models

Reuse frozen ESM-2 t12 35M residue embeddings (revision6fbf070e...), M2 residue graphs/M3 descriptors, and the existing M7 function-only GNN/bidirectional encoder. The fixed M7 encoder trained on old amylase function and selected epochs on NucB; neither overlaps new supervised proteins under the declared similarity screen (>=30% aligned identity and >=80% coverage both ways). No M6 stability head or M8 predicted stability scalar is consumed. In particular, invalid OOD structural stability extrapolations are not thermodynamic inputs. Frozen PLM pretraining exposure is unknown.

Reference graphs require exact full-sequence matching,8A contacts/backbone,960-point SASA, no DSSP. Audit graph coverage/connectivity, pLDDT<50, mapping exclusions and sequence/structure masks. Geometry describes AlphaFold reference predictions, not mutant rearrangements. An unmappable selected genotype fails preparation rather than being silently excluded after outcome inspection.

Compare ESM mutation-site ridge961, structural ridge270, frozen multimodal ridge100; simple separate endpoint-family ridge heads; assay-ID ridge heads with global fallback for unseen IDs. Ridge alpha [.1,10,100] by validation macro Spearman only. Inputs standardized from training; SD<1e-6 uses scale1 to suppress floating-point noise amplification. No old M8 scaler changes. Predictions clipped[0,1] by prespecified contract.

The conditioned head sums16-dimensional categorical embeddings with numeric/mask projection and concatenates to the100-dimensional frozen representation, then116→16→1 ReLU/sigmoid. All five modes instantiate identical capacity/initialization: `zero` disables metadata; `structured` omits assay/protein IDs; `with_assay_id` includes assay ID; `shuffled_train` uses structured metadata but swaps whole endpoint records within each training protein; `protein_proxy` enables only protein ID. Two endpoints mean the label-blind nonidentity permutation is a swap; across seeds it is the same swap. Validation/test metadata remain correct. No row or label crosses roles. `structured_missing_at_test` zeroes all context at inference using the selected structured head; this is a missing-context stress, not a novel-family experiment.

Frozen encoder and identical head families isolate incremental conditioning. No new attention conditioning is necessary for this first test. Train seeds0,1,2, CPU4threads, hidden16, AdamW.003/decay.01, batch64, clipping5, max60epochs/patience10. Choose earliest best validation macro Spearman, never test. Numeric projection initialized zero to avoid random predictions from unseen numeric/status dimensions. Fit MSE to training-assay midrank percentiles, **not** pooled native scores. Predictions are bounded ranking scores, not calibrated probabilities or held-out empirical percentiles.

## Evaluation and reproducibility

Compute Spearman against native measurements separately per assay; average endpoints per protein, then equal-protein macro. Never pool unlike native outcomes or regression errors. Undefined constant prediction rho remains null and contributes zero to macro, without dropping a model/protein. Report individual protein/endpoint results, three-seed variation, paired conditioned-minus-zero changes, shuffled/proxy controls, training/validation traces and failures. Three proteins are not enough for population confidence claims. No unseen-family claim or novel-assay-family holdout: two families cannot supply three disjoint family roles.

Commit this protocol, config and implementation before the first head benchmark. Cache token extraction by pinned source/model identity; record features/checkpoints/config/source hashes, dependency versions, seeds, device, runtime/RSS. Verify all historical hashes before/after. Repeat the full experiment with only output changed; compare all predictions and metrics/selections except checkpoint serialization hashes and runtime. Replay all saved heads and validate metadata masking/dimensions. Dense binary artifacts stay uncommitted.

## Reproduction

Use Python3.12 and requirements-m3.txt; frozen encoder prerequisites follow docs/m7_protocol.md (M4/M5/M6 input preparation and M7 function-only training). If missing, preparation reconstructs the exact configured M7 function-only checkpoint and checks its encoder tensor hash. Frozen ESM weights must be cached locally (existing download workflow in M1/M5).

```
python -m proteinmod.assay_experiments download --config configs/m9.json
python -m proteinmod.assay_experiments prepare --config configs/m9.json
python -m proteinmod.assay_experiments run --config configs/m9.json
python scripts/verify_m9.py --input runs/m9
MPLCONFIGDIR=/tmp/proteinmod-m9-matplotlib python scripts/report_m9.py --input runs/m9 --output runs/m9
ruff check .
ruff format --check .
pytest -q
```

ESM uses MPS when available; fitted heads use deterministic CPU. For a full repeat copy config changing only output to runs/m9_repeat. For report-only reproduction pass --input benchmarks/m9 --output runs/m9-report; raw downloads and neural weights are unnecessary.
