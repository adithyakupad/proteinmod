# M11: explicit mutation interactions and measured epistasis

## Real benchmark and results

Original measured Wu2016 GB1 landscape, not the fitness-biased FLIP sampled subset. Same-experiment input/selected counts define all fitness labels, including singles; source149,361 measurements, bounded cohort2,648 (WT1/singles76/doubles2,091/triples360/quads120).64 probe-only lower-order genotypes support factorial evaluation and never train. C1 roles1,877/385/386, C2 roles1,569/425/654. C2 excludes pair39–54 on every training/validation background and retains all alleles through training singles. No whole-protein or unseen-position/family claim.

54 model/seed/protocol cases produce28,080 matched held-out predictions. Neural means average three seeds; deterministic baselines have one fit. Ranking and RMSE describe ln(max(count-normalized enrichment,0.01)); no prediction clipping. This is not a common-assay percentile or thermodynamic scale.

| Model | C1 rho | C1 RMSE | C2 rho | C2 RMSE |
|---|---:|---:|---:|---:|
| constant | undefined | 1.8279 | undefined | 1.5875 |
| single_additive | 0.7705 | 1.9853 | 0.7064 | 2.6643 |
| mutation_count | 0.1870 | 1.8094 | 0.2903 | 1.6936 |
| onehot_ridge | 0.8419 | 1.1023 | 0.6783 | 1.6630 |
| esm_ridge | 0.8568 | 0.8228 | 0.6904 | 1.2263 |
| structural_ridge | 0.7522 | 1.2154 | 0.6608 | 1.3137 |
| hybrid_ridge | 0.8322 | 1.0825 | 0.6640 | 1.6785 |
| sequence_pair_ridge | 0.7611 | 1.3719 | 0.5541 | 1.7808 |
| structural_pair_ridge | 0.7694 | 1.3327 | 0.4176 | 2.5846 |
| sequence_pairs | 0.7591 | 1.3512 | 0.5641 | 1.6852 |
| structural_pairs | 0.7468 | 1.3512 | 0.5571 | 2.0607 |
| hybrid_pairs | 0.8123 | 1.0655 | 0.4535 | 2.4696 |
| hybrid_zero3d | 0.8118 | 1.0641 | 0.4983 | 2.2086 |
| hybrid_shuffled_distance | 0.8174 | 1.0474 | 0.4763 | 2.2108 |
| hybrid_shuffled_biochemical | 0.8092 | 1.0853 | 0.4762 | 2.2668 |

The findings are mixed. C1 hybrid pair modeling improves WT epistasis over additivity, while ESM has better fitness ranking but weaker WT epistasis correlation. One-hot ridge ranks fitness well while implying exactly zero epistasis: predictive accuracy alone is not evidence of interaction recovery. Explicit3D does not beat the zero3D or shuffled-distance controls. On C2, the real hybrid's epistasis MAE is worse than additivity, and its fitness ranking degrades strongly. No structural/causal benefit is established.

## Direct factorial epistasis

Measured and predicted epistasis each use their own four terms: f(AB)-f(A)-f(B)+f(BG). Predicted contrasts use model predictions of all four, not noisy observed constituent labels as an oracle offset. Require all four actually measured and >=0.01; never fabricate singles/doubles or exact epistasis for censored observations. WT-background and non-WT background-specific pair contrasts are separate.

| Model | C1 WT epsilon rho | C1 MAE | C2 WT epsilon rho | C2 MAE |
|---|---:|---:|---:|---:|
| constant | undefined | 1.0340 | undefined | 1.2286 |
| single_additive | undefined | 1.0340 | undefined | 1.2286 |
| mutation_count | undefined | 1.0340 | undefined | 1.2286 |
| onehot_ridge | undefined | 1.0340 | undefined | 1.2286 |
| esm_ridge | 0.2942 | 1.0936 | 0.4509 | 1.0571 |
| structural_ridge | 0.1417 | 1.0971 | 0.3925 | 1.0187 |
| hybrid_ridge | 0.1903 | 1.1673 | 0.4379 | 1.6067 |
| sequence_pair_ridge | 0.4368 | 0.9800 | 0.3975 | 1.0440 |
| structural_pair_ridge | 0.4830 | 0.9463 | 0.4238 | 1.4438 |
| sequence_pairs | 0.4844 | 0.9284 | 0.3246 | 1.0847 |
| structural_pairs | 0.4496 | 0.9452 | 0.3238 | 1.1891 |
| hybrid_pairs | 0.6547 | 0.7658 | 0.4981 | 1.3520 |
| hybrid_zero3d | 0.6639 | 0.7502 | 0.5361 | 1.2069 |
| hybrid_shuffled_distance | 0.6798 | 0.7442 | 0.5237 | 1.1588 |
| hybrid_shuffled_biochemical | 0.6394 | 0.7819 | 0.5646 | 1.1689 |

C1 tests172 WT-background and68 non-WT contrasts; C2 tests213 WT-background and275 non-WT contrasts. Non-WT pairs share target genotypes and lower-order terms, so these counts are not independent experimental replicates:

| Model | C1 non-WT epsilon rho | C1 MAE | C2 non-WT epsilon rho | C2 MAE |
|---|---:|---:|---:|---:|
| single_additive | undefined | 1.0834 | undefined | 1.0053 |
| esm_ridge | 0.5008 | 0.8774 | 0.3420 | 0.9749 |
| hybrid_pairs | 0.2524 | 1.0992 | 0.2020 | 1.2279 |
| hybrid_zero3d | 0.2756 | 1.1064 | 0.2351 | 1.1745 |

The explicit hybrid does not improve non-WT contrast MAE over additivity. Its C2 fitness RMSE increases with mutation burden (doubles1.9180, triples2.4933, quadruples3.5734). That failure combines unseen-pairing/depth shift with a model constrained to background-independent pair contributions; it is not repaired or retuned after inspecting tests.

Across the declared cohort,1499 contrasts are eligible (1,125 WT,374 non-WT); exclusions: {'censored_constituent': 2333, 'missing_measured_constituent': 59}. Per-test counts and non-WT metrics are in runs.json/metrics.json. Constant/additive/count/one-hot predictions have zero model epistasis and undefined correlation, not a successful zero correlation. Numerical four-point roundoff below1e-10 is zeroed.

- C1 hybrid_pairs versus single_additive: fitness RMSE improvement +0.9198 (seed SD 0.0207); WT epistasis MAE improvement +0.2682 (SD 0.0119).
- C1 hybrid_pairs versus hybrid_zero3d: fitness RMSE improvement -0.0014 (seed SD 0.0178); WT epistasis MAE improvement -0.0155 (SD 0.0136).
- C1 hybrid_pairs versus hybrid_shuffled_distance: fitness RMSE improvement -0.0181 (seed SD 0.0039); WT epistasis MAE improvement -0.0216 (SD 0.0081).
- C1 hybrid_pairs versus hybrid_shuffled_biochemical: fitness RMSE improvement +0.0198 (seed SD 0.0279); WT epistasis MAE improvement +0.0161 (SD 0.0036).
- C2 hybrid_pairs versus single_additive: fitness RMSE improvement +0.1947 (seed SD 0.2213); WT epistasis MAE improvement -0.1234 (SD 0.2286).
- C2 hybrid_pairs versus hybrid_zero3d: fitness RMSE improvement -0.2610 (seed SD 0.0976); WT epistasis MAE improvement -0.1451 (SD 0.0709).
- C2 hybrid_pairs versus hybrid_shuffled_distance: fitness RMSE improvement -0.2588 (seed SD 0.2069); WT epistasis MAE improvement -0.1932 (SD 0.2396).
- C2 hybrid_pairs versus hybrid_shuffled_biochemical: fitness RMSE improvement -0.2028 (seed SD 0.1550); WT epistasis MAE improvement -0.1831 (SD 0.1085).

- C1 fitness explicit3D improvement, seed-mean ensemble:95% paired block interval [-0.0109, +0.0152]; 11 blocks.
- C1 WT_epistasis explicit3D improvement, seed-mean ensemble:95% paired block interval [-0.0320, -0.0006]; 6 blocks.
- C2 fitness explicit3D improvement, seed-mean ensemble:95% paired block interval [-0.3151, -0.2465]; 4 blocks.
- C2 WT_epistasis explicit3D improvement, seed-mean ensemble:95% paired block interval unavailable (one position-pair block); 1 blocks.

paired_bootstrap.json uses1,000 paired block resamples of seed-mean predictions, with mutation-position-set blocks for fitness and position-pair blocks for WT epistasis. C1 has few blocks and C2 WT only one, so that interval is unavailable. These conditional pilot intervals do not quantify shared single-measurement noise, training uncertainty or biological replicate variance. The original counts have no replicate-level error model here.

## Data and scale limitations

Primary source: [Wu et al.2016](https://doi.org/10.7554/eLife.16965), [measured supplement1](https://doi.org/10.7554/eLife.16965.024), [official FLIP source README](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/README.md). Archive2.66 MB, extracted46.4 MB, commit/checksums/licensing pinned in configs/m11.json. Original CC-BY-4.0, FLIP derivative AFL-3.0. No imputed supplement2 labels or keep/fitness-biased source sampling.

The published Fitness singles were imported from Olson2014 and scaled1.159. M11 instead computes every genotype from Wu selected/input counts divided by the WT ratio (WT input92,735/selected338,346). Non-single published values reproduce this formula; median absolute difference between count-derived and published singles is 0.0564. This avoids mixing assay batches but uses noisier single counts (minimum input14, some selected0). Global log-normalization cancels in four-point contrasts, but shared constituent/WT-background measurements still induce dependence. No precision or causal-mechanism claim.

The floor0.01 reflects an approximate reported detection limit. Fitness evaluation retains censored values, so RMSE evaluates a floored observation, not an exact latent fitness. Direct epistasis excludes censored constituents, selecting a more measurable subset and potentially biasing its distribution. scale_sensitivity.json/plot compare raw-enrichment and log-enrichment contrasts of the same four measurements; nonlinear transformation changes both magnitude and sometimes sign. Do not conflate these units or treat epistasis as scale independent.

FLIP inferred a full reference from a structure; only the compatible56-residue GB1 core is modeled. The inferred fusion tail is not claimed as an experimentally verified assay construct. IgG-Fc mRNA-display affinity selection can reflect folding and binding, not isolated thermodynamic stability. Numerical selection pH/temperature were not invented from library-preparation conditions.

## Leakage, structure and interpretation

All alleles are known from permitted training singles. C1 permits pairs seen in different training backgrounds, so it tests whole-genotype holdout only. C2 guarantees the test position pairing is absent from both training and validation; all quadruples are also test, creating mutation-depth shift. mutation_count_slices.csv separates doubles/triples/quads; rare_pair_slices.csv reports unseen and rare exact mutation-pair occurrences in training. No test/probe outcome enters a feature, scaler, additive delta or selection. Test-label perturbations reproduce fitted models unchanged.

Validated1PGA geometry has55 compatible nodes; known Q2T at residue2 is excluded, and all four mutated sites map exactly. Distances, contact neighborhoods, components/paths, source-confidence and missing masks remain distinct. B-factors are A², not pLDDT or predictive uncertainty. Only six position pairs exist; geometry can encode positional identity, so improvements over shuffled/zero controls cannot establish a physical mechanism. Reference geometry is not measured mutant rearrangement.

Measured-single additivity plus symmetric ridge/MLP pair residuals supports any mutation set with known features. WT/singles have empty pairs and unchanged additive predictions. Pair contributions use only that unordered pair and frozen reference-single context, so these explicit predictors are pairwise sums; they do not estimate true higher-order coefficients. Background-specific measured pair contrasts can vary for noise or higher-order reasons. The full-mutant ESM/hybrid probes model implicit interactions, but fitness accuracy alone is not evidence of measured epistasis.

Plain sequence_pairs and structural_pairs have identical420-input MLP capacity (explicit3D zeroed for sequence). Hybrid uses720 dimensions including symmetric frozen reference-single ESM/GNN/bidirectional context; zero3D and shuffled hybrid controls have identical capacity. The no-explicit3D hybrid still has implicit structure. Shuffled3D moves descriptor records among the six unordered position pairs; biochemical control permutes explicit destination identity/property records within each position. Frozen contexts stay correct, so controls isolate explicit increments rather than removing all biological information. Reference permutations are label-blind and consistent across roles.

nonadditive_examples.csv selects large measured WT contrasts by absolute truth magnitude, including errors, rather than successful model examples. Generalization and control differences should be interpreted alongside scarce structural diversity, noisy singles, censoring and low-order sampling. No source/model was changed after inspecting held-out outcomes.

## Reproducibility and compute

Run:26.62s on arm64 CPU/four threads; peak RSS 0.988 GiB. Frozen preparation:38.10s including local MPS ESM token extraction/cache access; checkpoint/frozen pretraining costs excluded. Versions:torch 2.14.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, biopython 1.88, transformers 4.57.6.

Frozen ESM t12 35M revision6fbf070e... and M7 function-only encoder trained old amylase/validated NucB, with GB1 held out. No historical GB1-supervised head is loaded; PLM pretraining exposure unknown. No M8 stability scalar. Source/reference/structure/model/config/code hashes and devices are archived; binary caches/weights/raw archive stay ignored. Protocol/config/code/tests committed before fitting.

Exact reproduction follows docs/m11_protocol.md: download pinned archive, prepare, run epistasis_experiments, verify_m11 and report_m11. Report-only: `MPLCONFIGDIR=/tmp/proteinmod-m11-matplotlib python scripts/report_m11.py --input benchmarks/m11 --output runs/m11-report`. Full repeat changes only output. Replay/repeat results are archived separately. All238 M0–M10 benchmark artifacts and15 prior configurations are unchanged. Stacked on verified #20 → #18 → #16 → #14 → #12, all50 dependency checks passed; main unchanged.

No M12 family-held-out expansion, new encoder training, formal uncertainty calibration, LoRA or epistasis-as-causality claim.
