# M10: confidence-aware structure input and reliability fusion

## Actual matched experiments

The exact M9 six-assay cohort and G1/G2 role IDs are retained:1,080 observations from540 paired-endpoint genotypes across PTEN, CYP2C9 and VKOR1. 96 selected model/seed/fold fits,29,808 held-out predictions plus the same29,808 missing-structure stress predictions. All native correlations are assay-local before equal-endpoint/protein aggregation; three-seed means and variation are not population confidence intervals. All candidates and failures remain visible.

| Model | G1 rho | G2 rho | Missing-structure G2 stress |
|---|---:|---:|---:|
| unconditioned | 0.3171 | 0.3491 | 0.2192 |
| confidence_input | 0.2420 | 0.3090 | 0.2141 |
| weighted_late | 0.2977 | 0.3443 | 0.2391 |
| gate | 0.2725 | 0.3365 | 0.2074 |
| combined | 0.2354 | 0.3082 | 0.2353 |
| gate_zero | 0.2856 | 0.3381 | 0.2203 |
| input_shuffled | 0.2512 | 0.2996 | 0.2140 |
| combined_shuffled | 0.2506 | 0.3048 | 0.2254 |

G2 paired seed changes:

- confidence_input minus unconditioned: -0.0401, seed SD 0.0103, range -0.0485 to -0.0256.
- gate minus gate_zero: -0.0015, seed SD 0.0040, range -0.0064 to +0.0033.
- combined minus gate_zero: -0.0298, seed SD 0.0032, range -0.0343 to -0.0274.
- combined minus combined_shuffled: +0.0035, seed SD 0.0047, range -0.0029 to +0.0084.

The result is negative: confidence input is worse than the matched neutral-input baseline in all three G2 seeds. The learned gate does not improve on its zero-confidence control, and combined fusion's small advantage over shuffled confidence changes sign across seeds. Do not infer incremental biological reliability from these comparisons. This tests a frozen pretrained encoder plus small heads, not every possible confidence-aware architecture.

## Protein and quality failures

| Held-out protein | Assay | Unconditioned | Input | Gate | Combined | Shuffled combined |
|---|---|---:|---:|---:|---:|---:|
| PTEN | abundance | 0.4596 | 0.3805 | 0.4214 | 0.3673 | 0.3641 |
| PTEN | activity | 0.4145 | 0.4147 | 0.4433 | 0.4386 | 0.4322 |
| CYP2C9 | abundance | 0.3955 | 0.3645 | 0.3773 | 0.3577 | 0.3598 |
| CYP2C9 | activity | 0.3464 | 0.3231 | 0.3386 | 0.3254 | 0.3160 |
| VKOR1 | abundance | 0.3690 | 0.2933 | 0.3392 | 0.3182 | 0.3174 |
| VKOR1 | activity | 0.1093 | 0.0778 | 0.0995 | 0.0422 | 0.0391 |

| Held-out protein | Site pLDDT | Variants per assay | Unconditioned rho | Combined rho |
|---|---|---:|---:|---:|
| PTEN | low_plddt | 35 | 0.1306 | 0.1977 |
| PTEN | high_plddt | 145 | 0.3838 | 0.3731 |
| CYP2C9 | low_plddt | 4 | 0.0000 | -0.1333 |
| CYP2C9 | high_plddt | 176 | 0.3703 | 0.3386 |
| VKOR1 | low_plddt | 13 | 0.0458 | 0.1346 |
| VKOR1 | high_plddt | 167 | 0.2344 | 0.1710 |

Slice rho is averaged over the two native assays and three seeds, without pooling raw targets. PTEN/VKOR1 low-pLDDT gains coexist with high-pLDDT regressions; CYP2C9 has only four low-pLDDT genotypes, so its slice is extremely noisy. Local predicted geometry quality is not a functional-effect error estimate. Frozen encoder quality weights learned on different proteins may not transfer; global source and full-coverage constants cannot teach cross-source or missingness effects here.

quality_slices.csv contains per-assay low(<70), high(>=70) and missing site-pLDDT strata for every fold/seed/model, including empty or too-small slices as null. All actual source strata are AlphaFold and every genotype is a single mutant; experimental-vs-predicted source contrast and mutation-burden performance differences are unidentifiable here. Do not average unlike-assay raw measurements or silently drop small slices. generalization_gaps.csv retains train/validation/test scores and selected epochs.

Missing structure forces mixing weight exactly0 and recomputes sequence-query features with structural attention masked. Mixture uses its trained sequence expert; concatenation uses its same trained head on missing geometry. The stress is inference-only: no absent-structure examples were used in training, so these scores do not establish calibrated fallback quality.

## Scientific semantics and architecture

AlphaFold pLDDT describes predicted local reference geometry, not mutation-effect prediction uncertainty. Experimental B-factors have units A², while experimental resolution has units A; neither is converted into pLDDT. Values/observed fractions and source/mapping provenance remain distinct. No M8 stability extrapolation or thermodynamic scalar is consumed. Reference geometry does not establish mutant rearrangements.

The existing frozen GNN already contains B-factor/pLDDT input columns. The neutral profile masks those values and their availability consistently; real input enables typed, physically normalized confidence BEFORE GNN message passing, using existing pretrained weights. The real-input encoder exactly reproduces M9 frozen features. The M10 head uses raw frozen LayerNorm representations and fixed physical descriptor scales for every comparison, not M9 within-protein StandardScaler. Historical M9 head scores therefore are not a matched architecture baseline.

Gate input is both cross-attended pooled branches plus18 typed confidence descriptors (site/neighborhood quality, source, mapping, coverage, missingness and optional resolution). Gate mixes sequence-query and structure-query expert predictions. Both branches already include bidirectional cross-attention: mixing weight is NOT total structural influence. Plain weighted late fusion learns one scalar on these same experts. Registered capacity is identical across modes, but active capacity differs between concatenation, scalar mix and gate; gate_zero and shuffled comparisons isolate the incremental confidence channel.

Shuffling moves whole confidence value/mask records within each reference protein for each fixed seed; geometry/sequence/labels stay unchanged and local summaries are recomputed. This preserves protein-level confidence marginals and can expose protein-proxy or nuisance effects; it does not constitute a broad randomization test. Physical structure-source type is explicit; no protein/assay identity or target statistics enter the reliability descriptors. Training-assay midranks and validation-only epoch selection remain unchanged.

gates.csv reports per-assay weight distributions and correlations with real pLDDT. Constant0.5 values in concatenation modes are inert diagnostics, not learned reliability; weighted_late has a learned scalar; gate_zero receives no explicit quality. No monotonic confidence-gate relationship was imposed. Weights must not be presented as predictive probabilities, error bars or mechanistic explanations.

| Held-out protein | Combined mean structural-query weight | Weight/pLDDT Spearman |
|---|---:|---:|
| PTEN | 0.4984 | 0.1282 |
| CYP2C9 | 0.5212 | 0.1594 |
| VKOR1 | 0.4731 | 0.2661 |

## Reproducibility, costs and limits

Head run: 18.25s on arm64 CPU, four threads; peak RSS 0.426 GiB. Cache-hit feature preparation: 5.91s; original ESM tokens were extracted with MPS in M9. Frozen backbone: ESM-2 t12 35M revision6fbf070e..., M7 function-only GNN/bidirectional encoder trained on old amylase/validated on NucB. M9 similarity/provenance audits remain valid; PLM pretraining exposure is unknown. Registered parameters per head: 5125. Environment: torch 2.14.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, biopython 1.88.

Protocol/config/tests were committed before head fitting; full source/config/checkpoint/feature fingerprints and role IDs are archived. Reproduction follows docs/m10_protocol.md: prepare frozen profiles, run reliability_experiments, verify_m10 and report_m10. Report-only reproduction needs only committed evidence: `MPLCONFIGDIR=/tmp/proteinmod-m10-matplotlib python scripts/report_m10.py --input benchmarks/m10 --output runs/m10-report`. Dense token arrays and weights remain ignored.

The bounded all-AlphaFold three-protein, single-mutant sample limits identifiability and statistical power. Full structures make missingness/mapping/resolution effects engineering-tested rather than scientifically validated. No cross-source, unseen-family, calibrated uncertainty or mutant-geometry claims. No M11, new epistasis, PLM fine-tuning, uncertainty calibration or multitask training.

All219 M0–M9 benchmark artifacts and old configurations remain unchanged. M10 is stacked on verified draft #18 → #16 → #14 → #12; all40 dependency checks passed, main unchanged.
