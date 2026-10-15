# M9: assay-aware mutation-effect prediction

## Verified comparison

Real local experiments on six ProteinGym v1.3 assays: PTEN, CYP2C9 and VKOR1, each with abundance/activity measurements of the same180 genotypes. 1,080 observations; G1 has756/162/162 train/validation/test rows, each G2 fold360/360/360. All92 model/seed/fold cases use matched test IDs (28,566 predictions). Native outcomes remain assay specific. Table values are assay-local Spearman, then endpoint/protein macro; neural means average three seeds, ridge is deterministic.

| Model | G1 rho | G2 rho |
|---|---:|---:|
| esm_sites_ridge | 0.3545 | 0.2810 |
| structural_ridge | 0.3352 | 0.2650 |
| multimodal_ridge | 0.2585 | 0.0000 |
| family_heads | 0.3489 | 0.0000 |
| assay_id_heads | 0.2848 | 0.0000 |
| zero | 0.3050 | 0.2278 |
| structured | 0.2772 | 0.2008 |
| structured_missing_at_test | 0.2767 | 0.2005 |
| with_assay_id | 0.2612 | 0.1771 |
| shuffled_train | 0.2775 | 0.2810 |
| protein_proxy | 0.2919 | 0.2329 |

Primary structured-minus-zero change: G1 -0.0278 (seed SD 0.0152; range -0.0419 to -0.0067); G2 -0.0270 (seed SD 0.0490; range -0.0948 to +0.0196). Shuffled-training-minus-zero G2: +0.0531 (seed SD 0.0664; range -0.0130 to +0.1440). These are seed variation, not population confidence intervals.

## Held-out protein/endpoint failures

| Held-out protein | Endpoint | Zero | Structured | Shuffled |
|---|---|---:|---:|---:|
| PTEN | abundance | 0.3551 | 0.3635 | 0.3365 |
| PTEN | activity | 0.3135 | 0.3241 | 0.3347 |
| CYP2C9 | abundance | 0.1079 | 0.0000 | 0.2540 |
| CYP2C9 | activity | 0.0827 | 0.0000 | 0.2516 |
| VKOR1 | abundance | 0.2995 | 0.3107 | 0.3073 |
| VKOR1 | activity | 0.2084 | 0.2066 | 0.2017 |

Conditioning is not uniformly beneficial. Interpret changes alongside the shuffled control and the stronger classical baselines; an improvement over the same neural head alone does not establish a general benefit over existing representations. Individual assay IDs remain protein-specific and map to UNK in G2. The protein-proxy control is diagnostic, not a deployable input. Separate assay-ID ridge heads fall back to the global head for all unseen test assays.

The bounded multimodal ridge collapses to a constant in 3/3 G2 folds; its undefined per-assay correlations are null in runs.json and contribute zero to the macro. The conditioned neural head also collapses on CYP2C9 in 3/3 seeds. Do not interpret these zero macros as measured zero correlation. The clipping/sigmoid contract was fixed before fitting; it was not relaxed after this failure.

Frozen multimodal feature shift is substantial: PTEN maximum absolute standardized coordinate 95.8, 18.6% of test coordinates above10 SD; CYP2C9 maximum absolute standardized coordinate 134.6, 28.0% of test coordinates above10 SD; VKOR1 maximum absolute standardized coordinate 296.9, 31.2% of test coordinates above10 SD. Only one protein supplies training labels in each G2 fold. Reference-global encoder features vary little within that protein and can shift strongly across proteins, consistent with saturation. The 1e-6 scaling guard prevents roundoff amplification but does not solve genuine distribution shift. feature_shift.json, saturation.csv and generalization_gaps.csv retain the evidence; no posthoc scaler retuning was performed.

Shuffling swaps complete endpoint records within each training protein (two families, so every nonidentity permutation is a swap). Correct validation/test metadata stay unchanged. This is an adversarial mismatch control, not a null distribution from many independent random annotations. Structured metadata includes endpoint-specific technologies/substrates that can be unseen at test; explicit UNK handling limits transfer.

## Data, mapping and provenance

| Protein | Common genotypes | Selected | Original rows by assay |
|---|---:|---:|---|
| PTEN | 4839 | 180 | PTEN_HUMAN_Matreyek_2021: 5083; PTEN_HUMAN_Mighell_2018: 7260 |
| CYP2C9 | 4421 | 180 | CP2C9_HUMAN_Amorosi_2021_abundance: 6370; CP2C9_HUMAN_Amorosi_2021_activity: 6142 |
| VKOR1 | 692 | 180 | VKOR1_HUMAN_Chiasson_2020_abundance: 2695; VKOR1_HUMAN_Chiasson_2020_activity: 697 |

Sampling is label-blind sequence SHA256 on the intersection of experimental genotypes. Unpaired counts and cap exclusions remain in audit.json/source counts; a small intersected cohort can select for variants measurable by both assays. All selected mutations are single substitutions. Experimental score is processed DMS_score with higher-is-better direction, not kcal/mol; raw phenotype names/directions and source hashes are in the configuration. The ProteinGym MIT curated archive and original paper citations are retained. Structures are AlphaFold DB v6, CC-BY-4.0.

Every selected genotype maps to a full exact reference/structure alignment: zero mapping exclusions. Geometry is reference-predicted geometry, not observed mutant rearrangement. Fractions of all reference residues with pLDDT<50: PTEN 0.171; CYP2C9 0.000; VKOR1 0.031. Low confidence was not excluded after examining outcomes. Connectivity and mappings are in audit.json.

Temperature and pH are unavailable in all six curated metadata records (0/6 observed each); unavailable is distinct from not_applicable and observed zero. No fabricated physiological defaults. Numerical condition effects cannot be estimated here. Activity is a heterogeneous proxy: PTEN yeast growth, CYP probe activity, VKOR carboxylation reporter. This collection supports known-endpoint-family protein holdout, not novel-assay or unseen-family generalization. Only three proteins limit certainty.

## Leakage, representation and compute

Both endpoint observations of a genotype stay together in G1. G2 proteins are entirely disjoint. Vocabularies, standardization and target midranks fit training only; alpha/epoch selection uses validation only. Primary metadata excludes protein/assay IDs and all outcomes. Per-assay native correlations are aggregated only after evaluation. Test-label perturbation tests reproduce both model fits and selections unchanged.

Frozen ESM-2 t12 35M revision6fbf070e65b0b7291e7bbcd451118c216cff79d8 and M7 function-only GNN/bidirectional encoder are reused; only the new heads train. Fixed encoder supervision was old amylase training/NucB validation; similarity audits screen the new proteins against both. PLM pretraining exposure is unknown. No M6 stability prediction head or M8 OOD stability scalar is read; no thermodynamic inference. The pretrained frozen encoder may underfit new human proteins; end-to-end adaptation is outside this bounded first test.

All metadata/zero/proxy heads have identical registered capacity within a fold: G1/within 2561 parameters; G2/PTEN 2385 parameters; G2/CYP2C9 2401 parameters; G2/VKOR1 2401 parameters. Category vocabularies change with training proteins, so capacity can differ across folds. Unused control parameters remain registered but receive no task gradients. ESM/structural/multimodal ridge coefficient counts including intercept are962/271/101; group-head counts are retained per run.

Head experiment: 12.03s, CPU arm64, four threads; peak process RSS 0.399 GiB. Feature preparation 2.54s with graph/token cache hits (not a cold-start timing); original token extraction used the recorded encoder device. Environment: torch 2.14.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, biopython 1.88. Exact configuration/seeds/source hashes/checkpoints are in manifest.json. Dense embeddings/weights are ignored, not committed.

## Reproduction and historical preservation

Follow docs/m9_protocol.md and configs/m9.json: acquire pinned selected members/structures, prepare cached features, run assay_experiments, verify_m9, report_m9. For report-only reproduction: `MPLCONFIGDIR=/tmp/proteinmod-m9-matplotlib python scripts/report_m9.py --input benchmarks/m9 --output runs/m9-report`. All199 M0–M8 benchmark hashes remain unchanged. Repeat/replay evidence is archived separately. Training traces report overfitting; no test-led tuning or cohort revision was performed after the prespecified commit.

Stacked on draft PR #16, then #14 and #12; all30 dependency CI checks passed, but drafts remain unmerged. Main is unchanged. No M10 work, new multitask training, PLM fine-tuning or uncertainty calibration.
