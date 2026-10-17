# M11 prespecified explicit interaction and epistasis protocol

## Decision and dependency discipline

Issue #21 asks whether explicit symmetric mutation interactions improve held-out fitness and measured factorial epistasis beyond single-effect additivity and implicit sequence/structure representations. Negative results are valid. No M12 family benchmark, formal uncertainty calibration, new structural encoder or PLM fine-tuning.

Stack on verified draft PR #20/head624fb3e, then #18/9610fe4 → #16/16a53a1 → #14/dd9e8bc → #12/1753cbb. All50 exact-head push/PR checks passed; drafts remain reviewable, main/ad2e564 unchanged. Preserve all238 M0–M10 benchmark files and15 old configurations. The only compatibility edit bounds M10's historical enumerator to M0–M9, as M9 already does, so future artifacts do not invalidate old replays. Check runner/verifier/test callers and a regression test. No old data/config/results change.

Change reaches a measured-landscape adapter, symmetric pair features, frozen feature cache, additive/ridge/neural fitters, tests/CI, config, replay/report and README. Reuse existing substitution parser, M2 graph, frozen ESM and M7 function encoder. No new dependency or broad dataset download.

## Source and same-experiment target contract

Use official FLIP original GB1 archive at commit62cace8735f5610e2743cf06ce0f944b37fffaa6 (2.66 MB), not its fitness-biased8,733-row sampled subset. Pin archive and46.4 MB extracted CSV SHA256s in configs/m11.json. Table contains149,361 measured four-site genotypes: WT1, singles76, doubles2,091, triples26,019, quadruples121,174. No imputed supplement2 values; input count>=10 required. Source Wu et al.2016, DOI10.7554/eLife.16965, supplement1 DOI10.7554/eLife.16965.024; raw CC-BY-4.0, FLIP table derivative AFL-3.0. Full inferred FLIP reference is provenance-checked, but only the56-residue GB1 core is modeled; inferred fusion tail is not claimed as a measured assay construct. Four sites39,40,41,54 have WT VDGV.

**Do not silently use published Fitness singles as same-batch measurements.** The paper imported Olson2014 single-mutant fitness and rescaled it1.159. Primary M11 instead derives *every* label from the same table's Wu input/selected read counts:

w(g) = [selected(g)/input(g)] / [selected(WT)/input(WT)].

WT is1 by definition (input92,735; selected338,346). Non-single published Fitness must reproduce this formula within rounding; singles retain their own raw counts. Archive both published and count-derived values and quantify their difference. All76 singles exist, but counts are much noisier (minimum input14; some selected0). This avoids cross-experiment single harmonization at the cost of greater noise. No fabricated observation or pseudo-single is inserted. Read counts are not biological replicates; no variance/causal claim from them.

Assay is mRNA-display affinity selection for IgG-Fc binding, not pure stability. Selection temperature/pH unavailable in this curated snapshot; do not substitute library-preparation conditions. Original publication: https://pmc.ncbi.nlm.nih.gov/articles/PMC4985287/ ; official source README: https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/README.md .

Before fitting define y=ln(max(w,0.01)), consistent with the reported approximate detection limit. This is a censored/floored log-fitness benchmark, **not exact uncensored log fitness** below0.01. Keep zero/low variants in fitness evaluation; never clip predictions to the floor (clipping would make an additive predictor artificially non-additive). No percentile transform or thermodynamic units.

## Cohort, splits and leakage

Select all measured WT/singles/doubles plus360 triples and120 quadruples by smallest SHA256(salt,genotype), without using outcomes. Total2,648 fitness observations. The data are connected around WT and include all measured two-site sublandscapes, not a complete higher-order factorial. Missing75 theoretical doubles remain missing.

C1: WT/all76 singles are permitted training anchors. Independently by mutation count, hash multi-mutants into70/15/15 train/validation/test:1,877/385/386 rows. C1 holds out complete genotypes, but the same mutation pair can appear in other training backgrounds; audit that explicitly.

C2: declare test position pair39–54 and validation pair40–54 before fitting. Any genotype containing the test pair goes to test; among the remainder, any with the validation pair goes to validation; all others train. Roles1,569/425/654. Test pair is absent from every training **and validation** genotype, including triples/quads. All individual alleles remain known from training singles. All quadruples end up in test and no quadruple trains in C2; report burden-specific performance to distinguish pairing shift from mutation-depth shift. These are new M11 protocols on a new source/cohort; retain all old M0–M10 partitions unchanged. No whole-protein or unseen-position/family claim.

Validate source mutations, full sequences, count filters, duplicates and reference correspondence; complete sequences cannot cross roles. Same rows in all models. Additional64 lower-order *probe-only* rows are needed for eligible evaluation contrasts; never train on these labels or tune with them. Their features use sequences/reference only. Feature fitting/scaling/label transforms use training only. Test-label perturbation must preserve predictions, hyperparameters and training traces.

## Direct measured epistasis

For every held-out target AB and each unordered changed-site pair, BG is the target with those two changes reverted; A and B add each change separately to BG. Require all four genotypes actually measured and all four w>=0.01. No invented lower-order values; censored or absent terms produce an exclusion, not an exact epistasis estimate.

Measured epsilon_log = y(AB)-y(A)-y(B)+y(BG).
Predicted epsilon_log = fhat(AB)-fhat(A)-fhat(B)+fhat(BG).

**Use model predictions for all four predicted terms**, avoiding spurious signal from subtracting the same noisy observed singles from both truth and prediction. Evaluate WT-background doubles separately from background-specific pair contrasts in3+/4-mutants. Across the declared cohort there are1,125 eligible WT contrasts and374 eligible non-WT contrasts; actual test counts reported per protocol. Additional missing/censored contrasts are recorded. No true higher-order epistasis is claimed from pair contrasts or fitness improvement.

Also compute the same four-point contrast on untransformed w to show nonlinear-scale sensitivity, with sign changes/examples and a plot. Do not treat raw- and log-scale magnitudes as commensurate. Pure additive/one-hot models imply zero model epistasis; numerical contrast roundoff<1e-10 is zeroed before correlation. Undefined constant-prediction correlations stay null, never treated as successful correlation.

## Features, additive baseline and models

Training-only measured-single baseline: yWT + sum[y(single)-yWT]. WT and singles are unchanged exact anchors. General missing-single policy: zero delta plus an explicit missing count, never a fabricated measurement. The selected collection has no missing singles. Keep constant and mutation-count regressors.

Classical baselines: position-aware one-hot ridge80, frozen ESM mutation-site probe961, M3 structural ridge270, and frozen M7 GNN/bidirectional multimodal probe100. Reuse ESM t12 35M revision6fbf070e... and old M7 function-only encoder trained amylase/validated NucB (GB1 was held out); never load a GB1-supervised historical encoder. PLM pretraining exposure unknown. Frozen tokens include all probes but no outcomes. No M8 stability scalar.

Use existing validated1PGA experimental reference graph (55 compatible residues). Q2T is the already documented mismatch at position2 and excluded from graph nodes; all four mutation sites match exactly. Geometry is reference geometry, not mutant rearrangement. Experimental B-factors retain A² semantics; no pLDDT conversion. Path availability and coordinate masks are explicit for missing/disconnected sites.

Per unordered pair, form symmetric source/destination identity and property features: sums/absolute differences/products of two44-dimensional mutation vectors, unordered destination-AA outer-product210, constant1, absolute sequence separation and its fixed exponential basis. Sequence width345. Structural fields15 include CA distance, contact, shared-neighbor Jaccard, component/path and observation flags, typed confidence/source; cross with four biochemical delta products gives total420. Missing/unmapped sites yield zero-filled unobserved values/masks, never fabricated measured distances. Sum over all pairs; no pairs for WT/singles.

Ridge residuals: measured additive prediction + linear sum of sequence-only345 or structural420 pair features; no intercept/centering in residual term, so WT/singles stay unchanged. Classical ridges alpha[.1,10,100], validation RMSE only. All standardization train-only, SD<1e-6 uses scale1, float64 ridge arithmetic.

PairResidual MLP is shared across pairs and sums masked scalar contributions, then adds measured additive prediction. No background/global target features enter pair contributions; this is an explicit pairwise model for arbitrary mutation sets, not a true higher-order interaction expansion. Hidden16, CPU4, AdamW.003/decay.01, batch128, clipping5, max60epochs/patience10; residual training SD scales loss, chosen solely from training labels. Earliest minimum validation RMSE selects epoch. Seeds0,1,2.

Neural comparisons:
- sequence_pairs:420 input with explicit3D slots zero;
- structural_pairs:420 real features (same capacity);
- hybrid_pairs:720 features adding symmetric300 reference-single-mutant multimodal coordinates;
- hybrid_zero3d: same720, explicit3D channels zero;
- hybrid_shuffled_distance: same720, 3D descriptor records permuted among the six reference position pairs;
- hybrid_shuffled_biochemical: same720, destination identities/properties permuted within each position's19 substitutions, only in explicit biochemical features.

Controls are label-blind fixed reference permutations per seed, shared across roles, preserving reference pair/allele marginals and symmetry. Frozen multimodal contexts remain unchanged, including implicit structure/biochemical information; these controls isolate **explicit** feature increments, not destruction of all biological information. Hybrid_zero3d is not a sequence-only model because its context still contains GNN/attention. Registered hybrid/control capacities match exactly. Sequence-vs-structural plain neural widths also match.

## Reporting and reproducibility

Fitness Spearman/RMSE on declared floored-log scale; WT/non-WT epistasis Spearman/MAE, count/censor/rare-pair slices, strong non-additive examples, train/validation traces and failures. All models use matched IDs. Fixed paired block bootstrap1,000 draws/seed2026 for seed-mean prediction comparisons: mutated-position-set blocks for fitness, position-pair blocks for WT epistasis; fewer than3 blocks gives no interval (C2 WT has only one pair). These are conditional pilot intervals, not biological measurement uncertainty, with few dependent blocks/shared anchor measurements. No uncertainty-calibration milestone.

Commit this protocol/config/code/tests before fitting. Full repeat changing only output must reproduce predictions, contrasts, metrics and selections (excluding runtime/checkpoint serialization hashes). Replay every saved model, reconstruct all model-implied contrasts and preserve all historical hashes. Record exact source/model/structure/config/code identities, seeds/devices, runtime/RSS and caching costs. Dense arrays/weights/source archive stay ignored.

```
python -m proteinmod.epistasis_experiments download --config configs/m11.json
python -m proteinmod.epistasis_experiments prepare --config configs/m11.json
python -m proteinmod.epistasis_experiments run --config configs/m11.json
python scripts/verify_m11.py --input runs/m11
MPLCONFIGDIR=/tmp/proteinmod-m11-matplotlib python scripts/report_m11.py --input runs/m11 --output runs/m11
ruff check .
ruff format --check .
pytest -q
```

Use Python3.12 requirements-m3.txt and existing M4/M7 graph/encoder preparations. ESM tokens extract locally with MPS, deterministic heads/ridges run CPU. Report-only reproduction uses --input benchmarks/m11 --output runs/m11-report without weights/raw data. No M12 work.

Archived M10 replay now reads its recorded feature-cache root explicitly, rather than selecting a new code-hash cache after the history-only compatibility edit. Cache checksum and checkpoint-feature identity still fail closed. The loader's optional root preserves its existing default API; unit corruption checks and actual archived-head replay cover the change. No feature values or old checkpoints are replaced.
