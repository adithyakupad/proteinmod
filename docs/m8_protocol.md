# M8 prespecified stability-informed function protocol

## Goal and decision

Issue #15 tests whether a predicted, experimentally grounded stability scalar adds useful function information beyond existing sequence/structure representations. Primary outcome is per-protein native-measurement Spearman, with equal-protein macro and paired three-seed changes. Negative results are acceptable. Do not select a source or architecture on function test performance. No M9 assay conditioning, PLM tuning, uncertainty calibration or epistasis model.

Stacked on draft PR #14 (`dd9e8bc`), which is stacked on draft PR #12 (`1753cbb`). All 20 dependency push/PR CI checks were verified successful. Their experimental provenance, scientific limits and historical hashes are retained. Work stays isolated while those drafts remain reviewable; main is unchanged. All M0–M7 benchmark/configuration artifacts are immutable.

## Data and source roles

Reuse M7's exact 1,731 structure-matched M4/M5 function variants and 1,800 M6 stability variants. G1 retains original variant partitions. G2 retains whole-protein train/validation/test roles. Prespecified source-role pairs are those of M7: within/within, amylase/1LP1, gb1/1A32, nucb/1TUD. Both sources see the same permitted M6 stability training and validation roles. Every function example, including training examples, receives a predicted scalar; no actual function-protein stability measurements exist in this collection. Never attach an experimental label from another protein.

Stability sources: M6 ESM mutation-site ridge (961 features) and structural ridge (272 features), fixed alpha grid [.1,10,100], train-only StandardScaler and native folding ΔΔG targets, choose alpha by stability validation macro RMSE. Return kcal/mol, positive destabilization, with no sign-flipping during conditioning. Source-held-out stability predictions and errors are archived separately from function results. No source selection using function validation/test scores: report both sources throughout.

M6's original frozen neural encoder had amylase function training and NucB function validation. It is excluded as a universal source because that would contaminate G2. The two ridge sources use frozen ESM or label-free reference geometry and experimental stability supervision only. M6 features are independently reconstructed from the same tokens/WT tokens/graphs, not author PCA or outcome-derived structural columns. Preserve mutation counts and reference mapping. WT mutation coverage is zero, not fabricated one. Full source/calibration/feature/cache/checkpoint fingerprints and role IDs are recorded.

All function proteins are **OOD** relative to the three short stability training constructs, regardless of G1/G2. They are not in-domain validated stability examples. M6 source tests are reported as within-protein (S1) or held-out-protein (S2) assessments, distinct from function-cohort OOD use. All confidence/uncertainty is unavailable. Single-mutant training does not validate multi-mutant or WT extrapolation. Record both flags. Geometry is reference geometry, not measured mutant rearrangement. Existing sequence/homology audits reject cross-task close homologs (>=30% aligned identity and >=80% coverage in both directions); this is not verified unseen-family separation. PLM pretraining exposure is unknown.

## Models and controls

Fixed seeds 0,1,2. Reproduce all 12 archived M7 function-only checkpoint predictions exactly on G1/G2 test IDs before conditioning. Reuse frozen M7 function-only ESM/GNN/bidirectional architecture to isolate the incremental scalar; no new joint pretraining or encoder adaptation. If original ignored weights are absent, rerun the unchanged M7 function-only fitter and verify baseline identity before use.

1. Fixed M7 function-only model.
2. Classical ridge on 67 sequence descriptors + 270 structural descriptors (337), no scalar.
3. Predicted stability alone with compact ridge function head.
4. Same classical features + stability scalar and availability bit, ridge head.
5. Fixed M7 prediction + additive ridge correction from stability scalar/availability; base function model stays frozen. Training correction targets are training-assay rank residuals on the training examples, not cross-fitted residuals. This is a small supervised second stage, potentially biased by in-sample base overfit; validation controls selection.
6. New function MLP on frozen M7 100-dimensional multimodal representation + scalar + availability bit. Compare with the exact same 102→16→1 head receiving zero scalar and the same availability bit. Only the new head trains. Same initialization and training settings across conditioning/control variants; no PLM, GNN or attention update.

For models3–6, run real predictions, shuffled predictions and matched-scale dummy noise. Shuffling is seeded and **within protein and role**; no train/validation/test value crossing, no labels read. This is a cohort-level negative control, not an online inference method. It preserves protein/role scalar marginals while destroying mutation association. Dummy noise is deterministically ID-seeded and rescaled using only training noise/scalar statistics to exactly match training mean/SD. It does not preserve each protein's marginal. A constant scalar produces constant dummy values. Availability patterns stay explicit. These controls probe extra capacity/nuisance scalar effects, not a biological mechanism.

Scalar standardization uses function training predictions only. Missing scalar values require a false availability bit and a zero standardized value, never an observed zero kcal/mol. No observed scalar values may be nonfinite. Real selected cohorts have all required features; missing-scalar behavior is engineering-tested, not performance-calibrated.

Function targets remain M4 training-assay midrank percentiles; use train-only mean/SD for neural loss and invert to rank scores. Native assay measurements are used for per-protein Spearman; do not compute pooled native regression errors across unlike assays. Ridge alpha [.1,10,100] and neural epoch (up to40, patience8) chosen only by function validation macro Spearman; earliest optimum, smallest alpha tie. AdamW .003, decay.01, clipping5, batch32, CPU4, hidden16. All model/source/control candidates retained. No posthoc clipping, source replacement, sample dropping or test-led retuning.

## Evidence and limitations

Archive complete function test predictions, OOD stability features for every function role, per-protein and equal-protein macro results, paired seed changes, source dependence, shuffle/dummy controls, baseline identity, training/validation traces and test errors. Include M6 source prediction error and extrapolation feature-range audits. Numerical extreme OOD predictions remain visible; source cross-protein errors and source sample size limit any thermodynamic interpretation. M6 compact-table replicate/censoring limits and 1TUD upper-bound pile-up remain unchanged.

No paired observations or established mechanistic relationship are inferred. This pilot asks whether a noisy model-derived feature improves held-out functional ranking. Three proteins per endpoint, fixed auxiliary fold pairing and assay shift confounding cannot establish general thermodynamic transfer. No inferential confidence intervals across proteins from only three seeds.

## Reproduce

Use the pinned Python3.12 environment and ordered M4/M5/M6 preparation in docs/m7_protocol.md. M7 baseline weights can be recovered automatically if absent. No M6 neural checkpoint is needed. WT ESM tokens are locally extracted once and checksum-cached; dense tokens, representations and weights remain ignored.

```
python -m proteinmod.conditioning_experiments --config configs/m8.json
MPLCONFIGDIR=/tmp/proteinmod-m8-matplotlib python scripts/report_m8.py --input runs/m8 --output runs/m8
ruff check .
ruff format --check .
pytest -q
```

The experiment protocol/config/code is committed before inspecting M8 test results. Historical checksums are checked before/after. Source-model caches serialize coefficients, intercepts, preprocessing, unit/sign convention, role IDs and input/weight fingerprints; neural checkpoints record function calibration, frozen function checkpoint identity and stability-source identity. Cache hashes fail closed on corruption. Run manifests record source/config revisions, dependency versions, seed, device, runtime and peak process RSS.

### Numerical fidelity correction before the final benchmark

The first local pilot reconstructed the source feature arrays as float64. A subsequent M6 prediction replay detected micro-kcal numerical differences and exposed the importance of preserving M6 precision under severe extrapolation. Both source arrays now explicitly retain M6's float32 contract, and stability WT tokens are read from the original M6 cache. A regression test asserts that dtype. Source checkpoints record input/coefficient dtypes as well as weights/calibration. The original pilot and its repeat remain locally under runs/m8_float64_pilot*; final benchmark and repeat are rerun after this correction. No cohort, model family, alpha grid, seed, control, selection criterion or training budget changed. This is a numerical fidelity correction, not test-led model optimization; raw OOD extrapolation failures remain visible.

After a run, `python scripts/verify_m8.py` reloads all 84 saved neural heads, restores preprocessing/calibration and compares all 41,853 neural test predictions. `python scripts/audit_m8.py` reconstructs the posthoc descriptor-shift decomposition and compares all 4,140 native M6 source predictions. Its additional legacy feature-array comparison requires the original M6 transfer checkpoint (rebuild it with the M6 protocol if absent); M8 training itself never loads that function-pretrained checkpoint. A full repeat can use a copy of configs/m8.json changing only output to runs/m8_repeat; both prediction CSVs and all runs.json metrics/selections should match exactly. The archived reproducibility.json records the actual complete repeat.

For report-only reproduction without checkpoints or data preparation:

```
MPLCONFIGDIR=/tmp/proteinmod-m8-matplotlib python scripts/report_m8.py --input benchmarks/m8 --output runs/m8-report
```
