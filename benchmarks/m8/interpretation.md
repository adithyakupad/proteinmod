# M8 scientific interpretation and failure analysis

## What was established

A complete stability-conditioned function pathway runs using experimental M6 ridge stability sources and frozen M7 sequence–structure/bidirectional representations. Every function example receives **predicted** stability at training and inference. There are **zero paired experimental stability labels** for the function proteins. No stability measurements are transferred between proteins. Function models preserve all G1/G2 IDs, native-measurement per-protein Spearman and training-assay rank handling.

The result is **source-dependent and limited**, rather than a general endorsement of thermodynamic transfer. M6 ESM-site and structural sources use only permitted stability training proteins, with source alpha selected on stability validation RMSE. Function heads/alphas use only function validation Spearman. Both sources and all controls are retained. M6's original function-pretrained neural stability encoder is excluded because its amylase/NucB function supervision would contaminate some G2 folds.

## Final held-out results

Equal-protein G2 macro Spearman, averaged over three fixed seeds:

| Function model | Its matched baseline | ESM stability source | Structural stability source |
|---|---:|---:|---:|
| Feature ridge + scalar | .191013 | .196131 | .192179 |
| Frozen M7 + residual correction | .102592 | .122329 | .166501 |
| New frozen-representation neural head | .092811 | .099704 | .118238 |

These models have different baselines. Residual correction's gain must not be presented as beating the stronger feature ridge. Stability-alone ridge reaches .064873 with ESM source and .210730 with structural source; the latter does not validate its physically unusable stability calibration.

Paired seed differences against each model's own baseline:

- ESM feature-ridge addition: **+.005118**, deterministic predictions repeated across seed-indexed comparisons. Zero seed SD does not mean uncertainty is zero; only three proteins were evaluated. The improvement is mainly amylase (+.017838), while GB1 worsens (−.002563).
- Structural feature-ridge addition: **+.001166**, also deterministic. A structural scalar is a linear recombination of already available structural descriptors, so it can change regularization rather than add independent information.
- ESM residual: **+.019738 ± .021127** paired seed SD; amylase +.058406, GB1 +.000381, NucB +.000426.
- Structural residual: **+.063909 ± .054027**; amylase +.191414, GB1 +.001282, NucB −.000969. This is dominated by one function protein.
- ESM neural head: **+.006893 ± .015839** against the zero-scalar head, but **−.011746 ± .021252 against matched dummy noise** (real exceeds dummy in only1/3 seeds). Its gain is not specific evidence for useful stability information.
- Structural neural head: **+.025427 ± .015004** against the zero-scalar head, **+.006788 ± .003844 against dummy**, and **+.022696 ± .011801 against shuffled predictions** (real exceeds each control in3/3 seeds). This is limited evidence for a useful source-trained scalar projection under this fixed benchmark; the predictor's catastrophic physical extrapolation prevents interpreting it as reliable mutant ΔΔG.

Within-protein G1 diagnostics also give small/mixed effects: fixed M7 .592716; residual ESM .592200 / structural .590052. New zero-scalar neural head .582234; ESM .586162 / structural .583002. Feature ridge .473096; ESM .470238 / structural .473245. G1 and G2 have different test populations and are not interchangeable.

## Critical source failure: numerical noise and descriptor-domain shift

M6 source-held-out S2 macro RMSE remains **1.326421 kcal/mol (ESM)** and **1.350311 (structural)**. These are measurements on different short stability proteins, not error estimates for the function proteins. All function proteins are OOD for stability prediction.

On function examples, the structural source produces roughly **−286,636 to −1,372,637 kcal/mol** depending on source fold. These values are physically unusable and must not be served as validated thermodynamic estimates. They are retained without clipping or posthoc removal.

The posthoc, label-free decomposition in `source_failure_audit.json` identifies structural feature indices **86,174,262** (zero-based): mean/sum/max of the neighborhood missing-B-factor fraction. M6 AlphaFold structures have unavailable experimental B-factors, so these neighborhood fractions are nominally1. Float32 neighborhood matrix multiplication introduces tiny variation, approximately **5.7–6.8×10⁻⁸** SD. StandardScaler amplifies it; fitted ridge coefficients become large when applied to function experimental structures whose missing-B-factor fraction is0. Three corresponding terms dominate the enormous negative offset. At million-scale float32 outputs, quantization can also affect within-protein ranks.

This is a pre-existing numerical/descriptor-shift failure exposed by OOD application of the fixed M6 source. M8 does not silently change historical graph construction, scaler behavior, source coefficients or results to improve its test scores. The model and source hyperparameters remain prespecified. A future numerical hardening experiment should be independently declared and validated; the current experiment demonstrates that apparently useful function ranking does not establish thermodynamic validity.

ESM-source function predictions range roughly −1.20 to2.87 kcal/mol. Plausible magnitude alone does not establish correctness: no function-protein stability ground truth is available. Its S2 error varies substantially by source test protein (.515 to2.257 kcal/mol). Neither source has calibrated confidence or uncertainty on these function examples.

## Leakage and causal interpretation

All 12 fixed M7 function-only models reproduce **5,979 archived test predictions exactly**. The same representation, baseline and evaluation IDs are used for real, shuffled and dummy conditioning. M7 pathways and stability sources stay frozen; only compact new function heads or corrections train. Separate observed-outcome contracts, role IDs, source weights, preprocessing and feature/checkpoint hashes are archived. Tests perturb function and source-held-out stability labels independently without changing fitted predictions/selection.

Shuffling is within protein and split role, preserving marginal effects while destroying variant association. It is a cohort-level negative control, not a deployable single-example method. Dummy noise matches training scalar mean/SD and availability, but not every protein distribution. Its neural head has exactly the same1665 parameters as the real-conditioned and zero-scalar controls. Controls establish neither a mechanism nor a causal benefit of knowing true stability.

The classical ESM-scalar addition also introduces an ESM projection beyond the composition/structural baseline, so an improvement does not isolate physical stability from generic sequence information. The nonlinear M7 representation already includes frozen ESM, structural GNN and cross-attention, but an added scalar can still recover information lost by compression. Residuals use in-sample training-base predictions, not cross-fitted residuals; this can bias correction training. All correction selection remains validation-only, with failures retained.

## Training-sample and source limitations

Each G2 function fit trains on one protein with only553–600 examples. Each S2 stability source trains on600 single substitutions of one55–63-residue construct. Three auxiliary fold pairs preserve M7 roles but do not exhaust all nine possible source/function pairings. Protein and assay shifts remain confounded. About **94.0% (1,627/1,731)** of matched function variants contain multiple substitutions, beyond M6's single-mutant training support. The cache records this explicitly; no additivity, epistasis or reliable multi-mutant thermodynamics claim is made.

Reference geometry is not measured mutant rearrangement. Exact sequences/role separation and close-homolog screening are enforced, but remote-family separation and PLM pretraining exclusion are not established. M6 compact-table limitations remain: experimental proteolysis-inferred folding ΔΔG, source sign inverted to kcal/mol positive destabilization, unavailable raw replicate/censoring information, and60/600 1TUD targets at its upper observed bound. The cohorts were not changed after seeing errors.

## Reproducibility and numerical fidelity

The protocol/config/model grid was committed at **b7e4661** before the initial run. A fidelity audit then detected accidental float64 reconstruction of M6's float32 feature arrays. **ad26d68** restores the exact M6 precision and WT-cache contract, with a dtype regression test; no cohort, seed, model family, loss, grid or selection rule changed. Initial pilots remain locally under runs/m8_float64_pilot*. All reported primary results are from the corrected full run.

After correction, both M6 feature matrices match legacy arrays exactly, and all **4,140 M6 source test predictions match exactly**, as do the **5,979 fixed M7 function predictions**. A complete second run reproduces **161,433 function predictions**, **13,848 stability inference rows**, and all **324 function model/seed/fold metrics and selection traces exactly**. Only local CPU/fixed-cache reproducibility is established.

Final run: **312 newly fitted models** (8 stability sources +304 function fits), **35.01 s CPU**, four threads, peak process RSS **1,473.7 MiB** on arm64. M7 supervised weights and frozen ESM/mutant caches were reused; the run excludes historical training and token extraction costs. The numerical-fidelity change invalidated the first-run representation cache, so the corrected run includes its reconstruction. These are descriptive local times, not isolated hardware benchmarks. Expensive arrays/weights remain ignored; predictions, reports and audit metadata are committed. All **176 M0–M7 evidence files** and previous configurations remain unchanged.

No M9 assay-aware conditioning, PLM fine-tuning/LoRA, calibrated uncertainty, biochemical graph extension or explicit epistasis learning was implemented.
