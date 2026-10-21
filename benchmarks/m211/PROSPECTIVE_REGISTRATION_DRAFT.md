# M21.1 prospective statistical registration — DRAFT / NON-EXECUTABLE

Date: 2026-10-21. Planning reference: [PLANNING_REFERENCE.md](PLANNING_REFERENCE.md). Qualification: [REPORT.md](REPORT.md). **NO GO. This is a design document, not an issued protocol, outcome-access authorization, final cohort or executable manifest.** Missing exact endpoint, construct, archive, exposure and model-state qualifications cannot be replaced by this draft. No measurement, genotype list, prediction or model execution is produced.

## Question, scope and frozen comparisons

Question: on independently measured benign human proteins absent from supervised development, does the fixed unscaled-versus-scaled ESM ridge rank contrast transport? Candidates are only GCK, ASPA, PRKN and SOD1, abundance endpoints only. There are currently fewer than four admitted families. No replacement search, pathogen/resistance protein, clinical-classification endpoint or optimization is part of this study.

Primary arms are original M17 `esm_unscaled_ridge` and `esm_local_ridge` (train-scaled), full training fraction, G3 roles. The encoder is the historical frozen ESM2 t12/35M, revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`; mutation-site feature construction, historical validation-selected alpha, scaler and development-label transform remain fixed. Historical alpha candidates were [0.1,10,100]; there is no new selection. M19 expanded search is excluded. No external fitting, calibration, tuning or architecture change is planned.

Use the four original G3 development folds as separate fixed predictors. For this draft, deterministic ridge uses historical seed 0, not five pseudo-replicates. Compute assay-local correlation for each fold/arm, then equal-fold means within a family. Do not choose the best externally performing fold or seed. This aggregation is a new declared transportability estimand, not exact replication of M17's original test cohort. Historical checkpoint existence, numerical replay identity and exact state hashes must be verified in a separately authorized software-only stage before any final registration. None was loaded or run here, and no checkpoint download is authorized. If any required original state cannot be verified, STOP; no substitute refit.

Secondary arms, only if exact original states and role/budget comparability pass: structural ridge, structural nonlinear probe, original/rich GNN, zero-message, degree-preserving graph null, chemistry shuffle, WT-only and archived shuffled-training controls. These are descriptive transfer comparisons. No new control training or graph generation is authorized here. Effective capacity differences are reported; controls do not establish a causal physical mechanism.

## Admission and cohort rules

All four families must pass exact-item rights, phenotype-production/export lineage, construct and structure suitability, evolutionary independence under inherited thresholds and exposure adjudication before a final protocol can exist. Preserve full-reference failures and old masks as records. Adopting any mature-reference mask requires explicit prospective justification; no outcome-based geometry rescue or cutoffs relaxed from history.

One endpoint and one source-defined construct per family; no pooling activity, toxicity, clinical categories or model-derived stability with abundance. Higher-is-favorable direction must be established from assay definitions before outcomes, not inferred from numerical patterns. GCK's exact definition is still incomplete here. Variant eligibility is a rule for future existing published observations, not an instruction to generate new variants: canonical single amino-acid substitutions with consistent WT/numbering, excluding WT, synonymous, stops, indels and compound alleles. No genotype identities or count-derived eligibility decisions are made now.

A later approved custodian would freeze the existing eligible IDs and a source-defined deduplication rule without reading labels. All eligible published observations are preferred over a favorable subset. This draft does not invent numeric sample adequacy cutoffs: minimum evaluable positions/substitutions must be justified and registered using genotype-only support before outcome access. That unresolved requirement makes this document non-executable.

Primary paired comparisons use the common structure-mapped cohort with identical IDs for every arm. A separately registered sequence-all cohort is secondary; comparing sequence-all against a smaller structural cohort is invalid. Missing/nonfinite outcome labels would be removed identically for all arms, counted and never replaced. Do not use observed magnitude, SE, clinical classification, score-based category or outcome-selection flag for eligibility. Upstream support filtering or censoring must be disclosed even when impossible to undo.

Any undefined family-level correlation makes the planned four-family primary contrast non-estimable; do not silently discard that family or convert undefined to zero. The existing M17 convention and any different historical zero handling remain immutable. Low variation is not a reason to replace a candidate after outcomes.

## Statistical estimand and decisions

For admitted family f, arm a and historical fold k, compute rho(f,a,k): Spearman correlation with average ranks for ties, within the single assay/construct and the common cohort. Let delta(f) be the mean over the four folds of rho(f,unscaled,k) minus rho(f,scaled,k). Primary delta is the equal-weight mean across four families. Model folds, substitutions and seeds are not independent families. No raw-unit RMSE across assays, externally fitted rescaling or invented thermodynamic units.

Preserve historical descriptive support criteria: macro delta >= +0.02 and positive delta in at least 3 of 4 families. For this NEW estimand only, macro delta <= -0.02 with negative directions in at least 3/4 is descriptive contradiction of its specific predicted direction. Mixed, intermediate, underqualified or non-estimable results are inconclusive. These labels do not adjudicate every original H5 subclaim and do not retroactively reclassify M17/M19. No result is assigned now.

There is one primary contrast. All structural/control contrasts are secondary descriptive results with complete reporting, not alternatives available for choosing a positive headline. No post-hoc endpoint, subgroup, fold, margin or seed selection. With four convenience-sampled families and shared laboratories, no confirmatory population p-value or well-calibrated population confidence claim is planned. Even under an ideal independent sign-null, four concordant signs alone have one-sided probability 1/16, not conventional 0.05 evidence.

## Uncertainty and diagnostics

Report every per-family/fold arm correlation, delta, undefined case, denominator, exclusions and deviations. Family-level summary: range, SD and leave-one-family-out sensitivity. These quantify a finite panel; four families are insufficient for reliable family-population bootstrap inference.

Conditional within-family sensitivity: if separately authorized later, 2,000 paired bootstrap resamples of reference positions, seed 21; retain all substitutions at sampled positions and identical paired draws across arms/folds, recomputing ranks each time. Report percentile 2.5/97.5 intervals and the number of undefined draws, without hiding them. Position clustering does not model spatial dependence or experimental replicate error fully. These are conditional sensitivity intervals, not experimental-noise or population-generalization confidence guarantees. If genotype-only coverage is inadequate for this scheme, preregister a no-interval descriptive report before outcome access rather than choosing uncertainty methods after results.

No inverse-SE weighting, sigma-as-physical-noise or abundance-to-free-energy calibration. Replicate uncertainty propagation would require separately qualified replicate rights, provenance and an amended pre-outcome plan. No patient or clinical classifications are accessed.

Constant/WT-only predictors can produce undefined rho; report that honestly. Archived shuffled-training/null arms are only usable when frozen and comparable. A future label-permutation diagnostic requires a separately specified exchangeability model and authorization; it is omitted from this draft's runnable scope. Invariance of cohort selection, feature/scaler state and predictions to held-out labels would be a required leakage check, never a new fit using those labels.

## Identifiable hypotheses

| Original question | Identifiability here |
| --- | --- |
| H1 label budget/sample complexity | Not a primary test. Only descriptive transport of archived nested-budget contrasts could be considered in another registered secondary plan; no external learning curve is identified. |
| H2 familiarity/family shift | Not identified: all external families are held out, with no matched familiar-family intervention. Assay and family shifts remain confounded. |
| H3 capacity/inductive bias | Conditional, partial: matched frozen null comparisons can measure ranking differences. No isolated causal mechanism or matched external retraining is established. |
| H4 graph information | Metadata/mapping audit only; reference geometry does not establish mutant rearrangement or physical interaction causality. |
| H5 representation/preprocessing mismatch | Conditional yes for the fixed scaled/unscaled ranking contrast only, after all admission gates; no general mechanistic proof. |

## Limits and stopping state

Shared investigators/assay methods reduce study independence despite different genes. Published data already exposed to pretrained models or human literature cannot be certified unseen in every sense. New domain IDs do not rule out remote homology. Full-length human proteins and proxies differ from small-domain stability development cohorts; the shift has multiple causes. Exact constructs and observed-residue coverage remain unresolved for three candidates, and SOD1 has a retained historical strict-mapping failure.

The earlier search narratives, current ASPA inline missingness annotation and inherited SOD1 incident prohibit pristine-blinding claims. A same-context assistant memo cannot supply independent adjudication. No final outcome access, variants, states, runs, tags, release or merge are authorized. **Stop here unless a separately authorized metadata/registration stage resolves every hard gate; this report does not initiate that stage.**
