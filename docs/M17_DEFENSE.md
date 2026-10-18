# Neuralink ML evidence example and technical defense

## Evidence statement suitable for an ML discussion

ProteinMod is an AI-assisted research software project demonstrating controlled
machine-learning evaluation under sample scarcity and distribution shift. The M17
study preregistered five explanations for an existing structural-baseline result,
froze exact paired evaluation roles, and ran 1,096 fresh fits with 34,304 held-out
predictions. All selected states were reloaded and replayed exactly. The strongest
lesson was not a larger model winning: preprocessing changed the mean model ordering,
and a degree-preserving graph null remained competitive with physical contacts.
An external-data provenance gate was left blocked rather than presenting reused or
unresolved data as independent validation.

This is relevant evidence of experimental design, leakage prevention, ablations,
uncertainty skepticism and reproducible engineering. It is **not neural decoding,
brain–computer-interface performance, medical validation or evidence of work at
Neuralink**. Protein sequences and neural recordings have different data-generating
processes. The transferable skill is how to test an inductive bias and an OOD claim,
not a claim that this protein model transfers to neural signals.

For an interview or application, describe only responsibilities personally owned
and verified. Implementation, analysis and documentation here were produced with
AI assistance; do not imply sole manual authorship or unperformed wet-lab work.
Scientific ownership means being able to explain and reproduce the protocol,
recognize its confounds, and defend the negative results. No independent biological
confirmation has been obtained. Do not claim a calibrated deployment system.

## Questions and defensible answers

1. **What is the experimental unit?** Variants are nested in proteins and families.
   Five seeds quantify training sensitivity; only four development families were
   tested. Seeds and variants cannot be treated as independent family replications.
2. **Why no population confidence interval?** Four selected families are too few
   for a stable family-bootstrap interval. Report paired family effects, range and
   seed variation; avoid significance from thousands of correlated variants.
3. **Did G1/G2/G3 use the same examples and label budget?** H2 has the same original
   test variants, fixed validation family, four training proteins and 128 labels.
   Familiarity changes which proteins contribute; composition remains a confound.
4. **How did you isolate family diversity?** Two proteins and 128 labels in both
   conditions; enumerate two narrow and four diverse pairings. Even this small
   intervention changes protein identity and cannot establish a universal law.
5. **Where can labels leak?** Reference construction, cohort filtering, scaling,
   assay normalization, hyperparameter selection and residual calibration. Here
   references/cohorts are existing pinned inputs, target ranks/scalers fit training
   only, alpha/epoch use validation only, and exact role IDs are checked.
6. **Did you change the experiment after seeing a negative result?** No model grid,
   role or decision threshold changed after fitting. The initial external-source
   wording was corrected before fitting and the first plan preserved. The
   independent-confirmation gate remained blocked. Graph and chemistry reversals
   are retained rather than selecting a favorable label fraction.
7. **What actually supports a mechanism?** Unscaled ESM improves over scaled ESM in
   all four full-label family comparisons. That supports the specific preprocessing
   intervention descriptively. It does not prove the encoder lacks information or
   that unscaled ESM is a universal winner. Independent confirmation is absent.
8. **Why doesn't edge-over-zero prove geometry matters?** Zero-message disables
   effective parameters. Degree null preserves degree, feature marginals and node
   descriptors while breaking physical endpoints; it matches/beats physical edges
   at full labels. The nominal capacity match is not an effective capacity match.
9. **What does the graph audit establish?** Exact contact completeness and canonical
   heavy-atom availability for eight reference constructs. It does not validate
   chemistry, dynamics, protonation or mutant rearrangements. Unknown confidence
   is not inferred from complete coordinates.
10. **What is your uncertainty?** Across-seed/subsample sensitivity, not measured
    aleatoric variance. At full labels deterministic ridges have zero seed spread;
    that does not imply certainty. Selective prediction is descriptive, with a
    fixed shuffled control and no test-fitted cutoff.
11. **Can the new models output calibrated ΔΔG?** No. M17 predicts assay ranks.
    Native coverage/width results are clearly marked M13 reanalysis with its own
    heads and calibration roles. One calibration family implies unbounded 80/90%
    family-block intervals; finite variant-residual intervals do not fix this.
12. **What is independent about the external candidate?** Nothing has been accepted
    at label lineage. The archive mixes sources; metadata attribution is not enough.
    Independent measurement, independent protein family and unknown PLM pretraining
    exposure are distinct questions.
13. **Could the results be optimization failure?** Yes. Compact 12-epoch GNN and
    max_iter200 MLP budgets are deliberately bounded. Record warnings, gaps,
    parameter counts and runtimes; do not claim optimal model performance.
14. **How would you translate this rigor to neural ML?** Predefine session/subject/
    device holdouts, preserve chronology and calibration roles, separate correlated
    samples from independent units, compare simple baselines and shuffles, and
    distinguish measured uncertainty from distribution shift. This is a proposed
    evaluation discipline, not an executed neuroscience experiment.
15. **What would change your conclusion?** A resolved independent primary source,
    precommitted additional-family cohort, sufficient calibration families and
    replicated controlled effects. Do not tune on this study's test families again
    and then call them independent confirmation.

## Audit pointers

- [Preregistered criteria](../benchmarks/m17/PREREGISTRATION.md), commits `5e0cfac` and `9172e9a`.
- [Exact roles](../benchmarks/m17/split-manifest.json) and [source gate](../benchmarks/m17/ELIGIBILITY.md).
- [Per-family contrasts](../benchmarks/m17/per-family.csv), [all predictions](../benchmarks/m17/predictions.csv), [hypothesis decisions](../benchmarks/m17/hypotheses.json).
- [Graph audit](../benchmarks/m17/graph-audit.json), [physical atom audit](../benchmarks/m17/atom-audit.json).
- [Calibration reanalysis](../benchmarks/m17/native-calibration-reanalysis.csv), [family-block limits](../benchmarks/m17/family-block-calibration.json).
- [Selected-state replay](../benchmarks/m17/state-replay.json), [qualification matrix](../benchmarks/m17/QUALIFICATION.md).
