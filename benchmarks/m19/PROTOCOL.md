# Approved M19 ridge development protocol

Approved by the user on 2026-10-20. No independent confirmation is claimed.
Use exactly the H1 entries of the hash-pinned M17 split manifest: four families,
four fractions, five seeds (80 cases). Historical test outcomes have already been
studied. No SOD1 source, outcome, aggregate metadata or admission gate is read.

Four arms: historical scaled/unscaled ridge, expanded scaled/unscaled ridge.
Historical preprocessing and alpha [.1,10,100] reproduce M17 exactly.
Expanded geometries use identical nine dimensionless penalties 10^-6 through 10^2.
For each training subset, coordinate-standardize using training means/stds for
scaled geometry; retain original coordinates for unscaled geometry. Define
E = sum((X_train - mean(X_train))^2) / n_train, in float64, after the geometry
transform. Reject nonfinite or zero E. Divide all features by sqrt(E), then fit
Ridge(alpha=lambda, fit_intercept=True, solver=lsqr, tol=1e-8, max_iter=10000).
This is equivalent to original-geometry alpha=lambda*E. Centering in E measures
predictive variation, excluding the fitted intercept. No validation/test features
contribute to preprocessing. Scikit-learn's fitted intercept already handles
centering: no additional centering-only arm is necessary.

Labels remain original train-assay midrank percentiles. Select by equal-protein
validation Spearman only, undefined correlations contribute zero with counts
retained. Enumerate penalties ascending; strict greater-than selects first on ties.
Joint expanded selection compares unscaled first, then scaled, with the same tie
rule. Never select a test winner. Primary endpoint: joint validation-selected
expanded model minus historical unscaled, full fraction, equal-family macro rho.
Report both fixed-geometry arms regardless of joint selection. Full-fraction
ridge repeats across seeds are deterministic duplicates, not replications.

Success requires >=.02 macro gain and strictly positive effects in >=3/4 families,
plus passing controls/replay. Partial: a diagnostic repair without that gain.
Failure: no gain or failed numerical/leakage controls. No adaptive grid expansion,
new architectures, extra datasets or test-driven reruns. Secondary low-label curves
use all existing fractions/seeds; report family differences, range, sample SD and
leave-one-family-out means, no population CI/p-value from four development families.
Secondary error: per-protein absolute difference between predicted within-test
midrank percentile and observed within-test midrank percentile (evaluation only).
No cross-assay native RMSE or predictive-interval calibration claim.

Before authentic fits: synthetic scalar-rescaling equivalence (c=7, alpha*c^2;
normalized predictions also equivalent), deterministic ties, held-out-label
perturbation and train-only preprocessing tests. On authentic data repeat scalar
rescaling and held-out-test-label perturbation without changing selected state.
Fixed negative control: permute training labels within assay with RNG seed314159;
validation remains genuine. Save all candidates, state arrays, predictions, family
and protein scores, input/cache hashes, versions, timings, device and selected
penalties. Replay saved states exactly; independently refit all cases once and
compare selected states/predictions exactly. Check historical predictions within
1e-12; if unavailable or different, stop and report, never overwrite history.
Historical file hashes must remain identical. Commit this protocol before fitting.
