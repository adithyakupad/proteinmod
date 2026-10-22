# M23 prospective synthetic representation-geometry study v1

Base: integrated main 9f3fc9b6d158539e5758fe88c04d5af4f5b39b64. Commit this protocol
and config before implementation, tests or numerical execution. Preserve every existing
tracked file, branch ref and archive/m203-interrupted. Study is exclusively artificial
numeric regression; no biological inputs, architectures or benchmark tuning.

## Independent units and data

40 independent master seeds 0–39. Each seed is one simulation unit shared/paired across
all scenarios, methods and conditions; never count configurations/test rows as independent
replications. PCG64 SeedSequence([23, namespace, seed, role]); role 0/1/2=train/validation/test.
Namespaces: 100 latent standard-normal Z, 101 independent N(0,1) observation noise,
200 MLP initialization, 300 minibatch schedule, 400 training-label permutation.
Train64, master validation256, test8192; p=12. beta=(1,-.7,.5,0,...,0), noise SD1,
y=z beta+epsilon. Validation16 is the first16 rows of the same master validation256.
All IDs (seed, role, row) fixed; outcomes never determine retention or selection.

A: z=Z in every role; x=z (isotropic informative features).
B: z=Z in every role; x=z diag(s), s_j=10^(-1.3+2.6*j/11), j=0..11.
A/B share latent samples, noise and targets exactly: only feature units differ.
C: same s as B and identical training arrays/labels. Validation z=Z L(.6)^T;
test z=Z L(-.6)^T, where L(rho) is the lower Cholesky factor of R_ij=rho^|i-j|.
Training R=I. Thus all three C role covariances differ, but the linear conditional
function y|x has unchanged coefficients beta/s and homoscedastic noise. B/C fitted
candidates must be identical; covariance shift is not concept shift. No other scenarios.

## Fixed candidates, budgets and preprocessing

All ridge fits use float64 direct SVD, intercept via training-only x/y centering,
objective mean squared error + penalty. Let E=mean_j training population variance(x_j).
For dimensionless lambda: unscaled ridge uses penalty lambda*E on raw coefficients;
global energy normalization uses x_centered/sqrt(E) and penalty lambda. These are
exactly the same estimator in different scalar coordinates, a prespecified equivalence
negative control, not two independent mechanisms. Coordinate-standardized ridge uses
x_centered/training_SD_j and penalty lambda: the induced raw-coordinate penalty is
lambda*diag(training_variances). Match average feature energy to avoid attributing a
scalar penalty change to coordinate geometry. Zero scales are replaced by1.

Small grid: [1e-3,1e-1,1e1,1e3]. Large grid: 10^(-3+.4*k), k=0..15;
small indices [0,5,10,15], exactly nested. Each method gets exactly4 or16 candidates,
independent of validation size. All methods also have the same prespecified fixed
comparator lambda=.1 (already in both grids). Selection minimizes validation MSE;
ascending lambda then strict smaller-than chooses deterministic ties. Never use test
labels, analytic population risks or coefficient errors for selection. Fixed baselines
are not chosen by outcomes. K and n_val form a paired 2x2 factorial (16/256 by4/16).

Secondary MLP: original M22.1 initialization/gradient primitives, fixed 12→16 tanh→1
float64, Adam lr.003, betas.9/.999 eps1e-8, epochs120, batch32, no early stopping,
scheduler or dropout; training-only coordinate standardization and y centering. Add
2*lambda*weight to gradients of w1/w2 only (biases unpenalized), matching objective
MSE+lambda*(||w1||²+||w2||²). Same lambda grids and fixed .1. Same initialization/order
for every candidate within seed, independent PCG64 namespaces. No architecture or
learning-rate search. B/C share fitted states; no redundant training. Scenarios jointly
alter representation units or covariance as declared, not label noise/sample-size grids.

## Estimands and primary decisions

Only test MSE is the primary performance metric. Noise variance=1 defines meaningful
absolute thresholds; no predetermined percentage gain. Signed gain=baseline MSE minus
candidate MSE. All six primary decisions are fixed below. Use paired Student-t intervals
across40 seeds. For H_A/H_B/H_V/H_K/H_S use confidence 1-.05/6 (99.1666667%); HG's four
baseline comparisons use 1-.05/(6*4) (99.7916667%) conservatively. No population claim
beyond these DGPs. Other contrasts have descriptive95% paired intervals, not extra tests.

H_A (equivalence): fixed coordinate vs fixed unscaled .1 in A; interval of signed gain
entirely within[-.05,.05] supports little systematic scaling benefit; otherwise inconclusive.
H_B (geometry): fixed coordinate vs fixed unscaled .1 in B; lower gain bound>.10 and
>=30/40 positive supports meaningful geometry benefit. Distinguish coefficient recovery
from predictive benefit; failed improvement is acceptable. Direct diagonal-penalty
identity and scalar controls must pass before mechanistic interpretation.
H_V (validation size): coordinate ridge in B, K16; MSE(nval16)-MSE(nval256). Lower bound>.05
supports a meaningful small-validation penalty, upper<-.05 contradicts it; else inconclusive.
H_K (multiplicity): coordinate ridge in B, nval16; MSE(K16)-MSE(K4). Lower bound>.05 supports
harm; upper<-.05 supports benefit; otherwise no identified meaningful directional effect.
Nested grids change candidate opportunity as well as multiplicity, so report population
oracle regret and selection optimism; never equate this with independent multiple tests.
H_S (covariance shift): fixed unscaled .1, identical B/C trained state. Difference in
analytic population test risk C-B has interval outside[-.10,.10] to identify meaningful
risk change due solely to declared covariance. Report empirical test change separately.
HG (strong fixed comparator): coordinate-selected (nval256,K16) in B must beat EACH of
four fixed .1 baselines (unscaled, global, coordinate, MLP) by lower gain bound>.10.
No test-selected fixed winner. Global duplicates unscaled intentionally. Failure means
no qualified meaningful gain over the strongest prespecified fixed comparator.

## Selection reliability and diagnostic decomposition

For all scenarios/methods/nval/K, report selected penalty, test MSE/Spearman, validation
MSE, val-to-test gap and seed-paired selected-minus-fixed effects. Count wins/losses/ties
against the SAME method's fixed .1 baseline (absolute MSE tie tolerance1e-12); Wilson95%
intervals over40 independent seeds. Retain all candidate metrics and fixed identities.
After selection is frozen, compute finite-test oracle regret for every method; for
linear models additionally compute exact Gaussian population risk =1+intercept²+
(beta_hat_latent-beta)^T R_role (beta_hat_latent-beta). Population oracle regret and
true-validation-risk minus observed-validation-MSE expose selection versus noise.
Oracle quantities are diagnostics only, never model-selection inputs.

Counterfactual B/C decomposition per method/nval/K, on paired test identities:
C-selected/C-test minus B-selected/B-test = (B-selected/C-test minus B-selected/B-test)
+ (C-selected/C-test minus B-selected/C-test). First term freezes fitted/selected model;
second freezes C-test data. Report exact numerical sum check; do not call both terms
pure independent causal effects because validation and test covariances jointly change.
For fixed models, B/C training equality proves only covariance intervention differs.

Report training representation effective rank exp(entropy(normalized squared singular
values)), feature-variance anisotropy max/min, matrix condition number and numerical
rank; raw/coordinate/global representations. Also coefficient recovery error for ridge
in latent coordinates, normalized by ||beta||², and all selected penalties. Coefficient
error, accuracy, and selection reliability are separate endpoints. MLP recovery undefined.

## Mandatory controls, replay and finite stopping

Training-only scalers/intercepts; adversarial validation/test label perturbations must
leave all fitted states unchanged (validation choices MAY change); test-label changes
must leave choices unchanged. B/C training states and schedules identical. A/B latent
target equality. Every candidate is refitted independently and must have bitwise-equal
states/predictions within the captured environment, with source/input/seed/schedule
hashes. Reuse M22 primitives; do not edit historical trainers.

Unscaled/global scalar-rescaling identity and standardized/generalized diagonal-ridge
identity: prediction rtol1e-8/atol1e-9, fixed before execution. Redundant multiplicity
control repeats the four small candidates4 times without new fits; selection must be
identical to K4, including ties. Fixed lambda .1 label-permutation controls for all four
methods, same seed permutation independent of features. Expected rank correlation near0
is diagnostic; if A unscaled mean abs(Spearman)>.10 AND its descriptive95% interval excludes
0, flag negative-control failure and stop mechanistic claims. No forced zero outcome.

Stop after exactly40 seeds, three scenarios, four methods, two K and two nval; no seed
replacement or new substitutes. Any nonfinite fit, missing case, identity/leakage,
replay/numerical-control failure -> engineering FAIL; preserve all finished records and
errors, do not relax tolerances. Scientific hypotheses may be supported, contradicted
or inconclusive independently of an engineering PASS. Software bug fixes require an
explicit amendment and preserved failed evidence, not silent protocol edits.

## Deliverables and limitations

Per-seed/all-candidate records, paired effects/CIs, selection rates, geometry diagnostics,
SVG diagnostic plots, source/config/environment hashes, reproduction commands, final
report and draft PR. Freeze implementation before registered full run. Run focused tests,
Ruff and scoped CI; CI independently repeats unchanged numerical study with exact local
replay and same control tolerances, not universal cross-platform bit equality. No merge,
release, biological interpretation or claim that these mechanisms explain M17/M19.
