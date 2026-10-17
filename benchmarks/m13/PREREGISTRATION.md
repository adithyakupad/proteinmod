# M13 fixed uncertainty/calibration plan

Registered before any M13 fitting or held-out interval/error inspection. Use the exact
M12 768-genotype cohort, original test IDs and G1/G2/G3 fold order; preserve all M0–M12
artifacts. M12 and M6 configs/source hashes and all new budgets are pinned in m13.json.

Uncertainty definitions: within-protein genotype-bootstrap ridge sensitivity and
random-initialization/optimization ensemble dispersion are epistemic *diagnostics*,
not posterior probabilities. They condition on this small training collection and do
not measure unknown-family uncertainty completely. Residuals mix measurement noise,
misspecification and epistemic error. No measured aleatoric decomposition is possible
from these processed source tables. Structure confidence is an input, not predictive
uncertainty. Nearest-training ESM-feature distance, sequence identity, family status
and structural coverage are OOD diagnostics, not calibrated probabilities.

M12 outputs are assay-local ranks, not physical units. Preserve each source target.
Use ordinal pairwise-discordance error on *all* held-out variants, without fitting a
test-label CDF/scale. For native intervals, crosswalk the independently verified M6
primary Figure 3 table onto the same reference+single-mutant sequence, with canonical
folding ΔΔG in kcal/mol (positive destabilizing, minus source positive-stabilizing ddg).
347/768 genotypes have compatible labels under the existing M6 Cys/missing/censor rules.
The rest remain eligible for rank diagnostics but not native interval scoring. These
are two released estimates of the same experimental variant, not fabricated paired
stability/function labels. No M12 DMS score is reinterpreted as kcal/mol. Only common
physical labels from the same M6 task/conditions may share native residual widths.

Reserve calibration independently of model-selection validation. G1: label-blind
hash-select 12 native-eligible genotypes/protein from original training (52 train,
12 calibration, 16 validation, 16 test per protein). G2: remove one whole protein from
each of the two non-test/non-validation families for calibration; four proteins train,
one validates, two calibrate, one tests; both test and validation have training-family
partners. G3: test family j, validate j+1, calibrate j+2, train j+3. All four roles are
family-disjoint. No test-family calibration. Calibration selection uses availability,
protein/family identities and hashes, never target values or prediction errors.

Point baselines: transferable sequence ridge, frozen ESM-site ridge, structural ridge,
structural GNN, weighted fusion and bidirectional attention. Refit using reduced
training roles, fixed M12 grids/12-epoch maximum/validation-only early stopping.
Original M12 fits remain historical references, not calibration-safe fits. Ridge:
12 within-protein genotype bootstraps, fixed train-only scales/rank transformation and
validation-selected alpha. Two bootstrap replicates. Neural ensembles: seeds [0,1,2]
and [3,4,5]. No historical supervised checkpoints. Retain member and deterministic
point predictions. A fixed training-only affine native head on each member maps its
rank to primary physical targets; this is a new, explicit transfer head, not test-assay
scale fitting. Member dispersion therefore omits uncertainty in a fixed fitted head.

Nominal levels 80/90%. Compare validation-residual width (not valid split conformal
because selection uses validation), independent-calibration residual width,
ensemble-scaled conformal, label-free distance-scaled conformal, shuffled-ensemble-scale
control and uncalibrated Gaussian ensemble bars. G1 calibration is per protein; G2/G3
pool only compatible primary kcal/mol residuals. Fixed native scale floor 0.05 kcal/mol,
distance floor 0.05; no test tuning. Finite-sample quantile index ceil((n+1)*level).
Insufficient calibration units yield explicit unbounded intervals, never a clipped
quantile or fake finite guarantee. Family-block G3 calibration has one independent
family: no finite 80/90% family-conformal interval. Variant-wise G3 intervals are
*empirical shift diagnostics*, not distribution-free family coverage claims.

Evaluate native coverage/width/interval score, dispersion-vs-absolute-error Spearman,
ordinal uncertainty-vs-discordance association, and selective risk at 25/50/75/100%
retention. Include ensemble, distance, shuffled and constant ranking controls; shuffle
within protein and role without labels. Report per protein/family and equally weighted
family macros; only four M12 families are independent replication units. Report seed/
bootstrap replicate dispersion without mutation-level significance. Retain original
single/double and structural-eligibility breakdowns, and fixed sequence-identity bins.

Add an existing M6 native-unit ridge sanity control on its original 1800 single-mutant
cohort and S1/S2 test IDs: sequence, frozen ESM and structural ridge. S1 removes 60
original training variants/protein for calibration. S2 removes 120 from its training
protein; calibration is therefore training-protein-distributed, not new-protein
exchangeable. These are explicitly separate results, with no family-held-out claim
and no combination with M12 supervision.

Archive per-example member/point/UQ predictions, intervals, selection/calibration
records, audits, source versions, devices, costs, plots and exact replay checks.
No heteroscedastic noise head without replicate evidence; no M14/M15 or fine-tuning.
