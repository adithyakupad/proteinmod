# M16.1 prespecified portability qualification

Recorded before new CPU/MPS runs. No hyperparameters are retuned against test labels.
The legacy absolute 1e-6 failure remains a failure in the historical record.

Tier 1: identical-device independent forwards on 64 variants selected by SHA256
of sequence, plus WT, must be bitwise identical. The full original M1 cohort is
extracted independently on CPU and MPS, with identical pinned weights, tokenizer,
float32, seed 0, batch size 8, deterministic algorithms, dropout disabled.

Tier 2: normalized activation differences must satisfy
`abs(a-b) <= 32 * 12 * eps(float32) * (1 + max(abs(a),abs(b)))`.
This is an engineering roundoff screen: 32 elementary rounding allowances per
transformer layer, accumulated over depth. It is not a rigorous nonlinear error
bound or a threshold inferred from the old maximum. Report failures without
relaxation. Compare intermediate layers on a label-blind probe to locate divergence.
Float32 non-associative reductions and backend kernels can differ even with
identical weights and deterministic algorithms; exact device parity is not promised
by [PyTorch](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

Tier 3: separately fitted ridge probes must choose identical validation-selected
alphas; maximum held-out prediction change divided by training target SD, absolute
Spearman change, and RMSE/MAE change divided by training SD must each be <=0.001.
The margin is 0.1% of training outcome variation / 0.001 rank correlation: a
prespecified practical benchmark resolution, not experimental measurement noise
(which has not been established). For fixed CPU ridge weights also report the
linear propagated bound `sum(abs(coef_j)*activation_budget_j)`.
Reject nonfinite tensors, changed shapes, IDs, labels or splits. Include deliberately
corrupted negative controls. Passing applies only to tested software/hardware/model
and cohort, not all M0–M15 devices or platforms. Preserve same-device strict replay.

Cold-source experiment: empty dedicated cache; redownload FLIP source, immutable
ESM files with existing SHA pins, regenerate all 3,706 variant features plus WT,
and refit original M0/M1 comparisons and learning curves. Download/rebuild a real
PDB/mmCIF structure pair and validate the native stability source adapter where
possible. Other milestones remain explicitly cache-dependent pending independent
acquisition and regeneration; no universal cold-release claim.
