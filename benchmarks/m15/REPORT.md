# M15 scientific results and failure analysis

**The prespecified last-block LoRA adaptation did not improve the strongest frozen
or structural baselines.** M15 completed 442 selected fits (780 LR/alpha candidate
executions) and 56,576 held-out prediction records on unchanged M12 G1/G2/G3 roles,
including a separately matched 25% label regime. Seven models use identical test
variants and structure eligibility. Neural means use seeds 0/1/2; deterministic
ridges fit once per fold. The complete [numeric tables](RESULTS.md),
[per-family](per_family.csv), [per-protein](per_protein.csv),
[single/multiple slices](failure_slices.csv) and [raw predictions](predictions.csv)
are retained. Negative results are not a failed implementation.

## Main comparison

At full training-label availability, macro per-protein Spearman is:

| Model | G1 | G2 | G3 |
|---|---:|---:|---:|
| Frozen ESM ridge | .7040 | .5401 | .4008 |
| Frozen ESM learned head | .6626 ±.0458 | .4461 ±.0143 | .3401 ±.0207 |
| Parameter-matched frozen head | .6746 ±.0321 | .4496 ±.0123 | .3317 ±.0156 |
| ESM LoRA | .6297 ±.0443 | .4460 ±.0126 | .3426 ±.0197 |
| Structural ridge | .5533 | .3996 | .4298 |
| Original GNN | .4170 ±.0142 | .4014 ±.0186 | .4020 ±.0104 |
| Edge-aware GNN | .4440 ±.0127 | .3991 ±.0091 | .3799 ±.0065 |

LoRA's G3 gain over the frozen learned head is only **.0025**, and **.0109** over
the nominally parameter-matched head. It remains **.0582 below frozen ridge** and
**.0872 below structural ridge**. G1 deteriorates relative to both learned frozen
controls, while G2 is effectively unchanged relative to the frozen learned head
and clearly below frozen ridge. This does not justify promoting the adapted model.
Seed SD is descriptive optimization variability, not a significance test or
calibrated predictive uncertainty. Four families are the effective family-level
sample size; there is no claim of a universal failure of LoRA.

The common head has 15,409 parameters. Both LoRA and matched-head controls have
19,249, adding exactly 3,840 parameters, with identical initial predictions and
common-head initialization. The control matches nominal count and initial state,
not identical effective function classes or optimization geometry. It therefore
cannot by itself prove a causal benefit of contextual adaptation. Structural
models retain their prior 12-epoch budget; ESM models use the prespecified six-epoch
two-LR budget, so structural models are scientific baselines rather than matched
neural-capacity controls.

## Low-label behavior

At 25% of each original training protein's labels, G3 rho is **.2142 ±.0472** for
LoRA, **.2147 ±.0473** for frozen head, **.2165 ±.0490** for matched head,
**.3307** for frozen ridge and **.3266** for structural ridge. No low-label adaptation
advantage is observed. Original/edge GNN G3 seed SD rises to **.1453/.1955**.
Larger nonlinear heads remain difficult to fit robustly at this sample size.

The effective low-label training counts are G1 128 (16/protein), G2 144
(24/protein), G3 96 (24/protein). Their original validation/test records are unchanged.
The [learning-curve plot](learning_curves.png) shows only two measured budgets,
25% and 100%; connecting lines do not estimate a scaling law. Subset selection is
label blind and nested, not repeated until favorable outcomes are found.

## Family failures and overfitting

Full-label G3 per-family Spearman, averaging the same two proteins and neural seeds:

| Model | SH3 | Chromo | Homeodomain | WW |
|---|---:|---:|---:|---:|
| Frozen ridge | .4442 | .4545 | .3130 | .3915 |
| Frozen learned head | .2349 | .5080 | .1836 | .4339 |
| Matched head | .2208 | .4878 | .1896 | .4287 |
| LoRA | .2346 | .5073 | .1871 | .4414 |
| Structural ridge | .3591 | .5460 | .4331 | .3808 |

LoRA remains weak on SH3 and homeodomains. The richer nonlinear prediction heads
share much of this failure pattern; the aggregate does not identify an adaptation
specific transfer benefit. Slightly stronger WW ranking does not offset those
failures. All eight protein results and mutation-burden slices are preserved, not
filtered after observing scores.

Across 39 full-label ESM fits, median train-minus-validation Spearman is **.3367**
for LoRA, **.3367** for frozen head and **.3328** for matched head. This large gap
is consistent with memorization and/or validation-family shift; it is not proof
of one mechanism. Nineteen of 39 LoRA fits select the maximum epoch six, versus
16 frozen-head and 14 matched-head fits. A short budget and limited rank/layer
coverage remain limitations. Extra epochs, all-layer adaptation, additional ranks
or test-informed LR changes were not tried after inspecting results.

G1 tests 128 original within-protein variants; G2/G3 each test all 768 selected
variants. G1-to-G3 gaps mix test composition, training diversity, validation roles
and evolutionary novelty. A smaller gap caused by lower G1 performance is not
better generalization. All primary comparisons within a protocol/fraction have
identical test IDs. No cross-assay raw RMSE or native thermodynamic scale is claimed.

## Provenance and leakage checks

The ESM-2 nominal 35M checkpoint/tokenizer revision and all local source files are
pinned; the loaded `EsmModel` has **33,269,521 original parameters**, excluding the
new head. LoRA trains only last-block query/value rank-2 adapters (alpha4, scale2,
dropout0) plus the common head. Every original tensor stays fixed. Original model
dropout remains disabled, but gradients flow through both adapted mutant and WT.

All 776 unlabelled sequences pass exact whole-forward versus frozen-prefix+tail
parity, every residue-token alignment check and frozen-feature cache parity.
No adapted hidden-state cache is shared between fits. All candidates start fresh;
no adapter trained in another fold/family supplies initialization. Only training
labels determine rank transforms, ridge scaling and gradients. Validation alone
selects LR/alpha/epoch. G3 uses the assigned validation family and excludes both
held-out families from supervised training. A held-out-label perturbation test
produces identical selected adapter tensors and predictions.

Actual cached mutation masks are checked under reversed mutation metadata for all
768 variants. Synthetic tests cover multiple sites, padding, token/shape failures,
WT gradients, active LoRA updates, unchanged base weights, exact matched-control
initialization, checkpoint save/load/schema, cache invalidation and protected output
paths. Historical file hashes and old model prediction equivalence are independently
audited, not inferred from passing unit tests. No selected protein correlation is
undefined in these runs; the inherited undefined-to-zero macro policy remains
explicit for reproducibility.

The old baselines reproduce **3,328 M12 frozen/structural ridge predictions** and
**9,984 M14 original/edge GNN predictions** exactly at full-label availability.
All 354 historical benchmark/config files remain byte-identical to PR #28.
No native stability/function labels are pooled, no new fusion is trained and no
M16 integration has been started.

## Measured local costs and repeat evidence

First complete run: **588.1 s**, peak process RSS **1,081,917,440 bytes** (~1.01 GiB),
float32 CPU, four threads. Measured prefix extraction across eight references and
all variants totaled **10.63 s** and is reused without supervised labels. The first
and independent repeat overlap briefly with local checks/each other; timings are
practical measured costs, not a controlled hardware speed benchmark. Only whole
process peak RSS was measured, not isolated per-architecture GPU memory.

Median selected-fit seconds (including each ESM model's two validation LR candidates):

| Model | 25% labels | 100% labels | Trainable parameters |
|---|---:|---:|---:|
| Frozen ridge | .022 | .124 | 962 |
| Frozen learned head | .106 | .165 | 15,409 |
| Matched head | .139 | .241 | 19,249 |
| LoRA | 2.905 | 8.906 | 19,249 |
| Structural ridge | .021 | .025 | 271 |
| Original GNN | .191 | .563 | 3,473 |
| Edge GNN | .351 | .992 | 5,361 |

LoRA costs roughly 54 times the full-label frozen-head fit here, despite a small
trainable parameter count; backpropagation through contextual attention still costs
more than cached-feature regression. The additional cost produced no improvement
over either strong ridge baseline. Parameter efficiency is not equivalent to
runtime efficiency or improved held-out prediction.

[validation.json](validation.json) records actual disk-checkpoint replay and the
complete fresh-repeat comparison, including every selected state, candidate trace,
role and held-out prediction. The complete independent repeat took **590.3 s**
with **1,104,183,296 bytes** peak process RSS; all 442 selected state tensors,
all candidate/selection traces and 56,576 predictions are exactly identical.
An independent full raw-sequence forward with selected real G3 adapters on all
eight proteins bypasses the activation cache and matches with **zero error**.
Failed supervised candidate executions: **none**;
all 780 candidate fits completed. Every LoRA/head candidate passed explicit
finite-loss/gradient checks, with nonzero adapter/control gradients where applicable. No failed seeds,
families or candidate scores were dropped. Reproduction and exact software pins
are in [M15 documentation](../../docs/M15.md); large binary checkpoints, original
PLM and prefix arrays remain ignored caches.

Unknown ESM pretraining exposure, four families from one study, small label budgets
and bounded last-block/rank-2 adaptation limit claims. G3 is supervised-family-held-out,
not a guarantee of unseen pretraining families, unseen folds or functional transfer.
