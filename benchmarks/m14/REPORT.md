# M14 scientific interpretation and failure analysis

**Richer structural message passing did not establish a robust generalization
improvement.** The fixed experiment ran nine models/controls across the original
13 M12 G1/G2/G3 folds and four M6 S1/S2 folds: 391 fits, 85,882 held-out prediction
records. Seven neural variants use three seeds; two deterministic ridges use one
fit per fold. All models share exactly the same test variants within each protocol.
See [the complete numeric table](RESULTS.md), [per-family metrics](per_family.csv),
[per-protein metrics](per_protein.csv) and [raw predictions](predictions.csv).

## Main comparison

On family-held-out G3, structural ridge achieves **0.4298 Spearman**, the original
M3 GNN **0.4020 ±0.0104**, the richer edge GNN **0.3799 ±0.0065**, and the independently
pretrained ProteinMPNN probe **0.4147**. The zero-message control is **0.3874 ±0.0163**;
chemistry-removed **0.3798**, shuffled chemistry **0.3792**, shuffled topology **0.3746**.
The tiny full-versus-shuffled differences do not demonstrate useful chemistry.
The richer GNN also trails original GNN and ridge on this family-held-out task.

For held-out native stability proteins (S2), richer GNN RMSE is **1.2237 ±0.0534
kcal/mol**, compared with original GNN **1.2521 ±0.0322** and structural ridge
**1.3503**. However, zero-message **1.2157 ±0.0605**, shuffled chemistry **1.2175**
and shuffled topology **1.2233** match or beat it. This prevents attributing the small
improvement over the smaller original model to correct spatial/chemical messaging.
The neural controls have identical 5,361 parameters; original GNN has 3,473.
Zero-message still has mutation properties, reference node descriptors and global
context; it specifically removes incoming messages, not all structural information.

The frozen ProteinMPNN probe improves known-protein G1 Spearman from structural
ridge **0.5533** to **0.5662**, held-out-protein G2 from **0.3996** to **0.4538**,
and native S1 RMSE from **0.7728** to **0.6373 kcal/mol**. Its S2 RMSE is **1.2166**,
similar to zero-message, and its G3 rho remains below structural ridge. These are
actual frozen encoder features from the official small inverse-folding checkpoint,
not supervised models renamed as pretraining. The frozen representation and probe
also differ from the GNN head; architecture/pretraining effects are not isolated.

## Family and protein failures

G3 family-macro Spearman, averaging the same two proteins and neural seeds:

| Model | SH3 | Chromo | Homeodomain | WW |
|---|---:|---:|---:|---:|
| Structural ridge | .3591 | .5460 | .4331 | .3808 |
| Original GNN | .2482 | .5777 | .3383 | .4439 |
| Rich edge GNN | .2318 | .6000 | .3236 | .3640 |
| Zero-message | .2382 | .5891 | .3149 | .4074 |
| Frozen ProteinMPNN ridge | .0139 | .5562 | .5538 | .5348 |

The pretrained probe's SH3 collapse is hidden by a reasonable aggregate; family
heterogeneity is materially larger than seed variation. A single aggregate is
insufficient evidence of family robustness. The edge model's Chromo ranking is
stronger, but other held-out families regress. No family-specific model was selected
using these test results.

Native S2 RMSE is dominated by **1TUD**: original/richer GNN **1.9608/1.9409**,
zero-message **1.9391**, and ProteinMPNN **1.9283 kcal/mol**. For 1A32 the corresponding
values are **.5367/.5057/.5074/.5660**; for 1LP1 **1.2587/1.2246/1.2006/1.1555**.
This is a three-protein benchmark: neither a small mean gain nor three random seeds
supports a significance or broad thermodynamic-transfer claim.

G3 single/multiple mutation slices (macro per-protein rho, separately ranked within
slice) are **.2020/.3726** for rich GNN, **.2545/.3874** for original GNN,
**.2936/.4795** for ridge and **.3575/.3563** for ProteinMPNN. Mutation order invariance
does not imply accurate multi-mutant interaction prediction. Slice sizes differ
between proteins; these are diagnostics, not independent trials or causal effects.

G1-to-G3 rho falls by **.0641** for richer GNN and **.1515** for ProteinMPNN.
G1 tests only the original within-protein subset; G2/G3 test all selected variants.
Thus these descriptive gaps mix protein/family novelty, training diversity/size,
validation roles and test composition; they do not isolate evolutionary distance.

## Audits and validity

- Original M12 roles: G1 512/128/128; G2 576/96/96 per fold; G3 384/192/192.
  Original M6 roles: S1 1,260/270/270; S2 600/600/600 per fold. Train/validation/test
  order is shown. No M13 calibration-role substitution or extra filtering.
- All 2,568 variants retain accepted mutation mapping. Full heavy-atom reparsing
  matches M2 author IDs, chain, reference identity and polymer positions on all
  11 structures (596 nodes, 5,042 directed edges). All required backbone/interaction
  atoms are present here. Missing-atom masks are tested on synthetic fixtures;
  observed missingness robustness cannot be claimed from a complete-atom cohort.
- Existing coordinate checksums, unique alignments, Pfam annotations, sequence
  similarity, reverse-pair and held-out protein/family leakage audits remain active.
  Shuffles are fixed, label-blind and symmetric across edge directions. Their
  assignments are not selected from validation or test outcomes.
- Only training labels determine rank targets, native normalization, ridge scaling
  and gradients. Validation determines alpha/epoch; test determines neither. A
  test-label perturbation test produces identical selected states and predictions.
  Primary targets are not pooled or compared using arbitrary cross-assay RMSE.
- All features describe WT/reference geometry. Hydrogen bonding is reported only
  as a carbonyl-geometry-qualified heavy-atom candidate, never a confirmed bond.
  Experimental B factors, unknown confidence and AF pLDDT remain distinct, with
  missingness. They are not epistemic/aleatoric predictive uncertainty.
- Mutation input order is checked on every selected variant. Actual reference
  structures are checked under rigid transforms and node permutations for the
  learned graph pipeline and frozen structural encoder. Floating-point tolerance
  is explicitly recorded in [validation](validation.json); mathematical invariance
  does not require pretending float32 transforms are bit-identical.
- Primary rich-model median train-minus-validation Spearman is **.1908**, versus
  original GNN **.1640** and zero-message **.1662**. Thirteen of 39 primary rich
  fits select epoch 12. Both overfitting and a short fixed training budget remain
  plausible limits; neither justifies post-hoc test-driven retuning. Complete
  selected-epoch traces and [overfitting evidence](overfitting.csv) are retained.
- [Feature shifts](feature_shifts.json) record protein-macro edge feature means in
  each train/validation/test role and held-out-minus-train differences. Some
  proteins have no salt candidates. These are WT feature shifts, not label leakage
  or energy estimates; small domain graphs and family composition remain confounded.

## Reproducibility and compute

The prespecified config/plan commit is `087712c`; final fitting source is `aad6afe`.
The complete independent refit uses `6bbaa72`; its only training-runner change is
an output-checkpoint overwrite guard, with no model/target/optimization changes.
Source hashes are verified against the immutable actual fitting Git revision;
all saved models are then inferred using the final implementation. Original M12
structural ridge's **1,664 test predictions reproduce exactly** after restoring
fit-time StandardScaler float32 semantics in M14's serialization path.

The actual first final run took **271.4 s** on Apple Silicon CPU, four threads;
frozen encoder inference across 11 references took **.212 s** after model loading.
Median fits: original GNN **.342 s**, rich GNN **.709 s**, zero-message **.726 s**,
structural ridge **.015 s**, pretrained probe ridge **.025 s**. These small-run wall
times include local background/check contention; they are indicative costs, not a
controlled hardware speed benchmark. Zero-message intentionally retains parameter
and message-computation capacity. The independent repeat took **228.7 s** and recorded **726,368,256 bytes**
(~693 MiB) peak process RSS. All 391 selected state tensors/arrays, selection
traces and 85,882 predictions were exactly equal; saved-model inference also
reproduced every prediction with zero absolute error.
No new heavy graph framework, GPU service or full structural corpus was needed.

See [validation.json](validation.json) for actual serialized-model replay and complete
fresh-repeat evidence, [costs.json](costs.json) for all models and
[reproduction instructions](../../docs/M14.md). All M0–M13 benchmark/config artifacts
are byte-identical to PR #26. No large binary checkpoints or token caches are committed.

ProteinMPNN structural pretraining may include exact/homologous PDB information;
this was not exhaustively excluded. Four supervised held-out families, all from
one study, do not establish unseen-fold, unseen-pretraining-family or general
functional generalization. Richer chemistry, more epochs or larger encoders may
behave differently, but were not pursued to improve observed test scores.
M15/M16, cross-attention changes, fine-tuning and multitask expansion are excluded.
