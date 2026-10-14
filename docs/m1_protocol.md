# M1 experiment contract

Question: do fixed pretrained sequence features improve this single-protein
activity assay, particularly when fewer training labels are available?

The project owner requested the same FLIP2 Amylase by-mutation distribution as M0.
Use the M0 compressed-data checksum, WT provenance and effective partition rules.
Compare every model against the same 702 test variants and fixed 601 validation
variants. Fail if historical test identities/targets or M0 baseline predictions
and metrics differ beyond numerical tolerance. Never overwrite `benchmarks/amylase`.

Features fixed before inspecting M1 test results:

- Training mean, M0 mutation-count ridge, M0 full-sequence one-hot ridge.
- ESM-2 `facebook/esm2_t12_35M_UR50D`, revision
  `6fbf070e65b0b7291e7bbcd451118c216cff79d8`, frozen eval, last hidden state,
  float32. Same checkpoint revision for tokenizer.
- Residue-only mutant mean pooling (480 features).
- Mutant mean minus WT mean (480 features).
- Concatenated mean mutant states at changed sites, mean mutant-minus-WT states
  at those sites, and number of changed sites (961 features). Pool symmetrically;
  WT (zero sites) gets zeros. Positions are aligned against the provenance WT,
  not guessed from variant frequency. Special tokens/padding excluded; no truncation.

All ridge models use M0's six alphas (0.01 to 1000), selected by validation RMSE,
without scaling or refitting on validation. LSQR/intercept/tolerance stay as in M0.
Cast pretrained features to float64 for regression; the encoder stays float32.
Mean/delta are a constant translation under this single-WT/intercept model and
should have effectively identical ridge results; they are not independent evidence.

Training fractions: 1%, 5%, 10%, 25%, 50%, 100%, with ceil and at least two examples.
Seeds 0, 1, 2 each define one permutation of training rows; prefixes are nested,
then sorted into source order. Every model gets the same subset at each seed/size.
Validation/test membership never changes. Reuse the identical 100% fit across seeds.
Report Spearman, Pearson, MAE and RMSE and sample SD across training subset seeds.
This SD is not a test confidence interval. The full 601 validation labels remain
available even when training uses only 25 labels: this is a low-*training*-label
study, not a study of a 25-label total annotation budget.

No architectural changes or additional model choices will be made in response to
test outcomes. A negative result is useful. No GNNs, structures, cross-attention,
multitask heads or fine-tuning. Checkpoints and embedding caches remain ignored.

Known interpretation limits: one protein, 410 shared train/test substitutions,
fixed test set, only three subset seeds, potential unknown overlap with ESM's
pretraining corpus. Test sequence embedding extraction is label-free transductive
evaluation. No cross-protein/family transfer claim or claim to replicate the
paper's complete protocol.
