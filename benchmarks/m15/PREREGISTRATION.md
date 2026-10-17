# M15 prespecified LoRA experiment

Issue #29, stacked on verified unmerged PR #28 head
6ce820fce2b8f287b79ff92a1a576c5e761f6cf3. Main and all M0–M14 benchmark/config bytes
remain unchanged. Scope reaches new LoRA model/cache/experiment/report modules,
synthetic offline tests, fixed config, replay verifier, README and neural CI test
selection. Existing graph, embedding, dataset and model APIs remain unchanged.

Scientific question: does limited supervised ESM adaptation improve ranking on
held-out proteins/families, beyond frozen features, extra trainable head capacity
and validated structural baselines? A negative result is complete; no promotion
or M16 integration is justified by a test-score win alone.

Use all 768 original M12 variants (eight domains, four externally validated Pfam
families), exact original G1/G2/G3 train/validation/test roles. G3 adapters and heads
never train on validation/test families. Labels retain assay-specific training-only
rank transforms; only per-protein/family Spearman is primary, never pooled raw RMSE.
No native stability/function label pooling or new dataset. Same test records and
structure eligibility for every model. Separate 25% training-label regime: nested
label-blind hash selection within each original training protein; no row moves
between original roles, validation/test remain unchanged. Salt fixed in config.

Pin existing ESM-2 35M and tokenizer revision plus checkpoint/tokenizer file SHA256.
Freeze every original parameter. Add LoRA only to attention query/value projections
in the final encoder block (zero-based index 11), rank 2, alpha 4 (scale 2), dropout
0. Random A / zero B initialization; original backbone dropout stays disabled.
This is a deliberately bounded last-block adaptation, not all-layer fine-tuning.
The frozen eleven-block prefix can be cached without labels; cache identity binds
checkpoint/tokenizer hashes, sequence IDs/content, boundary and implementation.
Verify parity against the entire original forward pass before training. No adapted
hidden states are cached across fits. Both WT and mutant pass through the currently
adapted final block, with gradients through their mutation-site difference. Verify
BOS/EOS/padding and every residue token; never truncate mutation sites.

Representation: mean mutant embedding and mean mutant-minus-WT embedding at unique
mutation sites, plus count (961 features), invariant to mutation-set input order.
Common learned head: non-affine LayerNorm, Linear(961,16), ReLU, Linear(16,1).
Four model comparisons: frozen ESM ridge; frozen ESM learned head; LoRA plus that
head; exactly parameter-matched frozen-feature head with two rank-2 residual
480→2→480 branches, one for mutant-site and one for delta features. Both LoRA and
matched head add exactly 3,840 parameters to the common 15,409-parameter head.
Use identical common-head initialization. This controls nominal parameter count;
it does not prove identical effective function class or optimization geometry.

Also fresh-fit structural ridge, original M3 GNN and M14 edge-aware GNN on identical
roles/subsets using existing M14 settings (12 epochs). Structural models retain
historical architecture/selection budget; they are not parameter-matched LoRA
controls. Ridge alpha .1/10/100, validation selected. Neural ESM grid: common head
LR .001/.003, adapter/control LR one tenth head LR, AdamW decay .01, batch32,
max6 epochs, patience3, seeds0/1/2, CPU4 threads. Rank/alpha/dropout are fixed, not
searched after results. Select LR/epoch only by assigned validation macro-protein
Spearman, then evaluate test once. No test-driven retuning. A label-free timing
probe found ~.0315 sec/step for batch16 through the final block, justifying CPU.

Record all candidate traces, gradient/finite loss checks, train/validation overfit,
per-protein/family/seed results, single/multiple mutation slices, low-label curves,
trainable counts, runtime and process peak RSS, actual software/device, frozen
backbone tensor hashes before/after fits and adapter checkpoint hashes. Replay
all selected checkpoints and execute a second complete fresh fit matrix. Report
failed attempts and do not hide numerical/engineering corrections.

Test frozen weights, LoRA zero identity and gradient flow, WT gradient pathway,
masking/no truncation, mutation order, padding, cache invalidation, save/load,
held-out target perturbation, low-label role nesting and historical preservation.
Real sequence audit must compare prefix+tail against full ESM and frozen caches.
Four families from one study and unknown ESM pretraining exposure limit claims.
No function endpoints, full fine-tuning, new fusion, design or M16 consolidation.
