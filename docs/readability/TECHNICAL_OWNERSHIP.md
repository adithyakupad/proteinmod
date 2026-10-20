# Ten functions to master for a machine learning engineering interview

These are learning priorities, not claims about independent authorship. Read each
function, trace a tiny numerical example on paper, then attempt the reconstruction
without the repository open. Exercises deliberately have no completed solutions.
The Python names of public arguments remain unchanged even where they are terse.

## 1. Sequence ingestion — `data.load_dataset`

Source: [src/proteinmod/data.py](../../src/proteinmod/data.py), `load_dataset(path, reference=None)`.

Inputs are a filesystem path to plain/gzipped FLIP2 CSV and an optional canonical
reference string. Outputs are `list[dict]` variant records and a QC dictionary.
Each sequence has length L; targets are finite Python floats; mutation sets are
tuples of substitution strings; partitions are train/validation/test. This is a
development-data reader, **not permission to unblind external M18 outcomes**.

Its purpose is to establish biological correspondence and official split semantics
before encoding. Validation flags override training assignments; test/validation
overlap, duplicate sequences/IDs, mixed lengths and invalid residues fail. It does
not infer WT from observed variants. Sequential validation makes the first failed
assumption visible. Missing/censored targets, WT rows and missing reference
provenance deserve deliberate handling; do not silently impute or deduplicate.

- **Unaided reconstruction:** Build a CSV reader that returns validated records and partition counts for three tiny partitions. Write down each rejection before coding.
- **Debugging:** A validation-flagged training row enters a model fit. Find where raw assignment and effective partition diverged.
- **Follow-up question:** Why can identical sequences with different IDs still leak across partitions?

## 2. Mutation-site features — `embeddings.site_features`

Source: [src/proteinmod/embeddings.py](../../src/proteinmod/embeddings.py), `site_features(residues, wt_residues, positions)`.

Inputs: mutant/WT NumPy arrays `[L,D]` and unique zero-based site positions.
Output: float32 vector `[2D+1]`; for frozen ESM-2 D=480, width=961. It concatenates
mean mutant site embeddings, mean mutant-minus-WT site embeddings and site count.
The empty mutation set returns zeros. Positions are normalized before pooling;
shape disagreement and out-of-range sites reject. One-based mutation strings must
be converted exactly once, before this call.

Symmetric means support multi-mutation order invariance while retaining contextual
representations from the **whole mutant sequence**. This is not the same as
substituting independent WT embedding rows. The module remains byte-identical
because its source hash participates in frozen embedding cache identity.

- **Unaided reconstruction:** Implement the readout for a 5-residue, 3-feature array and show how the output width follows from D.
- **Debugging:** Reversing mutation tokens changes predictions. Determine whether the error is in site pooling, sequence reconstruction or the downstream model.
- **Follow-up question:** What interaction information could a frozen contextual encoder contain even if the final pooling is permutation invariant?

## 3. Train-only scaling — `assay_models.scale_fit`

Source: [src/proteinmod/assay_models.py](../../src/proteinmod/assay_models.py), `scale_fit(x, train, floor=1e-6)`.

Input: finite NumPy feature matrix `[N,F]`, integer training indices `[N_train]` and
a scale-floor threshold. Output: fitted sklearn `StandardScaler`, with `[F]`
means/variances/scales. Only training rows fit the scaler. Features with standard
deviation below the threshold get scale **1**, not the threshold itself. The
caller applies this same transform to every partition.

This helper isolates fitted state from prediction-time transformation. Constant
features, tiny variance and nonfinite input are important cases. The specific
floor policy is part of the existing numerical definition; it was not replaced
with another normalization during refactoring. Feature-level QC may inspect test
inputs; fitting preprocessing statistics on them is a different operation.

- **Unaided reconstruction:** Fit preprocessing on one subset and transform two held-out subsets, then demonstrate that changing held-out values cannot change fitted means/scales.
- **Debugging:** A near-constant feature becomes numerically dominant. Inspect variance thresholds, float precision and which rows fitted the transform.
- **Follow-up question:** Why does centering one-hot sparse matrices have different memory implications from centering dense ESM features?

## 4. Ridge selection and representation scaling — `mechanism_experiments.probe_fit`

Source: [src/proteinmod/mechanism_experiments.py](../../src/proteinmod/mechanism_experiments.py), `probe_fit(matrix, rows, roles, alphas, seed, scaled=True, nonlinear=False)`.

Inputs: `[N,F]` features, row metadata, disjoint role-index arrays and a fixed
regularization grid. Returns `[N_test]` predictions, selection metadata and a
NumPy checkpoint dictionary (coefficients/intercept, and scaler state when used;
the optional MLP has a separate preserved schema).

For ridge, minimize the existing squared-error/L2 objective with the configured
solver and tolerance. Targets are transformed per **training assay**; validation
selects alpha by protein-macro Spearman. Scaled ESM uses a train-fitted scaler;
unscaled ESM casts the original matrix without that scaler. Structural ridge
remains a separate representation. Equal numerical alpha is not equal effective
regularization under arbitrary feature scaling. Undefined correlations, empty
roles, collinearity and grid ties require the existing surrounding contracts.

The visible sequence is target construction → optional train-only scaling →
candidate fitting → validation comparison → selected test prediction → state
export. Never refit on test labels or compare scaled/unscaled variants using
unequal model-selection budgets.

- **Unaided reconstruction:** Write a fixed-alpha ridge sweep using separate train, validation and test arrays; keep a separate record of selection and final evaluation.
- **Debugging:** Unscaled ESM unexpectedly becomes identical to scaled ESM. Inspect the actual scaler branch and saved state rather than just model labels.
- **Follow-up question:** How can feature scaling change a ridge comparison without adding predictive information?

## 5. Physical graph construction — `structure_features.contact_graph`

Source: [src/proteinmod/structure_features.py](../../src/proteinmod/structure_features.py), `contact_graph(rows, features, cutoff=8. 0, backbone_cutoff=1. 8)`.

Inputs: complete ordered reference mapping rows and per-residue feature dictionaries
of length L. Output: undirected edge-record list plus backbone-break records; it
also updates contact descriptors in `features`. Each edge retains one-based
reference endpoints, distance in Å and spatial/backbone flags. GNN loaders later
convert endpoints to zero-based compatible-node indices and bidirectional edges.

Spatial edges require CA distance ≤8 Å. Peptide edges additionally require polymer
adjacency, compatible segment and observed C/N distance ≤1. 8 Å. Missing residues,
TER breaks, insertion codes and construct/reference differences cannot be fixed
by assuming sequence adjacency. Full row order is essential: filtering rows first
would corrupt reference indexing. The masked adapter changes `has_ca` eligibility
without deleting reference rows. Coordinates describe deposited geometry, not
experimentally observed mutant rearrangements. Pairwise loops keep edge decisions
inspectable; this is not a learned graph constructor.

- **Unaided reconstruction:** Construct spatial and backbone edges for five hand-specified residues including one missing position and one chain break.
- **Debugging:** A mutation maps to the right amino acid but the wrong node. Trace reference position → polymer position → author identifier → graph lookup.
- **Follow-up question:** Why is a missing peptide connection different from the absence of a spatial contact?

## 6. Message passing — `structural_gnn.ResidueGNN.encode`

Source: [src/proteinmod/structural_gnn.py](../../src/proteinmod/structural_gnn.py), `ResidueGNN.encode(dynamic, graph=None)`.

Inputs: dynamic float tensor `[B,V,25]`; stored or supplied static `[V,S]`, directed
edge endpoints `[E]`, edge features `[E,3]` and degree `[1,V,1]`. Normal structural
S=34. Output: node states `[B,V,H]`. Initial states combine static/mutation inputs;
messages gather source states, apply learned edge gates, accumulate at targets,
normalize by degree, then apply residual updates. The separate forward readout
combines mutation sum/mean and global mean with mutation count.

Shared topology avoids copying edges for every variant. Preserve `index_add_`
direction, normalization and nonlinear/residual order. Zero-message controls,
isolated nodes, zero mutation count and node permutations matter. CPU execution
is the existing deterministic training policy; MPS aggregation cannot be assumed
bitwise equivalent. Registered parameter/buffer names stay unchanged for loading
historical checkpoints.

- **Unaided reconstruction:** Implement one message-passing step on a three-node bidirectional graph, keeping batch and node dimensions explicit.
- **Debugging:** Prediction changes under node permutation. Inspect endpoint remapping, static/dynamic permutation and graph-level aggregation.
- **Follow-up question:** What does a degree-preserving null control test that a zero-message control does not?

## 7. LoRA forward computation — `lora_model.LowRank.forward`

Source: [src/proteinmod/lora_model.py](../../src/proteinmod/lora_model.py), `LowRank.forward(x)`; read `LoRALinear.forward` alongside it.

Input `x` is a float tensor `[...,D]`; rank projection produces `[...,r]`, output
projection returns `[...,D]`. The adapter computes `(alpha/r) B A dropout(x)` and
`LoRALinear` adds it to the frozen base linear output. A is initialized with the
existing Kaiming rule; B starts at zero. M15 inserts adapters only into last-block
query/value projections; module attributes `a`, `b`, `base`, `adapter` must retain
their names because checkpoint keys depend on them.

The intermediate names expose the numerical path without reassociation or extra
random draws. Dropout position, rank, scaling, initialization order and train/eval
mode are part of the model definition. Zero initialization is not equivalent to
freezing the adapter. Validate dimensions and understand why both low-rank factors
cannot be initialized to zero indiscriminately.

- **Unaided reconstruction:** Implement an adapter wrapper for a square linear projection and list its trainable parameters and checkpoint keys.
- **Debugging:** Adapters initially preserve the base output but never learn. Inspect which factors receive gradients at initialization and after one update.
- **Follow-up question:** How do rank and alpha affect capacity and update magnitude differently?

## 8. LoRA gradient paths — `lora_model.AdaptedESM.features`

Source: [src/proteinmod/lora_model.py](../../src/proteinmod/lora_model.py), `AdaptedESM.features(prefix, extended_mask, mutation_mask)`.

Input prefix `[B+1,T,D]` includes B mutants plus one WT; extended attention mask
`[B+1,1,1,T]`; mutation mask `[B,T-2]`. Output `[B,2D+1]`. The frozen cached prefix
feeds the adapted final block, terminal special tokens are removed, and the last
WT row is expanded for mutant-minus-WT pooling. Padding/special correspondence
must agree with mutation positions; each mutant needs a valid mutation site.

Both mutant and shared WT final-block paths must remain differentiable. An
`eval()` module can still propagate gradients; `no_grad()`/`detach()` are different.
The frozen base is checked by digest, while selected adapter/head state is loaded
with exact key/shape checks. This separation permits cached expensive prefixes
without pretending the adapted final block is frozen.

- **Unaided reconstruction:** Draw the computation graph for two mutants and one WT, mark every frozen/trainable segment, then write a test asserting a WT-path gradient.
- **Debugging:** Frozen weights stay unchanged but the WT delta path has zero gradient. Locate an unintended detach/no-grad boundary or masking error.
- **Follow-up question:** Why does WT sharing save computation while still requiring gradient accumulation from multiple mutants?

## 9. Training workflow — `structural_gnn.train_gnn`

Source: [src/proteinmod/structural_gnn.py](../../src/proteinmod/structural_gnn.py), `train_gnn(graph, dynamic, targets, train, validation, test, settings, seed, layers)`.

Inputs: graph dictionaries, `[N,V,25]` mutation array, `[N]` targets, role indices
and fixed settings. Outputs: metrics/selection dictionary, `[N_test]` predictions,
and checkpoint with state_dict, target mean/std, settings, layers and seed.

State progresses through input/role validation → train-only target scaling →
seeded candidate initialization → shuffled training batches → forward/loss →
zero/backward/clip/step → validation → early stopping → selected-state restoration.
Learning rate/weight-decay candidates, strict tie comparisons, batch order and RNG
consumption must not change during a style refactor. The code deliberately does
not refit the selected model on validation. Nonfinite loss, empty/overlapping
roles and unavailable devices fail. Changing test targets may change reported
metrics but must not change selected state or test predictions.

- **Unaided reconstruction:** Implement a deterministic three-epoch loop with validation-only early stopping and restoration; keep test evaluation outside selection.
- **Debugging:** Restored predictions differ from the best epoch. Check whether the checkpoint was deep-copied and restored before evaluation.
- **Follow-up question:** Which state besides model weights is needed to resume training exactly, rather than merely reproduce selected inference?

## 10. Family-held-out evaluation — `family_data.folds`

Source: [src/proteinmod/family_data.py](../../src/proteinmod/family_data.py), `folds(rows, families)`; read its `audit` caller and `mechanism.study_roles` next.

Inputs: row dictionaries with protein/family/original-partition metadata and an
ordered family list. Output: list of `(protocol, fold_id, roles)` where each role
is a one-dimensional NumPy integer index array. G1 retains variant partitions;
G2 holds out proteins within represented families; G3 excludes entire test and
validation families from supervised training. The existing benchmark requires at
least four unambiguous families and two proteins per family.

Explicit family/protein indexing makes held-out units and validation rotation
visible. Ordering is scientific state: do not sort or reshuffle it casually.
`audit` rejects variant/family leakage and invalid G2 family coverage. Evolutionary
classification/provenance and sequence-similarity audits complement, rather than
replace, index disjointness. Per-family aggregation prevents many correlated
variants from masquerading as many independent families. ESM unsupervised exposure
is not guaranteed absent merely because supervised G3 roles are correct.

- **Unaided reconstruction:** Create four families with two proteins each, construct G1/G2/G3 roles, and write leakage assertions before training anything.
- **Debugging:** G3 looks unusually strong. Check duplicated sequences, family aliases, train-fitted transformations and any calibration/selection access to test families.
- **Follow-up question:** Why can seed variability across thousands of variants underestimate uncertainty about unseen-family performance?

## Suggested MATLAB-to-Python practice order

Start with ingestion/index tracing and mutation pooling. Then reconstruct ridge
selection and graph indexing in NumPy. Move to message passing and autograd only
after you can state every tensor shape. Finish with checkpoint restoration,
train-only preprocessing and family-level evaluation. For cross-attention, trace
`hybrid.CrossAttention.forward` next: Q `[B,Q,D]`, context `[B,K,D]`, boolean masks,
per-head weights `[B,heads,Q,K]`. For multitask learning, inspect `masked_loss` and
explain why missing NaN labels must be indexed away **before** subtraction.
