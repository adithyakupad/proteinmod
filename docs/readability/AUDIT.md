# Repository-wide readability audit

The frozen inventory covers **158 Python files**:65 maintained source modules,
52 existing scripts and41 existing test files. Every file was inventoried for
module/function structure, local bindings, public signatures, imports and source
identity sensitivity. Implementations selected for editing and the critical
scientific/cache/checkpoint boundaries were inspected in detail. This is a
repository-wide structural audit plus targeted close review, not a claim that
renaming variables constitutes an independent scientific review of every line.

The actual refactor changes **28 source modules /46 functions**. 130 original
Python files remain byte-identical. No public function names/parameters, class
attributes, model state keys, scientific schemas, configurations, old tests or
historical reports change. The complete changed/unchanged/new file inventory,
with before/after hashes and per-file disposition, is in `inventory.json`.

## Findings by subsystem

| Subsystem | Readability improvement | Invariants and retained boundaries |
|---|---|---|
| Sequence ingestion/encoding | Explicit reference/mutant residue names in `data.mutations` | Canonical alphabet, FLIP2 split validation and public contracts unchanged; ingestion/QC already has visible stages |
| Biological alignment | Polymer index, residue record, alignment quality and declared range in `map_reference` | Original unique alignment, conformer handling, mismatch errors and thresholds unchanged; lower-level parser retained |
| M18 genotype eligibility | Reference/mutant/source indices and residue eligibility | Only genotype field read; no relaxation of masks, provenance or failed admission |
| ESM extraction/site pooling | Retained: coherent contract/extractor/token-validation separation | `embeddings.py` must remain byte-identical for existing cache identity; no unnecessary new extractor |
| Residue/LoRA prefix caches | Retained: descriptive contract and checksum boundaries | Source bytes themselves are cache keys; no hash substitution or invalidation |
| Structural graph | Residue/contact records, endpoints, descriptors, mutation indices and coordinates | One-based reference versus zero-based node indices, edge order, units, missingness, feature concatenation unchanged |
| Physical descriptors | Left/right residue records and indices in contact construction | SASA, RSA, typed confidence and backbone-gap checks preserved |
| GNN/edge GNN | Node hidden states, edge messages, mutation count/sum | Layer construction, message direction, residual order, zero-message/null controls and registered names preserved |
| Cross-attention | Visible batch/node sizes, validity masks, sequence/graph states and query/readout names | Q/K/V construction, normalization, mask behavior and state_dict keys unchanged |
| LoRA | Explicit dropout/rank/output projections and two mutation-site readout terms | Last-block Q/V, initialization, scale, eval/gradient distinction and WT gradient path preserved |
| Multi-protein/multitask | Clear local targets, protein graphs/indices, task generators and task errors | Unpaired losses index observed values before subtraction; task balance and progressive schedules unchanged |
| Family/low-label evaluation | Protein/family indices, replacement roles and balanced subsets | Original family order, G1/G2/G3 roles, sampling and four-family criteria unchanged |
| M17 probes/nulls | Probe feature name, balanced roles and selected edges | Scaled/unscaled branches, alpha budget, degree-null random draws and statistical units unchanged |
| Calibration/uncertainty | Calibration/validation/group indices, quantile and scaled-feature names | Existing finite-sample quantiles, exchangeability caveats and unavailable family-conformal claims retained |
| Epistasis | Pair geometry, shuffled alleles, sequence/geometry/context feature names | Symmetry, biochemical controls, distance scaling and RNG order unchanged |
| Assay metadata/confidence/stability conditioning | Clear validation scores, quality/representation tensors, scales and batches | Assay labels not pooled as thermodynamic units; OOD physical-stability guard untouched |
| Runners | Meaningful state in seven model-training workflows plus shared GNN training | No changed initialization budget, optimizer, loss, batch/RNG consumption, patience, tie comparison or selected-state restoration |
| Historical reporting/reproduction scripts | Original scripts retained; new explicit baseline/current verification and replay entry points | Frozen source assertions remain strict; old reports and verifiers are not rewritten to conceal changed source bytes |
| Tests/CLI | Existing regression oracles/CLI retained; four style-contract checks added | All pre-existing tests byte-identical; public keyword arguments retained, even terse historical ones |

## Why some dense code remains

Some dense expressions encode consequential operation order or stable schemas.
The refactor does not turn them into a general framework, replace vectorization
with loops, or rewrite subtle parser logic merely to touch every file. Existing
helpers already separate acquisition, QC, mapping, graph construction, fitting
and evaluation. Scientific comments predominantly explain those boundaries and
were retained; there was no bulk removal of biological/reproducibility comments.

The three source-hashed cache modules are deliberately isolated from style edits.
Historical verifier/report scripts and existing tests are retained as stable
oracles. Broader algorithmic rewrites or changes to public argument names would
need a separate behavior/API migration and stronger validation. This is an
explicit limit on the style scope, not a scientific equivalence claim for an
unexecuted rewrite.

## Checkpoint and protocol inventory

All original model/experiment configuration files and their hashes are listed in
BASELINE.json and retained unchanged. No weights are committed by this refactor.

- M0/M1: original feature/scaler definitions, alpha grids, official partitions,
  nested fractions/seeds and prediction/report formats unchanged.
- M3/shared GNN: registered static/source/target/edge_features/degree buffers and
  learned layer/head names unchanged; train checkpoint wrapper retains state_dict,
  target_mean/std, settings, layers and seed.
- M5–M14: fusion, task heads, biochemical edges and confidence/uncertainty schemas
  remain the original attributes/dictionary fields. New provenance records
  correctly describe the new source revision rather than impersonating old hashes.
- M15: selected adapter/head tensor keys and base-weight digest stay intact;
  cached last-block prefixes retain their original source identity. A real
  PF00018 held-out-family checkpoint is replayed across 192 variants.
- M17: coefficient/intercept and optional scaler NPZ fields, neural state_dicts,
  H1–H5 configurations, roles and result files unchanged. All1,096 selected states
  replayed; no retraining/selection on test outcomes.
- M18: original four-family criteria and blocked admission cases remain immutable;
  no final outcome manifest/checkpoint issued, no external labels read. All384
  masked structural candidates replay with exact family audit records.

## Before / after examples

Original LoRA forward:

```python
return self.b(self.a(self.dropout(x))) * self.scale
```

Maintained forward:

```python
dropped_inputs = self.dropout(x)
rank_features = self.a(dropped_inputs)
projected_features = self.b(rank_features)
return projected_features * self.scale
```

The mathematical call order remains dropout → A → B → scale. Registered names
`a`/`b` deliberately stay unchanged for checkpoint compatibility.

Original message passing:

```python
m = message(h)[:, source] * torch.sigmoid(gate(edge_features))[None]
aggregate = torch.zeros_like(h)
aggregate.index_add_(1, target, m)
h = h + torch.relu(update(torch.cat([h, aggregate / degree], dim=-1)))
```

Maintained message passing:

```python
edge_messages = message(node_hidden)[:, source] * torch.sigmoid(gate(edge_features))[None]
aggregate = torch.zeros_like(node_hidden)
aggregate.index_add_(1, target, edge_messages)
node_hidden = node_hidden + torch.relu(update(torch.cat([node_hidden, aggregate / degree], dim=-1)))
```

Original loop state versus maintained names:

```python
# Before
x = torch.as_tensor(dynamic, dtype=torch.float32, device=device)
y = torch.as_tensor((targets - mean) / scale, dtype=torch.float32, device=device)
# After
mutation_inputs = torch.as_tensor(dynamic, dtype=torch.float32, device=device)
scaled_targets = torch.as_tensor(
    (targets - target_mean) / target_scale, dtype=torch.float32, device=device
)
```

These examples expose data flow rather than imitate MATLAB array growth or
invent authorship claims. The active source retains only necessary scientific
comments, not the explanatory before/after comments in this teaching example.
