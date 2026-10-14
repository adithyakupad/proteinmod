# M5: mutation-query sequence–structure cross-attention

Predeclared pilot: `configs/m5.json`, inherited **unchanged** collection, split construction, exclusions and target policy from `configs/m4.json`. PR #8 is the stacked dependency. No new dataset, label-dependent selection or test-guided architecture search. The objective is a valid comparison, including negative results.

## Models and tensor contract

Frozen ESM-2 `facebook/esm2_t12_35M_UR50D` revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8` produces mutant residue tokens `[B,L,480]`. BOS, EOS and padding are removed and every residue token is checked. Existing M1/M4 pooled caches cannot reconstruct those tokens; a separate checksum-validated, label-free float32 memory-mapped cache stores them. The pretrained model is frozen and never part of the optimizer.

The existing M3 GNN exposes its residue encoder with unchanged message-passing operations and checkpoint parameter names. Its static descriptors and 25 dynamic mutation channels use the M2 audited graph. Its two message-passing layers and the learned sequence projection produce dimension 16 states. A one-to-one, zero-based graph-node → reference-token position vector excludes missing/mismatched structural sites; missing descriptors retain M2/M3 missingness indicators. All selected structures are experimental: B factors are not pLDDT. These are reference geometries, not measured mutant rearrangements.

Seven predefined models: learned sequence-only readout; structure-only readout; sigmoid scalar weighted fusion; concatenation; sequence mutation queries attending to all structural nodes; structural mutation queries attending to all sequence residues; both directions in parallel. Weighted fusion is not attention. All hybrid models have a single function regression head. Attention uses learned Q/K/V projections, two heads, scaled dot products, masked softmax and output projection. Both directions consume the original streams, followed by residual updates and LayerNorm. Mutation queries (rather than all residue queries) reduce cost to approximately `mutation_count × context_length`. This is a mutation-focused architecture, not full all-residue bidirectional updating.

Mutation queries are gathered by reference correspondence and pooled by sum and mean; global masked context means and mutation count complete the readout. Mutation-set order has no effect. WT has zero mutation queries and retains global context. Fully masked attention keys return exact zero contributions with finite gradients; padded queries return zero. Mutations at masked sites are rejected. Graph node removal happens in M2, before encoding; optional graph masks control attention/readout, not a new per-example graph-construction mechanism.

All models: dimension 16, GNN depth 2, AdamW LR .003, decay .01, clipping norm 5, batch 32, at most 20 epochs, patience 6, seeds 0/1, four CPU threads and deterministic algorithms. Earliest maximum validation macro Spearman selects the epoch. No architecture grid or test-driven retuning. Parameter budgets differ and must be reported rather than treated as perfectly matched.

## Evaluation and leakage controls

The exact archived M4 `structure_matched` train/validation/test IDs are checked before each fold. G1 preserves source variant partitions; G2 holds out an entire protein for test and another for validation. Training labels alone define assay-specific midrank-percentile targets. Validation native-label Spearman selects epochs. Test labels only enter post-selection metrics. No calibration or raw RMSE is pooled across assays; NucB is ordinal. Per-protein Spearman and unweighted protein macros are primary. Undefined correlations score zero in selection/macros, matching M4. Existing M4 baselines are immutable archived comparators on identical IDs; their full sequence-only cohort is contextual, not a matched comparison.

The pilot contains only three proteins and two seeds. Family separation and pretrained ESM family overlap are not verified. G1 and G2 have different test populations; any claimed change between protocols must either disclose that or restrict G2 predictions to the G1 test IDs. Attention entropy/peak diagnostics describe learned weights, not mechanistic explanations or causal residue interactions. Report train–validation gaps, seed variability, mapping exclusions and per-protein failures without retuning on test outcomes.

## Reproduction

Use the same Python environment as M4 (`requirements-m3.txt`, plus the pinned Transformers version recorded in M4 embedding provenance). Run from repository root. Prepare original datasets, graphs and pooled features as documented in `docs/m4_protocol.md`, then:

```sh
python -m proteinmod.hybrid_experiments prepare --config configs/m5.json
python -m proteinmod.hybrid_experiments run --config configs/m5.json
python scripts/report_m5.py --input runs/m5 --output runs/m5
ruff check .
pytest
```

Preparation uses the already cached pinned ESM checkpoint (`local_only=True`); obtain it using M1/M4 preparation first on a fresh machine. CPU is supported; ESM extraction can use available MPS. GNN training remains deterministic CPU because MPS indexed aggregation is unsupported for this reproducibility contract. Cache tokens and checkpoints under ignored `data/cache/m5`; never commit binary features. Preserve M0–M4 benchmark artifacts and configurations. No M6 heads, multitask objectives, fine-tuning, LoRA or generative design.
