# M3 protocol: structural prediction on FLIP2 Alpha Amylase

## Scope and frozen design

M3 tests whether reference geometry helps predict the existing continuous FLIP2 Alpha Amylase functional-effect labels. It is a single-protein study; it cannot establish cross-protein generalization. PR #6 (M2) was unmerged when work began. The M3 branch is based on its final commit `b1265b6`; main and every historical M0/M1/M2 artifact are preserved. Until #6 merges, the draft PR targets the M2 branch so its review diff contains M3 only. Retarget to main after M2 merges.

The predeclared configuration is `configs/amylase_m3.toml`. The structural ridge was run before neural model implementation; its first-run evidence is saved separately. Neural settings were fixed before the GNN test results: width 16, two message-passing layers, AdamW learning rate 0.003, weight decay 0.001/0.01, batch size 64, at most 80 epochs, validation patience 12, seeds 0/1/2. A zero-message-passing neural control uses the same inputs/readout and optimization grid. The experiment compares 10% and 100% nested training subsets; it does not search architectures to beat M0/M1.

## Input contracts and reference limitations

Use only the accepted, checksum-verified M2 1UA7 experimental residue graph. Verify the M2 manifest hashes, ordered reference positions and reference sequence against the pinned assay reference. Remove graph nodes with incompatible residue identities or missing C-alpha coordinates, together with all incident edges. A variant is eligible only if **every** mutated site remains in the graph. This decision depends on sequences/coordinates, never labels. WT (empty mutation set) has zero mutation features and is eligible.

The assay reference has 425 residues. In 1UA7, reference positions 1–3 have no observed coordinates; positions 105, 126, 142, 163, 214, 254 and 356 differ from the assay reference. The induced graph has 415 nodes and 2,137 undirected edges. It remains connected, with no isolated nodes. Mapping-compatible does not mean the whole construct equals assay WT: neighboring residues, experimental ligands and missing loops can still influence the deposited geometry.

Reference geometry is fixed across all variants. It does not measure mutant relaxation, activity mechanisms, stability changes, or mutation-induced rearrangements. No AlphaFold/external structure selection is performed using fitness labels.

## Features and architecture

Graph nodes contain reference amino-acid one-hot identity, four residue properties (Kyte–Doolittle hydropathy /4.5, Biopython free-amino-acid molecular mass /200, K/R versus D/E charge category, F/W/Y aromatic category), structural descriptors and descriptor-missing flags. Structural descriptors are RSA, M2 contact and nonlocal contact counts /20, experimental C-alpha B-factor /100 and pLDDT /100. Fixed-cutoff contact count is proportional to M2 contact density (count / sphere volume at 8 Å); no redundant density column is needed. RSA and M2 degrees retain the deposited context; neighborhood pooling and GNN edges use the induced identity-compatible graph.

Each mutation adds a node indicator, substituted-residue identity and differences in the four residue properties. Unmutated dynamic node features are zero. The experimental structure supplies B-factors, not pLDDT: pLDDT is explicitly missing at all nodes, represented by zero plus a missing flag. B-factors are not treated as a prediction confidence probability. DSSP was unavailable in M2, so secondary structure is omitted rather than guessed. Unknown residues are excluded, not imputed as a canonical residue.

Handcrafted ridge features concatenate site identity/properties/descriptors, substituted identity/property differences, and one-hop neighborhood mean identities/descriptors. Aggregate site vectors by mean, sum and componentwise maximum, then add mutation count, pair count, a pair-availability indicator and mean/minimum/maximum C-alpha mutation-pair distance /30 Å. A single-site/WT variant has zero pair distances with an explicit availability indicator. Sort mutation positions before aggregation. Structural feature scaling is fitted only on the current training subset.

The GNN projects node inputs to width 16. Each layer computes source-node linear messages gated by a learned sigmoid of edge distance /8 Å, spatial-contact flag and validated-backbone flag. Sum incoming messages, divide by node degree, update from self plus neighborhood, and add a residual. Undirected M2 edges are expanded into both directions. The readout concatenates sum/mean of mutated-node states, global mean of all residue states and mutation count. A small MLP predicts the continuous effect. All mutation and node aggregations are permutation invariant (floating-point tolerance for node ordering).

The zero-layer control has the identical input projection and readout, with no learned message passing. Both models therefore have structural descriptors; the ablation isolates learned neighborhood propagation, not all structural information. Native PyTorch index aggregation suffices for one shared graph: no extra graph framework dependency is justified. Graph construction/validation and model implementation are separate modules.

## Matched evaluation and leakage control

Preserve the official partition membership; never move examples between splits. Filter all partitions by the same label-independent structural eligibility mask. Refit M0 one-hot ridge and M1 cached frozen ESM-2 mutation-site ridge on the same eligible training subsets used by M3. Their unscaled representation and validation-selected ridge protocol remain unchanged. Historical full-coverage scores are not substituted for matched-subset results.

Resulting counts: original 2,403/601/702 train/validation/test; matched 2,296/577/700. Exclusions: 107/24/2. Save all excluded IDs, substitutions and mapping reasons; save matched partition IDs and exact seeded subset IDs. Test labels are used only after model selection, for reporting; validation RMSE selects ridge alpha, neural weight decay and epoch. No refit on validation. GNN target mean/std are computed on the training subset alone. All models at a seed/fraction receive exactly the same rows.

The official split has shared substitutions across training/test although complete sequences are unique; this is interpolation within one protein rather than novel-site or cross-protein evaluation. Features include reference structure and label-free frozen ESM extraction of held-out sequences. There is no target in the feature cache contract. The tiny training test replaces test labels and confirms selection and predictions do not change.

## Reproduction

From repository root, Python 3.12:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,esm,structure,m3]' -r requirements-m3.txt
.venv/bin/proteinmod download --config configs/amylase_m1.toml
.venv/bin/proteinmod extract --config configs/amylase_m1.toml
.venv/bin/proteinmod run-m3 --config configs/amylase_m3.toml --baseline-only --output runs/m3_baseline
.venv/bin/proteinmod run-m3 --config configs/amylase_m3.toml
.venv/bin/python scripts/report_m3.py --output runs/amylase_m3
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
```

M3 reads committed, validated M2 feature/graph artifacts; it does not need to reacquire raw structures. To independently re-extract them, follow `docs/m2_protocol.md` and verify their numerical agreement. Frozen ESM extraction requires the pinned model; subsequent runs can use `extract --local-only`. M3 execution itself is offline. Torch is required for neural experiments; base-only CI still tests M0/M1/M2 and structural inputs, while a dedicated CPU job tests tiny neural/integration cases.

CPU runs work on Apple Silicon without optional native graph extensions. `gnn.device = "mps"` is supported explicitly, with unavailable-device errors and no silent fallback; CPU is the measured deterministic reference. Backend support and numerical reproducibility are version/device specific; do not expect bitwise equivalence between CPU and MPS.

## Artifacts and interpretation

Save predictions for each model/seed/fraction, validation candidates and epoch traces, train/validation/test metrics, learning-curve summaries, audit, source/data/graph hashes, dependency versions, device, runtime and ignored checkpoint hashes. Expensive M2 geometry and ESM features are reused; derived structural inputs and model weights live under ignored `data/cache/m3`. No large binary feature or checkpoint files are committed.

Seed SD reports neural initialization/training variation (and training-subset variation at 10%); it is not a confidence interval over future proteins. Full-fraction ridge seeds have identical fits. Analyze error slices by mutation burden, largest residuals, validation/test differences, and train/validation gaps. A worse GNN result is valid evidence that this small architecture adds insufficient useful signal under this design; it does not refute structural information generally. Preserve the actual unsuccessful results and do not tune on them.

No cross-attention, multimodal fusion, multitask learning, LoRA, protein-LM fine-tuning or future milestones.
