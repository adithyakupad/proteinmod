# M14 prespecified structural encoder comparison

Stack: PR #26 at 0a1faedad5cb9aa5f318b0dc09f19ef59d03ed07; main remains unchanged.

Scope reaches new graph annotation, model, experiment runner, offline tests, config,
report, README and CI neural test list. Existing graph/API/cache and historical
benchmark/config files remain unchanged. Output paths must not overlap history.

Primary: M12's original 768 variants / eight domains / four externally classified
families, exact original G1/G2/G3 roles. Secondary: M6's original 1,800 experimental
native kcal/mol labels, positive destabilizing, original S1/S2 roles. No label
rescaling between tasks: train-only assay ranks for M12; train-only target
standardization for M6. No calibration split or M13 split substitutions.

Use seeds 0/1/2, CPU four threads, 12 epochs, patience four, batch 32, AdamW .003,
weight decay .01, hidden 16, two layers. No test-driven architecture search.
Ridge selects alpha .1/10/100 using validation only. Neural epochs use validation
macro protein Spearman (primary) or RMSE (native). Refit all comparators fresh.

Compare structural ridge; original M3 gated GNN; edge-conditioned message MLP;
same-width distance-only, chemistry-removed, shuffled chemistry, shuffled topology,
and zero-message controls; frozen ProteinMPNN structural encoder plus site ridge.
All learned edge models keep the same parameter count, initialization and optimizer.
Zero-message retains node transforms but suppresses incoming messages.
Existing frozen ESM / bidirectional fusion APIs remain available, unchanged;
M14 isolates structural encoding rather than adding another fusion comparison.

Keep M2 node mapping, descriptors and contact/backbone topology. Annotate existing
edges from checksum-verified full heavy atoms, never from CA alone: salt-bridge
candidates (Lys/Arg versus Asp/Glu, terminal charged atoms <=4 A); hydrophobic
sidechain-carbon proximity <=4.5 A; backbone donor/acceptor proximity qualified
by O--N <=3.5 A and C--O--N >=120 degrees. The last is explicitly a **candidate**,
not a confirmed hydrogen bond: M2 removes hydrogens and protonation is unknown.
Exclude proline donors and sequential backbone neighbors from this candidate.
Missing required atoms mask the entire relevant interaction; no false absence.
No histidine charge assignments, disulfides, metal coordination or inferred mutant
rearrangements. Geometry is WT/reference geometry. Experimental B factors and
AF pLDDT retain existing M2 semantics; neither is predictive uncertainty.

Edges include 16 fixed radial bases spanning 0–12 A, original distance/spatial/
backbone flags, three interaction candidates with validity masks, and symmetric
charge/hydrophobic/aromatic category pairs. Chemistry shuffle permutes all chemical
channels jointly over undirected edges and mirrors directions. Topology shuffle
permutes node labels on endpoints only, preserving degrees/connectivity while
breaking correspondence to residues. Controls use one fixed label-blind seed.

ProteinMPNN v_48_020: independently trained inverse-folding backbone encoder,
MIT official code and checkpoint at immutable revision pinned in config. Freeze
all weights; extract encoder before sequence-conditioned decoder, no generation.
Use N/CA/C/O backbone, no coordinate augmentation. Report missing backbone masks,
checkpoint parameters, compute and pretraining PDB-overlap caveat. ProteinMPNN
pretraining is not family-held-out; G3 describes supervised task training only.
Train ridge on mean/sum/max mutation-site encoder plus substituted-AA/delta
properties and global encoder mean; unordered mutations. No supervised checkpoint
transfer from previous milestones. All models use identical evaluation records.

Record per-protein/per-family scores, seeds, costs, feature shifts, failure slices,
role IDs, fit traces, source/data hashes and retained local weights. Run fresh
second execution to check predictions exactly, including ridge and frozen encoder.
Offline tests cover missing atoms, chemical thresholds, invariances, gradients,
held-out label blindness, deterministic fitting and serialization. Preserve all
M0–M13 artifacts byte-for-byte. No M15/M16.
