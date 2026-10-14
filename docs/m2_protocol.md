# M2 structural pipeline: protocol and scientific contract

This milestone builds audited residue-level features before any structural learning.
No GNN, cross-attention, modality fusion, multitask, LoRA or fine-tuning is performed.
The optional structure-derived fitness regression is deliberately **not** included:
the assay-linked experimental structure is a related, shortened construct with
seven differences, so this PR establishes context/mapping quality rather than a
claim of structure-based predictive improvement. No M0/M1 test scores are reused
for feature tuning. Historical benchmark artifacts are hash-checked unchanged.

## Reference and acquisition

The assay's published provenance links `data/AMY_BACSU/1UA7.pdb` in
[Van der Flier's source repository](https://github.com/florisvdf/mutation-predictability/tree/main/data/AMY_BACSU).
This justifies checking 1UA7; its identity is not guessed from the protein name.
The independent 425-residue assay reference remains bound to its original Git blob.
[RCSB 1UA7](https://www.rcsb.org/structure/1UA7) is an acarbose-bound X-ray structure
at 2.21 Å. Alignment to the assay reference is unique, but it is **not identical WT**:
residues 1–3 are absent and there are seven differences, `S105F;P126Q;S142L;D163E;
V214A;V254S;N356Q`. The reference/construct differences are not all attributed to
engineering in the deposited structure; they are simply observed differences from
the assay reference. Configs explicitly approve these reviewed differences for
context extraction; mutation sites at those positions remain unusable.

A second protein, [1UBQ ubiquitin](https://www.rcsb.org/structure/1UBQ), uses its
traceable 76-residue deposited entity sequence as reference. This tests the adapter
on another protein, without inventing a second fitness benchmark or claiming an
independently validated genomic WT. Both structures are checked as PDB and mmCIF.

The predicted fallback example is the native full-length P00691 model,
[AF-P00691-F1](https://alphafold.ebi.ac.uk/entry/P00691), v6 identified by the
provider API, not an invented filename. It covers the whole assay span at author
positions 42–466, with four differences (`P126Q;S142L;V214A;V254S`). Its other 234
residues are outside the assay reference and are excluded from reference-node
contacts, while remaining SASA occluders. It is an alternative **related reference
context**, not a model of the exact engineered assay WT or of mutant structures.
No automatic splicing of experimental and predicted coordinates occurs.

Structure downloads are explicit. Acquisition supports an exact PDB ID, an
AlphaFold accession plus exact entry ID resolved from the official API, or a
verified local PDB/mmCIF path. Extraction is offline and requires SHA256, source,
license and confidence semantics. Byte checksums pin coordinate revisions; provider
changes fail instead of silently replacing the scientific input. Download manifests
retain URLs, acquisition time, license and AlphaFold provider metadata. wwPDB data
are [CC0](https://www.wwpdb.org/about/usage-policies); AlphaFold DB data are CC-BY-4.0.
Cite the original structure/primary publication and AlphaFold database/model papers.

## Parsing and alignment

- Parse one explicit model (zero-based `model_index`, default 0); audit model count.
  There is no ensemble averaging. Coordinates remain in Å without alignment-driven
  coordinate transforms, interpolation or missing-loop reconstruction.
- mmCIF: align the full polymer scheme, then bind its `label_seq_id` to actual
  `atom_site` author chain/number/insertion code. Author numbering is never treated
  as a FASTA index. Ambiguous label/author mappings fail.
- PDB: align full SEQRES to observed residues uniquely, then align reference to the
  declared polymer sequence. If SEQRES is absent, use observed sequence only and
  explicitly report that limitation. Repetitive missing regions can make PDB
  alignment ambiguous; use mmCIF rather than guessing.
- Global affine alignment: match +3, mismatch −3, gap-open −6, gap-extension −0.5.
  More than one optimal alignment is rejected. Tags, truncations and internal
  insertions/deletions are audited. Explicit author chain selection is preferred;
  automatic selection requires a unique highest-scoring chain. Real configs use A.
- Minimum aligned identity is 98%; minimum reference Cα coverage is 95% in the real
  configs. Thresholds are explicit, not a substitute for reviewing differences.
  Any unapproved mismatch rejects extraction and writes the mapping audit. Known
  missing coordinates, missing Cα and mismatched sites remain separate statuses.
- Standard residues plus explicit parent mappings (MSE/FME→Met, SEP→Ser, TPO→Thr,
  PTR→Tyr, HYP→Pro, CSO→Cys; mmCIF parent chemistry and PDB MODRES where provided).
  Original residue names and noncanonical flags remain visible. Unknown peptide
  residues remain X and cannot silently become a matching canonical residue.
  Canonical free amino-acid ligands are not incorrectly inserted into the chain.
- Choose one coherent residue alternate label by greatest summed occupancy; ties
  use lexical order. Shared blank atoms are retained. Disordered residue identities
  (microheterogeneity) fail. Waters/H/D and unoccupied, missing-occupancy or nonfinite
  atoms are excluded; heavy ligands/other chains remain in the SASA environment.
  Missing experimental B-fields remain null, never artificial pLDDT=0.
- PDB TER boundaries, polymer gaps, missing C/N atoms and C–N distances >1.8 Å
  prevent backbone links. mmCIF uses polymer indices plus observed geometry; it does
  not reconstruct TER information absent from that format.

See the [PDBx polymer scheme dictionary](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/pdbx_poly_seq_scheme.html).

## Features and missingness

Use an undirected, no-self-loop graph of mapped reference Cα atoms. An edge is a
spatial contact at distance ≤8 Å, or a validated peptide backbone link. Spatial,
backbone, sequential-neighbor and nonlocal (reference separation >2) flags are
separate. Local contact density is contact count / `(4πr³/3)` in residues/Å³;
this counts reference nodes, not all ligand/extra-domain atoms. Graphs are
single-chain, not biological-assembly or crystal-symmetry graphs.

SASA uses [Biopython Shrake–Rupley](https://biopython.org/docs/latest/api/Bio.PDB.SASA.html),
probe radius 1.4 Å and 960 sphere points per atom. The selected deposited model's
occupied heavy atoms, including ligands/other chains, exclude water/H/D; no symmetry
mates or hydrogens are reconstructed. Missing atoms/loops can inflate accessibility.
RSA divides residue ASA by [Tien et al. 2013 theoretical Gly-X-Gly maxima](https://doi.org/10.1371/journal.pone.0080635)
(the Wilke table), explicitly recorded in each audit. Parent maxima for modified
residues are an approximation. RSA is not clipped; values above 1 are flagged.

[Biopython DSSP](https://biopython.org/docs/latest/api/Bio.PDB.DSSP.html) runs if
`mkdssp`/`dssp` is installed and first-model coordinates are selected. SS uses the
original file and DSSP's alternate-conformer policy, separately labeled from SASA's
cleaned model. `dssp='auto'` records missing/failed dependency and leaves SS null;
`required` fails, `off` disables it. DSSP was **not installed** for the local evidence;
no actual SS assignment is claimed.

AlphaFold pLDDT is explicitly read from the coordinate B field into its own 0–100
field, with `<70` flagged. Nodes/edges are annotated, not silently removed by that
threshold. Experimental Cα B-factors remain Å² and never receive pLDDT thresholds.
Unknown semantics preserve the raw field without interpreting it as either measure.
[pLDDT is local prediction confidence](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/):
low values may mean flexibility/disorder **or** insufficient information; high
values do not establish relative domain orientation (PAE is not analyzed) or stability.

Mutation sets validate WT letters and indices, reject conflicting/duplicate/no-op
sites, and sort positions. Per-variant outputs include each site's author mapping,
spatial and ±2 sequence neighborhoods, RSA, contact degree, confidence and symmetric
Cα distance matrix. Mismatched or absent sites give null geometry/features, while
mapped/unmapped counts stay explicit. Aggregation uses site means, neighborhood
union and pairwise distance mean/min without ordering dependence. WT has an empty
set and null summary means, not fabricated zero distances between missing sites.

## Verification protocol

`python scripts/verify_structures.py` is opt-in, local-only and not part of CI. It:

1. Runs five pinned cases: 1UA7 PDB/mmCIF, 1UBQ PDB/mmCIF, and AF-P00691-F1 mmCIF.
2. Requires author/reference mapping, Cα coordinates (≤1e−6 Å difference), RSA
   (≤1e−6 difference) and contact edge records to agree between PDB/mmCIF pairs.
3. Demonstrates deliberate rejection of unreviewed 1UA7 differences.
4. Checks every historical M0/M1 artifact hash before/after.
5. Saves residue/variant CSV and JSON, contacts, PNG/SVG, audit, package/source/file
   hashes, timings, provider provenance and human-readable reports.

Tiny handcrafted PDB/mmCIF fixtures are entirely synthetic, not cropped downloads.
They include insertion 10A, a missing declared loop residue, missing Cα, a TER break,
spatial-but-not-sequential neighbors, alternate conformers and MSE. Offline tests
also cover multiple chains/models, tags, negative numbering, mapping rejection,
mutation symmetry, analytical single-atom SASA, confidence nulls and source checksums.
