# Structural mapping: 1UA7 pdb experimental context

All variant features are reference-structure context. No mutant coordinates were predicted.

Selected author chain: A; coordinate coverage: 0.992941.
Aligned identity: 0.983412; exact compatible Cα positions: 415.
Missing/unobserved reference positions: [1, 2, 3].
Reference/construct differences (explicitly reviewed): ['S105F', 'P126Q', 'S142L', 'D163E', 'V214A', 'V254S', 'N356Q'].
Structure insertions/tags (polymer positions): [].
Chain break diagnostics: [].

Contacts: 2195 undirected edges; 421 peptide backbone links; 1354 nonlocal Cα contacts.
ASA: {'algorithm': 'Biopython Shrake-Rupley', 'probe_radius_a': 1.4, 'sphere_points': 960, 'environment': 'entire selected deposited model, occupied heavy atoms including ligands/other chains; no waters/H/D; no symmetry mates', 'rsa_table': 'Tien et al. 2013 theoretical maxima (Wilke), doi:10.1371/journal.pone.0080635', 'max_asa_a2': {'A': 129, 'C': 167, 'D': 193, 'E': 223, 'F': 240, 'G': 104, 'H': 224, 'I': 197, 'K': 236, 'L': 201, 'M': 224, 'N': 195, 'P': 159, 'Q': 225, 'R': 274, 'S': 155, 'T': 172, 'V': 174, 'W': 285, 'Y': 263}, 'rsa_clipped': False, 'secondary_structure': {'status': 'unavailable', 'reason': 'mkdssp/dssp not installed; no SS invented'}}; mean RSA over identity-compatible Cα sites: 0.18964918520657528.
Confidence semantics: **experimental_bfactor**. Experimental B-factors are not pLDDT.

| Variant | Mutation set | Usable sites | Unmapped/mismatched sites | Mean site RSA | Mean pair distance (Å) |
|---|---|---:|---:|---:|---:|
| reference | reference | 0 | 0 | None | None |
| synthetic-multi | T10A;T200A | 2 | 0 | 0.10230756295808872 | 22.22408502479472 |
| synthetic-N-terminus | L1A | 0 | 1 | None | None |
| synthetic-construct-mismatch | P126A | 0 | 1 | None | None |
| FLIP2-row-1 | T10V;T103P;I209W | 3 | 0 | 0.003512517820557897 | 22.43448523749508 |
| FLIP2-row-2 | A98V | 1 | 0 | 0.001950303966472733 | None |
| FLIP2-row-3 | R296H;P307Q;G326N | 3 | 0 | 0.04375653338021987 | 18.71573925305879 |
| FLIP2-row-4 | R133D;R164K | 2 | 0 | 0.25601764625550755 | 30.615146595457052 |
| FLIP2-row-5 | W130Y;L141W;G143S;Y252W | 4 | 0 | 0.10297005064447691 | 18.564751921299177 |

![Reference structure context](structure_context.png)

## Limits and interpretation

Sequence neighbors are nearby FASTA indices. Spatial neighbors can be distant in sequence but within the Cα distance cutoff after folding. A sequence-adjacent pair is marked backbone only with adjacent polymer indices, no TER break and a C–N distance ≤1.8 Å. Local contact density is spatial degree divided by a sphere volume of radius cutoff (residues/Å³). Missing loops are never bridged as peptide bonds. Contacts are single-chain and omit unmapped insertions; SASA includes occupied heavy atoms of the selected deposited model, excluding water/H/D, including ligands/other chains and no crystal symmetry expansion. Missing atoms can inflate SASA.

Author residue numbers need not start at one, can be negative or include insertion codes, and differ from polymer and assay reference indices. Tags, missing coordinates and engineered substitutions require explicit alignment; equally optimal alignments are rejected.

pLDDT describes local prediction confidence. Low values may reflect disorder/flexibility or insufficient information, not a measured stability change. High pLDDT does not establish reliable relative domain placement; PAE is not analyzed here. Experimental B-factors depend on displacement, occupancy and refinement and are retained in Å² without pLDDT thresholds.

Mutation sites at mismatched or missing residues produce explicit null features/distances, not guessed mappings. Symmetric aggregation sorts mutation positions, averages available site features and tracks missing counts. Reference geometry says nothing about mutant relaxation, epistasis, activity or stability improvements. No predictive baseline is trained.

Coordinate source: https://www.rcsb.org/structure/1UA7
License: wwPDB CC0; cite PDB deposition and primary publication
Reference provenance: https://github.com/florisvdf/mutation-predictability/blob/main/data/AMY_BACSU/reference.fasta
See audit.json, residues.csv/json, contacts.csv, variants.csv/json and manifest.json for full evidence.
