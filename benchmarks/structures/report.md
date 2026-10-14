# M2 real-structure verification

All extraction was local and used pinned official coordinates; no predictive model was trained.

| Structure | Cα coverage | Aligned identity | Exact compatible sites | Edges |
|---|---:|---:|---:|---:|
| amylase_m2 | 0.992941 | 0.983412 | 415 | 2195 |
| amylase_m2_pdb | 0.992941 | 0.983412 | 415 | 2195 |
| ubiquitin_m2 | 1.000000 | 1.000000 | 76 | 326 |
| ubiquitin_m2_pdb | 1.000000 | 1.000000 | 76 | 326 |
| amylase_m2_alphafold | 1.000000 | 0.990588 | 421 | 2235 |

Format consistency: {'amylase_m2': {'passed': True, 'max_ca_coordinate_difference_a': 0.0, 'max_rsa_difference': 0.0, 'identical_contact_edges': 2195}, 'ubiquitin_m2': {'passed': True, 'max_ca_coordinate_difference_a': 0.0, 'max_rsa_difference': 0.0, 'identical_contact_edges': 326}}.
Historical M0/M1 files unchanged: 19.
An unreviewed 1UA7-to-assay mapping was rejected for its seven sequence differences.
Reviewed differences are visible in the accepted audit; mutation sites at those residues stay unusable.
The AlphaFold native sequence is a related context, not the assay WT, and has four flagged differences.
DSSP is optional; inspect each audit for actual availability. Null SS annotations are not guessed.
No GNN, cross-attention, fitness score, mutant rearrangement or stability effect is inferred.

## Reports

- [amylase_m2](amylase_m2/report.md)
- [amylase_m2_pdb](amylase_m2_pdb/report.md)
- [ubiquitin_m2](ubiquitin_m2/report.md)
- [ubiquitin_m2_pdb](ubiquitin_m2_pdb/report.md)
- [amylase_m2_alphafold](amylase_m2_alphafold/report.md)
