# M14 data and pretrained-model provenance

No new mutation-effect dataset or split was introduced. Immutable config references
are in `configs/m14.json`; `audit.json` records source, reference, graph and feature
manifest checksums. All 329 M0–M13 benchmark/config files match PR #26's Git blobs.
The stack and exact-head checks are retained in `dependencies.json`; main remains
`ad2e56478ec1c23c18ad9e46932259f865f7ff55`.

## Original M12 cohort: 768 experimental variants

ProteinGym curation is MIT licensed; original experimental attribution is retained.
Eight selected ProteinGym v1.3 Tsuboyama 2023 assay files, 96 variants each, 438 single
and 330 double mutants. Source and provenance remain
[the M12 evidence](../m12/DATA_PROVENANCE.md). Only eight members of the public
ProteinGym substitutions archive were acquired (296,298 compressed payload bytes);
no full ProteinGym collection was added. Assay-specific DMS scores are preserved;
only training labels enter per-assay rank transforms. There is no DMS-to-kcal/mol
conversion. Family assignments are externally supported Pfam 34.0 domain annotations,
not clusters invented after observing test scores. Existing sequence-similarity
and near-duplicate audits are preserved and rerun by the M12 verifier.

| Family | Domains | Observed reference structures |
|---|---|---|
| SH3 PF00018 | BBC1, MYO3 | 1TG0, 2BTT |
| Chromo PF00385 | CBX4, SR43C | 2K28, 2N88 |
| Homeodomain PF00046 | NKX31, PITX2 | 2L9R, 2L7M |
| WW PF00397 | PIN1, SAV1 | 1I6C, 2YSB |

Structure source: official RCSB mmCIF; wwPDB CC0, config-pinned SHA256. The first
selected model and existing M2 conformer policy are reused. PITX2 uses chain P;
SAV1 uses the already-validated exact reference/polymer range 7–49 to exclude tags.
All reference alignments have identity and coordinate coverage 1.0. Full-atom
mapping is checked against M2's author numbers, insertion codes and polymer indices.
No mutation is remapped based on its label. Every selected reference has complete
required interaction atoms and N/CA/C/O for the pretrained encoder; no new exclusions.

1TG0 is X-ray, not NMR; M12's already-documented display-name error is preserved in
its historical config. All primary confidence fields remain conservatively unknown
as previously configured. The other seven selected structures are NMR ensembles;
only the original first model is used. No NMR B factor is called pLDDT, and no
experimental structure receives invented AlphaFold confidence.

## Original M6 cohort: 1,800 native stability variants

Three author-validated constructs, 600 measured single mutants each: 1LP1, 1A32,
1TUD. Primary experimental data and author AlphaFold PDB archive:
[Zenodo 7992926](https://zenodo.org/records/7992926), v2_230420, CC-BY-4.0,
[Tsuboyama et al.](https://doi.org/10.1038/s41586-023-06328-6).
Figure data ZIP SHA256 `69f6d5a68ba961759879ab60b3f168b8fad1bce0b0b810ac490fe85001515890`;
exact table member SHA256 `79ff1e8676c50e9cdfefd1976a0bdcbbefdd42cc4c7f354fd5d146165f5a04aa`.
Structure archive SHA256 `70c6da80c91493b603d4319a16b1f59be0f261dfd2079f0c8710ccb49829575d`.

The existing adapter checks every WT site and source label. Source positive-stabilizing
ΔΔG is sign-negated to canonical **ΔΔG_fold kcal/mol, positive destabilizing**.
No ΔTm, fabricated pairing or OOD function-to-stability thermodynamic proxy is used.
Original S1 is 420/90/90 variants per protein; S2 holds out one complete protein,
validates on a second, and trains on the remaining third. Existing reverse-pair,
shared-structure and homolog leakage tests remain active. AF B fields carry pLDDT
only because these author files and metadata establish that meaning. These are
reference predictions, not mutant coordinates or experimental structure confidence.

## Independently pretrained ProteinMPNN

Official source: [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57), MIT.
Immutable revision `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`.

| File | SHA256 |
|---|---|
| protein_mpnn_utils.py | 74c8f9b7553422a7a0bbd705874844ee103c8926c2c96f154a87e0b824071e1b |
| vanilla_model_weights/v_48_020.pt | c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd |
| LICENSE | 82009d25ce585631f452b2b24589bdb29c559ccfefa2f200ef312ed5b501a586 |

The checkpoint is about 6.4 MiB. Frozen encoder inference, without decoder, sampling
or coordinate augmentation, was executed on all 11 structures locally. All paths,
file checksums, per-structure inference costs and parameter counts are recorded.
Code, license and weights remain cached together, uncommitted.

Pretraining task: backbone-conditioned sequence recovery, PDB assembly snapshot
August 2, 2021; see the official
[training documentation](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/training/README.md)
and [paper](https://doi.org/10.1126/science.add2187).
Exact and homologous structural pretraining exposure has not been excluded.
G3 disjointness applies to supervised experimental labels and specified families,
not the external structural pretraining corpus. No target-fitted GNN is described
as independently pretrained. Frozen ESM-2 remains available unchanged at its existing
pinned revision; it is not substituted for a pretrained structural encoder.
