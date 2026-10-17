# M15 data, sequence encoder and split provenance

No new assay or protein collection was selected. All 354 M0–M14 benchmark/config
files match the Git blobs at PR #28 head
`6ce820fce2b8f287b79ff92a1a576c5e761f6cf3` byte-for-byte. This branch preserves main
and the entire open draft stack. `dependencies.json` records PR #28's ten successful
exact-head checks. `historical_hashes.json` and `artifact_hashes.json` bind evidence.

## Existing M12 experimental records

The exact selected ProteinGym v1.3/Tsuboyama 2023 DMS score files, reference sequences,
source row numbers, Pfam 34.0 annotations, coordinates, exclusions and split salt
are inherited from checksum-bound `configs/m12.json` and
[M12 provenance](../m12/DATA_PROVENANCE.md). ProteinGym curation is MIT licensed;
original experimental attribution and licensing are retained. Source publication:
[Tsuboyama et al., Nature 2023](https://doi.org/10.1038/s41586-023-06328-6).

The eight assays contain 768 selected experimental variants, 438 single and 330
double mutations, 96 variants per protein. This is the same bounded stability-effect
ranking task, not a new functional experiment or physical ΔΔG conversion.

| Family | Domains | Reference structures |
|---|---|---|
| SH3 PF00018 | BBC1, MYO3 | 1TG0, 2BTT |
| Chromo PF00385 | CBX4, SR43C | 2K28, 2N88 |
| Homeodomain PF00046 | NKX31, PITX2 | 2L9R, 2L7M |
| WW PF00397 | PIN1, SAV1 | 1I6C, 2YSB |

Original structure mapping, complete eligibility and M14 biochemical edges are
reused. There are no additional exclusions or geometry changes. PITX2 retains
chain P; SAV1 retains its exact tag-disambiguating reference range. The M12 display
name error for X-ray 1TG0 is preserved in historical files and remains documented
in [M12 errata](../m12/ERRATA.md). B factors, unknown confidence and AF pLDDT are not
reinterpreted or conflated with predictive uncertainty.

G1 train/validation/test is 512/128/128. G2 uses 576/96/96 per held-out protein;
G3 uses 384/192/192 with complete families held out from supervised training.
Every original test ID remains unchanged for all models. The low-label condition
selects 25% of original training rows independently within each training protein,
using a fixed label-blind ID hash. The unused original training rows enter no
supervised loss; validation/test roles never change. `audit.json` records both
original and effective role IDs. Existing family, sequence similarity, duplicate
and structure-reference audits remain active.

Training targets are empirical midrank percentiles **within each training assay**.
Validation/test raw experimental scores enter Spearman calculation only, not target
normalization or gradients. No cross-assay pooled RMSE, fabricated paired function
labels or OOD stability-to-function thermodynamic input is introduced.

## Pinned original ESM-2 checkpoint and tokenizer

Model: [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D),
immutable revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`, the same model used
by the existing frozen baselines. Original ESM repository/model license: MIT;
see [official ESM](https://github.com/facebookresearch/esm) and the model card.

| Cached file | SHA256 |
|---|---|
| model.safetensors | e35647818e0e064351d4531ed480d225a002567b4b2b93ad3a9246d753150fc0 |
| config.json | ae07364efa51e0fa1d81da3fe608f6d948abdbbb69785ba6e1df379a458e58ee |
| special_tokens_map.json | 3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1 |
| tokenizer_config.json | 7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad |
| vocab.txt | 0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03 |

The frozen first eleven layers may be evaluated on all unlabelled sequences.
No held-out experimental label enters this cache. Its manifest hashes bind ordered
IDs, complete sequence content including WT, all checkpoint/tokenizer hashes,
implementation, settings and prefix boundary. Full original forward and cached
prefix+tail parity are checked on every sequence, including residue token IDs.
Adapted final-layer outputs are recomputed per fit/batch and never reused from a
frozen feature cache. LoRA begins from fresh adapters for every role/seed/LR candidate;
no supervised state transfers between proteins, families or low-label conditions.

Only last-block query/value adapters and task heads are trained. The original PLM
file hash and all original model tensor digests remain fixed. Adapter-only files
and raw/prefix caches remain ignored, not committed. `metrics.json` binds each
selected checkpoint path/hash and validation trace; the verifier reloads them onto
the exact frozen checkpoint, rather than assuming a checkpoint name implies identity.

Exact/homologous family exposure in ESM's external sequence pretraining has not been
excluded. Family disjointness refers to **supervised experimental adaptation**, not
the historical PLM pretraining corpus. Four families from one experiment cannot
establish broad functional, unseen-fold or unseen-pretraining-family generalization.
