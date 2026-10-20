# Third-party notices and distribution boundary

ProteinMod's original source is MIT licensed (LICENSE, owner-approved before this
closeout). This does not relicense datasets, model weights, dependency wheels,
annotation responses or archived experimental evidence. Historical license reviews
are dated records; docs/LICENSE_REVIEW.md's earlier unresolved source-license
status is superseded by LICENSE, not rewritten.

The qualified distribution candidate is a **source-code wheel** with its MIT
license and these notices. No experimental data, model checkpoints, coordinate
archives, research caches or dependency wheels are bundled. The full Git repository
is an evidence archive with separate per-item terms; it is not cleared as one
uniformly MIT data/model bundle. Do not redistribute optional assets without their
own notices and applicable terms. No tag or package publication is authorized.

| Component/material | Attribution and terms | Boundary |
|---|---|---|
| ESM-2 | Meta Platforms, Inc. and affiliates; MIT. https://github.com/facebookresearch/esm/blob/main/LICENSE | Pinned frozen checkpoint is downloaded separately; no weights bundled |
| ProteinMPNN | Copyright2022 Justas Dauparas; MIT. https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/LICENSE | Official source/encoder weights cached separately |
| ProteinGym curation | Copyright2023 OATML-Markslab, Pascal Notin, Aaron Kollasch, Daniel Ritter, Lood van Niekerk; MIT. https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/LICENSE | Curation terms do not license every primary experiment |
| Tsuboyama2023 measurements | Tsuboyama et al., Nature, doi:10.1038/s41586-023-06328-6; deposited dataset https://zenodo.org/records/7992926, CC-BY4.0 as recorded in benchmark provenance | Selected constructs/variants, rank/sign/coordinate transformations documented in M6/M12; retain attribution and modification notices |
| FLIP2 Alpha Amylase | FLIP2 repository/source and experimental attribution in benchmarks/amylase and configs/amylase.toml; MIT source terms | Preserve upstream notices and assay-specific units/splits |
| FLIP GB1 derivative | FLIP AFL3.0 plus original Wu2016 measurement CC-BY4.0 chain; benchmarks/m11/DATA_PROVENANCE.md | Not relabeled MIT; source-availability/assent obligations need intended-bundle review |
| wwPDB coordinates | Deposition authors and structure IDs in configs; archive CC0. https://www.wwpdb.org/about/usage-policies | Scientific citation retained; coordinates not bundled in wheel |
| AlphaFold coordinates | AlphaFold DB/EMBL-EBI, CC-BY4.0 per recorded provenance | Attribution and modification notices; pLDDT is not B-factor or calibrated uncertainty |
| Pfam/InterPro | Family/version-specific profile records; CC0 downloadable annotation terms per https://interpro-documentation.readthedocs.io/en/latest/license.html | Family assignment not proof of pretraining separation |
| SIFTS/RCSB cross-resource metadata | Exact endpoint provenance retained | Aggregate redistribution clearance remains PARTIAL |
| SOD1 MaveDB item | Item urn:mavedb:00001217-a-3, CC0; AJHG2025 doi:10.1016/j.ajhg.2025.08.019, article CC-BY4.0 | Genotype/header qualification only; no score workbook in distribution; blinding deviation and external HOLD retained |
| CspA/CI2 GEO files | File-specific fitness_raw/sigma_raw permission unresolved | No new download, unblinding or redistribution clearance; permission request is unsent draft |
| Python dependencies | NumPy, SciPy, scikit-learn; optional Biopython, PyTorch, Transformers, matplotlib and their transitive dependencies | Separately installed under upstream licenses; wheels and bundled BLAS/runtime licenses not redistributed by this wheel |

The three MIT components above carry the following common grant and disclaimer;
their individual copyright holders remain as listed. This notice does not assert
that ProteinMod vendors their source or weights.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
