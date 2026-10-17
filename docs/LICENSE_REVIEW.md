# M16.1 licensing review — decision reserved to the owner

**Recommendation: Apache-2.0 for newly authored ProteinMod source**, subject to
rights-holder approval and contributor ownership review. Its express contributor
patent grant and termination provisions are useful for a research ML project;
redistributors must retain notices, identify changes, include the license, and
preserve applicable NOTICE material. [Authoritative text](https://www.apache.org/licenses/LICENSE-2.0).
MIT is a reasonable simpler alternative if minimal notice obligations are preferred;
it does not contain Apache's express patent grant. [MIT text](https://choosealicense.com/licenses/mit/).
Neither choice licenses third-party datasets, checkpoints or dependencies. No LICENSE,
package license field, release tag or license selection is made by this change.
Public GitHub visibility alone does not grant a project reuse license.

## Separate provenance and distribution obligations

| Material | Recorded terms | Distribution action / remaining gate |
|---|---|---|
| ProteinMod original source | No source-code license | **FAIL/HOLD** until owner approves a license and confirms rights |
| FLIP2 amylase source | MIT as recorded in M0 provenance | Preserve upstream copyright/license with redistributed material |
| FLIP GB1 derivative used by M4/M11 | AFL-3.0; original Wu2016 measurements CC-BY-4.0 | Preserve both chains of attribution, modification notices and source references; do not relabel AFL derivative as MIT/Apache |
| ProteinGym curation | MIT curation; each experiment retains its own terms | Curation license does not replace primary measurement licenses |
| Tsuboyama native stability / M12 labels | CC-BY-4.0 | Cite source/version, link license, indicate selection/sign/coordinate transformations |
| wwPDB coordinates | CC0 | Retain scientific deposition attribution; separate annotation services may differ |
| AlphaFold DB coordinates | CC-BY-4.0 | Preserve attribution/license and modification notices; confidence is pLDDT |
| Pfam downloadable annotations | CC0 per InterPro documentation | Cite version/family accessions; this does not establish all SIFTS aggregate terms |
| SIFTS cross-resource annotations | Aggregated mappings, separate resources | **PARTIAL**: retain provenance; confirm exact endpoint/resource terms before bundled redistribution |
| Frozen ESM-2 / ProteinMPNN code and weights | MIT as recorded in pinned model provenance | Retain upstream MIT notices if redistributing; original pretraining-data rights/exposure not fully audited |
| Installed dependencies | 49 distributions inventoried in evidence | Metadata/file hashes are an inventory, not legal clearance or a vulnerability audit |

[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en) requires
appropriate credit, license identification and modification indication; database
rights can also apply. Attribution stays with derived benchmark evidence.
[Pfam/InterPro terms](https://interpro-documentation.readthedocs.io/en/latest/license.html)
apply to the specified downloadable data, not every linked database or software tool.
See the immutable [GB1 attribution](../benchmarks/m11/DATA_PROVENANCE.md).

[AFL-3.0](https://opensource.org/license/AFL-3.0) permits commercial use; its name
does not mean academic-only. Its distribution conditions include retention of
attribution, notices of modifications and reasonable effort to obtain recipient
assent; external deployment is addressed explicitly. Downstream licensing must
not contradict its terms. Exact source-availability and redistribution obligations
need review for the intended bundle; a source-only wheel and a full repository
with derived data are different distribution scopes. Apache-2.0 for our source
would not remove these obligations or establish blanket compatibility by itself.

The inventory retains unknown/missing declarations instead of guessing. Scientific
Python wheels contain bundled components with their own notices (for example BLAS
and compiler runtimes); certifi's MPL-2.0 package terms and certificate root terms
also need separate retention review. Installing dependencies is not the same as
redistributing their binary wheels. No dependency/model/data binaries are newly
committed. A release should package source separately from optional downloaded
research assets and preserve a third-party attribution bundle before distribution.

This is an engineering license audit and recommendation, not a completed legal
clearance. Owner decision, third-party bundle review, and final release review remain
open gates. The recommended license has deliberately not been installed.
