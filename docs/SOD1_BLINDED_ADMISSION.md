# SOD1 blinded admission: initial independent-source audit

Status: **CANDIDATE / NOT ADMITTED**. This is a source-metadata review, not an executed mutation mapping or external predictive benchmark. No experimental score values were opened.

## Primary study
Axakova et al., *Landscapes of missense variant impact for human superoxide dismutase 1* (2025), DOI 10.1016/j.ajhg.2025.08.019.
MaveDB experiment: https://mavedb.org/experiments/urn:mavedb:00001217-a

- Abundance: urn:mavedb:00001217-a-3, 2,906 catalogued variants, HEK293 VAMP-seq.
- Activity: urn:mavedb:00001217-a-4, 2,910 catalogued variants, functional selection in yeast.
- Published data score sets and original analysis are linked at https://pmc.ncbi.nlm.nih.gov/articles/PMC12696502/ .
- MaveDB target sequence (154 aa, includes initiator M): `MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVVHEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ`.
- Target source is assay-specific; don't assume deposited WT coordinates or amino-acid numbering match automatically.
- MaveDB metadata states "No special data usage policies". That statement alone is not a comprehensive source-item redistribution license.

## Structural candidate
RCSB 2C9V: https://www.rcsb.org/structure/2C9V (human wild-type Cu/Zn SOD1 crystal structure, 1.07 Å; original 2006 metallation-state study). **Potential mapping only**. Verify reference-sequence/chain identity, missing initiator Met, exact observed C-alpha coverage, metal occupancy, oxidation/disulfide and construct differences in the actual coordinate file before accepting the graph. No structure mapping run in this audit.

## Independence and comparability
Existing M0–M18 supervised benchmark records reviewed during this chat do not feature SOD1; repository main code search returned no SOD1 hit. This is not an exhaustive audit of all raw data/configurations. Run the complete historical reference/profiles/homology screen before admission. SOD1 shares a biological topic with earlier personal PyMOL research, which should be documented but is not itself supervised label leakage.
Assay phenotypes are abundance and enzymatic function, **not native folding ΔΔG**. Evaluate independently using per-assay Spearman if admitted. Pre-register exactly one primary outcome (or independent co-primary policy), not chosen after score inspection.

## Remaining blockers
1. Full score-set data column and variant schema provenance using metadata and genotype-only read.
2. Coordinate-exact assay/2C9V chain mapping and masked/full structural eligibility.
3. Formal Pfam/profile and sequence-similarity checks against full historical references and external candidate families.
4. Source-item rights for analysis, prediction sharing, weights and data redistribution.
5. Frozen four-family confirmation still requires CspA/CI2 item license admission; do not unblind merely because SOD1 passes.

## Frozen state
No genotype tables downloaded in this audit; no target cells examined; no data or checkpoint hashes claimed; no local PyMOL or model run; no new experimental metrics; no changes to the M17 frozen protocol. Worktree/data acquisition and all computational checks remain for execution under issue #48.
