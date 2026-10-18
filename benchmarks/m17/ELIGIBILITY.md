# External outcome gate (before any external fitting)

**Independent confirmation BLOCKED. No candidate outcome labels have been inspected,
no external test performance computed.** The downloaded Hydro file was projected to
sequence/set/validation only; its target column remains unused. It is not an accepted
independent cohort and no independent-confirmation score will be invented.

| Candidate | Primary experiment / endpoint | Decision and reason |
|---|---|---|
| Existing M12 / additional Tsuboyama ProteinGym assays | Tsuboyama2023 proteolysis | DEVELOPMENT only; same primary measurement generation |
| FLIP2 Hydrophobic Core | README: Escobedo2024, Zenodo11175470, CC-BY4 | BLOCKED label-level lineage: archive contains both lab DiMSUM/CD/NMR data and `Fig1/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (697,658,024 bytes). Publication attribution alone cannot prove which produced the FLIP target column. Exact preprocessing/reference provenance not located in public FLIP main/release trees. Do not assert it is independent or assert it is definitely reused |
| ProteinGym Nutschel2020 | Independent BsLipA thermal inactivation T50, ACS doi10.1021/acs.jcim.9b00954 | DEFER native-stability confirmation: T50 is not ΔΔG, only one protein/family, primary redistribution terms need review; possible future rank-only endpoint-shift study |
| Nisthal2019 | Independent GB1 chemical denaturation, doi10.1073/pnas.1903888116 | DEFER unseen-family confirmation: GB1 was already used M4/M11, single protein/family; independent experiments do not mean independent protein lineage |
| Jacquier2013 / McLaughlin2012 / Starita2013 | Antibiotic resistance / PDZ binding / E3 ligase activity | DEFER stability mechanism confirmation: different functional endpoints; licensing, domain mapping and family matching would need separate qualification |

Correction to the initial preregistration's categorical phrase “this is NOT
Tsuboyama”: that was defensible at publication identity, but not at label lineage.
This audit supersedes that phrase BEFORE testing. The initial plan is retained
unchanged as an auditable negative qualification result. The exact source is unresolved,
not proven dependent. It would be misleading to substitute new folds of M12 for it.

Hydro first-hand README bytes MD5 eb762cd877939c4cee891bdc1c159650; data bytes MD5
 a13b01d71faf2e76dd91dc57b7311972 from Zenodo18433203 metadata. Primary archive directory
was read via bounded HTTP ranges, not a full 1.3GB download. No measured targets were
needed to identify the unresolved mixed-source lineage. Frozen PLM/MPNN pretraining
overlap is additionally unknown, even for an eventual experimentally independent cohort.

References: https://zenodo.org/records/18433203/files/hydro/README.md
https://zenodo.org/records/11175470
https://pubs.acs.org/doi/10.1021/acs.jcim.9b00954
https://pmc.ncbi.nlm.nih.gov/articles/PMC6697890/

Only the frozen, outcome-free **development** split manifest is approved for fitting.
No external label unblinding occurs in this study. Future independent confirmation
needs a new precommitted manifest after a resolved primary-label provenance chain.
