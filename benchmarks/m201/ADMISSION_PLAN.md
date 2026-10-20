# M20.1 outcome-blind Zenodo source investigation

Base integrated main402bd8815dcd8857d416f47b1b41fdb922cf471d. M20 remains open
PR55 at10a59d0c3e94a1cf8c49791696200293801ff528; reference its immutable review,
without silently merging it or calling it part of main.

Before new acquisition: allow only record identity/version, exact archive filenames,
publisher checksums/sizes, license/access fields and archived README/code inputs.
Exclude record descriptions/statistics and notebook outputs. ZIP central directories
are filename metadata, not permission for arbitrary member extraction. Pin selected
members before reading. No full multi-GB download, phenotype row extraction or
outcome/distribution/score-derived flags. For any permitted table projection, decode
only exact whitelisted aa_seq and nt_seq cells AFTER byte-level column selection;
never decode count, WT, indel, STOP, fitness, sigma, growthrate or inclusion columns.
Recompute genotype eligibility independently. Validate header/width before cells,
stream bounded lines, compute opaque checksums/CRC without phenotype interpretation.

A source/schema gate that cannot be made safe stops that acquisition. Code/README
is inspected without notebook outputs, plots or inline empirical statistics. Original
M17 H1-H5/alphas/seeds/roles/.02 thresholds/four-family >=3/4 rule remain frozen;
M19 failure and scalar failure remain immutable. No model fit or prediction here.
Processed fitness/sigma are a potential NEW source/endpoint, not presumed identical
to GEO fitness_raw/sigma_raw. Preserve primary author citations and assay semantics.

A NEW prospective four-family final manifest is prohibited until exact rights,
column/assay/WT/genotype/structure/homology/selection/censoring gates and explicit
SOD1 owner disposition all pass. Existing separate-AI conditional review cannot
restore pristine blinding. No author email (owner declined), tag, tuning or release.
If any gate remains, report precisely what is unqualified and issue no final manifest.
