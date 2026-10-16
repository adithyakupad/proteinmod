# Data attribution and modifications

Original experiment: Wu NC, Dai L, Olson CA, Lloyd-Smith JO, Sun R (2016),
*Adaptation in protein fitness landscapes is facilitated by indirect paths*,
eLife5:e16965. [Article](https://doi.org/10.7554/eLife.16965),
[measured supplementary file1](https://doi.org/10.7554/eLife.16965.024).
Original measurements: [CC BY4.0](https://creativecommons.org/licenses/by/4.0/).

Upstream derivative: [FLIP GB1](https://github.com/J-SNACKKB/FLIP/tree/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1),
commit62cace8735f5610e2743cf06ce0f944b37fffaa6. FLIP's modified table/reference
and splits are distributed under [Academic Free License3.0](https://opensource.org/license/afl-3-0),
as stated in its [source README](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/README.md).
The notices here concern benchmark data, not a new license assignment to project code.

M11 modifications: select a label-blind bounded combinatorial cohort; crop to the
56-residue GB1 core; reconstruct all fitness from same-experiment selected/input
counts and WT normalization; floor observations at0.01 for log-fitness regression;
exclude censored/incomplete factorials from direct epistasis; add new fixed C1/C2
roles and model predictions. Published single-mutant Fitness values, which were
imported/rescaled from Olson2014, are retained for provenance and not used as labels.
No imputed or fabricated experimental outcomes were introduced.

Archive SHA256:85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff.
Extracted CSV SHA256:14cfb23e04edcaf99daa106743782b9289c1c673fcb6193ed90f8f19d33755f3.
Full source archive/arrays are cached locally and excluded from Git. Configs/m11.json
and audit.json retain source, reference, graph, model and outcome-transform contracts.

Structure: existing [1PGA](https://www.rcsb.org/structure/1PGA), wwPDB CC0;
original deposition and primary structural publication remain attributed by M4/M2
artifacts. Reference mismatchQ2T is excluded from graph nodes; all four mutagenized
sites map. B-factors remain experimental A² values.
