# M12 metadata correction

The preregistered config's display name says "BBC1 experimental NMR model 0".
The authoritative RCSB/mmCIF experimental-method field, retained in `audit.json`, is
**X-RAY DIFFRACTION**, resolution **0.97 Å**, PDB **1TG0**. The other seven structures
are solution NMR ensembles. The human display name is incorrect; the PDB accession,
coordinate bytes/hash, sequence mapping and geometry are correct.

The fixed experiment explicitly set confidence semantics to `unknown` for all eight
structures. Thus both pLDDT and crystallographic B-factor input descriptors are missing/
masked for every model, including BBC1. No B-factor was interpreted as pLDDT, and this
correction does not change model inputs, selection or numerical results. Using the BBC1
B-factor in a future experiment would be a new configuration/ablation, not a relabeling
of these results. No held-out-guided feature change was made.

The original preregistered config/plan and their hashes remain unchanged as an honest
record. Reports/provenance use the verified experimental-method field instead of the
mistaken display name. One should not infer NMR from that config label.
