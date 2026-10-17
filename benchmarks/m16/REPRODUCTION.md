# Actual M16 clean-source reproduction

Pinned dependency install and unchanged model source at `7cc4dbbe9c47355d1afa46f6f46f502394e9e999`.
Each milestone training command executed afresh; acquisition/frozen features, fixed transfer encoders and explicit baseline replays are reused as declared by its canonical protocol. M2 rebuilt all five real structures.
Three initial provenance-guard failures were repaired without changing canonical scientific configs.

| Milestone | Wall seconds | Compared prediction CSV records | Max absolute difference |
|---|---:|---:|---:|
| M0 | 7.482 | 702 | 0 |
| M1 | 20.346 | 702 | 0 |
| M2 | 12.284 | structure audit | PDB/mmCIF parity passed |
| M3 | 256.616 | 33600 | 0 |
| M4 | 27.863 | 18700 | 0 |
| M5 | 214.081 | 27902 | 0 |
| M6 | 6.032 | 18900 | 0 |
| M7 | 494.391 | 73134 | 0 |
| M8 | 16.474 | 161433 | 0 |
| M9 | 7.048 | 28566 | 0 |
| M10 | 19.893 | 29808 | 0 |
| M11 | 16.349 | 47736 | 0 |
| M12 | 89.244 | 23296 | 0 |
| M13 | 145.394 | 70620 | 0 |
| M14 | 222.956 | 85882 | 0 |
| M15 | 359.295 | 56576 | 0 |

**677,557 CSV records compare exactly**, including calibration records, baseline repetitions and selected aliases. These are not independent experimental samples. Scientific JSON metrics/selections/traces also compare exactly after excluding timing and transport/checkpoint-serialization identities.

The separate fresh CPU M1 extraction differs from historical MPS and fails the stated 1e-6 prediction tolerance; see fresh-m1-extraction.json. No historical outputs were replaced.

Initial failed commands and their repairs remain in refit-attempts.json / refit-repairs.json. A reporting attempt also required the M9 posthoc shift audit first; the runner now includes that dependency.

M12/M13 execute in fresh canonical clones with unchanged output paths, empty supervised checkpoint directories, and copied pinned acquisition/activation caches. Their canonical clones remain source-clean.
