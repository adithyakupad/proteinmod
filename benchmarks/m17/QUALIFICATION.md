# M17 qualification matrix

**Overall: FAIL / HOLD for scientific release.** Controlled development work is
implemented; independent confirmation is BLOCKED. No merge, tag or release.

| Gate | Status | Evidence / limitation |
|---|---|---|
| Verified baseline and stack | PASS | PR #34 0908f907, 12/12 exact-head checks; #32/#34 remain draft/unmerged |
| Source license | PASS | Owner-committed MIT LICENSE at 0908f907; supersedes historical unresolved source-license gate |
| Third-party redistribution | PARTIAL | MIT code does not relicense AFL/CC-BY datasets/models or clear the full bundle |
| H1–H5 preregistration | PASS | 5e0cfac plan; 9172e9a exact outcome-free roles and eligibility before fitting |
| Independent experimental confirmation | FAIL / BLOCKED | Hydro target lineage unresolved in mixed primary archive; no external performance claim |
| Sample complexity | PASS execution / PARTIAL science | 10/25/50/100%, five seeds; H1 direction only 2/4 families |
| Matched family-shift design | PASS | Same test variants, four training proteins/128 labels; composition confound documented |
| Fixed-label family diversity | PASS execution / negative result | Two proteins/128 labels; no consistent diversity benefit |
| Capacity/topology controls | PARTIAL | Equal nominal rich-graph capacity; zero-message effective capacity differs; physical topology not consistently superior to degree null |
| Graph information audit | PASS bounded | 418 nodes/1,752 contacts, no missing expected pairs or canonical heavy atoms; reference geometry only |
| Frozen representation tests | PASS execution / PARTIAL science | Scaling changes ordering; readouts differ in information/dimension; no universal superiority |
| Calibration and uncertainty | PARTIAL | Seed spread is not calibrated uncertainty; M13-only native reanalysis; G3 family-block intervals unbounded |
| Historical preservation | PASS | All baseline benchmark/config blobs and negative results unchanged |
| Selected-state reproducibility | PASS | All 1,096 states/34,304 predictions replay exactly; one independent edge refit exact |
| Local tests/Ruff | PASS | 194 tests; formatting/lint verified before delivery |
| Exact-head CI | PENDING | Must verify live draft head before completion |
| M8 OOD physical safety | PASS handling / FAIL validity | Guarded export preserved; OOD raw values remain physically invalid; no clipping |
| Device portability | PARTIAL | M16.1 pooled/functional tests pass on tested CPU/MPS; raw-layer and strict 1e-6 failures retained |
| Complete cold reproduction | PARTIAL | M17 uses pinned cached raw graphs/ESM; no new full cold-source claim |
| Neuralink evidence/defense | PASS bounded statement | AI assistance and personal-ownership limits explicit; no neural/clinical validation claim |
| Release approval | FAIL | Independent science, scope/bundle gates and final review remain unmet; no tag |
