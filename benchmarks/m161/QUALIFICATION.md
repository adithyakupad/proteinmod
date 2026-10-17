# M16.1 release qualification

**Overall: FAIL / HOLD — no release tag.** Source licensing remains the owner's
unresolved decision. Scoped research verification does not qualify all deployments.
Numerical results and final verification records are in this directory; the original
[M16 failure matrix](../m16/QUALIFICATION.md) remains unchanged.

| Gate | Status | Evidence / scope |
|---|---|---|
| PR #32 basis | PASS | Exact head ea6f6ce4cf547fe34bbc729c30cf82216a24d186, 12/12 CI success; stacked, not merged |
| Historical preservation | PASS | Every prior benchmark/config byte bound in historical_hashes.json |
| Same-device deterministic forwards | PASS | 65-sequence probes repeated independently: exact CPU and MPS |
| Cross-device engineering activation screen | PARTIAL | All pooled features pass; raw residual layers 11/12 FAIL (max budget ratio 2.13738) |
| Cross-device functional equivalence | PASS | All three probes, identical selected alphas; max normalized prediction drift 6.41497e-5 |
| Legacy cross-device absolute 1e-6 | FAIL | Original failure preserved; no tolerance rewriting |
| Cold M0/M1 source/feature regeneration | PASS | Empty dedicated cache; 3,706 variants on CPU/MPS; full learning curves; 372.781 s |
| Cold real structural pipeline | PASS | 1UBQ PDB/mmCIF source downloads, mapping/contacts/RSA rebuilt |
| Cold native stability adapter | PASS | Primary source archives downloaded; 1,800 mapped singles |
| Complete M0–M15 cold reproduction | PARTIAL | Explicit per-milestone cache matrix; core paths strengthened |
| OOD physical stability export | PASS | 13,848 raw predictions invalidated, null units/results; no clipping |
| OOD physical model validity | FAIL | Native M6 models not qualified for M8 function proteins |
| Historical checkpoint/leakage replay | PASS | M5–M15 selected states/roles replayed; bookkeeping attempts retained |
| Local tests/Ruff/dependencies | PASS | 192 tests; Ruff/format and pip check pass |
| Exact-head CI | PASS | 12/12 at 6a64f93; final draft-head checks must stay green: [live PR #34](https://github.com/adithyakupad/proteinmod/pull/34/checks) |
| Source-code license | FAIL | Apache-2.0 recommended; owner decision reserved; no LICENSE added |
| Third-party redistribution clearance | PARTIAL | 49 distribution inventory; AFL/CC-BY notices and bundle terms reviewed, final clearance open |
| Universal combined checkpoint | PARTIAL | Separate evaluated paths; no all-extension inference claim |
| Final release review | FAIL | Not approved; draft/HOLD |
