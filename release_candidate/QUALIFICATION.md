# Release-candidate qualification

Two independent decisions: **scoped code-only research candidate** and **independent
scientific confirmation**. PASS means only the named bounded check; historical
FAIL/HOLD reports are preserved. Final publication/release approval is still pending.

| Gate | Status | Evidence/limit |
|---|---|---|
| Dependency consolidation | PASS | Eight requested merge commits, exact original heads and protected blobs; dependencies.json |
| Final integrated main CI | PASS | 14fa4c37fbafda9fbc4bd7f5be482f632062c9d1, checks workflow39479533051 |
| Source MIT and code-only boundary | PASS | LICENSE, package metadata, third-party notices, wheel inventory in verification.json |
| Full repository/data/model redistribution | PARTIAL | AFL derivative/CC-BY obligations and aggregate-resource terms remain separate; no asset bundle qualified |
| Original files/scientific evidence | PASS | 926 unchanged files; seven explicit guard/doc amendments with old/new hashes; no benchmark/config exception |
| Tests/Ruff/checkpoint regressions | PASS | verification.json; clean source tests plus current neural checks |
| Clean wheel install and dependency consistency | PASS | Fresh temporary environment and pip check; wheel has code/licenses only |
| Current cached numerical replay | PASS | M17 all1096 states, M15 one192-prediction fold, M19 all15360 records, max error0 |
| Bounded single/multiple inference | PASS | Real saved M15 states; inference_examples.json; source-matched ranks only |
| OOD thermodynamic output safety | PASS | All13848 unsupported predictions/units null; safe_ood.json |
| OOD thermodynamic model validity | FAIL | M8 function proteins not qualified for physical stability prediction |
| M19 gain and scalar invariance | FAIL | Negative primary result; original scalar threshold failure retained |
| CPU/MPS exact numerical equivalence | PARTIAL | Historical failures retained; no new device sweep |
| Complete M0–M19 cold-source reproduction | PARTIAL | Historical M16.1 cold core checks; current clean code tests do not regenerate every scientific asset |
| Universal combined checkpoint / OOD intervals | FAIL | No all-extension model; no calibrated OOD interval guarantee |
| External independent four-family confirmation | FAIL/BLOCKED | GEO permissions, SOD1 independent blinding review; no unblinding or confirmation |
| Candidate-head CI and final review | PARTIAL | See live candidate PR and verification.json; publication not authorized |

**Scoped code-only research candidate: PASS for the recorded local code/legal/safety
checks; exact final-head CI must also be green at review. Publication remains
PARTIAL/pending owner approval, not automatically authorized.**
**Independent scientific confirmation: FAIL/BLOCKED.** Software verification cannot
change that scientific status. Source release need not pretend this gate passed.
