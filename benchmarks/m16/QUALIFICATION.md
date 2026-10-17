# M16 release qualification — PARTIAL / HOLD

This is a research-code qualification, not an unrestricted predictor validation.
No tag/release was published. Final human review and project licensing remain open.
Evidence baseline: `7cc4dbbe9c47355d1afa46f6f46f502394e9e999`.

| Gate | Status | Actual evidence / limitation |
|---|---|---|
| Safe dependency consolidation | PASS | Ten merge commits, original heads preserved; tested base-tree equivalence; guarded merges |
| Exact dependency-head CI | PASS | 100/100 original-head jobs successful; stack-before.json |
| Exact merge-commit CI | PASS | 50/50 jobs successful, four Python versions plus neural CPU; main-ci.json |
| Historical M0–M15 artifacts/configs | PASS | Byte-for-byte checks against integrated main; no changed/deleted old evidence |
| Fresh pinned installation | PASS | Separate venv and clean main source; documented requirements/extras; pip check |
| Existing offline/neural suite | PASS | Clean main: 187 passed; Ruff and format passed |
| M16 checks | PARTIAL | Added comparison/inference/link tests; final-head CI recorded separately |
| M0–M15 fresh supervised fits | PARTIAL | Running isolated output identities; refit command record and numerical comparisons are authoritative |
| Structure reconstruction | PASS | All five real M2 source cases rebuilt from pinned cached coordinates |
| Selected checkpoint replay | PASS | M7 selected refit; M8–M15 verifiers all succeeded, including actual ESM adapted forward |
| Source-matched single/multiple inference | PASS | BBC1 P56N and D45S;N47R through real G3 LoRA; units/interval ineligibility explicit |
| Mutation/order/tensor/missingness contracts | PASS | Full suite plus actual graph/sequence replay; mismapped/unknown inputs rejected |
| Protein/family leakage controls | PASS | Existing declared role audits replay; test labels excluded from supervised fitting/selection |
| Stability units/sign | PASS | Native M6 canonical kcal/mol positive destabilizing; rank/enrichment heads kept separate |
| Multitask/assay/confidence/epistasis | PASS | Implemented tested research pathways and negative controls, not claims of improved performance |
| Predictive uncertainty | PARTIAL | Model-specific calibrated diagnostics only; one G3 calibration family gives unbounded family-block intervals |
| Universal all-extension checkpoint | PARTIAL | Extensions evaluated in separate workflows; no jointly validated universal model |
| Complete cold-source reproduction | PARTIAL | External downloads and most frozen activations reused; M1 activations independently regenerated |
| CPU versus historical MPS M1 extraction parity | FAIL | Fresh features max 2.9027e-5; refit predictions max 3.62634e-6, exceeding stated 1e-6 tolerance |
| External citation availability | PARTIAL | 48/49 HEAD requests succeeded; Science DOI 10.1126/science.add2187 returned 403 |
| Pretraining/family population claims | PARTIAL | Unknown pretraining overlap; four measured families; no universal transfer claim |
| Source-code license | FAIL | No LICENSE/COPYING or declared project license; rights-holder choice required |
| Data/model provenance/licenses | PASS | Source-specific pinned attribution; MIT/CC BY/AFL/CC0 distinctions retained |
| Credentials/large weight archives/local release paths | PASS | No pattern matches or tracked weights/arrays/archives; no private paths in release docs; limited pattern scan |
| Local links | PASS | 62 Markdown documents checked, zero unresolved local/case targets; external URLs assessed separately |
| Independent security/dependency vulnerability audit | PARTIAL | pip consistency/CI only; no complete security certification |
| Tag/release and final review | HOLD | Intentionally unpublished; final qualification PR requires review |

PASS means a stated bounded check ran successfully, not broad biological validity.
PARTIAL means an explicit scope/coverage limitation or outstanding check. FAIL identifies
a known unmet gate. Existing scientific negative results are not engineering failures.
See [technical report](../../docs/M16.md) and machine-readable execution evidence here.
