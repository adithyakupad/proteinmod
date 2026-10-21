# M22.4 integration scope

Base: main 402bd8815dcd8857d416f47b1b41fdb922cf471d.
Import exact PR #62 head 2e82ff477979fffa398ae778ba2bac1162e24205 and PR #63
head df58fe0dbeb0df772465533659a8c5c7b3a9fd78. Both are sibling drafts, not
prerequisites of one another. No original draft, archive or historical branch is moved.

Keep the independent numeric trainers and checkpoints separate. M22.1 supplies the
registered training-horizon fixtures; M22.3 supplies a stricter envelope, grouped/ragged
schedule and rejection contracts. No checkpoint interchangeability is claimed.

Only conftest.py overlaps. Preserve both original versions under source-routing,
and likewise preserve both original CI workflow files before adapting live workflow
verification commands. All reports, results, protocols, freezes, model source, tests
and original verification scripts remain byte-identical to their respective draft heads.
The integration verifier checks every original snapshot file at its explicit preserved
location, plus all main file bytes, rather than changing either frozen manifest.

The original standalone verification commands require their own frozen root router
and workflows. On this combined tree use scripts/verify_m224.py: it verifies stronger
whole-snapshot identity, then uses M22.1's existing numeric validator/comparator.
M22.3's runner retains its exact local replay and separate portable-fixture policy.
No frozen-checker success on a different live router is implied.

Gate: 47 combined tests; 22 optimized-Python rejection/control tests; exact 12+108 vs
120 for all 20 M22.1 pairs and all 3 M22.3 seeds; existing endpoint tolerance unchanged;
Ruff check/format; wheel package check; static historical verification; relevant CI.
CI may exclude historical tests only after an additive-scope guard checks all changes
relative to main. Historical model code must stay unchanged. No biological training,
held-out outcome acquisition, model/variant optimization, merges or tags.
