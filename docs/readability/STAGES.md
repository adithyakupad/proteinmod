# Bounded stage results

## Baseline
215 tests passed in9.36s. Ruff/check formatting PASS (241 files). Original
M16/M16.1/M17/M18/M18.1/M18.2/M18.3 checks passed before any source edit.

## 1 — sequence and structural data flow
Ten source files /15 functions: mutation correspondence, source-index decoding,
reference/polymer alignment, residue contacts, graph tensor construction,
interaction features and family folds. Only declared internal bindings changed;
public parameters, strings, attributes, control flow and numerical expressions
are AST-identical after reversing the renames. No blinding/provenance gate changed.

Targeted tests:92 passed in22.43s. Ruff PASS. Restricted AST check PASS. Original
historical source/artifact guards PASS in the exact frozen Git revision, plus
current protected-file hashes PASS. Old source-hash verifiers are intentionally
not edited; CI now distinguishes frozen historical verification from current
refactor verification. No historical scientific report or config was rewritten.
