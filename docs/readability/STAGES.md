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

## 2 — message passing, attention, LoRA and unpaired supervision
Six additional files /14 functions. Node states, edge messages, mutation pooling,
sequence/graph queries, fusion weights and task errors have descriptive local names.
LoRA exposes dropout → rank projection → output projection, and the site readout
exposes mutant mean / mutant-minus-WT mean. Inlining these five single-use
intermediates restores the exact original expression AST. All registered names,
state_dict keys, parameter construction, seeds and operations are unchanged.

Targeted kernel tests:34 passed in6.94s (including LoRA gradient/checkpoint tests).
Three new style-contract tests pass. Ruff and restricted AST checks PASS.
