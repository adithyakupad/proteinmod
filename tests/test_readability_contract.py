"""The style contract preserves operations, attributes and immutable research evidence."""

import ast
import importlib.util

import pytest

specification = importlib.util.spec_from_file_location(
    "readability", "scripts/verify_readability.py"
)
readability = importlib.util.module_from_spec(specification)
specification.loader.exec_module(readability)


def test_declared_refactor_and_protected_files():
    assert readability.verify()["restricted_AST_equivalence"] == "PASS"


def test_names_do_not_change_registered_attributes_or_schema_fields():
    original = ast.parse('def forward(self, x):\n    h = self.h(x)\n    return {"h": h}\n')
    changed = readability.Rename({"h": "hidden"}).visit(original)
    assert changed.body[0].args.args[-1].arg == "x"
    assert changed.body[0].body[0].value.func.attr == "h"
    assert changed.body[0].body[-1].value.keys[0].value == "h"


def test_intermediates_inline_to_original_expression():
    original = ast.parse("def f(x):\n    return b(a(x))\n").body[0]
    expanded = ast.parse(
        "def f(x):\n    reduced = a(x)\n    projected = b(reduced)\n    return projected\n"
    ).body[0]
    normalized = readability.inline_temporaries(expanded, ["reduced", "projected"])
    assert ast.dump(original) == ast.dump(normalized)
    repeated = ast.parse("def f(x):\n    value = a(x)\n    return value + value\n").body[0]
    with pytest.raises(AssertionError):
        readability.inline_temporaries(repeated, ["value"])
