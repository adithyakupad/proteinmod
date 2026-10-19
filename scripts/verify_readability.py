"""Verify declared style-only AST changes and immutable baseline research files."""

import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/readability"


def function_node(tree, qualified_name):
    node = tree
    for name in qualified_name.split("."):
        node = next(
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name
        )
    return node


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.names.get(node.arg, node.arg)
        return self.generic_visit(node)


def inline_temporaries(node, names):
    for name in reversed(names):
        matches = [
            item
            for item in node.body
            if isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == name
        ]
        assert len(matches) == 1, name
        assignment = matches[0]
        uses = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Name) and item.id == name and isinstance(item.ctx, ast.Load)
        ]
        assert len(uses) == 1, name

        class Inline(ast.NodeTransformer):
            def visit_Name(self, item, temporary_name=name, value=assignment.value):
                return (
                    copy.deepcopy(value)
                    if item.id == temporary_name and isinstance(item.ctx, ast.Load)
                    else item
                )

        node.body.remove(assignment)
        Inline().visit(node)
    return node


def verify():
    baseline = json.loads((EVIDENCE / "BASELINE.json").read_text())
    plan = json.loads((EVIDENCE / "refactor_plan.json").read_text())
    changed = {entry["path"] for entry in plan}
    changed.add(".github/workflows/ci.yml")
    preserved = 0
    for record in baseline["files"]:
        path = ROOT / record["path"]
        if record["path"] not in changed:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], record["path"]
            preserved += 1
    for path in sorted(changed - {".github/workflows/ci.yml"}):
        original = subprocess.check_output(
            ["git", "show", baseline["head"] + ":" + path], cwd=ROOT, text=True
        )
        expected = ast.parse(original)
        actual = ast.parse((ROOT / path).read_text())
        for entry in [item for item in plan if item["path"] == path]:
            before = function_node(expected, entry["function"])
            after = function_node(actual, entry["function"])
            names = entry["renames"]
            parameters = {item.arg for item in ast.walk(before.args) if isinstance(item, ast.arg)}
            identifiers = {item.id for item in ast.walk(before) if isinstance(item, ast.Name)}
            assert set(names) <= identifiers and not set(names).intersection(parameters), entry
            assert not set(names.values()).intersection(identifiers | parameters), entry
            Rename(names).visit(before)
            inline_temporaries(after, entry.get("temporaries", []))
        assert ast.dump(expected) == ast.dump(actual), path
    result = {
        "baseline": baseline["head"],
        "preserved_baseline_files": preserved,
        "changed_python_files": len(changed) - 1,
        "declared_functions": len(plan),
        "restricted_AST_equivalence": "PASS",
        "limits": "Declared local renames and inlined temporaries; not a general semantic proof or independent scientific replication.",
    }
    print(json.dumps(result))
    return result


def historical(stdlib_only=False):
    baseline = json.loads((EVIDENCE / "BASELINE.json").read_text())
    with tempfile.TemporaryDirectory(prefix="proteinmod-frozen-") as directory:
        subprocess.run(
            ["git", "clone", "--shared", "--quiet", "--no-checkout", str(ROOT), directory],
            check=True,
        )
        subprocess.run(
            ["git", "-C", directory, "checkout", "--quiet", "--detach", baseline["head"]],
            check=True,
        )
        environment = {**os.environ, "PYTHONPATH": str(Path(directory) / "src")}
        scripts = ["audit_m16", "verify_m18", "verify_m181", "verify_m182", "verify_m183"]
        if not stdlib_only:
            scripts.extend(["verify_m16_1", "verify_m17"])
        for script in scripts:
            subprocess.run(
                [sys.executable, "scripts/" + script + ".py"],
                cwd=directory,
                env=environment,
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--stdlib-only", action="store_true")
    args = parser.parse_args()
    verify()
    if args.historical:
        historical(args.stdlib_only)
