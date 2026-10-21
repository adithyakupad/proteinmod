"""Tests of the metadata-only gate, using no assay data or model imports."""

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m211_checks", ROOT / "scripts/verify_m211.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


class MetadataBoundaryTests(unittest.TestCase):
    def test_rejects_nested_excluded_fields(self):
        for field in CHECKS.FORBIDDEN_KEYS:
            with self.subTest(field=field), self.assertRaises(ValueError):
                CHECKS.metadata_boundary({"metadata": [{field: "sentinel"}]})

    def test_permits_deposit_descriptors(self):
        CHECKS.metadata_boundary({"files": [{"name": "held-out.csv", "size": 10}]})

    def test_prohibits_implicit_execution_or_admission(self):
        original = CHECKS.load(ROOT / "benchmarks/m211/qualification.json")
        CHECKS.qualification_boundary(original)
        for field in ("outcome_access_authorized", "cohort_admitted", "models_executed"):
            modified = copy.deepcopy(original)
            modified[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                CHECKS.qualification_boundary(modified)

    def test_prohibits_substitution_or_silent_gate_promotion(self):
        record = CHECKS.load(ROOT / "benchmarks/m211/qualification.json")
        modified = copy.deepcopy(record)
        modified["candidates"][0]["id"] = "substitute"
        with self.assertRaises(ValueError):
            CHECKS.qualification_boundary(modified)
        modified = copy.deepcopy(record)
        modified["candidates"][0]["gates"] = {
            key: "PASS" for key in modified["candidates"][0]["gates"]
        }
        with self.assertRaises(ValueError):
            CHECKS.qualification_boundary(modified)


if __name__ == "__main__":
    unittest.main()
