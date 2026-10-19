import importlib.util

import pytest

spec = importlib.util.spec_from_file_location("audit_sod1", "scripts/audit_sod1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_assay_coordinates_and_reference_identity():
    assert module.decode_genotype("p.Ala2Val", "MAT") == ("A2V", None)
    assert module.decode_genotype("p.Met1Val", "MAT") == (None, "unobserved_initiator_Met")
    with pytest.raises(ValueError, match="reference mismatch"):
        module.decode_genotype("p.Cys2Val", "MAT")
    with pytest.raises(ValueError, match="reference mismatch"):
        module.decode_genotype("p.Ala9Val", "MAT")


@pytest.mark.parametrize(
    "hgvs,reason",
    [
        ("p.Ala2=", "synonymous_or_WT"),
        ("p.Ala2Ter", "stop"),
        ("p.=", "synonymous_or_WT"),
        ("p.[Ala2Val;Thr3Ala]", "unsupported_not_single_substitution"),
    ],
)
def test_nonmissense_and_unsupported_genotypes_are_explicit(hgvs, reason):
    assert module.decode_genotype(hgvs, "MAT") == (None, reason)


def test_committed_evidence_preserves_blockers_and_history():
    spec = importlib.util.spec_from_file_location("verify_sod1", "scripts/verify_sod1.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify()
