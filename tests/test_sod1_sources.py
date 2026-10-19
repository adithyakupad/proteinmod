"""Never decode excluded metadata or read a CSV body before genotype-only validation."""

import importlib.util
import io
import json

import pytest

spec = importlib.util.spec_from_file_location("sod1_sources", "scripts/sod1_sources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_metadata_projection_does_not_decode_excluded_numbers(monkeypatch):
    original = json.loads
    decoded = []

    def tracked(text):
        decoded.append(text)
        if "SECRET" in text:
            raise AssertionError("Excluded metadata was decoded")
        return original(text)

    monkeypatch.setattr(module.json, "loads", tracked)
    document = (
        '{"scoreDistribution":{"SECRET":[0.2,{"escaped":"}\\"["}]},"urn":"test","numVariants":3}'
    )
    assert module.project_metadata(document) == {"urn": "test", "numVariants": 3}
    assert len(decoded) == 2


@pytest.mark.parametrize("text", ['{"urn":"a","urn":"b"}', '{"extra":[}', "[1,2]", '{"urn":'])
def test_malformed_metadata_rejected(text):
    with pytest.raises((ValueError, IndexError)):
        module.project_metadata(text)


class HeaderOnly:
    def readline(self):
        return b"accession,hgvs_pro,scores.score\n"

    def read(self, *args):
        raise AssertionError("Body must remain unread")


def test_unexpected_score_column_aborts_before_body():
    with pytest.raises(ValueError, match="Unverified genotype-only"):
        module.genotype_rows(HeaderOnly())


def test_genotype_export_validates_width_and_preserves_identifiers():
    columns, rows = module.genotype_rows(io.BytesIO(b"accession,hgvs_pro\na,p.Ala2Val\n"))
    assert columns == ["accession", "hgvs_pro"]
    assert rows == [{"accession": "a", "hgvs_pro": "p.Ala2Val"}]
    with pytest.raises(ValueError, match="width"):
        module.genotype_rows(io.BytesIO(b"accession,hgvs_pro\na,p.Ala2Val,extra\n"))
