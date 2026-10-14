import csv
import gzip
import tomllib
from pathlib import Path

import pytest

from proteinmod.data import load_dataset, mutations, verified_reference, verify_file


def fixture_file(tmp_path, rows=None, compressed=False):
    rows = rows or [
        ["a", "ACDE", "1", "train", "False"],
        ["b", "CCDE", "2", "train", "True"],
        ["c", "ACDF", "3", "test", "False"],
    ]
    path = tmp_path / ("fixture.csv.gz" if compressed else "fixture.csv")
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "target", "set", "validation"])
        writer.writerows(rows)
    return path


def test_multiple_mutations_sorted_and_wt():
    assert mutations("ACDE", "CCDF") == ("A1C", "E4F")
    assert mutations("ACDE", "ACDE") == ()


@pytest.mark.parametrize(
    "reference,variant", [("ACDE", "ACD"), ("ACXE", "ACDE"), ("ACDE", "acde"), ("", "")]
)
def test_incompatible_mutations(reference, variant):
    with pytest.raises(ValueError):
        mutations(reference, variant)


def test_reference_provenance():
    assert verified_reference({}) is None
    with pytest.raises(ValueError, match="provenance"):
        verified_reference({"sequence": "ACDE"})
    with pytest.raises(ValueError, match="Git blob"):
        verified_reference({"sequence": "ACDE", "source": "fixture", "git_blob": "wrong"})
    config = tomllib.loads(Path("configs/amylase.toml").read_text())
    assert len(verified_reference(config["reference"])) == 425


@pytest.mark.parametrize("compressed", [False, True])
def test_overlap_semantics(tmp_path, compressed):
    rows, qc = load_dataset(fixture_file(tmp_path, compressed=compressed), "ACDE")
    assert [row["partition"] for row in rows] == ["train", "validation", "test"]
    assert qc["validation_flag_set_overlap_counts"] == {"train": 1}
    assert qc["mutation_burden"]["train"] == {0: 1}


@pytest.mark.parametrize(
    "column,value,message",
    [
        (0, "", "ID"),
        (0, "a", "ID"),
        (1, "ACDE", "Duplicate"),
        (1, "ACDX", "canonical"),
        (1, "ACD", "length"),
        (2, "NaN", "finite"),
        (2, "inf", "finite"),
        (2, "missing", "numeric"),
        (3, "other", "split"),
        (4, "maybe", "boolean"),
        (4, "True", "leak"),
    ],
)
def test_invalid_data(tmp_path, column, value, message):
    rows = [
        ["a", "ACDE", "1", "train", "False"],
        ["b", "CCDE", "2", "train", "True"],
        ["c", "ACDF", "3", "test", "False"],
    ]
    rows[2][column] = value
    with pytest.raises(ValueError, match=message):
        load_dataset(fixture_file(tmp_path, rows), "ACDE")


def test_explicit_validation(tmp_path):
    rows = [
        ["a", "ACDE", "1", "train", "False"],
        ["b", "CCDE", "2", "validation", "True"],
        ["c", "ACDF", "3", "test", "False"],
    ]
    assert load_dataset(fixture_file(tmp_path, rows))[1]["counts"]["validation"] == 1
    rows[1][4] = "False"
    with pytest.raises(ValueError, match="Conflicting"):
        load_dataset(fixture_file(tmp_path, rows))


def test_checksum(tmp_path):
    path = fixture_file(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        verify_file(path, {"md5": "wrong"})


def test_unverified_reference_is_not_guessed():
    with pytest.raises(ValueError, match="checksum"):
        verified_reference({"source": "unverified", "sequence": "ACDE"})


def test_download_explicit_and_checksum_atomic(tmp_path, monkeypatch):
    import hashlib
    from io import BytesIO

    from proteinmod.data import download

    path = tmp_path / "cache.csv.gz"
    content = b"official fixture bytes"
    specification = {
        "path": str(path),
        "url": "https://example.test/fixture",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr("proteinmod.data.urlopen", lambda *args, **kwargs: BytesIO(content))
    assert download(specification) == path
    assert path.read_bytes() == content
    assert path.with_suffix(".gz.manifest.json").exists()
    monkeypatch.setattr("proteinmod.data.urlopen", lambda *args, **kwargs: BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="checksum"):
        download(specification)
    assert path.read_bytes() == content


def test_missing_partition_and_columns(tmp_path):
    path = fixture_file(tmp_path, [["a", "ACDE", "1", "train", "False"]])
    with pytest.raises(ValueError, match="nonempty"):
        load_dataset(path)
    path.write_text("sequence,target\nACDE,1\n")
    with pytest.raises(ValueError, match="columns"):
        load_dataset(path)
