"""Qualification comparisons reject row changes and misleading numeric matches."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("qualification", Path("scripts/qualify_m16.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_comparison(tmp_path, monkeypatch):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("id,prediction\na,0.5\n")
    b.write_text("id,prediction\na,0.5000001\n")
    assert module.compare_csv(a, b)["rows"] == 1
    for text in ("id,prediction\nb,0.5\n", "id,prediction\na,nan\n", "id,prediction\na,0.6\n"):
        b.write_text(text)
        with pytest.raises(AssertionError):
            module.compare_csv(a, b)
    a.write_text("id,prediction\n1,0.5\n")
    b.write_text("id,prediction\n1.0000001,0.5\n")
    with pytest.raises(AssertionError):
        module.compare_csv(a, b)
    original = {"seconds": 1, "state_path": "old", "selection": {"alpha": 1, "score": 0.5}}
    repeat = {**original, "seconds": 2, "state_path": "new"}
    assert module.scientific(original) == module.scientific(repeat)
    assert module.scientific(original) != module.scientific({**repeat, "selection": {"alpha": 2}})
    monkeypatch.chdir(tmp_path)
    Path("runs/m16-m0").mkdir(parents=True)
    with pytest.raises(ValueError, match="preserve"):
        module.commands("refit", tmp_path)
    locked_spec = importlib.util.spec_from_file_location(
        "locked", Path(__file__).resolve().parents[1] / "scripts/refit_locked_m16.py"
    )
    locked = importlib.util.module_from_spec(locked_spec)
    locked_spec.loader.exec_module(locked)
    with pytest.raises(ValueError, match="Only M12/M13"):
        locked.refit(11)
    Path("runs/m16-m12").mkdir()
    with pytest.raises(ValueError, match="preserve"):
        locked.refit(12)


def test_bounded_variant_lookup():
    pytest.importorskip("torch")
    inference_spec = importlib.util.spec_from_file_location("inference", "scripts/infer_m16.py")
    inference = importlib.util.module_from_spec(inference_spec)
    inference_spec.loader.exec_module(inference)
    rows = [{"protein_id": "A", "mutations": ["A1C", "D3E"]}]
    assert inference.variant_index(rows, "A", "D3E;A1C") == 0
    for protein, mutations in (("B", "A1C;D3E"), ("A", "A1C;A1C"), ("A", "A1C")):
        with pytest.raises(ValueError):
            inference.variant_index(rows, protein, mutations)


def test_release_links_and_secret_patterns(tmp_path):
    audit_spec = importlib.util.spec_from_file_location("audit", "scripts/audit_m16.py")
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    (tmp_path / "Exact.md").write_text("evidence")
    doc = tmp_path / "README.md"
    doc.write_text("[valid](Exact.md#result) [case](exact.md) [absent](absent.md)")
    assert audit.local_links(doc) == ["exact.md", "absent.md"]
    assert audit.SECRET.search(b"ghp_" + b"x" * 36)
    assert not audit.SECRET.search(b"a normal scientific report")
    assert not audit.SECRET.search(b"MS" + b"AKIA" + b"A" * 50)
