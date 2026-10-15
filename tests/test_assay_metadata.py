"""M9 observable metadata, crossed assay leakage controls and bounded inference."""

import copy
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.assay_data import folds, read_assay
from proteinmod.assay_experiments import history, output_path
from proteinmod.assay_metadata import (
    MODES,
    AssayHead,
    AssayMetadata,
    Condition,
    MetadataVocabulary,
    training_shuffle,
)
from proteinmod.assay_models import evaluate, group_ridge, neural, ridge, scale_fit
from proteinmod.data import checksum
from proteinmod.range_zip import selected_members


def metadata(assay="a", family="abundance", **changes):
    return AssayMetadata.from_dict(
        {
            "assay_id": assay,
            "family": family,
            "endpoint": family,
            "technology": "VAMP_seq" if family == "abundance" else "reporter",
            "context": None,
            "substrate": "not_applicable",
            "unit": "assay_specific_score",
            "direction": "higher_score_better",
            "temperature_c": {"value": None, "status": "unavailable"},
            "ph": {"value": None, "status": "unavailable"},
            "provenance": ["https://example.org/study"],
            **changes,
        }
    )


def fixture():
    rows, meta = [], {}
    for p in range(3):
        for g in range(6):
            for a, family in enumerate(("abundance", "activity")):
                assay = f"p{p}-{a}"
                meta[assay] = metadata(assay, family).to_dict()
                rows.append(
                    {
                        "id": f"{p}-{g}-{a}",
                        "protein_id": f"p{p}",
                        "assay_id": assay,
                        "family": family,
                        "target": float(g * (100 if a else 1)),
                        "sequence": f"seq{p}-{g}",
                        "genotype_id": f"{p}-{g}",
                        "partition": "train" if g < 3 else "validation" if g < 5 else "test",
                    }
                )
    roles = folds(rows, ["p0", "p1", "p2"])[1][2]
    h = np.random.default_rng(3).normal(size=(len(rows), 5))
    return rows, roles, h, meta


def test_metadata_missingness_unseen_values_and_trust_boundary():
    a = metadata()
    assert AssayMetadata.from_dict(a.to_dict()) == a
    for raw in (
        {"value": 37, "status": "unavailable"},
        {"value": np.nan, "status": "observed"},
        {"value": None, "status": "unknown"},
    ):
        with pytest.raises(ValueError):
            Condition(**raw)
    with pytest.raises(ValueError, match="forbidden"):
        metadata(test_target_mean=42)
    vocab = MetadataVocabulary.fit([a], ["train"])
    b = metadata(
        "unseen",
        "activity",
        ph={"value": 7, "status": "observed"},
        temperature_c={"value": None, "status": "not_applicable"},
    )
    c, n = vocab.transform([b], ["test"])
    assert c[0, 0] == c[0, 7] == c[0, 8] == 0
    assert n[0].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    assert "test" not in vocab.state["vocabularies"]["protein_proxy"]
    observed = metadata(ph={"value": 6, "status": "observed"})
    trained = MetadataVocabulary.fit(
        [observed, metadata(ph={"value": 8, "status": "observed"})], ["p", "p"]
    )
    assert trained.transform([b], ["p"])[1][0, 1] == 0


def test_training_shuffle_is_whole_record_role_local_and_within_protein():
    rows, roles, _, meta = fixture()
    original = [AssayMetadata.from_dict(meta[r["assay_id"]]) for r in rows]
    proteins = [r["protein_id"] for r in rows]
    shuffled, mapping = training_shuffle(original, proteins, roles["train"], 0)
    for i in range(len(rows)):
        assert shuffled[i] == (
            original[i]
            if i not in roles["train"]
            else next(m for m in original if m.assay_id == mapping[original[i].assay_id])
        )
        if i in roles["train"]:
            assert shuffled[i].family != original[i].family
            assert shuffled[i].assay_id[:2] == original[i].assay_id[:2]
    assert training_shuffle(original, proteins, roles["train"], 0) == (shuffled, mapping)


def test_crossed_genotypes_and_held_out_proteins_do_not_cross_roles():
    rows, _, _, _ = fixture()
    for protocol, _, roles in folds(rows, ["p0", "p1", "p2"]):
        sets = [{rows[i]["genotype_id"] for i in ix} for ix in roles.values()]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        if protocol == "G2":
            assert len({rows[i]["protein_id"] for i in roles["test"]}) == 1
    rows[1]["partition"] = "test"
    with pytest.raises(ValueError, match="sequence leakage"):
        folds(rows, ["p0", "p1", "p2"])


def test_assay_local_scores_do_not_mix_native_scales():
    rows, roles, _, _ = fixture()
    ix = roles["test"]
    pred = np.array([int(rows[i]["sequence"].split("-")[-1]) for i in ix])
    result = evaluate(rows, ix, pred)
    assert result["macro"] == pytest.approx(1)
    assert len(result["assays"]) == 2


def test_test_labels_do_not_affect_training_selection_and_saved_head_replays(tmp_path):
    rows, roles, h, meta = fixture()
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -999
    p, selected, _, _ = ridge(h, rows, roles, [0.1, 10])
    q, other, _, _ = ridge(h, changed, roles, [0.1, 10])
    np.testing.assert_array_equal(p, q)
    assert selected == other
    g, evidence = group_ridge(h, rows, roles, [0.1, 10], "assay_id")
    np.testing.assert_array_equal(g[roles["test"]], p[roles["test"]])
    assert evidence["unknown_test_rows"] == len(roles["test"])
    settings = {
        "device": "cpu",
        "cpu_threads": 1,
        "scale_floor": 1e-6,
        "hidden": 4,
        "max_epochs": 4,
        "patience": 3,
        "batch_size": 4,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
    }
    parameters = []
    for mode in MODES:
        p, a, state, _ = neural(h, rows, roles, meta, settings, 0, mode)
        q, b, _, _ = neural(h, changed, roles, meta, settings, 0, mode)
        np.testing.assert_array_equal(p, q)
        assert a == b
        parameters.append(a["parameters"])
        path = tmp_path / "head.pt"
        torch.save(state, path)
        saved = torch.load(path, weights_only=True)
        vocab = MetadataVocabulary(saved["vocabulary"])
        model = AssayHead(h.shape[1], vocab, saved["hidden"])
        model.load_state_dict(saved["head"])
        c, n = vocab.transform(
            [AssayMetadata.from_dict(meta[r["assay_id"]]) for r in rows],
            [r["protein_id"] for r in rows],
        )
        x = (h - saved["representation_mean"]) / saved["representation_scale"]
        with torch.no_grad():
            replay = (
                model(torch.tensor(x, dtype=torch.float32), torch.tensor(c), torch.tensor(n), mode)
                .double()
                .numpy()
            )
        np.testing.assert_array_equal(replay[roles["test"]], p[roles["test"]])
        with pytest.raises(ValueError, match="dimensions"):
            model(
                torch.tensor(x, dtype=torch.float32), torch.tensor(c[:, :2]), torch.tensor(n), mode
            )
    assert len(set(parameters)) == 1


def test_scaler_guards_roundoff_without_reading_test_values():
    h = np.array([[1, 1], [1 + 1e-8, 3], [1e9, 1e9]])
    scaler = scale_fit(h, [0, 1])
    assert scaler.scale_.tolist() == [1, 1]
    assert scaler.mean_[1] == 2


def test_remote_zip_reads_only_pinned_member_and_rejects_ignored_range():
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("wanted.csv", b"mutation,target\nA1V,2\n")
        archive.writestr("unrelated", np.random.default_rng(0).bytes(100000))
    blob = raw.getvalue()

    def opener(request, timeout):
        response = io.BytesIO()
        response.headers = {"Content-Length": str(len(blob)), "ETag": '"pinned"'}
        response.status = 206
        if request.get_method() != "HEAD":
            assert request.get_header("If-match") == '"pinned"'
            start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
            response = io.BytesIO(blob[start : end + 1])
            response.status = 206
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}
        return response

    result, evidence = selected_members("https://example.org/archive", ["wanted.csv"], opener)
    assert result["wanted.csv"] == b"mutation,target\nA1V,2\n"
    assert evidence["downloaded_bytes"] < len(blob) / 10

    def ignored(request, timeout):
        response = opener(request, timeout)
        response.status = 200
        return response

    with pytest.raises(ValueError, match="honor"):
        selected_members("https://example.org/archive", ["wanted.csv"], ignored)


def test_mutation_parser_is_order_invariant_and_validates_source_sequence(tmp_path):
    source = tmp_path / "assay.csv"
    source.write_text("mutant,mutated_sequence,DMS_score\nC2D:A1V,VDD,1\n")
    spec = {"dataset": {"path": str(source), "sha256": checksum(source)}}
    a = read_assay(spec, "ACD")
    source.write_text("mutant,mutated_sequence,DMS_score\nA1V:C2D,VDD,1\n")
    spec["dataset"]["sha256"] = checksum(source)
    assert read_assay(spec, "ACD") == a
    source.write_text("mutant,mutated_sequence,DMS_score\nA1V,AAA,1\n")
    spec["dataset"]["sha256"] = checksum(source)
    with pytest.raises(ValueError, match="mapping"):
        read_assay(spec, "ACD")


def test_historical_artifacts_are_protected():
    assert len(history()) == 199
    for path in ("benchmarks/m8/overwritten", "benchmarks", "benchmarks/m0"):
        with pytest.raises(ValueError, match="historical"):
            output_path(path)
    config = json.loads(Path("configs/m9.json").read_text())
    assert len(config["assays"]) == 6
    assert "stability" not in config["modes"]
