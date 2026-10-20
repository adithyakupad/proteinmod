from proteinmod.external_gate import INDEX_FIELDS, project_row, similarity_review


def test_metadata_projection_never_requests_target():
    class Blinded(dict):
        def __getitem__(self, key):
            assert key in INDEX_FIELDS
            return super().__getitem__(key)

    source = Blinded({k: k for k in INDEX_FIELDS})
    assert project_row(source) == source


def test_review_thresholds_and_short_local_alignment():
    settings = {
        "cluster_identity_thresholds": [0.4, 0.6, 0.8],
        "min_both_coverage": 0.7,
        "partial_min_aligned": 25,
        "partial_min_identity": 0.5,
        "partial_min_shorter_coverage": 0.6,
    }
    g = {"identity": 0.42, "coverage_a": 0.74, "coverage_b": 0.75}
    local = {"identity": 1, "aligned": 3, "coverage_a": 0.05, "coverage_b": 0.05}
    assert similarity_review(g, local, settings)
    g["identity"] = 0.39
    assert not similarity_review(g, local, settings)
    local.update(aligned=25, identity=0.5, coverage_a=0.6)
    assert similarity_review(g, local, settings)
