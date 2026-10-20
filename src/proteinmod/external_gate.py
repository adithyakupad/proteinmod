"""Metadata-only admission gate for external assay qualification."""

INDEX_FIELDS = ("UniprotCode", "aa_seq", "core", "Nham_aa", "WT")


def project_row(row):
    """Whitelist metadata before any downstream inspection of a source row."""
    return {key: row[key] for key in INDEX_FIELDS}


def similarity_review(global_alignment, local_alignment, settings):
    """Reuse the historical thresholds; a flag requires review, not a homology claim."""
    g, local = global_alignment, local_alignment
    return (
        g["identity"] >= min(settings["cluster_identity_thresholds"])
        and min(g["coverage_a"], g["coverage_b"]) >= settings["min_both_coverage"]
    ) or (
        local["aligned"] >= settings["partial_min_aligned"]
        and local["identity"] >= settings["partial_min_identity"]
        and max(local["coverage_a"], local["coverage_b"])
        >= settings["partial_min_shorter_coverage"]
    )
