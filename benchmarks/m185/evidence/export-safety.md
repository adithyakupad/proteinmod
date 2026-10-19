# Genotype export verification before request

Official MaveDB API source revision894b63294bdb29e0ac65ac7e99854df658ab2138:
`src/mavedb/lib/csv/columns.py` plan_csv_columns constructs CORE plus only the
explicitly requested namespaces. `specs.py` defines CORE as accession and assay
hgvs_nt/hgvs_splice/hgvs_pro only. REFERENCE_HGVS (`mavedb`) resolves mapped HGVS
and VRS identifiers. SCORES/SCORES_CUSTOM and COUNTS are distinct RowSource
namespaces. `routers/score_sets.py` forwards the explicit namespace list and
namespaced=True, without adding score namespaces. No CSV comment prelude is added.
The published OpenAPI specifies that naming namespaces replaces defaults.

Request only namespaces=mavedb, start0, limit3000, preserving all HGVS columns.
The response boundary checks a finite explicit genotype-header allowlist BEFORE
reading the first body cell. Unexpected score/count/clinical columns cause an
immediate error; no generic full-body download or score parsing is used.
The unfiltered `/variants` view is forbidden because LeanVariant contains score.

Source files are locally hash-pinned in the acquisition manifest. Source revision
is an audit of published code, not proof that a live server has identical code;
the runtime header boundary is therefore mandatory. Header-poison and metadata
pre-decode exclusion tests pass before acquisition. No rejected bulk CSV command
was executed. The new request is strictly narrower and validates the response
before reading a body, addressing the approval review's stated concern.
