"""Select metadata before JSON decoding; acquire only explicitly score-free genotypes."""

import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

METADATA_FIELDS = {
    "urn",
    "title",
    "numVariants",
    "license",
    "dataUsagePolicy",
    "targetGenes",
    "datasetColumns",
    "externalLinks",
    "primaryPublicationIdentifiers",
    "doiIdentifiers",
    "methodText",
    "publishedDate",
    "modificationDate",
    "recordType",
}


def value_end(text, start):
    stack = []
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
                if not stack:
                    return index + 1
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                return index
            expected = "[" if char == "]" else "{"
            if stack.pop() != expected:
                raise ValueError("Unbalanced JSON container")
            if not stack:
                return index + 1
        elif not stack and char in ", \r\n\t":
            return index
    if stack or quoted:
        raise ValueError("Truncated JSON value")
    return len(text)


def project_metadata(text, allowed=METADATA_FIELDS):
    text = text.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("Expected metadata object")
    index = 1
    selected = {}
    seen = set()
    decoder = json.JSONDecoder()
    while index < len(text) - 1:
        while text[index].isspace():
            index += 1
        key, index = decoder.raw_decode(text, index)
        if not isinstance(key, str) or key in seen:
            raise ValueError("Unique metadata keys required")
        seen.add(key)
        while text[index].isspace():
            index += 1
        if text[index] != ":":
            raise ValueError("Missing metadata separator")
        index += 1
        while text[index].isspace():
            index += 1
        end = value_end(text, index)
        if key in allowed:
            selected[key] = json.loads(text[index:end])
        index = end
        while text[index].isspace():
            index += 1
        if text[index] == "}":
            if index != len(text) - 1:
                raise ValueError("Trailing metadata content")
            break
        if text[index] != ",":
            raise ValueError("Missing field separator")
        index += 1
    return selected


def genotype_rows(handle):
    header = handle.readline().decode("utf-8-sig").rstrip("\r\n")
    columns = next(csv.reader([header]))
    allowed = {
        "accession",
        "hgvs_nt",
        "hgvs_splice",
        "hgvs_pro",
        "mavedb.hgvs_nt",
        "mavedb.hgvs_splice",
        "mavedb.hgvs_pro",
        "mavedb.post_mapped_hgvs_g",
        "mavedb.post_mapped_hgvs_c",
        "mavedb.post_mapped_hgvs_p",
        "mavedb.post_mapped_hgvs_at_assay_level",
        "mavedb.post_mapped_vrs_id",
    }
    if not columns or len(columns) != len(set(columns)) or not set(columns) <= allowed:
        raise ValueError(f"Unverified genotype-only header: {columns}")
    if "hgvs_pro" not in columns:
        raise ValueError("Assay-frame protein genotypes required")
    # Header is validated before reading any body cell, even if the server ignores the namespace.
    reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"), fieldnames=columns)
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed genotype row width")
    return columns, rows


def acquire_genotypes(urn, output):
    if urn != "urn:mavedb:00001217-a-3":
        raise ValueError("Only preregistered abundance genotypes authorized")
    if Path(output).exists():
        raise FileExistsError(
            "Choose a new genotype output; frozen evidence must not be overwritten"
        )
    url = f"https://api.mavedb.org/api/v1/score-sets/{urn}/variants/data?start=0&limit=3000&namespaces=mavedb&drop_unused_hgvs_columns=false"
    with urllib.request.urlopen(url, timeout=60) as response:
        columns, rows = genotype_rows(response)
    if len(rows) != 2906 or len({r["accession"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate genotype inventory")
    result = {
        "url": url,
        "columns": columns,
        "rows": rows,
        "experimental_score_columns_requested": False,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return {"variants": len(rows), "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genotypes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("data/cache/sod1/source")
    metadata = project_metadata((root / "abundance-metadata.json").read_text())
    (args.output / "abundance-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.genotypes:
        print(acquire_genotypes(metadata["urn"], args.output / "genotypes.json"))
