"""Metadata/opaque-byte qualification only; never decode a phenotype value."""

import argparse
import gzip
import hashlib
import json
import urllib.request
import zipfile
import zlib
from pathlib import Path

from proteinmod.data import checksum, write_json
from proteinmod.range_zip import RangeZipReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("benchmarks/m183/evidence"))
out = parser.parse_args().output
out.mkdir(parents=True, exist_ok=True)
url = "https://zenodo.org/api/records/11175470"
with urllib.request.urlopen(url, timeout=30) as r:
    metadata = json.load(r)
write_json(out / "Zenodo11175470.json", metadata)
source = metadata["files"][0]
assert metadata["metadata"]["license"]["id"] == "cc-by-4.0"
assert source["size"] == 1331618368 and source["checksum"] == "md5:716ab20afae843c6db145bbb72063453"


def opener(request, timeout):
    # Range-specific query avoids intermediary caches serving a different byte range.
    extra = request.get_header("Range") or "head"
    rebuilt = urllib.request.Request(
        request.full_url + "?m183=" + extra,
        headers=dict(request.header_items()),
        method=request.get_method(),
    )
    return urllib.request.urlopen(rebuilt, timeout=timeout)


reader = RangeZipReader(source["links"]["self"], opener)
assert reader.length == source["size"]
with zipfile.ZipFile(reader) as archive:
    directory = [
        {
            "name": i.filename,
            "bytes": i.file_size,
            "crc32": i.CRC,
            "compressed_bytes": i.compress_size,
        }
        for i in archive.infolist()
    ]
write_json(out / "Zenodo-directory.json", directory)
geo = Path("data/cache/m18/source/primary-scores.csv.gz")
assert checksum(geo) == "4cd224d56a0004be70f1b33f055cb2a0d2846065497c94e4fcb8bb289a167b84"
b = gzip.decompress(geo.read_bytes())
header = b.partition(b"\n")[0].decode()
assert (
    header
    == "Protein,UniprotCode,aa_seq,core,Nham_aa,WT,fitness_scaled,sigma_scaled,fitness_raw,sigma_raw"
)
write_json(
    out / "GEO-Zenodo-item-audit.json",
    {
        "GEO_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE266nnn/GSE266299/suppl/GSE266299_sparse_DTS_core_mutagenesis_scores.csv.gz",
        "GEO_sha256": checksum(geo),
        "uncompressed_bytes": len(b),
        "uncompressed_sha256": hashlib.sha256(b).hexdigest(),
        "uncompressed_crc32": zlib.crc32(b),
        "header": header,
        "archive": source,
        "archive_directory_entries": len(directory),
        "range_downloaded_bytes": reader.transferred,
        "equal_size_and_crc_members": [
            i["name"] for i in directory if i["bytes"] == len(b) and i["crc32"] == zlib.crc32(b)
        ],
        "raw_filename_members": [
            i["name"] for i in directory if "sparse_DTS_core_mutagenesis_scores" in i["name"]
        ],
        "outcome_values_decoded": False,
        "archive_whole_MD5_verified": False,
        "conclusion": "No byte-identical uncompressed GEO member exists in this directory. Derived/column-level equivalence is NOT established by filenames or shared publication. Archive MD5 is publisher-declared, not a verified full download.",
    },
)
print(
    "Archive directory entries",
    len(directory),
    "range bytes",
    reader.transferred,
    "GEO identity verified",
    False,
)
