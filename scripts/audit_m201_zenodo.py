"""Blinded metadata/schema projection. Never JSON-decode unknown record fields."""

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sod1_sources import value_end

from proteinmod.range_zip import RangeZipReader


def raw_fields(text, allowed):
    text = text.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("Object required")
    index = 1
    result = {}
    seen = set()
    decoder = json.JSONDecoder()
    while index < len(text) - 1:
        while text[index].isspace():
            index += 1
        key, index = decoder.raw_decode(text, index)
        if key in seen or not isinstance(key, str):
            raise ValueError("Duplicate or non-string key")
        seen.add(key)
        while text[index].isspace():
            index += 1
        if text[index] != ":":
            raise ValueError("Missing colon")
        index += 1
        while text[index].isspace():
            index += 1
        end = value_end(text, index)
        if key in allowed:
            result[key] = text[index:end]
        index = end
        while text[index].isspace():
            index += 1
        if text[index] == "}":
            if index != len(text) - 1:
                raise ValueError("Trailing content")
            return result
        if text[index] != ",":
            raise ValueError("Missing comma")
        index += 1
    return result


def array_values(text):
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("Array required")
    index = 1
    while index < len(text) - 1:
        while text[index].isspace() or text[index] == ",":
            index += 1
        if text[index] == "]":
            break
        end = value_end(text, index)
        yield text[index:end]
        index = end


def metadata_projection(text):
    fields = raw_fields(text, {"id", "doi", "created", "updated", "metadata", "files"})
    result = {k: json.loads(v) for k, v in fields.items() if k not in {"metadata", "files"}}
    result["metadata"] = {
        k: json.loads(v)
        for k, v in raw_fields(
            fields["metadata"], {"title", "license", "access_right", "publication_date"}
        ).items()
    }
    result["files"] = []
    for raw in array_values(fields["files"]):
        file = {
            k: json.loads(v)
            for k, v in raw_fields(
                raw, {"id", "key", "size", "checksum", "links", "license", "access"}
            ).items()
        }
        result["files"].append(file)
    return result


def project_tsv(handle):
    header = handle.readline(65537)
    if len(header) > 65536:
        raise ValueError("Oversized header")
    names = header.decode("utf-8-sig").rstrip("\r\n").split("\t")
    if len(names) != len(set(names)) or not {"aa_seq", "nt_seq"} <= set(names):
        raise ValueError("Unique genotype columns required")
    columns = {name: names.index(name) for name in ["nt_seq", "aa_seq"]}
    rows = []
    for line in handle:
        if len(line) > 65536:
            raise ValueError("Oversized row")
        cells = line.rstrip(b"\r\n").split(b"\t")
        if len(cells) != len(names):
            raise ValueError("Unexpected row width; abort before cell decoding")
        rows.append({name: cells[index].decode("ascii") for name, index in columns.items()})
    return names, rows


def opener(request, timeout):
    separator = "&" if "?" in request.full_url else "?"
    query = request.get_header("Range") or "head"
    request = urllib.request.Request(
        request.full_url + separator + "m201=" + query,
        headers={**dict(request.header_items()), "User-Agent": "ProteinMod blinded source audit"},
        method=request.get_method(),
    )
    return urllib.request.urlopen(request, timeout=timeout)


def acquire_metadata(output):
    output.mkdir(parents=True, exist_ok=False)
    for record in [16266645, 11175470, 15302825]:
        url = f"https://zenodo.org/api/records/{record}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError("Oversized metadata")
        data = metadata_projection(raw.decode("utf-8"))
        assert data["id"] == record
        (output / f"{record}-metadata.json").write_text(json.dumps(data, indent=2) + "\n")
        for file in data["files"]:
            if not file["key"].endswith(".zip"):
                continue
            reader = RangeZipReader(file["links"]["self"], opener)
            assert reader.length == file["size"]
            with zipfile.ZipFile(reader) as archive:
                directory = [
                    {
                        "name": i.filename,
                        "bytes": i.file_size,
                        "compressed_bytes": i.compress_size,
                        "CRC": i.CRC,
                    }
                    for i in archive.infolist()
                ]
            name = f"{record}-{file['key']}-directory.json"
            (output / name).write_text(
                json.dumps(
                    {"archive": file, "range_bytes": reader.transferred, "members": directory},
                    indent=2,
                )
                + "\n"
            )
        print(
            record, data["metadata"].get("license"), [(f["key"], f["size"]) for f in data["files"]]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--metadata-output", type=Path, required=True)
    acquire_metadata(parser.parse_args().metadata_output)
