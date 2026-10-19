"""Remote Data S2 package metadata and first-row headers only; never parse a data row."""

import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from proteinmod.data import write_json
from proteinmod.range_zip import RangeZipReader

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def first_row(handle):
    content = bytearray()
    while chunk := handle.read(1):
        content.extend(chunk)
        if len(content) > 65536:
            raise ValueError("First-row metadata exceeds bounded header limit")
        if not content.endswith(b"</row>"):
            continue
        # Expat may defer end events; isolate header bytes before any XML decoding.
        row = re.search(rb"<row(?:\s[^>]*)?>.*?</row>", content, re.DOTALL)
        if row is None:
            raise ValueError("Unsupported first-row XML")
        wrapper = b'<root xmlns="' + NS["s"].encode() + b'">' + row.group() + b"</root>"
        element = ET.fromstring(wrapper)[0]
        if element.get("r") != "1":
            raise ValueError("Only a declared first row may be inspected")
        cells = []
        for cell in element.findall("s:c", NS):
            cells.append(
                {
                    "coordinate": cell.get("r"),
                    "type": cell.get("t"),
                    "value": cell.findtext("s:v", namespaces=NS),
                    "inline": cell.findtext("s:is/s:t", namespaces=NS),
                }
            )
        return cells
    raise ValueError("No first-row metadata")


def selected_strings(content, indices):
    result = {}
    for index, match in enumerate(re.finditer(rb"<si(?:\s[^>]*)?>.*?</si>", content, re.DOTALL)):
        if index in indices:
            element = ET.fromstring(match.group())
            result[index] = "".join(element.itertext())
    if set(result) != set(indices):
        raise ValueError("Unresolved shared-string header indices")
    return result


def opener(request, timeout):
    return urllib.request.urlopen(request, timeout=timeout)


def acquire(output):
    if Path(output).exists():
        raise FileExistsError("Choose a new header output; preserve source evidence")
    url = "https://ars.els-cdn.com/content/image/1-s2.0-S0002929725003519-mmc2.xlsx"
    reader = RangeZipReader(url, opener)
    with zipfile.ZipFile(reader) as archive:
        directory = [
            {"name": i.filename, "bytes": i.file_size, "crc32": i.CRC} for i in archive.infolist()
        ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in relationships}
        sheets = []
        for sheet in workbook.findall("s:sheets/s:sheet", NS):
            target = targets[
                sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            ]
            path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            with archive.open(path) as handle:
                header = first_row(handle)
            sheets.append({"sheet": sheet.get("name"), "member": path, "first_row": header})
        indices = {
            int(cell["value"])
            for sheet in sheets
            for cell in sheet["first_row"]
            if cell["type"] == "s"
        }
        strings = selected_strings(archive.read("xl/sharedStrings.xml"), indices)
        for sheet in sheets:
            for cell in sheet["first_row"]:
                if cell["type"] == "s":
                    cell["header"] = strings[int(cell["value"])]
    write_json(
        output,
        {
            "url": url,
            "archive_bytes": reader.length,
            "etag": reader.etag,
            "range_bytes_transferred": reader.transferred,
            "directory": directory,
            "sheets": sheets,
            "sheet_data_rows_parsed": 0,
            "whole_archive_sha256_verified": False,
            "whole_sheet_CRC_verified": False,
            "shared_strings_CRC_verified": True,
            "scope": "Only first row per sheet and selected shared-string headers decoded; no worksheet data rows parsed",
        },
    )
    print(json.dumps(sheets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/m185/evidence/DataS2-headers.json")
    )
    acquire(parser.parse_args().output)
