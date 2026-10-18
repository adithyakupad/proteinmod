"""Outcome-blind diagnostics; these do not replace historical admission thresholds."""

from proteinmod.family_data import alignment


def alignment_diagnostics(a, b):
    result = {mode: alignment(a, b, mode) for mode in ("global", "local")}
    for row in result.values():
        blocks = row["blocks"]
        identical = sum(
            a[i] == b[j] for x, y in blocks for i, j in zip(range(*x), range(*y), strict=True)
        )
        row.update(
            identical=identical,
            identity_shorter=identical / min(len(a), len(b)),
            identity_longer=identical / max(len(a), len(b)),
            unpaired_a=len(a) - row["aligned"],
            unpaired_b=len(b) - row["aligned"],
        )
    return result


def genotype_cells(row):
    """Only primary S2 genotype columns; fitness R/S must never enter this projection."""
    return {k: row[k] for k in ("A", "B", "C")}


def tem_genotypes(path):
    """Project S2 A/B/C only, including numeric positions; never decode R/S targets."""
    import xml.etree.ElementTree as ET
    import zipfile

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    records = []
    with zipfile.ZipFile(path) as archive:
        strings = [
            "".join(node.itertext())
            for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(ns + "si")
        ]
        with archive.open("xl/worksheets/sheet2.xml") as handle:
            for _, element in ET.iterparse(handle, events=["end"]):
                if element.tag != ns + "row":
                    continue
                number = int(element.attrib["r"])
                if number <= 2:
                    element.clear()
                    continue
                row = {}
                for cell in element:
                    column = "".join(filter(str.isalpha, cell.attrib["r"]))
                    if column not in ("A", "B", "C"):
                        continue
                    value = cell.find(ns + "v")
                    if value is not None:
                        row[column] = (
                            strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                        )
                if row:
                    row = genotype_cells(row)
                    records.append(
                        {
                            "source_row": number,
                            "position": int(row["A"]),
                            "wt": row["B"],
                            "mutant": row["C"],
                        }
                    )
                element.clear()
    return records
