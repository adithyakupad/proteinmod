"""Acquire only pinned processing source; notebook outputs remain undecoded."""

import hashlib
import json
import zipfile
from pathlib import Path

from audit_m201_zenodo import notebook_code, opener

from proteinmod.range_zip import RangeZipReader


def project_source(name, body):
    if name.endswith(".ipynb"):
        return "\n\n".join(notebook_code(body.decode("utf-8")))
    if name in {
        "zenodo/Fig2/DiMSUM_run_Sparse_cores_DTS.sh",
        "zenodo/Fig2/DiMSUM_experimentDesign_Sparse_cores_DTS.txt",
    }:
        return body.decode("utf-8")
    raise ValueError("Not a registered processing source type")


def main():
    root = Path("benchmarks/m202")
    destination = root / "evidence/source-audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    records = []
    for group in json.loads((root / "source-plan.json").read_text()):
        file = group["archive"]
        reader = RangeZipReader(file["links"]["self"], opener)
        assert reader.length == file["size"]
        with zipfile.ZipFile(reader) as archive:
            for member in group["members"]:
                info = archive.getinfo(member["name"])
                assert (info.file_size, info.compress_size, info.CRC) == (
                    member["bytes"],
                    member["compressed_bytes"],
                    member["CRC"],
                )
                if info.file_size > 16 * 1024**2 or info.compress_size > 8 * 1024**2:
                    raise ValueError("Source member exceeds bound")
                body = archive.read(info)
                source = project_source(info.filename, body)
                output = root / "evidence" / (Path(info.filename).stem + ".source.txt")
                with output.open("x") as handle:
                    handle.write(source)
                records.append(
                    {
                        "archive": file["key"],
                        "member": member,
                        "member_sha256": hashlib.sha256(body).hexdigest(),
                        "source_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "source_path": str(output),
                        "CRC_verified": True,
                        "notebook_outputs_decoded": False,
                        "source_executed": False,
                    }
                )
        print(file["key"], "range bytes", reader.transferred)
    with destination.open("x") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
