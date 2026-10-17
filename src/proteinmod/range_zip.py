"""Read selected public ZIP members without downloading the full archive."""

import io
import urllib.request
import zipfile


class RangeZipReader(io.RawIOBase):
    def __init__(self, url, opener=urllib.request.urlopen):
        super().__init__()
        self.url, self.opener, self.position, self.transferred = url, opener, 0, 0
        with opener(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.length = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        if self.length <= 0:
            raise ValueError("Invalid remote archive length")

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        if value < 0 or value > self.length or whence not in (0, 1, 2):
            raise ValueError("Invalid remote seek")
        self.position = value
        return value

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size > 8 * 1024**2:
            raise ValueError("Remote read exceeds bounded member limit")
        if size == 0:
            return b""
        start, end = self.position, self.position + size - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        with self.opener(urllib.request.Request(self.url, headers=headers), timeout=60) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.length}"
            ):
                raise ValueError("Server did not honor bounded byte range")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("Truncated or oversized byte range")
        self.position += size
        self.transferred += size
        return data


def selected_members(url, names, opener=urllib.request.urlopen):
    reader = RangeZipReader(url, opener)
    result = {}
    with zipfile.ZipFile(reader) as archive:
        for name in names:
            info = archive.getinfo(name)
            if info.file_size > 10 * 1024**2 or info.compress_size > 8 * 1024**2:
                raise ValueError("Dataset member exceeds bounded collection limit")
            result[name] = archive.read(name)  # ZipFile verifies each member CRC.
    return result, {
        "url": url,
        "etag": reader.etag,
        "archive_bytes": reader.length,
        "downloaded_bytes": reader.transferred,
    }
