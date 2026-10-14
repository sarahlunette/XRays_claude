"""Smoke-test a running xrayml API with the standard library only (used by CI and deploy jobs).

    python scripts/smoke_test.py http://localhost:8000
"""

from __future__ import annotations

import json
import struct
import sys
import time
import urllib.request
import uuid
import zlib


def _png(w: int = 96, h: int = 96) -> bytes:
    """A gray 8-bit PNG with a bright diagonal 'bone', built without numpy/opencv."""
    rows = b"".join(b"\x00" + bytes(220 if abs(x - y) < 8 else 60 for x in range(w)) for y in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 - test helper, URL from CLI
        return json.loads(r.read())


def main(base: str) -> int:
    for _ in range(60):
        try:
            assert _get(f"{base}/health") == {"status": "ok"}
            break
        except Exception:
            time.sleep(1)
    else:
        print("service never became healthy", file=sys.stderr)
        return 1
    version = _get(f"{base}/version")
    print("version:", version)
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + _png() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{base}/predict", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Site-Id": "smoke"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
        pred = json.loads(r.read())
    print("predict:", pred)
    assert 0.0 <= pred["score"] <= 1.0
    assert pred["model_version"] == version["model_version"]
    assert pred["thresholds_version"] == version["thresholds_version"]
    assert pred["positive"] == (pred["score"] >= pred["threshold"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
