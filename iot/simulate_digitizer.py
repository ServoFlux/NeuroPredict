from __future__ import annotations

import argparse
import io
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.config import ETIOLOGY_CLASS_NAMES
from wmd.filmscan import contact_sheet_from_volume

def _load_volume(args: argparse.Namespace) -> np.ndarray:
    if args.source:
        from wmd.preprocessing import load_volume

        return load_volume(args.source)
    from wmd.synthetic import make_etiology_volume

    etiology = ETIOLOGY_CLASS_NAMES.index(args.etiology)
    rng = np.random.default_rng(args.seed)
    return make_etiology_volume(etiology, rng=rng)

def _encode_multipart(fields: dict[str, str], image_bytes: bytes, image_name: str):
    boundary = f"----neuropredict{uuid.uuid4().hex}"
    crlf = "\r\n"
    body = io.BytesIO()

    for key, value in fields.items():
        body.write(f"--{boundary}{crlf}".encode())
        body.write(f'Content-Disposition: form-data; name="{key}"{crlf}{crlf}'.encode())
        body.write(f"{value}{crlf}".encode())

    ctype = mimetypes.guess_type(image_name)[0] or "application/octet-stream"
    body.write(f"--{boundary}{crlf}".encode())
    body.write(
        f'Content-Disposition: form-data; name="sheet"; filename="{image_name}"{crlf}'.encode()
    )
    body.write(f"Content-Type: {ctype}{crlf}{crlf}".encode())
    body.write(image_bytes)
    body.write(crlf.encode())
    body.write(f"--{boundary}--{crlf}".encode())

    return body.getvalue(), f"multipart/form-data; boundary={boundary}"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Path to a .nii/.nii.gz/DICOM scan to digitize")
    parser.add_argument(
        "--etiology",
        default="vascular",
        choices=list(ETIOLOGY_CLASS_NAMES),
        help="Synthetic volume to generate when --source is not given",
    )
    parser.add_argument("--cols", type=int, default=8, help="Slices per row on the sheet")
    parser.add_argument("--age", type=float, default=55.0, help="Patient age to send")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--save", help="Also save the contact-sheet image to this path")
    parser.add_argument("--no-post", action="store_true", help="Only build the sheet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image

    volume = _load_volume(args)
    sheet = contact_sheet_from_volume(volume, cols=args.cols)
    depth = int(volume.shape[0])
    print(f"Built contact sheet {sheet.shape[1]}x{sheet.shape[0]} from {depth} slices.")

    if args.save:
        Image.fromarray(sheet).save(args.save)
        print(f"Saved film-sheet image to {args.save}")

    if args.no_post:
        return 0

    buf = io.BytesIO()
    Image.fromarray(sheet).save(buf, format="PNG")
    fields = {"cols": str(args.cols), "depth": str(depth), "age": str(args.age)}
    body, content_type = _encode_multipart(fields, buf.getvalue(), "film_sheet.png")

    url = args.server.rstrip("/") + "/ingest/film"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"POST {url} -> {resp.status}")
            print(resp.read().decode())
    except Exception as exc:
        print(f"Request failed: {exc}")
        print("Is the server running?  uvicorn webapp.main:app --port 8000")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
