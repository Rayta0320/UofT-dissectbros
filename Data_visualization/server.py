#!/usr/bin/env python3
"""Zero-dependency local viewer for paired NIfTI scans and segmentation masks."""

from __future__ import annotations

import argparse
import array
import gzip
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import zlib


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DEFAULT_DATASET = Path("/Users/raytamorimoto/Downloads/TORALIS CHALLENGE ")
CACHE = ROOT / ".cache" / "nifti"
LABEL_STATS_FILE = ROOT / "label_stats.json"

TYPE_INFO = {
    2: ("B", 1),     # uint8
    4: ("h", 2),     # int16
    8: ("i", 4),     # int32
    16: ("f", 4),    # float32
    64: ("d", 8),    # float64
    256: ("b", 1),   # int8
    512: ("H", 2),   # uint16
    768: ("I", 4),   # uint32
}


class Nifti:
    def __init__(self, source: Path, cache_dir: Path = CACHE):
        self.source = source
        self.cache_dir = cache_dir
        self.compressed = self._is_gzip(source)
        with (gzip.open(source, "rb") if self.compressed else source.open("rb")) as fh:
            header = fh.read(352)
        if len(header) < 348:
            raise ValueError(f"Truncated NIfTI header: {source}")
        if struct.unpack("<I", header[:4])[0] == 348:
            self.endian = "<"
        elif struct.unpack(">I", header[:4])[0] == 348:
            self.endian = ">"
        else:
            raise ValueError(f"Not a NIfTI-1 file: {source}")
        dims = struct.unpack(self.endian + "8h", header[40:56])
        self.shape = tuple(int(v) for v in dims[1:4])
        self.datatype, self.bitpix = struct.unpack(self.endian + "2h", header[70:74])
        if self.datatype not in TYPE_INFO:
            raise ValueError(f"Unsupported NIfTI datatype {self.datatype}: {source}")
        self.fmt, self.itemsize = TYPE_INFO[self.datatype]
        self.vox_offset = max(0, int(struct.unpack(self.endian + "f", header[108:112])[0]))
        self.slope = struct.unpack(self.endian + "f", header[112:116])[0] or 1.0
        self.intercept = struct.unpack(self.endian + "f", header[116:120])[0]
        self.spacing = tuple(round(float(v), 3) for v in struct.unpack(self.endian + "8f", header[76:108])[1:4])

    @staticmethod
    def _is_gzip(path: Path) -> bool:
        with path.open("rb") as fh:
            return fh.read(2) == b"\x1f\x8b"

    def materialized_path(self) -> Path:
        if not self.compressed:
            return self.source
        stat = self.source.stat()
        token = hashlib.sha1(f"{self.source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
        destination = self.cache_dir / f"{self.source.stem}-{token}.nii"
        if destination.exists():
            return destination
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock = _materialize_locks.setdefault(str(destination), threading.Lock())
        with lock:
            if not destination.exists():
                temporary = destination.with_suffix(".tmp")
                with gzip.open(self.source, "rb") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                os.replace(temporary, destination)
        return destination

    def slice(self, orientation: str, index: int, scaled: bool = True):
        x_size, y_size, z_size = self.shape
        axis_size = {"axial": z_size, "coronal": y_size, "sagittal": x_size}[orientation]
        index = max(0, min(axis_size - 1, index))
        if orientation == "axial":
            width, height = x_size, y_size
        elif orientation == "coronal":
            width, height = x_size, z_size
        else:
            width, height = y_size, z_size

        values = []
        path = self.materialized_path()
        with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if orientation == "axial":
                start = self.vox_offset + index * x_size * y_size * self.itemsize
                values = self._unpack(mm[start:start + width * height * self.itemsize])
                # Put superior/anterior in a consistent screen direction.
                values = [v for row in range(height - 1, -1, -1)
                          for v in values[row * width:(row + 1) * width]]
            elif orientation == "coronal":
                for z in range(z_size - 1, -1, -1):
                    start = self.vox_offset + (z * x_size * y_size + index * x_size) * self.itemsize
                    values.extend(self._unpack(mm[start:start + x_size * self.itemsize]))
            else:
                unpack_one = struct.Struct(self.endian + self.fmt).unpack_from
                for z in range(z_size - 1, -1, -1):
                    base = self.vox_offset + z * x_size * y_size * self.itemsize
                    for y in range(y_size - 1, -1, -1):
                        values.append(unpack_one(mm, base + (y * x_size + index) * self.itemsize)[0])
        if scaled and (self.slope != 1.0 or self.intercept != 0.0):
            values = [v * self.slope + self.intercept for v in values]
        return width, height, values

    def _unpack(self, data: bytes):
        values = array.array(self.fmt)
        values.frombytes(data)
        file_little = self.endian == "<"
        if self.itemsize > 1 and file_little != (sys.byteorder == "little"):
            values.byteswap()
        return values


_materialize_locks: dict[str, threading.Lock] = {}
_render_slots = threading.Semaphore(4)


def png_rgb(width: int, height: int, pixels: bytes) -> bytes:
    raw = b"".join(b"\0" + pixels[row * width * 3:(row + 1) * width * 3] for row in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


PALETTE = [(255, 82, 108), (40, 214, 196), (255, 196, 61), (159, 115, 255), (64, 156, 255)]


def render(scan: Nifti, mask: Nifti, orientation: str, position: float,
           mode: str, center: float, window: float, opacity: float, outline: bool) -> bytes:
    axis = {"axial": 2, "coronal": 1, "sagittal": 0}[orientation]
    index = round(position * (scan.shape[axis] - 1))
    width, height, values = scan.slice(orientation, index)
    mask_index = round(position * (mask.shape[axis] - 1))
    mw, mh, labels = mask.slice(orientation, mask_index, scaled=False)
    if (mw, mh) != (width, height):
        raise ValueError("Scan and mask slice dimensions differ")
    low = center - max(window, 1) / 2
    scale = 255 / max(window, 1)
    gray = [max(0, min(255, int((value - low) * scale))) for value in values]

    if outline:
        visible = bytearray(width * height)
        for i, label in enumerate(labels):
            if not label:
                continue
            x, y = i % width, i // width
            if (x == 0 or y == 0 or x == width - 1 or y == height - 1 or
                    labels[i - 1] != label or labels[i + 1] != label or
                    labels[i - width] != label or labels[i + width] != label):
                visible[i] = 1
    else:
        visible = labels

    def composed(label_only=False):
        out = bytearray(width * height * 3)
        for i, (g, label) in enumerate(zip(gray, labels)):
            base = (13, 22, 29) if label_only else (g, g, g)
            if label and visible[i]:
                color = PALETTE[(int(label) - 1) % len(PALETTE)]
                alpha = 1.0 if label_only else opacity
                base = tuple(int(channel * alpha + base[j] * (1 - alpha)) for j, channel in enumerate(color))
            out[i * 3:i * 3 + 3] = bytes(base)
        return out

    if mode == "scan":
        pixels = bytes(channel for g in gray for channel in (g, g, g))
    elif mode == "labels":
        pixels = bytes(composed(True))
    elif mode == "split":
        left = bytes(channel for g in gray for channel in (g, g, g))
        right = composed(False)
        rows = bytearray()
        stride = width * 3
        for row in range(height):
            rows.extend(left[row * stride:(row + 1) * stride])
            rows.extend(right[row * stride:(row + 1) * stride])
        pixels, width = bytes(rows), width * 2
    else:
        pixels = bytes(composed(False))
    return png_rgb(width, height, pixels)


class ViewerHandler(SimpleHTTPRequestHandler):
    dataset: Path = DEFAULT_DATASET
    subjects: dict[str, tuple[Nifti, Nifti]] = {}
    label_stats: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/subjects":
            payload = [{"id": key, "shape": scan.shape, "spacing": scan.spacing, "compressed": scan.compressed,
                        "peak": self.label_stats.get(key, {}).get("peak"),
                        "labeled_voxels": self.label_stats.get(key, {}).get("voxels")}
                       for key, (scan, _) in self.subjects.items()]
            return self._json({"dataset": str(self.dataset), "subjects": payload})
        match = re.fullmatch(r"/api/subjects/(\d+)/slice", parsed.path)
        if match:
            try:
                return self._slice(match.group(1), parse_qs(parsed.query))
            except (ValueError, KeyError) as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self.log_error("slice failed: %s", exc)
                return self._json({"error": str(exc)}, 500)
        return super().do_GET()

    def _slice(self, subject_id: str, query: dict[str, list[str]]):
        scan, mask = self.subjects[subject_id]
        orientation = query.get("orientation", ["axial"])[0]
        mode = query.get("mode", ["overlay"])[0]
        if orientation not in {"axial", "coronal", "sagittal"} or mode not in {"overlay", "split", "scan", "labels"}:
            raise ValueError("Invalid orientation or view mode")
        position = max(0.0, min(1.0, float(query.get("position", ["0.5"])[0])))
        if query.get("alignment", ["volume"])[0] == "peak":
            stats = self.label_stats.get(subject_id)
            if stats:
                axis = {"axial": 2, "coronal": 1, "sagittal": 0}[orientation]
                position = stats["peak"][axis] / max(1, scan.shape[axis] - 1)
        center = float(query.get("center", ["40"])[0])
        window = max(1.0, float(query.get("window", ["400"])[0]))
        opacity = max(0.0, min(1.0, float(query.get("opacity", ["0.62"])[0])))
        outline = query.get("outline", ["0"])[0] == "1"
        with _render_slots:
            body = render(scan, mask, orientation, position, mode, center, window, opacity, outline)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def discover(dataset: Path):
    subjects = {}
    for folder in sorted(dataset.glob("subject*")):
        match = re.fullmatch(r"subject0*(\d+)", folder.name)
        if not match:
            continue
        subject_id = str(int(match.group(1)))
        scan_files = list(folder.glob("orig*.nii"))
        mask_files = list(folder.glob("mask*.nii"))
        if len(scan_files) == len(mask_files) == 1:
            subjects[subject_id] = (Nifti(scan_files[0]), Nifti(mask_files[0]))
    return dict(sorted(subjects.items(), key=lambda item: int(item[0])))


def main():
    parser = argparse.ArgumentParser(description="View all TORALIS scans and labels together")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if not args.dataset.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset}")
    ViewerHandler.dataset = args.dataset.resolve()
    ViewerHandler.subjects = discover(ViewerHandler.dataset)
    if LABEL_STATS_FILE.exists():
        saved_stats = json.loads(LABEL_STATS_FILE.read_text())
        ViewerHandler.label_stats = {
            key: value for key, value in saved_stats.items()
            if key in ViewerHandler.subjects and tuple(value.get("shape", ())) == ViewerHandler.subjects[key][1].shape
        }
    if not ViewerHandler.subjects:
        parser.error("No orig*.nii / mask*.nii pairs found")
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"TORALIS viewer: http://{args.host}:{args.port} ({len(ViewerHandler.subjects)} subjects)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
