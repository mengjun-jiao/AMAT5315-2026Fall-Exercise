"""Minimal local Safetensors reader for the pinned CIFAR checkpoint."""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np


DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
}


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict, set[str]]:
    """Load tensors into independent NumPy arrays without modifying the file."""
    with path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    tensors: dict[str, np.ndarray] = {}
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        data_start = 8 + header_length
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            dtype = metadata["dtype"]
            shape = tuple(metadata["shape"])
            begin, end = metadata["data_offsets"]
            view = memoryview(mapped)[data_start + begin : data_start + end]
            if dtype == "BF16":
                bits = np.frombuffer(view, dtype=np.uint16).astype(np.uint32)
                array = (bits << 16).view(np.float32).reshape(shape).copy()
                del bits
            else:
                array = np.frombuffer(view, dtype=DTYPES[dtype]).reshape(shape).copy()
            del view
            tensors[name] = array
    return tensors, header, set(tensors)
