"""Readers for the raw MNIST and CIFAR-10 formats."""

from __future__ import annotations

import gzip
import struct
import tarfile
from pathlib import Path

import numpy as np


MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049
CIFAR_IMAGE_SHAPE = (32, 32, 3)
CIFAR_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError(f"unexpected MNIST image magic number in {path}: {magic}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = count * rows * columns
    if data.size != expected:
        raise ValueError(f"MNIST image payload has {data.size} bytes; expected {expected}")
    return data.reshape(count, rows, columns)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError(f"unexpected MNIST label magic number in {path}: {magic}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"MNIST label payload has {data.size} values; expected {count}")
    return data


def load_mnist(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train images, train labels, test images, and test labels."""
    root = Path(root)
    train_images = _read_idx_images(root / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx_labels(root / "train-labels-idx1-ubyte.gz")
    test_images = _read_idx_images(root / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_idx_labels(root / "t10k-labels-idx1-ubyte.gz")
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("MNIST image and label counts do not match")
    return train_images, train_labels, test_images, test_labels


def _read_cifar_batch(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    record_size = 1 + int(np.prod(CIFAR_IMAGE_SHAPE))
    if len(data) % record_size:
        raise ValueError("CIFAR batch size is not a whole number of records")
    records = np.frombuffer(data, dtype=np.uint8).reshape(-1, record_size)
    labels = records[:, 0].copy()
    channels_first = records[:, 1:].reshape(-1, 3, 32, 32)
    images = channels_first.transpose(0, 2, 3, 1).copy()
    return images, labels


def _extract_cifar_archive(archive_path: Path, destination: Path) -> Path:
    extracted = destination / "cifar-10-batches-bin"
    if extracted.is_dir():
        return extracted
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination)
    if not extracted.is_dir():
        raise ValueError(f"CIFAR archive did not contain {extracted.name}/")
    return extracted


def load_cifar10(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train RGB images, train labels, test RGB images, and test labels."""
    root = Path(root)
    batch_dir = _extract_cifar_archive(root / "cifar-10-binary.tar.gz", root)
    train_images_parts = []
    train_labels_parts = []
    for index in range(1, 6):
        images, labels = _read_cifar_batch((batch_dir / f"data_batch_{index}.bin").read_bytes())
        train_images_parts.append(images)
        train_labels_parts.append(labels)
    test_images, test_labels = _read_cifar_batch((batch_dir / "test_batch.bin").read_bytes())
    return (
        np.concatenate(train_images_parts),
        np.concatenate(train_labels_parts),
        test_images,
        test_labels,
    )
