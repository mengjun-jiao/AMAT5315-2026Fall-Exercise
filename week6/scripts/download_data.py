"""Download the raw MNIST and CIFAR-10 datasets used in Week 6 Part 1."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_FALLBACK_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-binary.tar.gz"


def download(url: str, destination: Path) -> None:
    """Download *url* to *destination*, skipping an existing nonempty file."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"download: {url}")
    with urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    print(f"saved: {destination}")


def main() -> None:
    mnist_dir = INPUTS / "mnist"
    cifar_dir = INPUTS / "cifar10"

    for filename, url in MNIST_URLS.items():
        download(url, mnist_dir / filename)
    cifar_archive = cifar_dir / "cifar-10-binary.tar.gz"
    try:
        download(CIFAR_URL, cifar_archive)
    except OSError as error:
        print(f"primary CIFAR URL failed ({error}); trying documented mirror")
        download(CIFAR_FALLBACK_URL, cifar_archive)


if __name__ == "__main__":
    main()
