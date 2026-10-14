"""Load Week 6 Part 1 datasets and write distribution figures and a summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import CIFAR_CLASS_NAMES, load_cifar10, load_mnist


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"


def class_counts(labels: np.ndarray) -> list[int]:
    return np.bincount(labels, minlength=10).astype(int).tolist()


def plot_mnist(train_images, train_labels, test_images, test_labels) -> None:
    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(2.2, 1.3))
    gallery = grid[0].subgridspec(2, 10, wspace=0.05, hspace=0.25)
    for row, (images, labels) in enumerate(((train_images, train_labels), (test_images, test_labels))):
        chosen = [int(np.flatnonzero(labels == digit)[0]) for digit in range(10)]
        for column, index in enumerate(chosen):
            axis = figure.add_subplot(gallery[row, column])
            axis.imshow(images[index], cmap="gray", interpolation="none", vmin=0, vmax=255)
            axis.set_title(f"{labels[index]}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.axes[-1].set_ylabel("train" if row == 0 else "test", rotation=0, labelpad=24, va="center")

    counts = grid[1].subgridspec(1, 1)
    axis = figure.add_subplot(counts[0])
    digits = np.arange(10)
    width = 0.38
    axis.bar(digits - width / 2, np.bincount(train_labels, minlength=10), width, label="train")
    axis.bar(digits + width / 2, np.bincount(test_labels, minlength=10), width, label="test")
    axis.set_title("MNIST class frequencies")
    axis.set_xlabel("digit label")
    axis.set_ylabel("number of examples")
    axis.set_xticks(digits)
    axis.set_xlim(-0.7, 9.7)
    axis.legend()
    figure.suptitle("MNIST dataset distribution", fontsize=16)
    figure.savefig(EVIDENCE / "mnist_distribution.png", dpi=160)
    plt.close(figure)


def plot_cifar(train_images, train_labels, test_images, test_labels) -> None:
    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(2.2, 1.3))
    gallery = grid[0].subgridspec(1, 10, wspace=0.08)
    for class_index, class_name in enumerate(CIFAR_CLASS_NAMES):
        sample_index = int(np.flatnonzero(train_labels == class_index)[0])
        axis = figure.add_subplot(gallery[0, class_index])
        axis.imshow(train_images[sample_index], interpolation="none")
        axis.set_title(class_name, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    axis = figure.add_subplot(grid[1])
    labels = np.arange(10)
    width = 0.38
    axis.bar(labels - width / 2, np.bincount(train_labels, minlength=10), width, label="train")
    axis.bar(labels + width / 2, np.bincount(test_labels, minlength=10), width, label="test")
    axis.set_title("CIFAR-10 class frequencies")
    axis.set_xlabel("class")
    axis.set_ylabel("number of examples")
    axis.set_xticks(labels, CIFAR_CLASS_NAMES, rotation=25, ha="right")
    axis.legend()
    figure.suptitle("CIFAR-10 dataset distribution", fontsize=16)
    figure.savefig(EVIDENCE / "cifar10_distribution.png", dpi=160)
    plt.close(figure)


def write_summary(mnist, cifar) -> None:
    mnist_train_images, mnist_train_labels, mnist_test_images, mnist_test_labels = mnist
    cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels = cifar
    summary = {
        "mnist": {
            "train_size": int(len(mnist_train_images)),
            "test_size": int(len(mnist_test_images)),
            "image_shape": list(mnist_train_images.shape[1:]),
            "pixel_range": [int(mnist_train_images.min()), int(mnist_train_images.max())],
            "class_counts": {
                "train": class_counts(mnist_train_labels),
                "test": class_counts(mnist_test_labels),
            },
        },
        "cifar10": {
            "train_size": int(len(cifar_train_images)),
            "test_size": int(len(cifar_test_images)),
            "image_shape": list(cifar_train_images.shape[1:]),
            "class_counts": {
                "train": class_counts(cifar_train_labels),
                "test": class_counts(cifar_test_labels),
            },
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "part1_dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    mnist = load_mnist(INPUTS / "mnist")
    cifar = load_cifar10(INPUTS / "cifar10")
    plot_mnist(*mnist)
    plot_cifar(*cifar)
    write_summary(mnist, cifar)
    print("wrote evidence/mnist_distribution.png")
    print("wrote evidence/cifar10_distribution.png")
    print("wrote artifacts/part1_dataset_summary.json")


if __name__ == "__main__":
    main()
