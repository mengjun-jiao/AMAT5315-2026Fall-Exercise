"""Plot CD-1 and CD-10 RBM training diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "artifacts" / "rbm_training.json"
OUTPUT_PATH = ROOT / "evidence" / "rbm_learning_curve.png"


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, color in (("CD-1", "tab:blue"), ("CD-10", "tab:orange")):
        history = data["models"][name]["history"]
        epochs = [item["epoch"] for item in history]
        surrogate = [item["cd_surrogate"] for item in history]
        error = [item["held_out_one_step_reconstruction_error"] for item in history]
        axes[0].plot(epochs[1:], surrogate[1:], marker="o", markersize=3, label=name, color=color)
        axes[1].plot(epochs, error, marker="o", markersize=3, label=name, color=color)

    axes[0].set_title("Contrastive divergence surrogate")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("mean F(v₀) − F(vₖ)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Held-out one-step reconstruction")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("fraction of changed pixels")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle("MNIST RBM learning curves")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=160)
    plt.close(figure)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
