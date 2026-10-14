"""Plot classical Hopfield instability and recall-capacity diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "artifacts" / "hopfield_results.json"
SUMMARY_PATH = ROOT / "artifacts" / "hopfield_summary.json"
EVIDENCE_PATH = ROOT / "evidence"
SIZES = (100, 200, 400, 800)
THEORETICAL_ALPHA = np.linspace(0.05, 0.25, 400)


def standard_error(values: list[float]) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def normal_tail(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.array([math.erfc(value / math.sqrt(2.0)) for value in x])


def aggregate(data: dict) -> dict[int, dict[str, list[float]]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for size in SIZES:
        records = [record for record in data["results"] if record["N"] == size]
        records.sort(key=lambda record: record["alpha"])
        grouped[size] = {
            "alpha": [record["alpha"] for record in records],
            "unstable_mean": [],
            "unstable_stderr": [],
            "overlap_mean": [],
            "overlap_stderr": [],
            "success_fraction": [],
        }
        for record in records:
            set_unstable = [item["unstable_fraction"] for item in record["independent_sets"]]
            recalls = [
                recall
                for item in record["independent_sets"]
                for recall in item["recalls"]
            ]
            overlaps = [recall["final_overlap"] for recall in recalls]
            grouped[size]["unstable_mean"].append(float(np.mean(set_unstable)))
            grouped[size]["unstable_stderr"].append(standard_error(set_unstable))
            grouped[size]["overlap_mean"].append(float(np.mean(overlaps)))
            grouped[size]["overlap_stderr"].append(standard_error(overlaps))
            grouped[size]["success_fraction"].append(
                float(np.mean(np.asarray(overlaps) >= 0.9))
            )
    return grouped


def plot_unstable_bits(grouped: dict[int, dict[str, list[float]]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(SIZES)))
    for size, color in zip(SIZES, colors):
        values = grouped[size]
        axis.errorbar(
            values["alpha"],
            values["unstable_mean"],
            yerr=values["unstable_stderr"],
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            label=f"N={size}",
            color=color,
        )
    axis.plot(
        THEORETICAL_ALPHA,
        normal_tail(1.0 / np.sqrt(THEORETICAL_ALPHA)),
        "k--",
        linewidth=1.5,
        label=r"theory $\Phi(-1/\sqrt{\alpha})$",
    )
    axis.set_xlabel(r"load $\alpha=P/N$")
    axis.set_ylabel("unstable bit fraction")
    axis.set_xlim(0.05, 0.25)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Hopfield stored-pattern instability")
    figure.savefig(EVIDENCE_PATH / "hopfield_unstable_bits.png", dpi=160)
    plt.close(figure)


def plot_capacity(grouped: dict[int, dict[str, list[float]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(SIZES)))
    for size, color in zip(SIZES, colors):
        values = grouped[size]
        axes[0].errorbar(
            values["alpha"],
            values["overlap_mean"],
            yerr=values["overlap_stderr"],
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            label=f"N={size}",
            color=color,
        )
        axes[1].plot(
            values["alpha"],
            values["success_fraction"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=f"N={size}",
            color=color,
        )

    for axis in axes:
        axis.axvline(0.138, color="black", linestyle="--", linewidth=1, label=r"$\alpha=0.138$")
        axis.set_xlabel(r"load $\alpha=P/N$")
        axis.set_xlim(0.05, 0.25)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean final overlap")
    axes[0].set_title("A: recall overlap")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel(r"fraction of recalls with $m\geq0.9$")
    axes[1].set_title("B: successful recall fraction")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Classical Hopfield recall capacity")
    figure.savefig(EVIDENCE_PATH / "hopfield_capacity.png", dpi=160)
    plt.close(figure)


def write_summary(grouped: dict[int, dict[str, list[float]]], data: dict) -> None:
    summary = {
        "capacity_threshold": 0.9,
        "transition_reference_alpha": 0.138,
        "sizes": {},
        "runtime_seconds": data.get("runtime_seconds"),
    }
    for size in SIZES:
        values = grouped[size]
        crossing = next(
            (alpha for alpha, fraction in zip(values["alpha"], values["success_fraction"]) if fraction < 0.9),
            None,
        )
        summary["sizes"][str(size)] = {
            "crossing_alpha_success_below_0.9": crossing,
            "alpha": values["alpha"],
            "mean_overlap": values["overlap_mean"],
            "mean_overlap_stderr": values["overlap_stderr"],
            "success_fraction": values["success_fraction"],
        }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    grouped = aggregate(data)
    EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_unstable_bits(grouped)
    plot_capacity(grouped)
    write_summary(grouped, data)
    print(f"wrote {EVIDENCE_PATH / 'hopfield_unstable_bits.png'}")
    print(f"wrote {EVIDENCE_PATH / 'hopfield_capacity.png'}")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
