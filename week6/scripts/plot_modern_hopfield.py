"""Create verification figures for the modern Hopfield experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_mnist
from modern_hopfield import binarize_to_spins, make_corrupted_queries, query_seed


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "inputs" / "mnist"
ATTENTION_PATH = ROOT / "artifacts" / "attention_equivalence.json"
RESULTS_PATH = ROOT / "artifacts" / "modern_hopfield_results.json"
EVIDENCE_PATH = ROOT / "evidence"

PLOT_PATTERN_COUNT = 1000
PLOT_BETAS = (0.003, 0.03, 0.3)
SUCCESS_BETA = 0.3
EXAMPLE_QUERY_INDICES = (0, 1, 2, 3, 4, 5)


def load_artifacts() -> tuple[dict, dict]:
    attention = json.loads(ATTENTION_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return attention, results


def load_training_spins() -> tuple[np.ndarray, np.ndarray]:
    train_images, train_labels, _, _ = load_mnist(INPUT_PATH)
    return binarize_to_spins(train_images), train_labels


def spin_image(state: np.ndarray) -> np.ndarray:
    return ((state.reshape(28, 28) + 1.0) / 2.0).clip(0.0, 1.0)


def result_index(results: list[dict], pattern_count: int, beta: float) -> dict:
    for record in results:
        if record["P"] == pattern_count and np.isclose(record["beta"], beta):
            return record
    raise KeyError(f"missing result for P={pattern_count}, beta={beta}")


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def plot_attention_weights(patterns: np.ndarray, query: np.ndarray, target_index: int) -> None:
    scores = patterns @ query
    figure, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for axis, beta in zip(axes, PLOT_BETAS):
        weights = softmax(beta * scores)
        maximum_index = int(np.argmax(weights))
        axis.plot(np.arange(len(weights)), weights, linewidth=0.8)
        axis.axvline(target_index, color="tab:green", linestyle="--", label="target")
        axis.axvline(maximum_index, color="tab:red", linestyle=":", label="maximum")
        axis.set_ylabel("attention weight")
        axis.set_title(
            f"beta = {beta:g}; maximum weight = {weights[maximum_index]:.4f} "
            f"at pattern {maximum_index}"
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("stored pattern index")
    figure.suptitle("Softmax attention weights for one corrupted MNIST query")
    figure.savefig(EVIDENCE_PATH / "attention_weights.png", dpi=160)
    plt.close(figure)


def plot_retrieval_examples(
    patterns: np.ndarray,
    queries: np.ndarray,
    target_indices: np.ndarray,
    records_by_beta: dict[float, dict],
    classical_record: dict,
    labels: np.ndarray,
) -> None:
    titles = [
        "target",
        "corrupted query",
        "modern beta=0.003",
        "modern beta=0.03",
        "modern beta=0.3",
        "classical Hopfield",
    ]
    figure, axes = plt.subplots(
        len(EXAMPLE_QUERY_INDICES), len(titles), figsize=(12, 12), constrained_layout=True
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row, query_index in enumerate(EXAMPLE_QUERY_INDICES):
        target_index = int(target_indices[query_index])
        states = [patterns[target_index], queries[query_index]]
        for beta in PLOT_BETAS:
            retrieved_index = records_by_beta[beta]["modern_hopfield"]["retrieved_indices"][query_index]
            states.append(patterns[retrieved_index])
        classical_index = classical_record["classical_hopfield"]["retrieved_indices"][query_index]
        states.append(patterns[classical_index])
        for column, state in enumerate(states):
            axis = axes[row, column]
            axis.imshow(spin_image(np.asarray(state)), cmap="gray_r", vmin=0, vmax=1, interpolation="none")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(f"q{query_index}, digit {labels[target_index]}")
    figure.suptitle("Modern and classical Hopfield retrieval from corrupted queries")
    figure.savefig(EVIDENCE_PATH / "modern_retrieval_examples.png", dpi=160)
    plt.close(figure)


def plot_success_comparison(results: list[dict]) -> None:
    pattern_counts = sorted({record["P"] for record in results})
    modern_success = []
    classical_success = []
    for pattern_count in pattern_counts:
        record = result_index(results, pattern_count, SUCCESS_BETA)
        modern_success.append(record["modern_hopfield"]["success_fraction"])
        classical_success.append(record["classical_hopfield"]["success_fraction"])
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(pattern_counts, modern_success, marker="o", label="modern Hopfield, beta=0.3")
    axis.plot(pattern_counts, classical_success, marker="s", label="classical Hopfield")
    axis.set_xscale("log")
    axis.set_xticks(pattern_counts)
    axis.set_xticklabels([str(value) for value in pattern_counts])
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("number of stored patterns P")
    axis.set_ylabel("retrieval success fraction")
    axis.set_title("Retrieval success comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(EVIDENCE_PATH / "retrieval_success.png", dpi=160)
    plt.close(figure)


def main() -> None:
    attention, result_artifact = load_artifacts()
    spins, labels = load_training_spins()
    patterns = spins[:PLOT_PATTERN_COUNT]
    target_indices, queries = make_corrupted_queries(patterns, query_seed(PLOT_PATTERN_COUNT))
    reference = result_index(result_artifact["results"], PLOT_PATTERN_COUNT, SUCCESS_BETA)
    if target_indices.tolist() != reference["modern_hopfield"]["target_indices"]:
        raise RuntimeError("recreated queries do not match the stored retrieval experiment")

    plot_attention_weights(patterns, queries[0], int(target_indices[0]))
    records_by_beta = {
        beta: result_index(result_artifact["results"], PLOT_PATTERN_COUNT, beta)
        for beta in PLOT_BETAS
    }
    plot_retrieval_examples(
        patterns,
        queries,
        target_indices,
        records_by_beta,
        reference,
        labels,
    )
    plot_success_comparison(result_artifact["results"])

    maximum_difference = max(
        record["maximum_absolute_difference"] for record in attention["records"]
    )
    print(f"attention maximum absolute difference: {maximum_difference:.9g}")
    for filename in (
        "attention_weights.png",
        "modern_retrieval_examples.png",
        "retrieval_success.png",
    ):
        print(f"wrote {EVIDENCE_PATH / filename}")


if __name__ == "__main__":
    main()
