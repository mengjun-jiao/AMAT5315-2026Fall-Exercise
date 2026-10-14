"""Modern Hopfield retrieval and attention-equivalence experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from datasets import load_mnist
from hopfield import asynchronous_recall, hebbian_coupling


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "inputs" / "mnist"
ATTENTION_OUTPUT_PATH = ROOT / "artifacts" / "attention_equivalence.json"
RESULTS_OUTPUT_PATH = ROOT / "artifacts" / "modern_hopfield_results.json"

PATTERN_COUNTS = (3, 10, 30, 100, 300, 1000)
BETA_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
QUERY_COUNT = 100
MASTER_SEED = 6609


def binarize_to_spins(images: np.ndarray) -> np.ndarray:
    """Convert thresholded MNIST pixels to a +/-1 spin representation."""
    binary = images.reshape(len(images), -1) >= 128
    return np.where(binary, 1.0, -1.0).astype(np.float32)


def modern_hopfield_update(pattern_matrix: jax.Array, query: jax.Array, beta: float) -> jax.Array:
    """Apply x_new = X softmax(beta X^T x), with patterns in columns."""
    scores = beta * (pattern_matrix.T @ query)
    return pattern_matrix @ jax.nn.softmax(scores)


def attention_update(pattern_matrix: jax.Array, query: jax.Array, beta: float) -> jax.Array:
    """Apply the equivalent single-query attention operation."""
    keys = pattern_matrix.T
    values = pattern_matrix.T
    scores = beta * (keys @ query)
    weights = jax.nn.softmax(scores)
    return weights @ values


def nearest_pattern(patterns: np.ndarray, state: np.ndarray) -> int:
    """Return the stored pattern with maximum spin dot product."""
    return int(np.argmax(patterns @ state))


def query_seed(pattern_count: int) -> int:
    return MASTER_SEED + pattern_count * 100


def make_corrupted_queries(patterns: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target_indices = rng.integers(0, len(patterns), size=QUERY_COUNT)
    queries = patterns[target_indices].copy()
    lower_start = 14 * 28
    queries[:, lower_start:] = rng.choice(
        np.array((-1.0, 1.0), dtype=np.float32),
        size=(QUERY_COUNT, patterns.shape[1] - lower_start),
    )
    return target_indices, queries


def overlap(target: np.ndarray, retrieved: np.ndarray) -> float:
    return float(np.mean(target * retrieved))


def summarize_retrieval(
    patterns: np.ndarray,
    target_indices: np.ndarray,
    retrieved_states: np.ndarray,
) -> dict:
    retrieved_indices = np.array(
        [nearest_pattern(patterns, state) for state in retrieved_states], dtype=int
    )
    overlaps = np.array(
        [overlap(patterns[target], state) for target, state in zip(target_indices, retrieved_states)]
    )
    nearest_target = retrieved_indices == target_indices
    successful = (overlaps >= 0.95) & nearest_target
    return {
        "success_fraction": float(np.mean(successful)),
        "mean_overlap": float(np.mean(overlaps)),
        "overlap_stderr": float(np.std(overlaps, ddof=1) / np.sqrt(len(overlaps))),
        "nearest_target_fraction": float(np.mean(nearest_target)),
        "overlaps": overlaps.tolist(),
        "target_indices": target_indices.tolist(),
        "retrieved_indices": retrieved_indices.tolist(),
        "successes": successful.tolist(),
    }


def attention_equivalence(patterns: np.ndarray, queries: np.ndarray) -> list[dict]:
    pattern_matrix = jnp.asarray(patterns.T)
    records = []
    for beta in BETA_VALUES:
        query_array = jnp.asarray(queries)
        modern = jax.vmap(
            lambda query: modern_hopfield_update(pattern_matrix, query, beta)
        )(query_array)
        attention = jax.vmap(
            lambda query: attention_update(pattern_matrix, query, beta)
        )(query_array)
        records.append(
            {
                "P": len(patterns),
                "beta": beta,
                "maximum_absolute_difference": float(jnp.max(jnp.abs(modern - attention))),
            }
        )
    return records


def main() -> None:
    started = time.perf_counter()
    train_images, _, _, _ = load_mnist(INPUT_PATH)
    all_spins = binarize_to_spins(train_images)

    attention_records = []
    retrieval_records = []
    for pattern_count in PATTERN_COUNTS:
        patterns = all_spins[:pattern_count]
        target_indices, queries = make_corrupted_queries(patterns, query_seed(pattern_count))
        attention_records.extend(attention_equivalence(patterns, queries))

        coupling = hebbian_coupling(patterns.astype(np.int8))
        classical_states = []
        for query_index, query in enumerate(queries):
            state, _, _ = asynchronous_recall(
                query.astype(np.int8),
                coupling,
                query_seed(pattern_count) + 100_000 + query_index,
            )
            classical_states.append(state.astype(np.float32))
        classical_summary = summarize_retrieval(
            patterns, target_indices, np.asarray(classical_states)
        )

        pattern_matrix = jnp.asarray(patterns.T)
        for beta in BETA_VALUES:
            modern_states = np.asarray(
                jax.vmap(
                    lambda query: modern_hopfield_update(pattern_matrix, query, beta)
                )(jnp.asarray(queries))
            )
            modern_states = np.where(modern_states >= 0.0, 1.0, -1.0).astype(np.float32)
            modern_summary = summarize_retrieval(patterns, target_indices, modern_states)
            retrieval_records.append(
                {
                    "P": pattern_count,
                    "beta": beta,
                    "modern_hopfield": modern_summary,
                    "classical_hopfield": classical_summary,
                }
            )
        print(f"P={pattern_count}: completed {len(BETA_VALUES)} beta values", flush=True)

    ATTENTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    attention_output = {
        "formula": "x_new = X @ softmax(beta * X.T @ x)",
        "attention": "query=x, keys=X.T, values=X.T",
        "pattern_counts": list(PATTERN_COUNTS),
        "beta_values": list(BETA_VALUES),
        "tested_queries_per_setting": QUERY_COUNT,
        "records": attention_records,
    }
    ATTENTION_OUTPUT_PATH.write_text(json.dumps(attention_output, indent=2) + "\n", encoding="utf-8")

    results_output = {
        "protocol": {
            "pattern_counts": list(PATTERN_COUNTS),
            "beta_values": list(BETA_VALUES),
            "queries_per_setting": QUERY_COUNT,
            "corruption": "replace lower 14 image rows with random +/-1 spins",
            "success": "overlap >= 0.95 and nearest stored pattern is the target",
            "master_seed": MASTER_SEED,
            "patterns": "first P MNIST training images in stored order",
        },
        "results": retrieval_records,
        "runtime_seconds": time.perf_counter() - started,
    }
    RESULTS_OUTPUT_PATH.write_text(json.dumps(results_output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ATTENTION_OUTPUT_PATH}")
    print(f"wrote {RESULTS_OUTPUT_PATH}")
    print(f"runtime: {results_output['runtime_seconds']:.2f} s")


if __name__ == "__main__":
    main()
