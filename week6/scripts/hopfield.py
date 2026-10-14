"""Classical Hopfield storage and asynchronous recall experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "artifacts" / "hopfield_results.json"

SIZES = (100, 200, 400, 800)
ALPHAS = np.round(np.arange(0.05, 0.2501, 0.01), 2)
INDEPENDENT_SETS = 20
MASTER_SEED = 6608
RECALL_PATTERNS = 10


def generate_patterns(pattern_count: int, size: int, seed: int) -> np.ndarray:
    """Generate independent +/-1 patterns with equal probability."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array((-1, 1), dtype=np.int8), size=(pattern_count, size))


def hebbian_coupling(patterns: np.ndarray) -> np.ndarray:
    """Return J_ij = sum_mu xi_i^mu xi_j^mu / N with J_ii = 0."""
    size = patterns.shape[1]
    coupling = patterns.T.astype(np.float32) @ patterns.astype(np.float32) / size
    np.fill_diagonal(coupling, 0.0)
    return coupling


def energy(state: np.ndarray, coupling: np.ndarray) -> float:
    """Evaluate E(s) = -1/2 s^T J s."""
    return float(-0.5 * state @ coupling @ state)


def local_field(state: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    """Return h_i = sum_j J_ij s_j."""
    return coupling @ state


def overlap(pattern: np.ndarray, state: np.ndarray) -> float:
    """Return m = N^-1 sum_i xi_i s_i."""
    return float(np.mean(pattern * state))


def unstable_fraction(pattern: np.ndarray, fields: np.ndarray) -> float:
    """Return the fraction of bits with xi_i h_i < 0."""
    return float(np.mean(pattern * fields < 0.0))


def asynchronous_recall(
    initial_state: np.ndarray,
    coupling: np.ndarray,
    seed: int,
    max_sweeps: int = 1000,
) -> tuple[np.ndarray, int, int]:
    """Recall a state with random-order zero-temperature asynchronous updates."""
    state = initial_state.astype(np.int8, copy=True)
    fields = local_field(state, coupling)
    rng = np.random.default_rng(seed)
    flips = 0
    for sweep in range(1, max_sweeps + 1):
        changed = 0
        for index in rng.permutation(len(state)):
            field = fields[index]
            new_spin = 1 if field > 0.0 else -1 if field < 0.0 else int(state[index])
            if new_spin != state[index]:
                delta = new_spin - int(state[index])
                state[index] = new_spin
                fields += coupling[:, index] * delta
                changed += 1
                flips += 1
        if changed == 0:
            return state, sweep, flips
    return state, max_sweeps, flips


def experiment_seed(size_index: int, alpha_index: int, set_index: int) -> int:
    return MASTER_SEED + size_index * 1_000_000 + alpha_index * 1_000 + set_index


def run_experiment(size_index: int, size: int, alpha_index: int, alpha: float) -> dict:
    pattern_count = int(round(alpha * size))
    set_records = []
    all_unstable = []
    all_overlaps = []

    for set_index in range(INDEPENDENT_SETS):
        pattern_seed = experiment_seed(size_index, alpha_index, set_index)
        patterns = generate_patterns(pattern_count, size, pattern_seed)
        coupling = hebbian_coupling(patterns)
        pattern_fields = patterns.astype(np.float32) @ coupling.T
        set_unstable = float(np.mean(patterns.astype(np.float32) * pattern_fields < 0.0))
        recalls = []
        for pattern_index in range(min(RECALL_PATTERNS, pattern_count)):
            recall_seed = pattern_seed + 100_000 + pattern_index
            final_state, sweeps, flips = asynchronous_recall(
                patterns[pattern_index], coupling, recall_seed
            )
            final_overlap = overlap(patterns[pattern_index], final_state)
            recalls.append(
                {
                    "pattern": pattern_index,
                    "seed": recall_seed,
                    "final_overlap": final_overlap,
                    "sweeps": sweeps,
                    "flips": flips,
                    "final_energy": energy(final_state, coupling),
                }
            )
            all_overlaps.append(final_overlap)
        all_unstable.append(set_unstable)
        set_records.append(
            {
                "set": set_index,
                "pattern_seed": pattern_seed,
                "unstable_fraction": set_unstable,
                "recalls": recalls,
            }
        )

    return {
        "N": size,
        "alpha": float(alpha),
        "P": pattern_count,
        "unstable_fraction": float(np.mean(all_unstable)),
        "final_overlap": float(np.mean(all_overlaps)),
        "seeds": {
            "master": MASTER_SEED,
            "pattern": [record["pattern_seed"] for record in set_records],
            "recall": [
                recall["seed"]
                for record in set_records
                for recall in record["recalls"]
            ],
        },
        "independent_sets": set_records,
    }


def main() -> None:
    started = time.perf_counter()
    results = []
    total = len(SIZES) * len(ALPHAS)
    completed = 0
    for size_index, size in enumerate(SIZES):
        for alpha_index, alpha in enumerate(ALPHAS):
            record = run_experiment(size_index, size, alpha_index, float(alpha))
            results.append(record)
            completed += 1
            print(
                f"[{completed:02d}/{total}] N={size:4d} alpha={alpha:.2f} "
                f"unstable={record['unstable_fraction']:.5f} "
                f"overlap={record['final_overlap']:.5f}",
                flush=True,
            )

    output = {
        "experiment": {
            "model": "classical Hopfield network",
            "sizes": list(SIZES),
            "alphas": [float(alpha) for alpha in ALPHAS],
            "independent_pattern_sets": INDEPENDENT_SETS,
            "recall_patterns_per_set": RECALL_PATTERNS,
            "update": "zero-temperature asynchronous random-order",
            "zero_field_rule": "keep previous spin",
            "master_seed": MASTER_SEED,
        },
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"runtime: {output['runtime_seconds']:.2f} s")


if __name__ == "__main__":
    main()
