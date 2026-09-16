"""Shared readers and aggregators for the Week 3 Part 2 analysis."""

import json
from collections import defaultdict
from pathlib import Path


WEEK3 = Path(__file__).resolve().parents[1]
ARTIFACTS = WEEK3 / "artifacts"
COARSE_NAMES = {32: "coarse-l32", 64: "coarse-l64"}
WINDOW_NAMES = {32: "window-l32", 64: "window-l64"}


def temperature_key(temperature: float) -> int:
    """Use millikelvin integer keys so decimal grid values match robustly."""
    return round(temperature * 1000)


def aggregate(path: Path) -> dict[int, dict[str, float]]:
    values = defaultdict(lambda: {"T": 0.0, "count": 0, "abs_m": 0.0, "m2": 0.0})
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = temperature_key(row["T"])
            value = values[key]
            value["T"] = row["T"]
            value["count"] += 1
            value["abs_m"] += abs(row["M"])
            value["m2"] += row["M"] ** 2
    if not values:
        raise ValueError(f"no data in {path}")
    return dict(values)


def load_run(directory: Path) -> dict:
    with (directory / "run.json").open() as stream:
        return json.load(stream)


def check_run(directory: Path, expected_l: int, expected_name: str) -> dict:
    run = load_run(directory)
    if run["L"] != expected_l or run["update"] != "metropolis":
        raise ValueError(f"unexpected run parameters in {directory}")
    if run["time_unit"] != "sweep" or run["sample_every"] != 1:
        raise ValueError(f"unexpected sampling parameters in {directory}")
    if directory.name != expected_name:
        raise ValueError(f"unexpected data grouping for {directory}")
    return run


def merged_data(lattice_size: int) -> dict[int, dict[str, float]]:
    coarse_dir = ARTIFACTS / COARSE_NAMES[lattice_size]
    window_dir = ARTIFACTS / WINDOW_NAMES[lattice_size]
    check_run(coarse_dir, lattice_size, COARSE_NAMES[lattice_size])
    check_run(window_dir, lattice_size, WINDOW_NAMES[lattice_size])
    coarse = aggregate(coarse_dir / "series.jsonl")
    window = aggregate(window_dir / "series.jsonl")
    merged = {key: value for key, value in coarse.items() if key not in window}
    merged.update(window)
    if len(merged) != 27:
        raise ValueError(f"expected 27 merged temperatures for L={lattice_size}, got {len(merged)}")
    return merged


def mean_abs_m(value: dict[str, float]) -> float:
    return value["abs_m"] / value["count"]


def susceptibility(value: dict[str, float], lattice_size: int) -> float:
    mean_abs = mean_abs_m(value)
    mean_m2 = value["m2"] / value["count"]
    return lattice_size * lattice_size * (mean_m2 - mean_abs * mean_abs) / value["T"]
