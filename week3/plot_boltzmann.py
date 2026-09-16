"""Plot the Part 1 Boltzmann energy-ratio check for T=3.0 and T=3.1."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WEEK3 = Path(__file__).resolve().parent
T30 = WEEK3 / "runs" / "T3.0" / "series.jsonl"
T31 = WEEK3 / "runs" / "T3.1" / "series.jsonl"
OUTPUT = WEEK3 / "evidence" / "boltzmann.png"
BIN_WIDTH = 40.0


def total_energies(path: Path) -> tuple[np.ndarray, int, float]:
    rows = [__import__("json").loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError(f"no series rows found in {path}")
    lattice_size = rows[0]["L"]
    temperature = rows[0]["T"]
    if any(row["L"] != lattice_size or row["T"] != temperature for row in rows):
        raise ValueError(f"inconsistent L or T in {path}")
    return np.array([row["E"] * lattice_size * lattice_size for row in rows]), lattice_size, temperature


def main() -> None:
    energy30, l30, t30 = total_energies(T30)
    energy31, l31, t31 = total_energies(T31)
    if l30 != l31 or (t30, t31) != (3.0, 3.1):
        raise ValueError("expected matching L and temperatures 3.0 and 3.1")

    minimum = np.floor(min(energy30.min(), energy31.min()) / BIN_WIDTH) * BIN_WIDTH
    maximum = np.ceil(max(energy30.max(), energy31.max()) / BIN_WIDTH) * BIN_WIDTH
    bins = np.arange(minimum, maximum + BIN_WIDTH, BIN_WIDTH)
    counts30, _ = np.histogram(energy30, bins=bins)
    counts31, _ = np.histogram(energy31, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    probabilities30 = counts30 / counts30.sum()
    probabilities31 = counts31 / counts31.sum()
    valid = (counts30 >= 5) & (counts31 >= 5)
    if not np.any(valid):
        raise ValueError("no common bins have at least five samples")

    slope = 1 / 3.0 - 1 / 3.1
    log_ratio = np.log(probabilities31[valid] / probabilities30[valid])
    intercept = np.mean(log_ratio - slope * centers[valid])
    line_x = np.array([centers[valid].min(), centers[valid].max()])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, (histogram, ratio) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    histogram.hist(energy30, bins=bins, alpha=0.65, label="T = 3.0")
    histogram.hist(energy31, bins=bins, alpha=0.65, label="T = 3.1")
    histogram.set_ylabel("Count")
    histogram.set_title("Total energy distributions")
    histogram.legend()
    histogram.grid(alpha=0.25)

    ratio.plot(centers[valid], log_ratio, "o", label="Observed log probability ratio")
    ratio.plot(line_x, slope * line_x + intercept, "--", label=f"Fixed theoretical slope = {slope:.7f}")
    ratio.set_xlabel("Total energy E")
    ratio.set_ylabel(r"ln[P$_{3.1}$(E) / P$_{3.0}$(E)]")
    ratio.set_title("Boltzmann ratio")
    ratio.legend()
    ratio.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=160)
    print(f"saved {OUTPUT}")
    print(f"samples: T=3.0 {len(energy30)}, T=3.1 {len(energy31)}")
    print(f"common bins used: {valid.sum()}, fitted intercept: {intercept:.6f}")
    print(f"fixed theoretical slope: {slope:.7f}")


if __name__ == "__main__":
    main()
