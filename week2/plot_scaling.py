from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for line in Path("scaling-results.csv").read_text().splitlines()[1:]:
    n, method, repeat, seconds, status, digest, command = line.split(",", 6)
    rows.append((int(n), method, float(seconds), int(status)))
if len(rows) != 18 or any(status != 0 for *_, status in rows):
    raise RuntimeError("expected 18 successful benchmark rows")
fig, ax = plt.subplots(figsize=(7, 4.5))
for method, color in [("naive", "tab:blue"), ("cells", "tab:orange")]:
    points = []
    for n in (100, 400, 1600):
        values = [seconds for nn, mm, seconds, _ in rows if nn == n and mm == method]
        points.append((n, median(values) / 600.0))
    ax.plot([n for n, _ in points], [value for _, value in points], "o-", label=method)
    for n, value in points:
        ax.annotate(f"{value:.3g}", (n, value), textcoords="offset points", xytext=(0, 6), ha="center")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("seconds per step")
ax.set_title("MD scaling (600 integration steps; initialization/output included)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(title="force method")
fig.tight_layout()
fig.savefig("scaling.png", dpi=150)
