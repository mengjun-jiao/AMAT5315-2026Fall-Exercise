#!/usr/bin/env python3
"""Create the official Treeverse action and work evidence figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (1, 3, 5, 10)
COLORS = {"store": "tab:blue", "restore": "tab:orange", "call": "tab:green", "grad": "tab:red", "fetch": "tab:purple"}
LABELS = {"store": "Store", "restore": "Restore", "call": "Call", "grad": "Grad", "fetch": "Fetch"}


def main() -> None:
    actions = json.loads((ROOT / "artifacts/checkpoint-5/actions-0.json").read_text())
    figure, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    for kind in ("store", "restore", "call", "grad", "fetch"):
        indices = [i for i, action in enumerate(actions) if action["action"] == kind]
        steps = [actions[i]["step"] for i in indices]
        axis.scatter(indices, steps, s=8 if kind == "call" else 22, color=COLORS[kind], label=LABELS[kind], alpha=0.8)
    axis.set_title("Treeverse schedule; reflector shot 0, budget 5")
    axis.set_xlabel("Operation index")
    axis.set_ylabel("Timestep")
    axis.set_ylim(-5, 245)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=5, loc="upper center")
    figure.savefig(ROOT / "artifacts/checkpoint-actions.png", dpi=150)
    plt.close(figure)

    forwards = []
    bytes_saved = []
    states = []
    for delta in BUDGETS:
        result = json.loads((ROOT / f"artifacts/checkpoint-{delta}/result.json").read_text())
        stats = result["statistics"]
        forwards.append(stats["per_shot"][0]["scheduler_forward_calls"])
        bytes_saved.append(stats["peak_saved_bytes"])
        states.append(stats["peak_saved_states"])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].plot(BUDGETS, forwards, "o-", color="tab:blue", label="Treeverse")
    axes[0].plot(BUDGETS, [240] * len(BUDGETS), "--", color="black", label="Full history")
    axes[0].set_yscale("log")
    axes[0].set_title("Scheduler recomputation cost")
    axes[0].set_xlabel("Additional checkpoint slots, delta")
    axes[0].set_ylabel("Scheduler forward steps per shot")
    axes[0].set_xticks(BUDGETS)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].plot(BUDGETS, bytes_saved, "o-", color="tab:green", label="Treeverse")
    axes[1].axhline(6481936, linestyle="--", color="black", label="Full history: 241 states")
    for delta, value, count in zip(BUDGETS, bytes_saved, states):
        axes[1].annotate(f"{count} states", (delta, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_title("Saved-state storage")
    axes[1].set_xlabel("Additional checkpoint slots, delta")
    axes[1].set_ylabel("Peak saved-state bytes")
    axes[1].set_xticks(BUDGETS)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    figure.savefig(ROOT / "artifacts/checkpoint-work.png", dpi=150)
    plt.close(figure)
    print("Wrote checkpoint-actions.png and checkpoint-work.png")


if __name__ == "__main__":
    main()
