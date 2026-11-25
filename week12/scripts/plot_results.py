"""Render the four learning-sheet figures and print the measurements from saved runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150
BLUE, ORANGE, GREEN = "#205d96", "#be5606", "#196d47"
FILL_BLUE, EDGE_BLUE = "#60a5fa", "#233c8e"
FILL_ORANGE, EDGE_ORANGE = "#f59e0b", "#88451a"
POLICIES = ("most-frac", "neural", "root-neural", "strong")
NAMES = ("Most-Frac", "Online Neural", "Root Neural", "Strong Branch")
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "font.size": 10.3,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.4,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.75,
    "legend.framealpha": 1.0,
    "legend.fontsize": 10,
    "legend.borderpad": 0.35,
    "legend.handlelength": 2.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
}


FILES = {
    "gap": "gap/run.json",
    "most": "bnb_most/run.json",
    "strong": "bnb_strong/run.json",
    "model": "model.json",
    "test": "eval_test/run.json",
    "transfer": "eval_transfer/run.json",
}
INPUTS = {
    "gap": ("gap",),
    "bnb": ("most", "strong"),
    "neural_bnb": ("model", "test"),
    "eval": ("test", "transfer"),
}


def read_runs(root, figure=None):
    keys = FILES if figure is None else INPUTS[figure]
    runs = {key: json.loads((root / FILES[key]).read_text()) for key in keys}
    if "strong" in runs and runs["strong"]["parameters"]["export_labels"]:
        archive = root / "bnb_strong/run_no_labels.json"
        if archive.exists():
            plain = json.loads(archive.read_text())
            studied = ("spins", "edges", "count", "seed", "policy")
            if not plain["parameters"]["export_labels"] and all(
                plain["parameters"][k] == runs["strong"]["parameters"][k]
                for k in studied
            ):
                runs["strong"] = plain
    return runs


def frame(width, height, left, mid, right, bottom):
    # These plot-area boundaries are measured from the sheet's embedded reference figures.
    panel = right - mid
    fig, axes = plt.subplots(1, 2, figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.subplots_adjust(
        left=left / width,
        right=right / width,
        bottom=bottom / height,
        top=1 - 42 / height,
        wspace=(mid - left - panel) / panel,
    )
    # Keep the light background grid separate from the vector data and lettering in PDF reviews.
    for ax in axes:
        ax.tick_params(axis="both", which="major", grid_rasterized=True)
    return fig, axes


def box(ax, text, y, kind, *, right=False):
    fills = {
        "warm": ("#fff7ed", "#ffbb80"),
        "green": ("#f0fdf4", "#86efac"),
        "blue": ("#f8fafc", "#cbd5e1"),
    }
    face, edge = fills[kind]
    ax.text(
        0.96 if right else 0.03,
        y,
        text,
        transform=ax.transAxes,
        ha="right" if right else "left",
        va="top",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=face, edgecolor=edge),
    )


def records(run, policy, metric):
    return np.array(
        [
            row["policies"][policy]["search"][metric]
            for row in run["evidence"]["instances"]
        ]
    )


def ranked(ax, run, rule, metric, label):
    color, marker = {
        "most-frac": (ORANGE, "o"),
        "neural": (BLUE, "^"),
        "strong": (GREEN, "s"),
    }[rule]
    values = np.sort(records(run, rule, metric))
    ax.plot(
        np.arange(len(values)),
        values,
        color=color,
        marker=marker,
        ms=4.8 if marker != "^" else 6,
        lw=1.7,
        label=label,
    )
    ax.set_xlabel(f"Sorted instance rank (k = 0..{len(values) - 1})")


def mismatches(run, rules):
    return sum(
        any(row["policies"][p]["error"] > 1e-6 for p in rules)
        for row in run["evidence"]["instances"]
    )


def gap_figure(runs):
    run = runs["gap"]
    a = run["parameters"]
    rows = run["evidence"]["instances"]
    s = run["summary"]
    ids = [r["id"] for r in rows]
    data = [r["result"] for r in rows]
    lp, exact, gap, excess = (
        [r[key] for r in data] for key in ("lp", "exact", "gap", "excess")
    )
    fig, (left, right) = frame(1655, 603, 108, 922, 1642, 79)
    left.vlines(ids, lp, exact, colors="#94a3b8", lw=1.8)
    left.scatter(ids, lp, color=BLUE, s=45, label="LP lower bound", zorder=3)
    left.scatter(
        ids,
        exact,
        color="#bd460a",
        marker="s",
        s=40,
        label="Exact ILP ground state",
        zorder=3,
    )
    left.set(
        title=f"(a) Spin Glass LP vs. ILP energy ({a['spins']} spins, {a['edges']} bonds)",
        ylabel=r"Energy $H(s)$",
        xlabel="Seeded Spin Glass ID",
    )
    left.set_ylim(min(lp) - 4, max(exact) + 8.5)
    left.legend(loc="upper right")
    box(
        left,
        f"Mean gap = {s['mean_gap']:.2f} ({100 * s['mean_relative_gap']:.1f}%)\n"
        f"Suboptimal round: {s['suboptimal_roundings']}/{len(rows)}",
        0.95,
        "warm",
    )
    w = 0.38
    right.bar(
        np.asarray(ids) - w / 2,
        gap,
        w,
        color=FILL_BLUE,
        edgecolor=EDGE_BLUE,
        lw=1.0,
        label=r"Integrality gap ($E^* - z_{\mathrm{LP}}$)",
    )
    right.bar(
        np.asarray(ids) + w / 2,
        excess,
        w,
        color=FILL_ORANGE,
        edgecolor=EDGE_ORANGE,
        lw=1.0,
        label=r"Rounded excess ($E_{\mathrm{round}} - E^*$)",
    )
    right.set(
        title="(b) Integrality gap vs. naive rounding excess",
        ylabel="Energy difference",
        xlabel="Seeded Spin Glass ID",
        ylim=(0, max(excess) * 1.28),
    )
    right.set_xlim(-1.2, len(ids) + 0.2)
    right.set_xticks(np.arange(0, len(ids) + 1, 2))
    left.set_xticks(np.arange(0, len(ids), 2))
    right.legend(loc="upper right")
    return fig


def bnb_figure(runs):
    most, strong = runs["most"], runs["strong"]
    m, s = most["summary"]["most-frac"], strong["summary"]["strong"]
    a = most["parameters"]
    fig, (left, right) = frame(1655, 603, 100, 928, 1648, 79)
    for rule, run, stats in (("most-frac", most, m), ("strong", strong, s)):
        ranked(left, run, rule, "nodes", f"{rule} (mean {stats['mean_nodes']:.1f})")
        ranked(
            right, run, rule, "lp_solves", f"{rule} ({stats['mean_lp_solves']:.1f} LPs)"
        )
    count = a["count"]
    left.set(
        title=f"(a) B&B tree size on {count} Spin Glasses ({a['spins']} spins, {a['edges']} bonds)",
        ylabel="B&B nodes explored",
    )
    right.set(
        title="(b) Total HiGHS LP solves (lookahead cost)",
        ylabel="LP relaxations solved per instance",
    )
    bad = sum(
        any(
            r["evidence"]["instances"][i]["policies"][p]["error"] > 1e-9
            for r, p in ((most, "most-frac"), (strong, "strong"))
        )
        for i in range(count)
    )
    box(
        left,
        f"Node cut: {m['mean_nodes'] / s['mean_nodes']:.2f}x smaller tree\nExact opt: {bad}/{count} mismatches",
        0.74,
        "green",
    )
    box(
        right,
        f"SB cost: {s['mean_lp_solves'] / s['mean_nodes']:.1f} LP solves / node\n"
        f"Wall time: {s['mean_wall_ms'] / m['mean_wall_ms']:.1f}x slower",
        0.74,
        "warm",
    )
    for ax in (left, right):
        ax.legend(loc="upper left")
    return fig


def neural_figure(runs):
    model, test = runs["model"], runs["test"]
    fig, (left, right) = frame(1656, 603, 92, 924, 1648, 79)
    history = model["history"]
    epochs = [r["epoch"] for r in history]
    left.plot(
        epochs, [r["train_ce"] for r in history], color=BLUE, lw=2, label="Train CE"
    )
    left.plot(
        epochs,
        [r["val_ce"] for r in history],
        color="#c4470c",
        lw=2,
        ls="--",
        label="Val CE (disjoint instances)",
    )
    left.set(
        title="(a) Masked softmax cross-entropy loss",
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
    )
    vals = [r[k] for r in history for k in ("train_ce", "val_ce")]
    left.set_ylim(min(vals) - 0.12, max(vals) + 0.32)
    left.legend(loc="upper right")
    box(
        left,
        f"Train: {model['train_nodes']} nodes ({len(model['train_instances'])} insts)\n"
        f"Val: {model['val_nodes']} nodes ({len(model['val_instances'])} insts)\n"
        f"Checkpoint val CE: {model['best_val_ce']:.3f}",
        0.74,
        "blue",
        right=True,
    )
    summary = test["summary"]
    count = test["parameters"]["count"]
    for p in ("most-frac", "neural", "strong"):
        ranked(right, test, p, "nodes", f"{p} (mean {summary[p]['mean_nodes']:.1f})")
    cut = 1 - summary["neural"]["mean_nodes"] / summary["most-frac"]["mean_nodes"]
    bad = mismatches(test, ("most-frac", "neural", "strong"))
    right.set(
        title=f"(b) B&B tree size on {count} unseen Spin Glasses",
        ylabel="B&B nodes explored",
        ylim=(0, None),
    )
    right.legend(loc="upper left")
    box(
        right,
        f"Node cut: {cut:.0%} vs most-frac\nExact opt: {bad}/{count} mismatches",
        0.64,
        "green",
    )
    return fig


def eval_figure(runs):
    fig, (left, right) = frame(1692, 639, 100, 947, 1685, 76)
    x = np.arange(4)
    w = 0.36
    for j, (key, name, face, edge) in enumerate(
        (
            ("test", "Held-out", FILL_BLUE, EDGE_BLUE),
            ("transfer", "Transfer", FILL_ORANGE, EDGE_ORANGE),
        )
    ):
        run = runs[key]
        a = run["parameters"]
        values = [run["summary"][p]["mean_nodes"] for p in POLICIES]
        bars = left.bar(
            x + (j - 0.5) * w,
            values,
            w,
            color=face,
            edgecolor=edge,
            lw=1,
            label=f"{name} ({a['spins']} spins, {a['edges']} bonds)",
        )
        left.bar_label(
            bars,
            labels=[f"{v:.1f}" for v in values],
            padding=2.4,
            fontsize=9.4,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
        )
    left.set_ylim(
        0,
        max(
            v["mean_nodes"]
            for key in ("test", "transfer")
            for v in runs[key]["summary"].values()
        )
        * 1.24,
    )
    left.legend(loc="upper right")
    left.set(title="(a) Rust B&B tree size across policies", ylabel="Mean B&B nodes")
    transfer = runs["transfer"]
    s = transfer["summary"]
    total = np.array([s[p]["mean_wall_ms"] for p in POLICIES])
    inference = np.array([s[p]["mean_inference_ms"] for p in POLICIES])
    rest = total - inference
    right.bar(x, rest, color="#568ff1", label="HiGHS LP + search ms")
    bars = right.bar(
        x, inference, bottom=rest, color="#ef4444", label="Neural infer ms"
    )
    total_labels = right.bar_label(
        bars,
        labels=[f"{v:.1f} ms" for v in total],
        padding=3.5,
        fontweight="bold",
        fontsize=9.4,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
    )
    # At the reference scale the two positive inference segments are subpixel.
    # Keep their true geometry and print the measured components above the totals.
    for label, part, rule in zip(total_labels, inference, POLICIES, strict=True):
        if rule in ("neural", "root-neural"):
            right.annotate(
                f"infer {part:.3f} ms",
                xy=label.xy,
                xytext=(0, 16),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color="#963833",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
            )
    a = transfer["parameters"]
    right.set(
        title=f"(b) Rust CPU time on larger transfer ({a['spins']} spins, {a['edges']} bonds)",
        ylabel="Mean CPU wall time (ms / instance)",
        ylim=(0, total.max() * 1.22),
    )
    right.legend(loc="upper left")
    for ax in (left, right):
        ax.set_xticks(x, NAMES, rotation=12)
    return fig


BUILDERS = {
    "gap": gap_figure,
    "bnb": bnb_figure,
    "neural_bnb": neural_figure,
    "eval": eval_figure,
}


def gap_report(runs):
    run = runs["gap"]
    ev, s = run["evidence"], run["summary"]
    sq, diet = ev["square"], ev["diet"]
    lines = [
        "Part 1. Mixed-integer formulation and LP relaxation",
        f"Square LP {sq['lp']:.6f}, x = {sq['lp_x']}, z = {sq['lp_z']}",
        f"Square exact {sq['exact']:.6f}, bits = {sq['exact_bits']}. Rounded {sq['rounded_energy']:+.6f}, excess {sq['excess']:.6f}.",
        f"Diet x = ({diet['x'][0]:.9f}, {diet['x'][1]:.9f}), cost {diet['cost']:.6f}.",
        f"Mean LP {s['mean_lp']:.2f}, exact {s['mean_exact']:.2f}, gap {s['mean_gap']:.2f} "
        f"({s['mean_relative_gap']:.2%} mean relative), fractional spins {s['mean_fractional']:.2f}.",
        f"Suboptimal roundings {s['suboptimal_roundings']}/{run['parameters']['count']}, mean excess {s['mean_rounding_excess']:.2f}.",
    ]
    return lines


def bnb_report(runs):
    lines = ["Part 2. Depth-first branch-and-bound"]
    for i, v in enumerate(runs["most"]["evidence"]["tetrahedron"]["trace"], 1):
        path = " / ".join(f"x{j}={b}" for j, b in v["path"]) or "root"
        split = f" on x{v['split']}" if v["split"] is not None else ""
        x = ", ".join(f"{abs(x):g}" for x in v["x"])
        lines.append(
            f"Node {i}  {path:16}  {v['status']}{split}  LP {v['bound']:g}  x = ({x})"
        )
    lines.append(
        "Policy     mean nodes    mean LP solves    mean wall ms    max HiGHS error"
    )
    for key, p in (("most", "most-frac"), ("strong", "strong")):
        row = runs[key]["summary"][p]
        lines.append(
            f"{p:10} {row['mean_nodes']:9.3f} {row['mean_lp_solves']:17.3f} "
            f"{row['mean_wall_ms']:15.3f} {row['max_error']:18.3g}"
        )
    largest = max(
        runs[key]["summary"][policy]["max_error"]
        for key, policy in (("most", "most-frac"), ("strong", "strong"))
    )
    lines.append(
        f"Largest absolute HiGHS error over both rules {largest:.3g} (tolerance 1e-9)."
    )
    return lines


def neural_report(runs):
    test, m = runs["test"], runs["model"]
    e = test["evidence"]["largest_saving"]
    lines = [
        "Part 3. Imitating Strong Branching",
        f"Train {m['train_nodes']} nodes / {len(m['train_instances'])} instances. "
        f"Validation {m['val_nodes']} nodes / {len(m['val_instances'])} instances. "
        f"Overlap {len(set(m['train_instances']) & set(m['val_instances']))}.",
        f"Best checkpoint epoch {m['best_epoch']}, validation CE {m['best_val_ce']:.6f}.",
        f"Maximum probability difference over six Rust/JAX probes {test['evidence']['parity']['max_probability_error']:.3g}.",
        f"Largest saving on instance {e['id']}, energy {e['energy']:.0f}.",
    ]
    for p in ("most-frac", "neural"):
        row = e[p]
        lines.append(
            f"{p:10} first two splits {row['first_splits']}, {row['nodes']} nodes, {row['wall_ms']:.3f} ms."
        )
    parts = ("most-frac", "neural", "strong")
    lines.append("Policy       mean nodes    mean wall ms     max error   LPs = nodes")
    for p in parts:
        v = test["summary"][p]
        lines.append(
            f"{p:10} {v['mean_nodes']:12.3f} {v['mean_wall_ms']:15.3f} "
            f"{v['max_error']:13.3g}   {v['lp_solves_equal_nodes']}"
        )
    lines.append(
        f"Largest held-out HiGHS error {max(test['summary'][p]['max_error'] for p in parts):.3g}."
    )
    return lines


def eval_report(runs):
    test, transfer = runs["test"], runs["transfer"]
    lines = [
        "Part 4. Per-search means from single-threaded, timed CPU runs",
        "Set       policy       nodes   LP solves    wall ms      LP ms   infer ms     max error    LPs = nodes",
    ]
    for name, r in (("held-out", test), ("transfer", transfer)):
        for p in POLICIES:
            v = r["summary"][p]
            lines.append(
                f"{name:9} {p:11} {v['mean_nodes']:7.3f} {v['mean_lp_solves']:10.3f} "
                f"{v['mean_wall_ms']:10.3f} {v['mean_lp_ms']:10.3f} {v['mean_inference_ms']:10.4f} "
                f"{v['max_error']:13.3g}    {v['lp_solves_equal_nodes']}"
            )
    lines.append(
        f"Largest evaluation error against HiGHS {max(v['max_error'] for r in (test, transfer) for v in r['summary'].values()):.3g}."
    )
    return lines


REPORTS = {
    "gap": gap_report,
    "bnb": bnb_report,
    "neural_bnb": neural_report,
    "eval": eval_report,
}


def report(runs, figure=None):
    names = (figure,) if figure is not None else BUILDERS
    return "\n\n".join("\n".join(REPORTS[name](runs)) for name in names) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--figure",
        choices=BUILDERS,
        help="render and print one completed worksheet part",
    )
    args = parser.parse_args()
    runs = read_runs(args.artifacts, args.figure)
    with plt.rc_context(STYLE):
        for name in (args.figure,) if args.figure else BUILDERS:
            fig = BUILDERS[name](runs)
            fig.savefig(args.artifacts / f"{name}.png", dpi=DPI)
            plt.close(fig)
    text = report(runs, args.figure)
    if args.figure is None:
        (args.artifacts / "report.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
