"""Check the plotted series and rebuild the four PNGs in memory."""

from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.testing import assert_allclose

import plot_results as plots


def heights(group):
    return [bar.get_height() for bar in group]


def values(run, policy, key):
    rows = run["evidence"]["instances"]
    return [r["policies"][policy]["search"][key] for r in rows]


def check_artists(name, fig, runs):
    a, b = fig.axes
    assert "(a)" in a.get_title() and "(b)" in b.get_title()
    if name == "gap":
        rows = [r["result"] for r in runs["gap"]["evidence"]["instances"]]
        for key, artist in zip(("lp", "exact"), a.collections[1:], strict=True):
            assert_allclose(
                artist.get_offsets(), list(enumerate(r[key] for r in rows)), atol=1e-10
            )
        for key, bars in zip(("gap", "excess"), b.containers, strict=True):
            assert_allclose(heights(bars), [r[key] for r in rows])
        for i, seg in enumerate(a.collections[0].get_segments()):
            assert_allclose(seg, [[i, rows[i]["lp"]], [i, rows[i]["exact"]]])
    elif name == "bnb":
        for ax, key in ((a, "nodes"), (b, "lp_solves")):
            for line, run, policy in zip(
                ax.lines, ("most", "strong"), ("most-frac", "strong"), strict=True
            ):
                assert_allclose(line.get_xdata(), np.arange(48))
                assert_allclose(
                    line.get_ydata(), sorted(values(runs[run], policy, key))
                )
        rows = zip(
            runs["most"]["evidence"]["instances"],
            runs["strong"]["evidence"]["instances"],
            strict=True,
        )
        mismatches = sum(
            max(x["policies"]["most-frac"]["error"], y["policies"]["strong"]["error"])
            > 1e-9
            for x, y in rows
        )
        assert any(
            f"Exact opt: {mismatches}/48 mismatches" in t.get_text() for t in a.texts
        )
    elif name == "neural_bnb":
        for line, key in zip(a.lines, ("train_ce", "val_ce"), strict=True):
            assert_allclose(line.get_xdata(), np.arange(1, 121))
            assert_allclose(
                line.get_ydata(), [row[key] for row in runs["model"]["history"]]
            )
        for line, policy in zip(
            b.lines, ("most-frac", "neural", "strong"), strict=True
        ):
            assert_allclose(line.get_xdata(), np.arange(16))
            assert_allclose(
                line.get_ydata(), sorted(values(runs["test"], policy, "nodes"))
            )
    else:
        policies = ("most-frac", "neural", "root-neural", "strong")
        for group, run in zip(a.containers, ("test", "transfer"), strict=True):
            expected = [np.mean(values(runs[run], p, "nodes")) for p in policies]
            assert_allclose(heights(group), expected)
        total = np.array(
            [np.mean(values(runs["transfer"], p, "wall_ms")) for p in policies]
        )
        inference = np.array(
            [np.mean(values(runs["transfer"], p, "inference_ms")) for p in policies]
        )
        assert_allclose(heights(b.containers[0]), total - inference)
        assert_allclose(heights(b.containers[1]), inference)
        assert_allclose([bar.get_y() for bar in b.containers[1]], total - inference)
        for bars, dataset in zip(a.containers, ("test", "transfer"), strict=True):
            for bar, p in zip(bars, policies, strict=True):
                expected = np.mean(values(runs[dataset], p, "nodes"))
                assert any(
                    t.get_text() == f"{expected:.1f}"
                    and np.allclose(t.xy, (bar.get_x() + bar.get_width() / 2, expected))
                    for t in a.texts
                )
        for i, p in enumerate(policies):
            assert any(
                t.get_text() == f"{total[i]:.1f} ms"
                and np.allclose(t.xy, (i, total[i]))
                for t in b.texts
            )
            if p in ("neural", "root-neural"):
                assert any(
                    t.get_text() == f"infer {inference[i]:.3f} ms"
                    and np.allclose(t.xy, (i, total[i]))
                    for t in b.texts
                )


def verify(root=Path("artifacts")):
    sizes = {
        "gap": (1655, 603),
        "bnb": (1655, 603),
        "neural_bnb": (1656, 603),
        "eval": (1692, 639),
    }
    runs = plots.read_runs(root)
    with plots.plt.rc_context(plots.STYLE):
        for name, builder in plots.BUILDERS.items():
            fig = builder(runs)
            check_artists(name, fig, runs)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            with Image.open(root / f"{name}.png") as img:
                assert img.size == sizes[name], name
                assert np.array_equal(
                    canvas.buffer_rgba(), np.asarray(img.convert("RGBA"))
                ), name
            plots.plt.close(fig)
    assert (root / "report.txt").read_text() == plots.report(runs)
    print(
        "All plotted series, four PNG renders and the printed report match their source runs."
    )


if __name__ == "__main__":
    verify()
