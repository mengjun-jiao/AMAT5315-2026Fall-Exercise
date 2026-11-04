"""Intercept rendered figures and check plotted artists against saved numerical results."""
from pathlib import Path
import contextlib
import io
import json
import runpy
import sys
import numpy as np
from matplotlib.figure import Figure

script = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("plot_evidence.py")
root = script.resolve().parent.parent

def get(p):
    return json.loads((root / p).read_text())

runs = {c: get(f"artifacts/{c}/run.json") for c in ["prior", "AD", "ADX"]}
bp = {c: get(f"artifacts/bp-{c}/run.json") for c in ["prior", "AD", "ADX"]}
plans = {c: get(f"runs/{c}/order.json") for c in ["random", "greedy"]}
checked = set()

def verify(fig, target, *args, **kwargs):
    name = Path(target).name
    checked.add(name)
    if name in ("graphs.svg", "graphs.png"):
        labels = {t.get_text() for t in fig.axes[0].texts}
        for token in [r"$p(a)$", r"$p(t\mid a)$", r"$p(s)$", r"$p(l\mid s)$",
                      r"$p(b\mid s)$", r"$p(e\mid t,l)$", r"$p(x\mid e)$",
                      r"$p(d\mid e,b)$", r"$T$"]:
            assert token in labels, token
        assert sum(t.get_text()=="(1, 0)" for t in fig.axes[0].texts)==2
        assert sum(t.get_text()=="(1, 1)" for t in fig.axes[0].texts)==1
    elif name == "posteriors.png":
        for ax,c in zip(fig.axes,["prior","AD","ADX"]):
            enumeration = [runs[c]["comparison_with_enumeration"]["enumeration_marginals"][v][0] for v in [1,3,4]]
            contraction = [runs[c]["marginals"][str(v)][0] for v in [1,3,4]]
            assert np.allclose([p.get_width() for p in ax.patches],enumeration,rtol=0,atol=1e-15)
            assert np.allclose(ax.lines[0].get_xdata(),contraction,rtol=0,atol=1e-15)
    elif name == "joint.png":
        for row,c in enumerate(["prior","AD","ADX"]):
            j=np.array(runs[c]["joint_TL"])
            p=np.outer(runs[c]["marginals"]["1"],runs[c]["marginals"]["3"])
            for col,expected in enumerate([j,p,j-p]):
                assert np.allclose(fig.axes[row*3+col].images[0].get_array(),expected,rtol=0,atol=1e-15)
    elif name == "orders.png":
        for ax,key in [(fig.axes[2],"total_work"),(fig.axes[3],"largest_tensor")]:
            assert [p.get_width() for p in ax.patches] == [plans[c][key] for c in ["random","greedy"]]
    elif name == "bp_asia.png":
        for i,c in enumerate(["prior","AD","ADX"]):
            truth=np.array(runs[c]["comparison_with_enumeration"]["enumeration_marginals"])
            residual=[row["residual"] for row in bp[c]["trace"]]
            errors=[float(np.max(abs(np.array(row["beliefs"])-truth))) for row in bp[c]["trace"]]
            assert np.allclose(fig.axes[2*i].lines[0].get_ydata(),residual,rtol=0,atol=1e-15)
            assert np.allclose(fig.axes[2*i+1].lines[0].get_ydata(),errors,rtol=0,atol=1e-15)
    elif name == "bp_adx_comparison.png":
        ax=fig.axes[0]; vids=[1,3,4,2]
        expected=[runs["ADX"]["marginals"][str(v)][0] for v in vids]+[bp["ADX"]["marginals"][v][0] for v in vids]
        assert np.allclose([p.get_width() for p in ax.patches],expected,rtol=0,atol=1e-15)
        truth=np.array(runs["ADX"]["comparison_with_enumeration"]["enumeration_marginals"])
        errors=[float(np.max(abs(np.array(row["beliefs"])-truth))) for row in bp["ADX"]["trace"]]
        assert np.allclose(fig.axes[1].lines[0].get_ydata(),[row["residual"] for row in bp["ADX"]["trace"]],rtol=0,atol=1e-15)
        assert np.allclose(fig.axes[1].lines[1].get_ydata(),errors,rtol=0,atol=1e-15)
    elif name == "promedus.png":
        check=get("artifacts/promedus_checks.json")
        for ax,key in [(fig.axes[0],"work"),(fig.axes[1],"largest_generated")]:
            expected=[check["plans_independently_recounted"][f"promedus-{c}/order.json"][key] for c in ["greedy","treesa"]]
            assert [p.get_height() for p in ax.patches]==expected
        report=get("artifacts/promedus-bp/run.json")
        exact=np.array([check["greedy_marginals"][str(v)] for v in [59,102,139]])
        errs=[float(np.max(abs(np.array(t["beliefs"])[[59,102,139]]-exact))) for t in report["trace"]]
        assert np.allclose(fig.axes[2].lines[0].get_ydata(),[t["residual"] for t in report["trace"]],rtol=0,atol=1e-15)
        assert np.allclose(fig.axes[3].lines[0].get_ydata(),errs,rtol=0,atol=1e-15)
    else:
        raise AssertionError(f"Unexpected figure: {name}")

Figure.savefig=verify
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(script),run_name="__main__")
expected={"graphs.svg","graphs.png","posteriors.png","joint.png","orders.png","bp_asia.png","bp_adx_comparison.png","promedus.png"}
assert checked==expected,(checked,expected)
print("All eight rendered figure outputs match their source results and labels.")
