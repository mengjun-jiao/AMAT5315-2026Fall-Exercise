#!/usr/bin/env python3
"""Independently audit every saved scan, peak fit, SVD reconstruction and plotted data."""
from pathlib import Path
import contextlib
import csv
import importlib.util
import io
import json
import math
import runpy
import sys
import numpy as np
from matplotlib.figure import Figure

ROOT=Path(__file__).resolve().parent.parent
TC=2/math.log(1+math.sqrt(2))

def fit_peak(ticks,lookup,stride):
    spacing=(ticks[stride]-ticks[0])/800.
    cv=[]
    for i in range(stride,len(ticks)-stride,stride):
        k=ticks[i]/800.
        cv.append((ticks[i],k*k*(lookup[ticks[i+stride]]-2*lookup[ticks[i]]+lookup[ticks[i-stride]])/spacing**2))
    i=int(np.argmax([v for _,v in cv]));assert 0<i<len(cv)-1
    (left,yl),(middle,ym),(right,yr)=cv[i-1:i+2]
    assert middle-left==right-middle
    delta=.5*(yl-yr)/(yl-2*ym+yr)*spacing
    k=middle/800.+delta
    assert abs(delta)<=spacing
    return k,1/k,cv

def audit_method(method,sizes):
    prefix="boundary" if method=="boundary" else "trg"
    summary=json.loads((ROOT/"artifacts"/f"{prefix}-summary.json").read_text())
    with (ROOT/"artifacts"/f"{prefix}-g.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    lookup={}
    for row in rows:
        l=int(row["L"]);chi=int(row["chi"]);k=float(row["K"]);tick=round(800*k)
        assert row["method"]==method and abs(k-tick/800.)<1e-14
        key=(l,chi,tick);assert key not in lookup
        run=json.loads((ROOT/row["path"]/"run.json").read_text())
        assert run["method"]==method and run["l"]==l and run["chi"]==chi
        assert abs(run["k"]-k)<1e-14 and float(row["g"])==run["g"] and float(row["log_Z"])==run["log_z"]
        assert float(row["seconds"])==run["seconds"] and row["record"]==run["record"]
        assert math.isfinite(run["g"]) and abs(run["g"]-run["log_z"]/(l*l))<1e-14
        lookup[key]=run
    coarse={};refined={};temps=[]
    for l in sizes:
        coarse_ticks=list(range(320,401,2))
        g={tick:lookup[(l,16,tick)]["g"] for tick in coarse_ticks}
        cv=[]
        for i in range(1,40):
            tick=coarse_ticks[i];k=tick/800.
            cv.append((tick,k*k*(g[coarse_ticks[i+1]]-2*g[tick]+g[coarse_ticks[i-1]])/.0025**2))
        center=max(cv,key=lambda x:x[1])[0]
        assert center/800.==summary["sizes"][str(l)]["coarse_center_k"]
        coarse[l]=cv
        fine_ticks=list(range(center-4,center+5))
        results={}
        for chi in [16,32]:
            fine={tick:lookup[(l,chi,tick)]["g"] for tick in fine_ticks}
            for tick in fine_ticks:
                assert lookup[(l,chi,tick)]["record"]=="full",f"missing full stencil {method} {l} {chi} {tick}"
            _,t,cvfine=fit_peak(fine_ticks,fine,1)
            results[chi]=t;refined[(l,chi)]=cvfine
            recorded=summary["sizes"][str(l)][f"chi{chi}_h00125"]
            assert abs(t-recorded["t_peak"])<1e-9
            if chi==32:
                _,wide,_=fit_peak(fine_ticks,fine,2)
                assert abs(wide-summary["sizes"][str(l)]["chi32_h0025"]["t_peak"])<1e-9
                assert abs(wide-t)/t<.01
        assert abs(results[16]-results[32])/results[32]<.01
        temps.append(results[32])
    x=1/np.array(sizes,dtype=float);y=np.array(temps)
    for key,xx,yy in [("all_sizes",x,y),("largest_two",x[-2:],y[-2:])]:
        slope,intercept=np.polyfit(xx,yy,1)
        assert abs(intercept-summary["fits"][key]["intercept"])<1e-12
        assert abs(intercept-TC)/TC<.01

    # Inspect the actual Matplotlib artists instead of inferring values from pixels.
    module_path=ROOT/"scripts/sweep.py"
    spec=importlib.util.spec_from_file_location("week10_sweep_audit",module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    original=Figure.savefig;seen=[]
    def check_artist(fig,target,*args,**kwargs):
        ax,fitax,tableax=fig.axes
        for i,l in enumerate(sizes):
            xy=sorted((1/(tick/800.),value) for tick,value in coarse[l])
            assert np.allclose(ax.lines[2*i].get_xdata(),[p[0] for p in xy],rtol=0,atol=1e-12)
            assert np.allclose(ax.lines[2*i].get_ydata(),[p[1] for p in xy],rtol=0,atol=1e-12)
            xy=sorted((1/(tick/800.),value) for tick,value in refined[(l,32)])
            assert np.allclose(ax.lines[2*i+1].get_ydata(),[p[1] for p in xy],rtol=0,atol=1e-12)
        assert np.allclose(fitax.collections[0].get_offsets(),np.column_stack([x,y]),rtol=0,atol=1e-12)
        assert len(tableax.tables)==1
        seen.append(Path(target).name)
    Figure.savefig=check_artist
    try:
        with contextlib.redirect_stdout(io.StringIO()):module.plot(method,coarse,refined,summary)
    finally:
        Figure.savefig=original
    assert seen==[f"{prefix}-critical.png"]
    return {"rows_checked":len(rows),"peak_temperatures":temps,"fits":summary["fits"]}

def audit_svd():
    source=ROOT/"artifacts/svd-data"
    a=np.load(source/"matrix.npy");s=np.load(source/"singular_values.npy")
    assert a.shape==(512,512) and s.shape==(512,) and np.all(s[:-1]>=s[1:])
    norm=np.linalg.norm(a);total=np.sum(s*s);errs={}
    for rank in [16,64,128,256,512]:
        x=np.load(source/f"reconstruction-{rank}.npy")
        actual=float(np.linalg.norm(a-x)/norm)
        predicted=float(np.sqrt(np.sum(s[rank:]**2)/total))
        assert abs(actual-predicted)<1e-8
        errs[str(rank)]={"observed":actual,"predicted":predicted}
    assert errs["512"]["observed"]<1e-8
    crossing=next(r for r in range(513) if np.sqrt(np.sum(s[r:]**2)/total)<=.1)
    saved=json.loads((ROOT/"artifacts/svd-checks.json").read_text())
    assert crossing==saved["ten_percent_first_rank"]
    original=Figure.savefig;seen=[]
    def check_artist(fig,target,*args,**kwargs):
        spectrum,error=fig.axes
        assert np.array_equal(spectrum.lines[0].get_xdata(),np.arange(1,513))
        assert np.allclose(spectrum.lines[0].get_ydata(),s,rtol=0,atol=1e-15)
        prediction=np.array([np.sqrt(np.sum(s[r:]**2)/total) for r in range(513)])
        assert np.allclose(error.lines[0].get_ydata(),prediction,rtol=0,atol=1e-14)
        expected=np.array([[rank,errs[str(rank)]["observed"]] for rank in [16,64,128,256,512]])
        assert np.allclose(error.collections[0].get_offsets(),expected,rtol=0,atol=1e-14)
        assert np.allclose(error.lines[1].get_ydata(),[.1,.1])
        assert np.allclose(error.lines[2].get_xdata(),[256,256])
        seen.append(Path(target).name)
    Figure.savefig=check_artist
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT/"scripts/plot_svd.py"),run_name="__main__")
    finally:
        Figure.savefig=original
    assert seen==["svd.png"]
    return {"errors":errs,"first_rank_at_10_percent":crossing}

if __name__=="__main__":
    result={"svd":audit_svd(),"boundary":audit_method("boundary",[16,32,64]),"trg":audit_method("trg",[64,256,1024])}
    (ROOT/"artifacts/independent-audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
