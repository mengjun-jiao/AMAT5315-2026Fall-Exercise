#!/usr/bin/env python3
"""Run the required coarse and refined Ising scans, cache every g(K), fit and plot."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
EXE=ROOT/".install/bin/thermo"
if not EXE.exists():
    EXE=ROOT/"target/release/thermo"
SIZES={"boundary":[16,32,64],"trg":[64,256,1024]}
TC=2/math.log(1+math.sqrt(2))

@dataclass(frozen=True)
class Job:
    method:str
    l:int
    chi:int
    tick:int
    full:bool=False
    @property
    def k(self):return self.tick/800.
    @property
    def out(self):return ROOT/"runs/sweep"/self.method/f"L{self.l}"/f"chi{self.chi}"/f"k{self.tick:04d}"

def run_job(job):
    path=job.out/"run.json"
    if path.exists():
        result=json.loads(path.read_text())
        if (result["method"]==job.method and result["l"]==job.l and result["chi"]==job.chi
            and abs(result["k"]-job.k)<1e-14 and (not job.full or result["record"]=="full")):
            return job,result,True
    result=subprocess.run([str(EXE),"--method",job.method,"--l",str(job.l),"--k",f"{job.k:.12g}",
        "--chi",str(job.chi),"--out",str(job.out),"--record","full" if job.full else "summary"],
        cwd=ROOT,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f"{job}: {result.stderr}\n{result.stdout}")
    return job,json.loads(result.stdout),False

def run_jobs(jobs,workers,label):
    values={};start=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_job,job):job for job in jobs}
        for i,future in enumerate(as_completed(futures),1):
            job,run,cached=future.result();values[(job.method,job.l,job.chi,job.tick)]=run
            if i==1 or i%10==0 or i==len(futures):
                print(f"{label}: {i}/{len(futures)} complete; elapsed {time.monotonic()-start:.1f}s",flush=True)
    return values

def cv_points(ticks,gs,stride):
    assert len(ticks)>2*stride and all(ticks[i]-ticks[i-1]==ticks[1]-ticks[0] for i in range(1,len(ticks)))
    step=(ticks[stride]-ticks[0])/800.;out=[]
    for pos in range(stride,len(ticks)-stride,stride):
        k=ticks[pos]/800.;cv=k*k*(gs[pos+stride]-2*gs[pos]+gs[pos-stride])/step**2
        out.append((ticks[pos],float(cv)))
    return out

def peak(ticks,gs,stride):
    points=cv_points(ticks,gs,stride)
    i=int(np.argmax([v for _,v in points]))
    if i==0 or i==len(points)-1:
        raise ValueError(f"peak not bracketed for three-point fit: {points}")
    x=np.array([points[j][0]/800. for j in [i-1,i,i+1]])
    y=np.array([points[j][1] for j in [i-1,i,i+1]])
    # Centering improves conditioning for a small interval around K~0.44.
    fit=np.polyfit(x-x[1],y,2)
    assert fit[0]<0,"peak parabola is not concave"
    vertex=x[1]-fit[1]/(2*fit[0])
    assert x[0]<=vertex<=x[-1],(x,vertex)
    return {"k_peak":float(vertex),"t_peak":float(1/vertex),
            "c_v_peak":float(np.polyval(fit,vertex-x[1])),
            "discrete_peak_tick":int(points[i][0]),
            "step":(ticks[stride]-ticks[0])/800.,"fit_k":x.tolist(),"fit_c_v":y.tolist()}

def save_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plot(method,coarse,refine,summary):
    fig=plt.figure(figsize=(11,7.2),layout="constrained");grid=fig.add_gridspec(2,2,height_ratios=[3.7,1.35])
    ax=fig.add_subplot(grid[0,0]);fitax=fig.add_subplot(grid[0,1]);tableax=fig.add_subplot(grid[1,:]);tableax.axis("off")
    colors=(["#555555","#bd641e","#266ba0"] if method=="boundary" else ["#266ba0","#835b9f","#4e8c62"])
    markers=(["s","^","o"] if method=="boundary" else ["o","s","^"])
    for color,marker,l in zip(colors,markers,SIZES[method]):
        points=coarse[l];xy=sorted((1/(tick/800.),v) for tick,v in points)
        ax.plot([x for x,_ in xy],[y for _,y in xy],color=color,label=f"L = {l}, χ = 16, h = 0.0025")
        points=refine[(l,32)]
        xy=sorted((1/(tick/800.),v) for tick,v in points)
        ax.plot([x for x,_ in xy],[y for _,y in xy],marker=marker,ls="none",ms=4,mfc="white",color=color)
    ax.plot([],[],"o",ms=4,mfc="white",color="#555",ls="none",label="χ = 32, h = 0.00125")
    ax.axvline(TC,color="#777",ls="--",lw=1,label="Onsager Tc")
    ax.set(title="(a) Specific-heat peaks",xlabel="Temperature T",ylabel="Specific heat per spin")
    ax.set_ylim(bottom=0)
    if method=="trg":
        ax.set_xlim(2.225,2.325)
        ax.set_title("(a) Periodic specific-heat peaks (zoom)")
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc="best")
    sizes=SIZES[method];x=1/np.array(sizes,dtype=float)
    y=np.array([summary["sizes"][str(l)]["chi32_h00125"]["t_peak"] for l in sizes])
    fitax.scatter(x,y,color=colors,zorder=3)
    for xx,yy,ll in zip(x,y,sizes):fitax.annotate(f"L={ll}",(xx,yy),xytext=(5,5),textcoords="offset points",fontsize=8)
    xx=np.linspace(0,max(x)*1.05,120)
    for key,style,label in [("all_sizes","-","All three sizes"),("largest_two","--","Largest two")]:
        fitax.plot(xx,summary["fits"][key]["intercept"]+summary["fits"][key]["slope"]*xx,style,label=label)
    fitax.axhline(TC,color="#777",ls=":",lw=1,label="Onsager Tc")
    fitax.set(title="(b) Extrapolate the peak",xlabel="Inverse side length 1/L",ylabel="Peak temperature")
    fitax.grid(alpha=.2);fitax.legend(fontsize=8)
    rows=[]
    for l in sizes:
        d=summary["sizes"][str(l)]
        rows.append([str(l),f'{d["chi16_h00125"]["t_peak"]:.6f}',f'{d["chi32_h00125"]["t_peak"]:.6f}',
                     f'{d["chi32_h0025"]["t_peak"]:.6f}',f'{d["rank_shift_percent"]:.3f}%',f'{d["step_shift_percent"]:.3f}%',
                     f'{d["refined_chi32_seconds_sum"]:.2f}'])
    tableax.table(cellText=rows,colLabels=["L","χ16, h=.00125","χ32, h=.00125","χ32, h=.0025","Rank shift","Step shift","χ32 9-run sum (s)"],cellLoc="center",loc="center")
    small=summary["small_checks"]
    tableax.text(.5,.02,f'Tc: all sizes = {summary["fits"]["all_sizes"]["intercept"]:.6f}; largest two = {summary["fits"]["largest_two"]["intercept"]:.6f}; Onsager = {TC:.6f}\n'
                 f'Small-square checks: max |Δg| = {small["max_g_abs_error"]:.2e}; |ΔcV| = {small["heat_capacity_abs_error"]:.2e}',
                 ha="center",va="bottom",fontsize=9,transform=tableax.transAxes)
    fig.suptitle(("Open boundary MPS" if method=="boundary" else "Periodic tensor renormalization group")+
                 ": finite-size specific heat",fontsize=13)
    fig.savefig(ROOT/"artifacts"/("boundary-critical.png" if method=="boundary" else "trg-critical.png"),dpi=180)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--method",choices=["boundary","trg","all"],default="all")
    args=parser.parse_args();assert args.workers>=1 and EXE.exists()
    methods=list(SIZES) if args.method=="all" else [args.method]
    out=ROOT/"artifacts";out.mkdir(exist_ok=True)
    digest=hashlib.sha256(EXE.read_bytes()).hexdigest()
    config_path=ROOT/"runs/sweep-config.json"
    config_path.parent.mkdir(parents=True,exist_ok=True)
    if config_path.exists():
        config=json.loads(config_path.read_text())
        if config["executable_sha256"]!=digest:
            raise RuntimeError("Cached sweep uses a different thermo binary; preserve runs/sweep and runs/sweep-config.json elsewhere before starting a fresh scan.")
    else:
        config_path.write_text(json.dumps({"executable_sha256":digest,"k_tick":.00125,"blas_threads":os.environ.get("OPENBLAS_NUM_THREADS")},indent=2))
    coarse_ticks=list(range(320,401,2))
    coarse_jobs=[Job(m,l,16,t) for m in methods for l in SIZES[m] for t in coarse_ticks]
    all_runs=run_jobs(coarse_jobs,args.workers,"Coarse scan")
    centers={}
    for m in methods:
        for l in SIZES[m]:
            g=[all_runs[(m,l,16,t)]["g"] for t in coarse_ticks]
            values=cv_points(coarse_ticks,g,1)
            centers[(m,l)]=max(values,key=lambda x:x[1])[0]
            print(f"{m} L={l}: coarse peak K={centers[(m,l)]/800:.5f}",flush=True)
    refined_jobs=[]
    for m in methods:
        for l in SIZES[m]:
            for chi in [16,32]:
                for t in range(centers[(m,l)]-4,centers[(m,l)]+5):
                    # Keep complete peak-stencil evidence at every size and rank.
                    refined_jobs.append(Job(m,l,chi,t,full=True))
    all_runs.update(run_jobs(refined_jobs,args.workers,"Refined scan"))
    for m in methods:
        coarse={};refined={};summary={"method":m,"geometry":"open" if m=="boundary" else "periodic","onsager_tc":TC,"sizes":{},"fits":{}}
        table=[]
        for l in SIZES[m]:
            gs=[all_runs[(m,l,16,t)]["g"] for t in coarse_ticks]
            coarse[l]=cv_points(coarse_ticks,gs,1)
            ticks=list(range(centers[(m,l)]-4,centers[(m,l)]+5))
            per={}
            for chi in [16,32]:
                vals=[all_runs[(m,l,chi,t)]["g"] for t in ticks]
                refined[(l,chi)]=cv_points(ticks,vals,1)
                per[f"chi{chi}_h00125"]=peak(ticks,vals,1)
                if chi==32:per["chi32_h0025"]=peak(ticks,vals,2)
            p16=per["chi16_h00125"]["t_peak"];p32=per["chi32_h00125"]["t_peak"];wide=per["chi32_h0025"]["t_peak"]
            per["rank_shift_percent"]=100*abs(p16-p32)/abs(p32)
            per["step_shift_percent"]=100*abs(wide-p32)/abs(p32)
            assert per["rank_shift_percent"]<1 and per["step_shift_percent"]<1,(m,l,per)
            per["coarse_center_k"]=centers[(m,l)]/800.
            per["coarse_chi16_seconds_sum"]=sum(all_runs[(m,l,16,t)]["seconds"] for t in coarse_ticks)
            per["refined_chi16_seconds_sum"]=sum(all_runs[(m,l,16,t)]["seconds"] for t in ticks)
            per["refined_chi32_seconds_sum"]=sum(all_runs[(m,l,32,t)]["seconds"] for t in ticks)
            summary["sizes"][str(l)]=per
            table.append({"L":l,"coarse_center_K":per["coarse_center_k"],"T_peak_chi16_h0.00125":p16,
                          "T_peak_chi32_h0.00125":p32,"T_peak_chi32_h0.0025":wide,
                          "rank_shift_percent":per["rank_shift_percent"],"step_shift_percent":per["step_shift_percent"],
                          "coarse_chi16_seconds_sum":per["coarse_chi16_seconds_sum"],
                          "refined_chi16_seconds_sum":per["refined_chi16_seconds_sum"],
                          "refined_chi32_seconds_sum":per["refined_chi32_seconds_sum"]})
        x=1/np.array(SIZES[m],dtype=float)
        y=np.array([summary["sizes"][str(l)]["chi32_h00125"]["t_peak"] for l in SIZES[m]])
        for key,xx,yy in [("all_sizes",x,y),("largest_two",x[-2:],y[-2:])]:
            slope,intercept=np.polyfit(xx,yy,1)
            error=100*abs(intercept-TC)/TC
            assert error<1,(m,key,intercept,TC)
            summary["fits"][key]={"intercept":float(intercept),"slope":float(slope),"onsager_error_percent":float(error)}
        checks=json.loads((out/"small-checks.json").read_text())
        label="open " if m=="boundary" else "periodic "
        gchecks=[check for check in checks if check["case"].startswith(label)]
        cvchecks=[check for check in checks if check["case"].startswith(m+" L=") and "heat capacity" in check["case"]]
        assert gchecks and len(cvchecks)==1
        summary["small_checks"]={"max_g_abs_error":max(check["absolute_error"] for check in gchecks),
                                 "heat_capacity_abs_error":cvchecks[0]["absolute_error"]}
        checkrows=[]
        for check in gchecks+cvchecks:
            checkrows.append({"case":check["case"],"K":check["K"],"computed_g":check.get("g",""),
                              "enumeration_g":check.get("enumeration_g",""),"g_abs_error":check.get("absolute_error","") if "g" in check else "",
                              "finite_difference_cv":check.get("finite_difference",""),"variance_cv":check.get("energy_variance",""),
                              "cv_abs_error":check.get("absolute_error","") if "finite_difference" in check else ""})
        rows=[]
        for (method,l,chi,tick),run in sorted(all_runs.items()):
            if method!=m:continue
            rows.append({"method":m,"L":l,"chi":chi,"K":run["k"],"g":run["g"],"log_Z":run["log_z"],
                         "seconds":run["seconds"],"record":run["record"],"path":f"runs/sweep/{m}/L{l}/chi{chi}/k{tick:04d}"})
        prefix="boundary" if m=="boundary" else "trg"
        save_csv(out/f"{prefix}-g.csv",rows)
        save_csv(out/f"{prefix}-peak-table.csv",table)
        save_csv(out/f"{prefix}-small-checks.csv",checkrows)
        (out/f"{prefix}-summary.json").write_text(json.dumps(summary,indent=2))
        plot(m,coarse,refined,summary)
        print(f"{m}: all-size Tc={summary['fits']['all_sizes']['intercept']:.8f}; largest-two Tc={summary['fits']['largest_two']['intercept']:.8f}",flush=True)

if __name__=="__main__":
    main()
