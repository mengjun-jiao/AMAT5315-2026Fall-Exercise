#!/usr/bin/env python3
"""Independent exact spin enumeration and heat-capacity checks."""
from pathlib import Path
import json
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "small-checks.json"
EXE = ROOT / ".install/bin/thermo"
if not EXE.exists():
    EXE = ROOT / "target/release/thermo"


def exact(l, k, periodic):
    n = l*l
    states = np.arange(1 << n, dtype=np.uint32)
    spins = 1 - 2*((states[:,None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
    h = np.zeros(len(states), dtype=np.int16)
    for row in range(l):
        for col in range(l):
            i = row*l+col
            if periodic or col+1 < l:
                j = row*l+(col+1)%l
                h -= spins[:,i]*spins[:,j]
            if periodic or row+1 < l:
                j = ((row+1)%l)*l+col
                h -= spins[:,i]*spins[:,j]
    w = np.exp(-k*h); z=w.sum(); mean=float(np.dot(w,h)/z)
    var=float(np.dot(w,h.astype(float)**2)/z-mean**2)
    return float(np.log(z)/n), float(k*k*var/n)


def run(method,l,k,chi,name):
    target=ROOT/"runs"/"small-checks"/name
    result=subprocess.run([str(EXE),"--method",method,"--l",str(l),"--k",f"{k:.12g}",
                           "--chi",str(chi),"--out",str(target)],
                          check=True,capture_output=True,text=True)
    return json.loads(result.stdout)


def main():
    results=[]
    for k in [0.,.44,.7]:
        got=run("boundary",4,k,8,f"open4-k{k:g}")
        truth,_=exact(4,k,False);err=abs(got["g"]-truth)
        assert err<1e-8,(got,truth,err)
        results.append({"case":"open L=4 chi=8","K":k,"g":got["g"],"enumeration_g":truth,"absolute_error":err})
    for l,chi,tol in [(2,16,1e-8),(4,8,2e-3)]:
        got=run("trg",l,.44,chi,f"periodic{l}-k0.44")
        truth,_=exact(l,.44,True);err=abs(got["g"]-truth)
        assert err<tol,(got,truth,err)
        results.append({"case":f"periodic L={l} chi={chi}","K":.44,"g":got["g"],"enumeration_g":truth,"absolute_error":err,"tolerance":tol})
    for method,l,chi,periodic in [("boundary",4,8,False),("trg",2,16,True)]:
        k=.44;h=.0025
        gs=[run(method,l,k+d,chi,f"{method}{l}-cv-{d:+.4f}")["g"] for d in [-h,0,h]]
        cv=k*k*(gs[2]-2*gs[1]+gs[0])/h**2
        _,truth=exact(l,k,periodic);err=abs(cv-truth)
        assert err<1e-3,(method,cv,truth,err)
        results.append({"case":f"{method} L={l} heat capacity","K":k,"h":h,"finite_difference":cv,"energy_variance":truth,"absolute_error":err})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
