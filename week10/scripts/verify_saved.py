#!/usr/bin/env python3
"""Reconstruct saved boundary/TRG closures and audit full peak-stencil evidence."""
from pathlib import Path
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parent.parent

def check_boundary(path):
    run=json.loads((path/"run.json").read_text())
    scales=json.loads((path/"scales.json").read_text())
    diag=json.loads((path/"truncations.json").read_text())
    assert run["record"]=="full" and len(scales)==run["l"] and all(x>0 for x in scales)
    assert len(diag)==run["l"]*(run["l"]-1)
    for entry in diag:
        s=np.asarray(entry["singular_values"])
        q=entry["retained"]
        assert len(s)==entry["available"] and q<=run["chi"] and np.all(s[:-1]>=s[1:])
        assert abs(float(np.sum(s[q:]**2))-entry["discarded_squared_weight"])<1e-10*max(1.,float(np.sum(s*s)))
    edge=np.load(path/"edge.npy")
    assert edge.shape==(2,)
    v=np.ones(1);max_isometry=0.
    for i in range(run["l"]):
        core=np.load(path/f"final-core-{i:03}.npy")
        assert core.ndim==3 and core.shape[0]==len(v) and core.shape[1]==2
        if i+1<run["l"]:
            m=core.reshape(-1,core.shape[-1])
            max_isometry=max(max_isometry,float(np.max(abs(m.T@m-np.eye(m.shape[1])))))
        v=np.einsum("a,apb,p->b",v,core,edge,optimize=True)
    assert len(v)==1 and v[0]>0
    logz=sum(math.log(x) for x in scales)+math.log(v[0])
    err=abs(logz-run["log_z"])
    assert err<1e-8 and max_isometry<1e-9,(path,err,max_isometry)
    return {"path":str(path.relative_to(ROOT)),"log_z_abs_error":err,"max_isometry_error":max_isometry}

def check_trg_closure(path):
    run=json.loads((path/"run.json").read_text())
    records=json.loads((path/"scales.json").read_text())
    assert run["record"]=="full" and len(records)==2*int(math.log2(run["l"]))
    count=run["l"]**2
    for i,record in enumerate(records):
        assert record["step"]==i+1 and record["tensors_before"]==count and record["scale"]>0
        d=record["input_dim"];q=record["retained_dim"]
        assert q<=run["chi"] and len(record["singular_values_cut1"])==d*d and len(record["singular_values_cut2"])==d*d
        for name,shape in [("s1",(d,d,q)),("s2",(d,d,q)),("s3",(q,d,d)),("s4",(q,d,d)),
                           ("singular-1",(d*d,)),("singular-2",(d*d,))]:
            arr=np.load(path/f"step-{i+1:02}-{name}.npy",mmap_mode="r")
            assert arr.shape==shape
        count//=2
    assert count==1
    final=np.load(path/"final-tensor.npy",mmap_mode="r")
    assert final.ndim==4 and len(set(final.shape))==1
    z=float(np.einsum("ijij->",final))
    assert z>0
    reconstructed=sum(r["tensors_before"]*math.log(r["scale"]) for r in records)+math.log(z)
    error=abs(reconstructed-run["log_z"])
    assert error<1e-8,(path,error)
    return {"path":str(path.relative_to(ROOT)),"log_z_abs_error":error}

def check_trg_steps(path):
    run=json.loads((path/"run.json").read_text());records=json.loads((path/"scales.json").read_text())
    k=run["k"];c=math.sqrt(math.cosh(k));s=math.sqrt(math.sinh(k));w=np.array([[c,s],[c,-s]])
    a=np.einsum("sr,su,sl,sd->ruld",w,w,w,w)
    max_tail_error=0.
    for rec in records:
        i=rec["step"];d=rec["input_dim"];q=rec["retained_dim"];a=a/rec["scale"]
        s1=np.load(path/f"step-{i:02}-s1.npy");s2=np.load(path/f"step-{i:02}-s2.npy")
        s3=np.load(path/f"step-{i:02}-s3.npy");s4=np.load(path/f"step-{i:02}-s4.npy")
        m1=a.transpose(2,1,0,3).reshape(d*d,d*d)
        m2=a.transpose(3,2,1,0).reshape(d*d,d*d)
        for matrix,x,y,discard in [(m1,s1,s3,rec["discarded_squared_weight_cut1"]),
                                   (m2,s2,s4,rec["discarded_squared_weight_cut2"])]:
            observed=np.linalg.norm(matrix-x.reshape(d*d,q)@y.reshape(q,d*d))
            predicted=math.sqrt(max(0.,discard))
            error=abs(observed-predicted)/max(1.,float(np.linalg.norm(matrix)))
            max_tail_error=max(max_tail_error,float(error))
            assert error<1e-8,(i,error)
        a=np.einsum("war,abu,lbg,dgw->ruld",s1,s2,s3,s4,optimize=True)
    final=np.load(path/"final-tensor.npy")
    error=float(np.max(abs(a-final)))/max(1.,float(np.max(abs(final))))
    assert error<1e-9,(path,error)
    return {"path":str(path.relative_to(ROOT)),"max_split_tail_error":max_tail_error,"final_tensor_relative_max_error":error}

def main():
    boundary=[ROOT/"runs/boundary64"]
    trg=[ROOT/"runs/trg1024"]
    for path in sorted((ROOT/"runs/sweep/boundary").glob("L*/chi*/k*")):
        if (path/"run.json").exists() and json.loads((path/"run.json").read_text())["record"]=="full":
            boundary.append(path)
    for path in sorted((ROOT/"runs/sweep/trg").glob("L*/chi*/k*")):
        if (path/"run.json").exists() and json.loads((path/"run.json").read_text())["record"]=="full":
            trg.append(path)
    b=[check_boundary(p) for p in boundary]
    t=[check_trg_closure(p) for p in trg]
    steps=check_trg_steps(ROOT/"runs/trg1024")
    summary=json.loads((ROOT/"artifacts/trg-summary.json").read_text())
    center=round(800*summary["sizes"]["1024"]["coarse_center_k"])
    chi32=check_trg_steps(ROOT/"runs/sweep/trg/L1024/chi32"/f"k{center:04d}")
    output={"boundary_runs":len(b),"trg_runs":len(t),
            "max_boundary_log_z_abs_error":max(x["log_z_abs_error"] for x in b),
            "max_boundary_isometry_error":max(x["max_isometry_error"] for x in b),
            "max_trg_log_z_abs_error":max(x["log_z_abs_error"] for x in t),
            "trg_step_reconstruction":steps,"trg_chi32_step_reconstruction":chi32}
    (ROOT/"artifacts/saved-evidence-checks.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=="__main__":
    main()
