#!/usr/bin/env python3
"""Independently verify saved SVD reconstructions and plot the spectrum and error."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent
source=root/"artifacts/svd-data"
a=np.load(source/"matrix.npy");s=np.load(source/"singular_values.npy")
assert a.shape==(512,512) and s.shape==(512,) and np.all(s[:-1]>=s[1:]) and np.all(s>=0)
ranks=np.arange(513)
tail=np.concatenate(([np.sum(s*s)],np.cumsum((s[::-1]**2))[-2::-1],[0.]))
assert len(tail)==513
pred=np.sqrt(np.maximum(tail,0)/np.sum(s*s))
norm=np.linalg.norm(a)
observed={}
for rank in [16,64,128,256,512]:
    approx=np.load(source/f"reconstruction-{rank}.npy")
    assert approx.shape==a.shape
    measured=float(np.linalg.norm(a-approx)/norm)
    assert abs(measured-pred[rank])<1e-8
    observed[rank]=measured
assert observed[512]<1e-8
cross=int(np.flatnonzero(pred<=.1)[0]);ratio=2*cross/512
report={"generator":json.loads((source/"metadata.json").read_text())["generator"],"seed":2026,
        "relative_errors":observed,"predictions":{rank:float(pred[rank]) for rank in observed},
        "ten_percent_first_rank":cross,"two_factor_storage_ratio":ratio,
        "storage_saving_at_ten_percent":cross<256}
(root/"artifacts/svd-checks.json").write_text(json.dumps(report,indent=2))
fig,ax=plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
ax[0].plot(np.arange(1,513),s,color="#256aa0");ax[0].set(xlabel="Singular-value index",ylabel="Singular value",title="(a) Singular spectrum")
ax[0].set_xlim(0,512);ax[0].grid(alpha=.18)
ax[1].plot(ranks,pred,color="#256aa0",label="SVD tail prediction")
ax[1].scatter(list(observed),list(observed.values()),marker="o",facecolors="none",edgecolors="#b55f21",label="Saved reconstructions",zorder=4)
ax[1].axhline(.1,color="#8f4242",ls="--",lw=1,label="10% error")
ax[1].axvline(256,color="#777",ls=":",lw=1,label="Equal storage (rank 256)")
ax[1].axvspan(256,512,color="#f4e6c8",alpha=.55)
ax[1].plot(cross,pred[cross],"ko",ms=4)
ax[1].annotate(f"first ≤10%: rank {cross}",(cross,pred[cross]),xytext=(260,.34),arrowprops={"arrowstyle":"->"},fontsize=9)
ax[1].text(383,.87,"No storage saving",ha="center",fontsize=9)
ax[1].set(xlabel="Retained rank",ylabel="Relative Frobenius error",title="(b) Error versus retained rank",xlim=(-5,520),ylim=(-.03,1.03));ax[1].legend(fontsize=7);ax[1].grid(alpha=.18)
fig.suptitle("Gaussian matrix compression: SVD and reconstructed errors")
fig.savefig(root/"artifacts/svd.png",dpi=180);plt.close(fig)
print(json.dumps(report,indent=2))
