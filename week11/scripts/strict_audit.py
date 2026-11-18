#!/usr/bin/env python3
"""Independent audit of saved Week 11 plans, numerical results and figure artists."""
from collections import Counter
from itertools import product
from pathlib import Path
import contextlib, io, json, math, random, runpy, subprocess, sys
import numpy as np
from PIL import Image
from matplotlib.figure import Figure

ROOT=Path(__file__).resolve().parent.parent
sys.setrecursionlimit(100_000)
def load(name):return json.loads((ROOT/'runs'/name/'run.json').read_text())['evidence']

def verify_tree(tree, factors):
    total=Counter(v for f in factors for v in f)
    seen=set()
    def visit(t):
        if 'Leaf' in t:
            i=t['Leaf']['tensor_index'];assert 0<=i<len(factors) and i not in seen
            seen.add(i);v=factors[i]
            return v,Counter(v),len(v),0
        node=t['Node'];args=node['args'];assert len(args)==2
        left,lc,ls,lt=visit(args[0]);right,rc,rs,rt=visit(args[1])
        eins=node['eins'];assert eins['ixs']==[left,right]
        count=lc+rc;union=set(left)|set(right);out=eins['iy']
        assert len(out)==len(set(out))
        assert set(out)=={v for v in union if count[v]<total[v]}
        return out,count,max(ls,rs,len(out)),lt+rt+2**len(union)
    out,_,sc,tc=visit(tree)
    assert out==[] and len(seen)==len(factors)
    return sc,math.log2(tc)

plans={}
for k in range(2,9):
    bundle=json.loads((ROOT/f'runs/pred/scale_{k}.bundle.json').read_text())['target']['data']
    n=bundle['graph']['num_vertices'];edges=bundle['graph']['edges']
    assert n==14*k*k+8*k and len(edges)==len(bundle['couplings'])
    factors=[[i] for i in range(n)]+edges
    for order in ('greedy','treesa'):
        run=load(f'plan_k{k}_{order}');sc,tc=verify_tree(run['contraction_tree'],factors)
        assert sc==run['sc'] and abs(tc-run['tc'])<1e-10
        plans[f'{k}-{order}']={'sc':sc,'tc':tc,'leaves':len(factors)}

g=load('gadgets');gates=g['gates'];circuit=g['circuit'];cell=g['multiplier_cell']['states']
def S(x):return 2*x-1
def and_energy(a,b,z):return a*b-2*a*z-2*b*z-a-b+2*z
def or_energy(a,b,z):return a*b-2*a*z-2*b*z+a+b-2*z
def xor_energy(a,b,z,w):
    x,y,q,r=((v+1)//2 for v in (a,b,z,w))
    return -4+2*(x+y-q-2*r)**2
for row in gates:
    bits=row['bits'];spins=list(map(S,bits));name=row['gate']
    if name=='AND': value=and_energy(*spins);valid=bits[2]==bits[0]*bits[1]
    elif name=='OR': value=or_energy(*spins);valid=bits[2]==int(bits[0] or bits[1])
    elif name=='XOR':
        vals=[xor_energy(*spins,w) for w in (-1,1)];value=min(vals);valid=bits[2]==(bits[0]^bits[1]);assert vals==row['helper_energies']
    elif name=='NOT':value=spins[0]*spins[1];valid=bits[1]==1-bits[0]
    elif name=='COPY':value=-spins[0]*spins[1];valid=bits[1]==bits[0]
    elif name=='SET0':value=spins[0];valid=bits[0]==0
    else:value=-spins[0];valid=bits[0]==1
    assert value==row['energy'] and valid==row['valid']
for row in circuit:
    a,b,t,o=map(S,row['bits']);assert row['energy']==and_energy(a,b,t)+t*o-o
for row in cell:
    p,q,a,b,t,u,v,w,c,z,r,k=map(S,row['bits'])
    e=and_energy(p,q,t)+and_energy(a,t,u)+xor_energy(a,t,v,r)+and_energy(b,v,w)+or_energy(u,w,c)+xor_energy(v,b,z,k)
    assert e==row['energy']
assert len(cell)==4096 and sum(r['energy']==-20 for r in cell)==16

def complete_queens(n,first):
    def search(r,cols,diag1,diag2):
        if r==n:return 1
        return sum(search(r+1,cols|{c},diag1|{r-c},diag2|{r+c}) for c in range(n) if c not in cols and r-c not in diag1 and r+c not in diag2)
    return search(1,{first},{-first},{first})
slices=[complete_queens(10,c) for c in range(10)]
assert slices==load('count_nq10')['first_row_slice_counts']

for n in (15,6,7):
    raw=(ROOT/f'runs/pred/factor_{n}-solve.txt').read_text()
    out=json.loads(raw[raw.find('{'):]);bits=out['solution'];m=2;qbits=len(bits)-m
    p=sum(b<<i for i,b in enumerate(bits[:m]));q=sum(b<<i for i,b in enumerate(bits[m:]))
    assert out['evaluation']==f'Min({abs(p*q-n)})'
    assert (p*q==n)==(n!=7)

sample=load('sample_f15');hist=Counter(s['energy'] for s in sample['samples'])
qn=load('sample_nq6');freq=Counter(tuple(x) for x in qn['samples']);boards=sorted(freq)
original=Figure.savefig;images=[]
def check_figure(fig,target,*args,**kwargs):
    name=Path(target).name
    if name=='gates.png':
        ax,bx=fig.axes; assert len(ax.collections)==len(gates)+2 and len(bx.collections)==len(circuit)
        assert [float(a.get_offsets()[0,1]) for a in ax.collections[:len(gates)]]==[r['energy'] for r in gates]
        assert [float(a.get_offsets()[0,1]) for a in bx.collections]==[r['energy'] for r in circuit]
    elif name=='factoring.png':
        ax,bx=fig.axes
        for line,order in zip(ax.lines[:2],('greedy','treesa')):
            assert list(line.get_xdata())==list(range(2,9))
            assert list(line.get_ydata())==[plans[f'{k}-{order}']['sc'] for k in range(2,9)]
        assert [round(x.get_height()) for x in bx.patches]==[hist[e] for e in sorted(hist)]
        assert [int(t.get_text()) for t in bx.get_xticklabels()]==sorted(hist)
        p0=2000*sample['ground_degeneracy']*math.exp(-2*sample['e_min']-sample['log_z'])
        assert abs(bx.lines[0].get_ydata()[0]-p0)<1e-10
    elif name=='nqueens.png':
        ax,bx,boardax,_=fig.axes
        assert list(ax.lines[0].get_ydata())==[load(f'count_nq{n}')['count'] for n in range(4,11)]
        assert [round(x.get_height()) for x in bx.patches]==[freq[b] for b in boards]
        for i,b in enumerate(boards):
            for r,col in enumerate(b):
                point=boardax.lines[6*i+r]
                assert abs(point.get_xdata()[0]-(i-.15+col*.06))<1e-10
                assert abs(point.get_ydata()[0]-(.65-r*.1))<1e-10
    else:raise AssertionError(name)
    buf=io.BytesIO();original(fig,buf,*args,**kwargs);buf.seek(0)
    assert np.array_equal(np.asarray(Image.open(buf).convert('RGBA')),np.asarray(Image.open(target).convert('RGBA')))
    images.append(name)
Figure.savefig=check_figure
try:
    for name in ('plot_gates.py','plot_factoring.py','plot_nqueens.py'):
        with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(ROOT/'scripts'/name),run_name='__main__')
finally:Figure.savefig=original

# Independent probabilistic test against the exact enumeration of a 5-spin system.
bundle=ROOT/'runs/audit_synthetic.json';data=json.loads(bundle.read_text())['target']['data']
beta=.7;fields=data['fields'];edges=data['graph']['edges'];couplings=data['couplings']
prob={}
for bits in product((0,1),repeat=len(fields)):
    spin=[2*b-1 for b in bits]
    e=sum(h*s for h,s in zip(fields,spin))+sum(j*spin[a]*spin[b] for (a,b),j in zip(edges,couplings))
    prob[bits]=math.exp(-beta*e)
z=sum(prob.values());prob={b:w/z for b,w in prob.items()}
out=ROOT/'runs/strict_sample_mixed'
subprocess.run([str(ROOT/'.install/bin/tnsample'),'--mode','sample','--input',str(bundle),'--beta',str(beta),'--samples','12000','--seed','2026','--out',str(out)],check=True,capture_output=True,text=True)
draws=load('strict_sample_mixed')['samples'];counts=Counter(tuple(s['bits']) for s in draws)
groups=[(counts[b],12000*p) for b,p in prob.items() if 12000*p>=20]
rare=[(counts[b],12000*p) for b,p in prob.items() if 12000*p<20]
if rare:groups.append((sum(x for x,_ in rare),sum(x for _,x in rare)))
chi2=sum((o-e)**2/e for o,e in groups);df=len(groups)-1
assert chi2<df+8*math.sqrt(2*df)

qout=ROOT/'runs/strict_sample_nq6'
subprocess.run([str(ROOT/'.install/bin/tnsample'),'--mode','sample','--n','6','--samples','4000','--seed','2026','--out',str(qout)],check=True,capture_output=True,text=True)
qcounts=Counter(tuple(b) for b in load('strict_sample_nq6')['samples'])
assert len(qcounts)==4 and set(qcounts)==set(boards)
qchi=sum((v-1000)**2/1000 for v in qcounts.values());assert qchi<25

result={'verified_plans':plans,'verified_gate_rows':len(gates),'verified_circuit_rows':len(circuit),
        'verified_cell_states':len(cell),'n10_first_row_slice_counts':slices,'plot_artists_and_pixels':images,
        'mixed_5spin_samples':len(draws),'mixed_5spin_chi_squared':chi2,'mixed_5spin_degrees_of_freedom':df,
        'queens6_test_draws':sum(qcounts.values()),'queens6_test_frequencies':list(qcounts.values()),'queens6_chi_squared':qchi}
(ROOT/'artifacts/strict-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
