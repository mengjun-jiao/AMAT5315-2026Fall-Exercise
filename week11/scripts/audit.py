#!/usr/bin/env python3
"""Independently check saved runs, exact small cases, and displayed plot pixels."""
from collections import Counter
from itertools import product
from pathlib import Path
import contextlib,io,json,math,runpy,subprocess,sys
import numpy as np
from PIL import Image
from matplotlib.figure import Figure

root=Path(__file__).resolve().parent.parent
def load(name):return json.loads((root/'runs'/name/'run.json').read_text())['evidence']

gates=load('gadgets');assert len(gates['gates'])==36 and len(gates['circuit'])==16
expected={'AND':(4,-3),'OR':(4,-3),'NOT':(2,-1),'COPY':(2,-1),'SET0':(1,-1),'SET1':(1,-1),'XOR':(4,-4)}
for name,(number,floor) in expected.items():
    rows=[r for r in gates['gates'] if r['gate']==name]
    def truth(b):
        if name=='AND':return b[2]==(b[0]&b[1])
        if name=='OR':return b[2]==(b[0]|b[1])
        if name=='XOR':return b[2]==(b[0]^b[1])
        if name=='NOT':return b[1]!=b[0]
        if name=='COPY':return b[1]==b[0]
        return b[0]==(1 if name=='SET1' else 0)
    assert all(r['valid']==truth(r['bits']) for r in rows)
    assert sum(r['valid'] for r in rows)==number
    assert all(r['energy']==floor if r['valid'] else r['energy']>=floor+2 for r in rows)
    if name=='XOR':assert all(min(r['helper_energies'])==r['energy'] for r in rows)
assert sum(r['energy']==-5 for r in gates['circuit'])==3
assert all((r['energy']==-5)==r['valid'] for r in gates['circuit'])
cell=gates['multiplier_cell']['states'];assert len(cell)==4096
floors=[r for r in cell if r['energy']==-20]
assert min(r['energy'] for r in cell)==-20 and len(floors)==16
assert all((lambda b:b[0]*b[1]+b[2]+b[3]==b[9]+2*b[8])(r['bits']) for r in floors)
assert len({tuple(r['bits'][:4]) for r in floors})==16

known=[2,10,4,40,92,352,724]
def independent_queens(n):
    def visit(row,used,diag1,diag2):
        if row==n:return 1
        return sum(visit(row+1,used|{c},diag1|{row-c},diag2|{row+c})
                   for c in range(n) if c not in used and row-c not in diag1 and row+c not in diag2)
    return visit(0,set(),set(),set())
for n,expected_count in zip(range(4,11),known):
    item=load(f'count_nq{n}');assert item['count']==expected_count==independent_queens(n)
    assert item['network']=='sparse 0/1 carry tensors' and item['contraction_tree']
    if n==10:assert len(item['first_row_slice_counts'])==10 and sum(item['first_row_slice_counts'])==724

sample_counts={}
for n in (6,8):
    item=load(f'sample_nq{n}');boards=item['samples'];assert len(boards)==200
    for b in boards:
        assert len(b)==n and sorted(b)==list(range(n))
        assert len({r-c for r,c in enumerate(b)})==n and len({r+c for r,c in enumerate(b)})==n
    sample_counts[str(n)]={'distinct':len(set(map(tuple,boards))),'frequencies':[v for _,v in sorted(Counter(map(tuple,boards)).items())]}
    assert sample_counts[str(n)]['distinct']==item['distinct']

ferro=load('count_p0');assert ferro['e_min']==-180 and ferro['ground_degeneracy']==2
f15=load('count_f15');sample=load('sample_f15')
assert (f15['variables'],f15['e_min'],f15['ground_degeneracy'])==(104,-299,1)
assert abs(f15['log_z']-598.317945217)<1e-8
assert len(sample['samples'])==2000 and sample['e_min']==-299
bundle=json.loads((root/'runs/pred/factor_15.bundle.json').read_text())['target']['data']
def energy(bits):
    spins=[2*b-1 for b in bits]
    return sum(h*s for h,s in zip(bundle['fields'],spins))+sum(j*spins[a]*spins[b] for (a,b),j in zip(bundle['graph']['edges'],bundle['couplings']))
assert all(len(s['bits'])==104 and energy(s['bits'])==s['energy'] for s in sample['samples'])
extracted=json.loads((root/'runs/sample_f15/extract.json').read_text())
assert extracted['solution']==[1,1,1,0,1] and extracted['evaluation']=='Min(0)'
hist=Counter(s['energy'] for s in sample['samples']);p=math.exp(-2*sample['e_min']-sample['log_z'])
assert abs(hist[-299]-2000*p)<5*math.sqrt(2000*p*(1-p))
plans={}
for order in ('greedy','treesa'):
    vals=[load(f'plan_k{k}_{order}') for k in range(2,9)]
    assert [v['variables'] for v in vals]==[14*k*k+8*k for k in range(2,9)]
    assert all(v['contraction_tree'] and v['planner'].startswith('omeco') for v in vals)
    plans[order]=[v['sc'] for v in vals]

# Independent brute force for a 2x2 ferromagnet, run through the installed solver.
testout=root/'runs/audit_ferro2';testout.mkdir(parents=True,exist_ok=True)
subprocess.run([str(root/'.install/bin/tnsample'),'--mode','count','--l','2','--p-af','0',
                '--beta','2','--seed','2026','--out',str(testout)],check=True,capture_output=True,text=True)
small=json.loads((testout/'run.json').read_text())['evidence']
energies=[]
for spins in product((-1,1),repeat=4):
    energies.append(-sum(spins[i]*spins[j] for i,j in [(0,1),(0,2),(1,3),(2,3)]))
minimum=min(energies);z=sum(math.exp(-2*e) for e in energies)
assert small['e_min']==minimum and small['ground_degeneracy']==energies.count(minimum)
assert abs(small['log_z']-math.log(z))<1e-12

# A mixed-sign five-spin instance independently checks fields and bond signs.
fields=[1,-2,0,2,-1];edges=[[0,1],[1,2],[2,3],[3,4],[0,4],[0,2]];couplings=[-2,1,3,-1,2,-2]
data={'target':{'data':{'fields':fields,'couplings':couplings,'graph':{'num_vertices':5,'edges':edges}}}}
synthetic=root/'runs/audit_synthetic.json';synthetic.write_text(json.dumps(data))
testout=root/'runs/audit_mixed';testout.mkdir(parents=True,exist_ok=True)
subprocess.run([str(root/'.install/bin/tnsample'),'--mode','count','--input',str(synthetic),
                '--beta','0.7','--seed','2026','--out',str(testout)],check=True,capture_output=True,text=True)
mixed=json.loads((testout/'run.json').read_text())['evidence']
mixed_energies=[sum(h*s for h,s in zip(fields,spin))+sum(j*spin[a]*spin[b] for (a,b),j in zip(edges,couplings)) for spin in product((-1,1),repeat=5)]
mixed_min=min(mixed_energies);mixed_logz=math.log(sum(math.exp(-.7*e) for e in mixed_energies))
assert mixed['e_min']==mixed_min and mixed['ground_degeneracy']==mixed_energies.count(mixed_min)
assert abs(mixed['log_z']-mixed_logz)<1e-12

# Recreate each figure in memory and compare with the actual PNG on disk.
old=Figure.savefig;checked=[]
def check_pixels(fig,target,*a,**kw):
    buffer=io.BytesIO();old(fig,buffer,*a,**kw);buffer.seek(0)
    generated=np.asarray(Image.open(buffer).convert('RGBA'))
    actual=np.asarray(Image.open(target).convert('RGBA'))
    assert generated.shape==actual.shape and np.array_equal(generated,actual),str(target)
    checked.append(Path(target).name)
Figure.savefig=check_pixels
try:
    for script in ('plot_gates.py','plot_factoring.py','plot_nqueens.py'):
        with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(root/'scripts'/script),run_name='__main__')
finally:Figure.savefig=old
assert checked==['gates.png','factoring.png','nqueens.png']

report={'gate_rows':36,'circuit_rows':16,'multiplier_cell_states':4096,'multiplier_cell_floor_states':16,'nqueen_counts':known,'nqueen_samples':sample_counts,
        'ferro_2x2_bruteforce':{'e_min':minimum,'degeneracy':energies.count(minimum),'log_z':math.log(z)},
        'mixed_five_spin_bruteforce':{'e_min':mixed_min,'degeneracy':mixed_energies.count(mixed_min),'log_z':mixed_logz},
        'factor15':{'e_min':f15['e_min'],'log_z':f15['log_z'],'histogram':dict(sorted(hist.items())),
                    'predicted_ground_draws':2000*p,'decoded_factors':[3,5]},
        'plans':plans,'plot_pixels_checked':checked}
(root/'artifacts/audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
