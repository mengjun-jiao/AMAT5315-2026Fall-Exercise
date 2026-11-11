use anyhow::{ensure,Result};
use serde::Serialize;
use crate::linalg::Backend;

#[derive(Clone)]
pub struct Core {pub left:usize,pub right:usize,pub data:Vec<f64>}
#[derive(Serialize)]
pub struct Truncation {
    pub row:usize, pub bond:usize, pub available:usize, pub retained:usize,
    pub discarded_squared_weight:f64, pub singular_values:Vec<f64>,
}
pub struct BoundaryResult {
    pub log_z:f64,pub cores:Vec<Core>,pub scales:Vec<f64>,pub truncations:Vec<Truncation>,
    pub edge:[f64;2],
}

pub fn site(k:f64)->[f64;16] {
    let c=k.cosh().sqrt();let s=k.sinh().sqrt();let w=[[c,s],[c,-s]];
    let mut a=[0.;16];
    for r in 0..2 {for u in 0..2 {for l in 0..2 {for d in 0..2 {
        a[((r*2+u)*2+l)*2+d]=w.iter().map(|x|x[r]*x[u]*x[l]*x[d]).sum();
    }}}} a
}

fn absorb(core:&Core,a:&[f64;16],edge:&[f64;2],left_edge:bool,right_edge:bool)->Core {
    let old_l=core.left;let old_r=core.right;
    let ldim=if left_edge{1}else{2};let rdim=if right_edge{1}else{2};
    let new_l=old_l*ldim;let new_r=old_r*rdim;let mut out=vec![0.;new_l*2*new_r];
    for al in 0..old_l {for l in 0..ldim {for d in 0..2 {for be in 0..old_r {for r in 0..rdim {
        let mut value=0.;
        for u in 0..2 {let old=core.data[(al*2+u)*old_r+be];
            let mut local=0.;for ll in 0..2 {for rr in 0..2 {
                if !left_edge && ll!=l {continue} if !right_edge && rr!=r {continue}
                let weight=(if left_edge{edge[ll]}else{1.})*(if right_edge{edge[rr]}else{1.});
                local+=weight*a[((rr*2+u)*2+ll)*2+d];
            }}
            value+=old*local;
        }
        out[((al*ldim+l)*2+d)*new_r+(be*rdim+r)]=value;
    }}}}}
    Core{left:new_l,right:new_r,data:out}
}

fn compress(backend:&Backend, cores:&mut [Core], chi:usize,row:usize,collect:bool,diagnostics:&mut Vec<Truncation>)->Result<()> {
    let l=cores.len();
    // Right-to-left exact SVDs put the environment in a right-canonical gauge.
    for i in (1..l).rev() {
        let left=cores[i].left;let right=cores[i].right;
        let svd=backend.svd(&cores[i].data,left,2*right)?;let q=svd.rank;
        let mut us=vec![0.;left*q];
        for x in 0..left {for j in 0..q {us[x*q+j]=svd.u[x*q+j]*svd.s[j];}}
        let pl=cores[i-1].left;
        ensure!(cores[i-1].right==left,"MPS bond mismatch");
        let prev=backend.multiply(&cores[i-1].data,&us,pl*2,left,q)?;
        cores[i-1]=Core{left:pl,right:q,data:prev};
        cores[i]=Core{left:q,right,data:svd.vt};
    }
    // Left-to-right SVD truncation leaves the first L-1 cores left-isometric.
    for i in 0..l-1 {
        let left=cores[i].left;let right=cores[i].right;
        let svd=backend.svd(&cores[i].data,left*2,right)?;let q=svd.rank.min(chi);
        let mut u=vec![0.;left*2*q];
        for x in 0..left*2 {for j in 0..q {u[x*q+j]=svd.u[x*svd.rank+j];}}
        let mut svt=vec![0.;q*right];
        for j in 0..q {for x in 0..right {svt[j*right+x]=svd.s[j]*svd.vt[j*right+x];}}
        let nr=cores[i+1].right;ensure!(cores[i+1].left==right,"MPS bond mismatch");
        let next=backend.multiply(&svt,&cores[i+1].data,q,right,2*nr)?;
        if collect {diagnostics.push(Truncation{row,bond:i+1,available:svd.rank,retained:q,
            discarded_squared_weight:svd.s[q..].iter().map(|x|x*x).sum(),singular_values:svd.s.clone()});}
        cores[i]=Core{left,right:q,data:u};
        cores[i+1]=Core{left:q,right:nr,data:next};
    }
    Ok(())
}

pub fn solve(backend:&Backend,l:usize,k:f64,chi:usize,full:bool)->Result<BoundaryResult>{
    ensure!(l>=2&&k.is_finite()&&k>=0.&&chi>=1,"invalid boundary parameters");
    let a=site(k);let edge=[1./k.cosh().sqrt(),0.];
    let mut cores=vec![Core{left:1,right:1,data:edge.to_vec()};l];
    let mut log_scale=0.;let mut scales=Vec::with_capacity(l);let mut diagnostics=Vec::new();
    for row in 0..l {
        cores=cores.iter().enumerate().map(|(i,c)|absorb(c,&a,&edge,i==0,i+1==l)).collect();
        compress(backend,&mut cores,chi,row+1,full,&mut diagnostics)?;
        let last=&mut cores[l-1];let scale=last.data.iter().map(|x|x*x).sum::<f64>().sqrt();
        ensure!(scale>0.&&scale.is_finite(),"invalid boundary scale");
        for x in &mut last.data {*x/=scale} log_scale+=scale.ln();scales.push(scale);
    }
    let mut v=vec![1.];
    for c in &cores {
        ensure!(v.len()==c.left,"boundary closure mismatch");let mut next=vec![0.;c.right];
        for al in 0..c.left {for p in 0..2 {for be in 0..c.right {
            next[be]+=v[al]*c.data[(al*2+p)*c.right+be]*edge[p];
        }}}
        v=next;
    }
    ensure!(v.len()==1&&v[0]>0.&&v[0].is_finite(),"invalid boundary closure");
    Ok(BoundaryResult{log_z:log_scale+v[0].ln(),cores,scales,truncations:diagnostics,edge})
}
