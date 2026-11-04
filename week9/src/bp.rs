use crate::model::Model;
use anyhow::{ensure,Result};
use serde::Serialize;
use std::{collections::BTreeMap,time::Instant};

#[derive(Serialize)]
pub struct Trace { pub sweep:usize,pub residual:f64,pub cumulative_work:u64,pub beliefs:Vec<[f64;2]> }
#[derive(Serialize)]
pub struct BpRun {pub method:String,pub observations:BTreeMap<usize,usize>,pub damping:f64,pub tolerance:f64,pub max_sweeps:usize,pub sweeps:usize,pub converged:bool,pub residual:f64,pub cumulative_work:u64,pub seconds:f64,pub marginals:Vec<[f64;2]>,pub trace:Vec<Trace>,pub variable_to_factor:Vec<Vec<[f64;2]>>,pub factor_to_variable:Vec<Vec<[f64;2]>>,pub work_definition:String}
fn normalize(a:[f64;2])->Result<[f64;2]>{let s=a[0]+a[1];ensure!(s>0. && s.is_finite(),"zero or invalid BP normalization");Ok([a[0]/s,a[1]/s])}
pub fn run(model:&Model,obs:&BTreeMap<usize,usize>,damping:f64,tolerance:f64,max_sweeps:usize)->Result<BpRun>{
 ensure!(damping>0.&&damping<=1.&&tolerance>0.&&max_sweeps>0,"invalid BP settings");let started=Instant::now();let mut fac=model.factors.clone();
 for f in &mut fac {let k=f.vars.len();for (index,x) in f.data.iter_mut().enumerate(){if f.vars.iter().enumerate().any(|(j,v)|obs.get(v).is_some_and(|s| *s!=((index>>(k-1-j))&1))){*x=0.;}}}
 let n=model.dims.len();let mut adj=vec![Vec::<(usize,usize)>::new();n];for (fi,f) in fac.iter().enumerate(){for (i,&v) in f.vars.iter().enumerate(){adj[v].push((fi,i));}}
 let mut v2f:Vec<Vec<[f64;2]>>=fac.iter().map(|f|vec![[0.5,0.5];f.vars.len()]).collect();let mut f2v=v2f.clone();let mut cumulative_work=0u64;let mut trace=Vec::new();
 for sweep in 0..=max_sweeps{
  let mut nv=v2f.clone();let mut nf=f2v.clone();let mut work=0u64;
  for (v,edges) in adj.iter().enumerate(){for &(fi,pos) in edges{let mut val=[1.,1.];for &(gi,gpos) in edges{if gi!=fi{for state in 0..2{val[state]*=f2v[gi][gpos][state];work+=1;}}}nv[fi][pos]=normalize(val)?;}let _=v;}
  for (fi,f) in fac.iter().enumerate(){let k=f.vars.len();for pos in 0..k{let mut out=[0.,0.];for (idx,&weight) in f.data.iter().enumerate(){let mut w=weight;for other in 0..k{if other!=pos{let state=(idx>>(k-1-other))&1;w*=v2f[fi][other][state];work+=1;}}let state=(idx>>(k-1-pos))&1;out[state]+=w;work+=1;}nf[fi][pos]=normalize(out)?;}}
  let mut residual=0f64;for fi in 0..fac.len(){for pos in 0..fac[fi].vars.len(){for state in 0..2{residual=residual.max((nv[fi][pos][state]-v2f[fi][pos][state]).abs()).max((nf[fi][pos][state]-f2v[fi][pos][state]).abs());}}}
  let mut beliefs=Vec::with_capacity(n);for edges in &adj{let mut b=[1.,1.];for &(fi,pos) in edges{b[0]*=f2v[fi][pos][0];b[1]*=f2v[fi][pos][1];}beliefs.push(normalize(b)?)}
  trace.push(Trace{sweep,residual,cumulative_work,beliefs:beliefs.clone()});
  if residual<=tolerance || sweep==max_sweeps{return Ok(BpRun{method:"sum-product synchronous damped".into(),observations:obs.clone(),damping,tolerance,max_sweeps,sweeps:sweep,converged:residual<=tolerance,residual,cumulative_work,seconds:started.elapsed().as_secs_f64(),marginals:beliefs,trace,variable_to_factor:v2f,factor_to_variable:f2v,work_definition:"multiplications plus accumulation additions in message proposals; excludes normalizations and belief calculation".into()})}
  for fi in 0..fac.len(){for pos in 0..fac[fi].vars.len(){for state in 0..2{v2f[fi][pos][state]=(1.-damping)*v2f[fi][pos][state]+damping*nv[fi][pos][state];f2v[fi][pos][state]=(1.-damping)*f2v[fi][pos][state]+damping*nf[fi][pos][state];}}}cumulative_work+=work;
 }unreachable!()
}


#[cfg(test)]
mod tests {
 use super::*;use crate::model::{Factor,Model};
 #[test]
 fn nonuniform_tree_matches_enumeration_with_and_without_evidence(){
  let model=Model{dims:vec![2,2,2],factors:vec![Factor{vars:vec![0],data:vec![0.7,0.3]},Factor{vars:vec![0,1],data:vec![0.9,0.1,0.2,0.8]},Factor{vars:vec![1,2],data:vec![0.8,0.2,0.1,0.9]}]};
  for obs in [BTreeMap::new(),BTreeMap::from([(2,0)])] {let (_,exact,_)=model.enumerate(&obs).unwrap();let result=run(&model,&obs,0.5,1e-12,1000).unwrap();assert!(result.converged);for (x,y) in result.marginals.iter().zip(exact){for state in 0..2{assert!((x[state]-y[state]).abs()<1e-9)}}}
 }
}
