use anyhow::{bail,ensure,Context,Result};
use serde::{Serialize,Deserialize};
use std::{collections::BTreeMap,fs,path::Path};

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Factor { pub vars:Vec<usize>, pub data:Vec<f64> }
#[derive(Clone,Debug)]
pub struct Model { pub dims:Vec<usize>, pub factors:Vec<Factor> }
impl Model {
 pub fn load(path:&Path)->Result<Self>{
  let raw=fs::read_to_string(path).with_context(||format!("reading {}",path.display()))?;
  let mut it=raw.split_whitespace();let kind=it.next().context("missing header")?;ensure!(kind=="MARKOV","only UAI MARKOV models are supported");
  let n:usize=it.next().context("missing variable count")?.parse()?;let mut dims=Vec::new();for _ in 0..n{dims.push(it.next().context("missing dimension")?.parse::<usize>()?)}
  ensure!(dims.iter().all(|&d|d==2),"this build supports binary variables only");
  let nf:usize=it.next().context("missing factor count")?.parse()?;let mut scopes=Vec::new();
  for _ in 0..nf{let k:usize=it.next().context("missing scope rank")?.parse()?;let mut vars=Vec::new();for _ in 0..k{let v:usize=it.next().context("missing scope label")?.parse()?;ensure!(v<n && !vars.contains(&v),"invalid or repeated variable in scope");vars.push(v)}scopes.push(vars)}
  let mut factors=Vec::new();for vars in scopes{let size:usize=it.next().context("missing table size")?.parse()?;ensure!(size==1usize<<vars.len(),"scope/table size mismatch");let mut data=Vec::new();for _ in 0..size{let x:f64=it.next().context("missing factor entry")?.parse()?;ensure!(x.is_finite() && x>=0.,"invalid factor entry");data.push(x)}factors.push(Factor{vars,data});}ensure!(it.next().is_none(),"trailing UAI data");Ok(Self{dims,factors})
 }
 pub fn condition(&self,obs:&BTreeMap<usize,usize>)->Result<(Vec<Factor>,f64)>{
  for (&v,&s) in obs {ensure!(v<self.dims.len() && s<2,"invalid observation");}
  let mut non_scalar=Vec::new();let mut log_scalar=0.;
  for f in &self.factors {let vars:Vec<_>=f.vars.iter().copied().filter(|v|!obs.contains_key(v)).collect();let mut data=vec![0.;1<<vars.len()];
   for (i,slot) in data.iter_mut().enumerate(){let mut full=0usize;let mut j=0;for v in &f.vars{let bit=if let Some(&e)=obs.get(v){e}else{let bit=(i>>(vars.len()-1-j))&1;j+=1;bit};full=(full<<1)|bit;}*slot=f.data[full];}
   if vars.is_empty(){if data[0]<=0.{bail!("zero scalar evidence factor")}log_scalar+=data[0].ln();}else{non_scalar.push(Factor{vars,data});}
  }Ok((non_scalar,log_scalar))
 }
 pub fn enumerate(&self,obs:&BTreeMap<usize,usize>)->Result<(f64,Vec<[f64;2]>,[[f64;2];2])>{
  ensure!(self.dims.len()<=20,"enumeration limited to <=20 variables");let n=self.dims.len();let mut z=0.;let mut marg=vec![[0.;2];n];let mut joint=[[0.;2];2];
  for mask in 0..(1usize<<n){if obs.iter().any(|(&v,&s)|((mask>>v)&1)!=s){continue}let mut w=1.;for f in &self.factors{let mut index=0;for &v in &f.vars{index=(index<<1)|((mask>>v)&1)}w*=f.data[index]}z+=w;for (v,row) in marg.iter_mut().enumerate(){row[(mask>>v)&1]+=w}if n>3{joint[(mask>>1)&1][(mask>>3)&1]+=w}}
  ensure!(z>0.,"zero evidence mass");for row in &mut marg{for x in row{x.div_assign(z)}}for row in &mut joint{for x in row{*x/=z}}Ok((z,marg,joint))
 }
}
use std::ops::DivAssign;
pub fn observations(name:&str,n:usize)->Result<BTreeMap<usize,usize>>{let mut o=BTreeMap::new();match name {"prior"=>{},"AD"=>{ensure!(n==8,"AD requires ASIA");o.insert(0,0);o.insert(7,0);},"ADX"=>{ensure!(n==8,"ADX requires ASIA");o.insert(0,0);o.insert(7,0);o.insert(6,0);},_=>{let raw=fs::read_to_string(name).with_context(||format!("reading evidence {name}"))?;let mut it=raw.split_whitespace();let count:usize=it.next().context("missing evidence count")?.parse()?;for _ in 0..count{let v:usize=it.next().context("missing variable")?.parse()?;let s:usize=it.next().context("missing state")?.parse()?;ensure!(v<n&&s<2&&!o.contains_key(&v),"invalid evidence");o.insert(v,s);}ensure!(it.next().is_none(),"trailing evidence data");}}Ok(o)}
