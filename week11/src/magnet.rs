use anyhow::{bail,ensure,Context,Result};
use omeco::{EinCode,GreedyMethod,TreeSA,NestedEinsum,optimize_code,contraction_complexity};
use rand::{Rng,SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::{json,Value};
use std::{collections::HashMap,fs};

#[derive(Clone)]
pub struct Magnet { pub n:usize, pub fields:Vec<i32>, pub edges:Vec<(usize,usize,i32)> }
impl Magnet {
    pub fn from_bundle(path:&str)->Result<Self> {
        let v:Value=serde_json::from_str(&fs::read_to_string(path)?)?;
        let d=&v["target"]["data"];
        let fields:Vec<i32>=serde_json::from_value(d["fields"].clone())?;
        let edges:Vec<[usize;2]>=serde_json::from_value(d["graph"]["edges"].clone())?;
        let couplings:Vec<i32>=serde_json::from_value(d["couplings"].clone())?;
        ensure!(fields.len()==d["graph"]["num_vertices"].as_u64().context("num_vertices")? as usize);
        ensure!(edges.len()==couplings.len());
        let n=fields.len();
        Ok(Self{n,fields,edges:edges.into_iter().zip(couplings).map(|(e,j)|(e[0],e[1],j)).collect()})
    }
    pub fn lattice(l:usize,p:f64,seed:u64)->Result<Self>{
        ensure!(l>=2 && (0. ..=1.).contains(&p));
        let mut rng=ChaCha20Rng::seed_from_u64(seed);let mut edges=Vec::new();
        for r in 0..l {for c in 0..l {let a=r*l+c;
            for b in [if r+1<l {Some(a+l)}else{None},if c+1<l {Some(a+1)}else{None}].into_iter().flatten(){
                edges.push((a,b,if rng.gen::<f64>()<p {1}else{-1}));
            }
        }}
        Ok(Self{n:l*l,fields:vec![0;l*l],edges})
    }
    pub fn energy(&self,bits:&[u8])->i32 {
        let s:Vec<i32>=bits.iter().map(|&b|2*(b as i32)-1).collect();
        self.fields.iter().zip(&s).map(|(&h,&x)|h*x).sum::<i32>()+
            self.edges.iter().map(|&(a,b,j)|j*s[a]*s[b]).sum::<i32>()
    }
    fn factors(&self)->Vec<Factor>{
        let mut out:Vec<Factor>=self.fields.iter().enumerate().map(|(i,&h)|Factor{vars:vec![i],energy:vec![-h,h]}).collect();
        for &(i,j,k) in &self.edges {out.push(Factor{vars:vec![i,j],energy:vec![k,-k,-k,k]})}
        out
    }
}
struct Factor {vars:Vec<usize>,energy:Vec<i32>}

fn plan(factors:&[Factor],n:usize,order:&str)->Result<(NestedEinsum<usize>,f64,f64)>{
    let code=EinCode::new(factors.iter().map(|x|x.vars.clone()).collect(),vec![]);
    let sizes:HashMap<usize,usize>=(0..n).map(|i|(i,2)).collect();
    let tree=match order {"greedy"=>optimize_code(&code,&sizes,&GreedyMethod::default()),
        "treesa"=>optimize_code(&code,&sizes,&TreeSA::default().with_ntrials(8)),_=>bail!("unknown order")}.context("planner failed")?;
    let c=contraction_complexity(&tree,&sizes,&code.ixs);Ok((tree,c.sc,c.tc))
}

#[derive(Clone,Copy)]
struct Entry {w:f64,e:i32,g:u64}
struct Node {vars:Vec<usize>, data:Vec<Entry>, scale:f64, children:Option<(Box<Node>,Box<Node>)>}
fn project(bits:usize,map:&[usize])->usize {map.iter().enumerate().fold(0,|a,(i,&j)|a|(((bits>>j)&1)<<i))}
fn maps(vars:&[usize],all:&[usize])->Vec<usize>{vars.iter().map(|x|all.iter().position(|y|x==y).unwrap()).collect()}
fn contract(tree:&NestedEinsum<usize>,factors:&[Factor],beta:f64)->Result<Node>{
    match tree {
        NestedEinsum::Leaf{tensor_index}=>{
            let f=&factors[*tensor_index]; let base=*f.energy.iter().min().unwrap();
            Ok(Node{vars:f.vars.clone(),data:f.energy.iter().map(|&e|Entry{w:(-beta*(e-base) as f64).exp(),e,g:1}).collect(),scale:-beta*base as f64,children:None})
        },
        NestedEinsum::Node{args,eins}=>{
            ensure!(args.len()==2,"non-binary contraction unsupported");
            let a=Box::new(contract(&args[0],factors,beta)?);let b=Box::new(contract(&args[1],factors,beta)?);
            let mut all=a.vars.clone();for &v in &b.vars{if !all.contains(&v){all.push(v)}}
            ensure!(all.len()<=27,"intermediate contraction too large: {} indices",all.len());
            let am=maps(&a.vars,&all);let bm=maps(&b.vars,&all);let om=maps(&eins.iy,&all);
            let mut data=vec![Entry{w:0.,e:i32::MAX,g:0};1usize<<eins.iy.len()];
            for k in 0..(1usize<<all.len()){
                let x=a.data[project(k,&am)];let y=b.data[project(k,&bm)];let t=&mut data[project(k,&om)];
                t.w+=x.w*y.w;let e=x.e+y.e;
                let g=x.g.checked_mul(y.g).context("ground-state degeneracy overflow")?;
                if e<t.e {t.e=e;t.g=g}else if e==t.e {t.g=t.g.checked_add(g).context("degeneracy overflow")?}
            }
            let scale=data.iter().map(|x|x.w).fold(0f64,f64::max);ensure!(scale>0.&&scale.is_finite());
            for e in &mut data {e.w/=scale}
            let logscale=a.scale+b.scale+scale.ln();
            Ok(Node{vars:eins.iy.clone(),data,scale:logscale,children:Some((a,b))})
        }
    }
}
fn draw(node:&Node,bits:&mut [Option<u8>],rng:&mut ChaCha20Rng)->Result<()> {
    if let Some((a,b))=&node.children {
        let mut all=a.vars.clone();for &v in &b.vars{if !all.contains(&v){all.push(v)}}
        let free:Vec<usize>=all.iter().copied().filter(|i|bits[*i].is_none()).collect();
        let am=maps(&a.vars,&all);let bm=maps(&b.vars,&all);
        let mut choices=Vec::with_capacity(1<<free.len());let mut total=0.;
        for t in 0..(1usize<<free.len()){
            for (i,&x) in free.iter().enumerate(){bits[x]=Some(((t>>i)&1) as u8)}
            let encoded=all.iter().enumerate().fold(0usize,|z,(i,&x)|z|((bits[x].unwrap() as usize)<<i));
            total+=a.data[project(encoded,&am)].w*b.data[project(encoded,&bm)].w;choices.push(total);
            for &x in &free {bits[x]=None}
        }
        ensure!(total>0.&&total.is_finite(),"empty conditional distribution");
        let u=rng.gen::<f64>()*total;let t=choices.iter().position(|&x|u<x).unwrap_or(choices.len()-1);
        for (i,&x) in free.iter().enumerate(){bits[x]=Some(((t>>i)&1) as u8)}
        draw(a,bits,rng)?;draw(b,bits,rng)?;
    }
    Ok(())
}

pub fn run(m:&Magnet,mode:&str,order:&str,beta:f64,seed:u64,samples:usize)->Result<Value>{
    let factors=m.factors();let (tree,sc,tc)=plan(&factors,m.n,order)?;
    let mut out=json!({"variables":m.n,"factors":factors.len(),"sc":sc,"tc":tc,"order":order,"planner":"omeco 0.2.6",
        "contraction_tree":tree,"energy_convention":"sum J_ij s_i s_j + sum h_i s_i"});
    if mode=="plan" {return Ok(out)}
    let root=contract(&tree,&factors,beta)?;ensure!(root.vars.is_empty()&&root.data.len()==1);
    let x=root.data[0];out["log_z"]=json!(root.scale+x.w.ln());out["e_min"]=json!(x.e);out["ground_degeneracy"]=json!(x.g);
    if mode=="sample" {
        let mut rng=ChaCha20Rng::seed_from_u64(seed);let mut states=Vec::with_capacity(samples);
        for _ in 0..samples {let mut bits=vec![None;m.n];draw(&root,&mut bits,&mut rng)?;
            let b:Vec<u8>=bits.into_iter().collect::<Option<Vec<_>>>().context("unassigned spin")?;
            let energy=m.energy(&b);states.push(json!({"bits":b,"energy":energy}));}
        out["samples"]=json!(states);out["rng"]=json!("ChaCha20Rng::seed_from_u64");
    }
    Ok(out)
}
