use anyhow::{ensure,Context,Result};
use omeco::{EinCode,GreedyMethod,TreeSA,NestedEinsum,optimize_code,contraction_complexity};
use serde_json::{json,Value};
use std::collections::HashMap;
use rand::{Rng,SeedableRng};
use rand_chacha::ChaCha20Rng;

#[derive(Clone)]
pub struct Factor { vars:Vec<usize>, entries:Vec<(u64,u64)> }
pub fn build(n:usize)->Vec<Factor>{
    let mut lines:Vec<(Vec<usize>,bool)>=Vec::new();
    for r in 0..n {lines.push(((0..n).map(|c|r*n+c).collect(),true));}
    for c in 0..n {lines.push(((0..n).map(|r|r*n+c).collect(),true));}
    for delta in -(n as isize-1)..=(n as isize-1){
        let mut d=Vec::new();let mut a=Vec::new();
        for r in 0..n {for c in 0..n {
            if r as isize-c as isize==delta {d.push(r*n+c)}
            if r as isize+c as isize==delta+n as isize-1 {a.push(r*n+c)}
        }}
        lines.push((d,false));lines.push((a,false));
    }
    let mut next=n*n;let mut factors=Vec::new();
    for (line,exact) in lines {
        if line.is_empty(){continue}
        let carry:Vec<usize>=(0..line.len().saturating_sub(1)).map(|_|{let i=next;next+=1;i}).collect();
        for (i,&x) in line.iter().enumerate(){
            let mut vars=vec![x];if i>0 {vars.push(carry[i-1])} if i+1<line.len(){vars.push(carry[i])}
            let mut entries=Vec::new();
            for bits in 0..(1u64<<vars.len()){
                let bit=bits&1;let cin=if i>0 {(bits>>1)&1}else{0};
                if cin+bit>1 {continue}
                let cout=if i+1<line.len(){(bits>>(if i>0{2}else{1}))&1}else{cin+bit};
                if cout!=cin+bit || (i+1==line.len()&&exact&&cout!=1){continue}
                entries.push((bits,1));
            }
            factors.push(Factor{vars,entries});
        }
    }
    factors
}
pub fn plan(factors:&[Factor],order:&str)->Result<(NestedEinsum<usize>,f64,f64)>{
    let code=EinCode::new(factors.iter().map(|f|f.vars.clone()).collect(),vec![]);
    let sizes:HashMap<usize,usize>=code.ixs.iter().flatten().map(|&v|(v,2)).collect();
    let tree=match order {"greedy"=>optimize_code(&code,&sizes,&GreedyMethod::default()),
        "treesa"=>{let mut o=TreeSA::default().with_ntrials(12);o.score=omeco::ScoreFunction::space_optimized(0.);optimize_code(&code,&sizes,&o)},_=>None}.context("planner failed")?;
    let c=contraction_complexity(&tree,&sizes,&code.ixs);Ok((tree,c.sc,c.tc))
}
pub fn plan_output(n:usize,order:&str)->Result<Value>{
    ensure!(n>=1 && n<=15);let f=build(n);let (tree,sc,tc)=plan(&f,order)?;
    let variables=f.iter().flat_map(|t|t.vars.iter()).copied().max().unwrap()+1;
    Ok(json!({"n":n,"variables":variables,"factors":f.len(),"sc":sc,"tc":tc,"planner":"omeco 0.2.6","order":order,"contraction_tree":tree}))
}

struct Node { vars:Vec<usize>, entries:HashMap<u64,u64>, children:Option<(Box<Node>,Box<Node>)> }
fn project(x:u64,map:&[usize])->u64{map.iter().enumerate().fold(0,|v,(i,&j)|v|(((x>>j)&1)<<i))}
fn mappings(vars:&[usize],all:&[usize])->Vec<usize>{vars.iter().map(|v|all.iter().position(|x|x==v).unwrap()).collect()}
fn expand(x:u64,map:&[usize])->u64{map.iter().enumerate().fold(0,|v,(i,&j)|v|(((x>>i)&1)<<j))}
fn merge(tree:&NestedEinsum<usize>,f:&[Factor],keep:bool,fixed:&HashMap<usize,u8>)->Result<Node>{
    match tree {
        NestedEinsum::Leaf{tensor_index}=>{
            let t=&f[*tensor_index];
            let entries=t.entries.iter().copied().filter(|(k,_)|t.vars.iter().enumerate().all(|(i,v)|fixed.get(v).is_none_or(|x|((k>>i)&1)==*x as u64))).collect();
            Ok(Node{vars:t.vars.clone(),entries,children:None})
        },
        NestedEinsum::Node{args,eins}=>{
            ensure!(args.len()==2,"expected binary tree");
            let a=Box::new(merge(&args[0],f,keep,fixed)?);let b=Box::new(merge(&args[1],f,keep,fixed)?);
            let mut all=a.vars.clone();for &x in &b.vars{if !all.contains(&x){all.push(x)}}
            ensure!(all.len()<64,"intermediate rank too large");
            let common:Vec<usize>=a.vars.iter().copied().filter(|x|b.vars.contains(x)).collect();
            let ac=mappings(&common,&a.vars);let bc=mappings(&common,&b.vars);
            let aa=mappings(&a.vars,&all);let ba=mappings(&b.vars,&all);let output=mappings(&eins.iy,&all);
            let mut index:HashMap<u64,Vec<(u64,u64)>>=HashMap::new();
            for (&k,&v) in &b.entries {index.entry(project(k,&bc)).or_default().push((k,v));}
            let mut out:HashMap<u64,u64>=HashMap::new();let mut joins=0u64;
            for (&ka,&va) in &a.entries {if let Some(group)=index.get(&project(ka,&ac)){
                for &(kb,vb) in group {
                    joins+=1;ensure!(joins<=30_000_000,"sparse join exceeded work limit");
                    let bits=expand(ka,&aa)|expand(kb,&ba);let key=project(bits,&output);
                    let val=va.checked_mul(vb).context("count overflow")?;let e=out.entry(key).or_default();
                    *e=e.checked_add(val).context("count overflow")?;
                }
            }}
            ensure!(out.len()<=2_000_000,"sparse output exceeded memory limit");
            Ok(Node{vars:eins.iy.clone(),entries:out,children:if keep {Some((a,b))}else{None}})
        }
    }
}
fn sample_node(node:&Node,assigned:&mut [Option<u8>],rng:&mut ChaCha20Rng)->Result<()>{
    if let Some((a,b))=&node.children {
        let mut all=a.vars.clone();for &x in &b.vars{if !all.contains(&x){all.push(x)}}
        let common:Vec<usize>=a.vars.iter().copied().filter(|x|b.vars.contains(x)).collect();
        let ac=mappings(&common,&a.vars);let bc=mappings(&common,&b.vars);
        let aa=mappings(&a.vars,&all);let ba=mappings(&b.vars,&all);
        let mut idx:HashMap<u64,Vec<(u64,u64)>>=HashMap::new();
        let mut bsorted:Vec<_>=b.entries.iter().map(|(&k,&v)|(k,v)).collect();bsorted.sort_unstable();
        for (k,v) in bsorted {idx.entry(project(k,&bc)).or_default().push((k,v));}
        let mut choices=Vec::new();let mut sum=0u64;
        let mut asorted:Vec<_>=a.entries.iter().map(|(&k,&v)|(k,v)).collect();asorted.sort_unstable();
        for (ka,va) in asorted {if let Some(group)=idx.get(&project(ka,&ac)){for &(kb,vb) in group {
            let bits=expand(ka,&aa)|expand(kb,&ba);
            if all.iter().enumerate().any(|(i,&v)|assigned[v].is_some_and(|x|x as u64!=((bits>>i)&1))){continue}
            sum+=va*vb;choices.push((bits,sum));
        }}}
        ensure!(sum>0,"empty sparse conditional");let r=rng.gen_range(0..sum);
        let bits=choices.iter().find(|x|r<x.1).unwrap().0;
        for (i,&v) in all.iter().enumerate(){assigned[v]=Some(((bits>>i)&1) as u8)}
        sample_node(a,assigned,rng)?;sample_node(b,assigned,rng)?;
    }
    Ok(())
}
pub fn run(n:usize,mode:&str,order:&str,seed:u64,samples:usize)->Result<Value>{
    ensure!((1..=15).contains(&n));let f=build(n);let (tree,sc,tc)=plan(&f,order)?;
    let slicing=mode=="count"&&n>=10;
    let mut slice_counts=Vec::new();
    let root=if slicing {
        for column in 0..n {
            let fixed:HashMap<usize,u8>=(0..n).map(|j|(j,(j==column) as u8)).collect();
            let part=merge(&tree,&f,false,&fixed)?;
            slice_counts.push(*part.entries.get(&0).unwrap_or(&0));
        }
        Node{vars:vec![],entries:HashMap::from([(0,slice_counts.iter().sum())]),children:None}
    }else{merge(&tree,&f,mode=="sample",&HashMap::new())?};
    let count=*root.entries.get(&0).unwrap_or(&0);
    let mut out=json!({"n":n,"count":count,"factors":f.len(),"sc":sc,"tc":tc,"planner":"omeco 0.2.6",
        "order":order,"contraction_tree":tree,"network":"sparse 0/1 carry tensors","first_row_slice_counts":slice_counts});
    if mode=="sample" {
        let maxvar=f.iter().flat_map(|t|t.vars.iter()).copied().max().unwrap()+1;
        let mut rng=ChaCha20Rng::seed_from_u64(seed);let mut boards=Vec::new();
        for _ in 0..samples{let mut assigned=vec![None;maxvar];sample_node(&root,&mut assigned,&mut rng)?;
            let mut board=Vec::new();for r in 0..n {let col=(0..n).find(|&c|assigned[r*n+c]==Some(1)).context("missing queen")?;board.push(col)}
            boards.push(board);
        }
        let distinct=boards.iter().collect::<std::collections::HashSet<_>>().len();
        out["samples"]=json!(boards);out["distinct"]=json!(distinct);out["rng"]=json!("ChaCha20Rng::seed_from_u64");
    }
    Ok(out)
}
