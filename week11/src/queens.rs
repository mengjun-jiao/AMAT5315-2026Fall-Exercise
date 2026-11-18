use anyhow::{ensure,Result};
use rand::{Rng,SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::{json,Value};
use std::collections::{HashMap,HashSet};

// Sparse contraction of the row/column/diagonal carry network. Eliminating
// each row's square bits and local carry tensors gives this boundary state.
type Key=(u32,u32,u32);
fn count(mask:u32,cols:u32,left:u32,right:u32,memo:&mut HashMap<Key,u64>)->u64{
    if cols==mask {return 1}
    let key=(cols,left,right);if let Some(&v)=memo.get(&key){return v}
    let mut free=mask&!(cols|left|right);let mut total=0;
    while free!=0 {let bit=free&free.wrapping_neg();free-=bit;
        total+=count(mask,cols|bit,((left|bit)<<1)&mask,(right|bit)>>1,memo);
    }
    memo.insert(key,total);total
}
pub fn run(n:usize,mode:&str,seed:u64,samples:usize)->Result<Value>{
    ensure!((1..=15).contains(&n),"N must be 1..15");let mask=(1u32<<n)-1;
    let mut memo=HashMap::new();let total=count(mask,0,0,0,&mut memo);
    let mut result=json!({"n":n,"count":total,"stored_sparse_boundary_states":memo.len(),
        "network":"row, column and diagonal 0/1 carry constraints; row-local bits eliminated into sparse boundary states",
        "contraction_order":"row frontier, exact sparse dynamic contraction"});
    if mode=="sample" {
        ensure!(total>0,"no solutions");let mut rng=ChaCha20Rng::seed_from_u64(seed);
        let mut boards=Vec::new();
        for _ in 0..samples {let (mut cols,mut left,mut right)=(0,0,0);let mut board=Vec::new();
            while cols!=mask {let mut free=mask&!(cols|left|right);let mut options=Vec::new();let mut sum=0u64;
                while free!=0 {let bit=free&free.wrapping_neg();free-=bit;
                    let nc=cols|bit;let nl=((left|bit)<<1)&mask;let nr=(right|bit)>>1;
                    let c=count(mask,nc,nl,nr,&mut memo);sum+=c;if c>0 {options.push((bit,sum,nc,nl,nr))}
                }
                let u=rng.gen_range(0..sum);let &(bit,_,nc,nl,nr)=options.iter().find(|x|u<x.1).unwrap();
                board.push(bit.trailing_zeros() as usize);cols=nc;left=nl;right=nr;
            }
            assert!(valid(&board));boards.push(board);
        }
        let distinct:HashSet<Vec<usize>>=boards.iter().cloned().collect();
        result["samples"]=json!(boards);result["distinct"]=json!(distinct.len());
        result["rng"]=json!("ChaCha20Rng::seed_from_u64");
    }
    Ok(result)
}
fn valid(b:&[usize])->bool{for i in 0..b.len(){for j in 0..i{
    if b[i]==b[j] || b[i].abs_diff(b[j])==i-j{return false}
}}true}
