mod linalg;
mod boundary;
mod trg;
use anyhow::{bail,ensure,Context,Result};
use linalg::{Backend,write_npy};
use serde_json::json;
use std::{collections::BTreeMap,env,fs,path::PathBuf,time::Instant};

fn main()->Result<()> {
    let mut it=env::args().skip(1);let mut args=BTreeMap::new();
    while let Some(key)=it.next() {
        ensure!(["--method","--l","--k","--chi","--out","--record"].contains(&key.as_str()),"unknown option {key}");
        let value=it.next().with_context(||format!("missing value for {key}"))?;
        ensure!(args.insert(key,value).is_none(),"duplicate option");
    }
    let get=|key:&str|args.get(key).with_context(||format!("missing {key}"));
    let method=get("--method")?.as_str();let l:usize=get("--l")?.parse()?;let k:f64=get("--k")?.parse()?;
    let chi:usize=get("--chi")?.parse()?;let out=PathBuf::from(get("--out")?);
    let record=args.get("--record").map(String::as_str).unwrap_or("full");
    ensure!(record=="full"||record=="summary","record must be full or summary");
    ensure!(l>=2 && k.is_finite() && k>=0. && chi>=1,"invalid L, K or chi");
    ensure!(method=="boundary"||method=="trg","method must be boundary or trg");
    if method=="trg" {ensure!(l.is_power_of_two(),"TRG requires power-of-two L")}
    fs::create_dir_all(&out)?;let backend=Backend::new()?;let start=Instant::now();
    let log_z=if method=="boundary" {
        let result=boundary::solve(&backend,l,k,chi,record=="full")?;
        if record=="full" {
            for (i,core) in result.cores.iter().enumerate() {
                write_npy(&out.join(format!("final-core-{i:03}.npy")),&[core.left,2,core.right],&core.data)?;
            }
            write_npy(&out.join("edge.npy"),&[2],&result.edge)?;
            fs::write(out.join("truncations.json"),serde_json::to_string(&result.truncations)?)?;
        }
        fs::write(out.join("scales.json"),serde_json::to_string(&result.scales)?)?;
        result.log_z
    } else if method=="trg" {
        let result=trg::solve(&backend,l,k,chi,record=="full",&out)?;
        if record=="full" {write_npy(&out.join("final-tensor.npy"),&[result.final_dim;4],&result.final_tensor)?}
        fs::write(out.join("scales.json"),serde_json::to_string(&result.steps)?)?;
        result.log_z
    } else {bail!("unknown method")};
    let seconds=start.elapsed().as_secs_f64();let g=log_z/(l*l) as f64;
    ensure!(log_z.is_finite()&&g.is_finite(),"invalid result");
    let run=json!({"model":"square Ising; H=-sum_edges s_i*s_j; spins +/-1; J=k_B=1; zero field",
       "method":method,"geometry":if method=="boundary"{"open"}else{"periodic"},
       "l":l,"k":k,"chi":chi,"record":record,"log_z":log_z,"g":g,"seconds":seconds,
       "backend":"f64 OpenBLAS LAPACKE dgesdd and cblas dgemm (ILP64)"});
    fs::write(out.join("run.json"),serde_json::to_string_pretty(&run)?)?;
    println!("{}",serde_json::to_string(&run)?);
    Ok(())
}
