mod gadgets;
mod magnet;
mod queens;
mod queens_network;
use anyhow::{bail,Context,Result};
use serde_json::json;
use std::{collections::BTreeMap,env,fs,time::Instant};

fn main()->Result<()> {
    let mut it=env::args().skip(1); let mut args=BTreeMap::new();
    while let Some(k)=it.next() {
        let v=it.next().with_context(||format!("missing value for {k}"))?;
        if args.insert(k.clone(),v).is_some(){bail!("duplicate argument {k}")}
    }
    let get=|k:&str| args.get(k).with_context(||format!("missing {k}"));
    let mode=get("--mode")?; let seed:u64=get("--seed")?.parse()?; let out=get("--out")?;
    fs::create_dir_all(out)?;let start=Instant::now();
    let payload=match mode.as_str(){
        "gadgets"=>gadgets::enumerate(),
        "plan" if args.contains_key("--n")=>{
            queens_network::plan_output(args["--n"].parse()?,args.get("--order").map(String::as_str).unwrap_or("treesa"))?
        },
        "count"|"sample" if args.contains_key("--n")=>{
            let n=args["--n"].parse()?;
            let samples=if mode=="sample" {get("--samples")?.parse()?}else{0};
            if env::var("WEEK11_ROW_FRONTIER").as_deref()==Ok("1") {queens::run(n,mode,seed,samples)?}
            else {queens_network::run(n,mode,args.get("--order").map(String::as_str).unwrap_or("treesa"),seed,samples)?}
        },
        "plan"|"count"|"sample"=>{
            let magnet=if let Some(path)=args.get("--input") {magnet::Magnet::from_bundle(path)?}
                else if let Some(l)=args.get("--l") {magnet::Magnet::lattice(l.parse()?,get("--p-af")?.parse()?,seed)?}
                else {bail!("provide --input or --l with --p-af")};
            let order=args.get("--order").map(String::as_str).unwrap_or("treesa");
            let beta=if mode=="plan" {0.}else{get("--beta")?.parse()?};
            let samples=if mode=="sample" {get("--samples")?.parse()?}else{0};
            magnet::run(&magnet,mode,order,beta,seed,samples)?
        },
        _=>bail!("unknown mode {mode}")
    };
    let elapsed=start.elapsed().as_secs_f64()*1000.;
    let run=json!({"schema_version":1,"mode":mode,"seed":seed,"parameters":args,"runtime_ms":elapsed,"evidence":payload});
    fs::write(format!("{out}/run.json"),serde_json::to_string_pretty(&run)?)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    if mode=="gadgets" {
        println!("{mode}\tgate_rows\t{}\t{elapsed:.3}",run["evidence"]["gates"].as_array().unwrap().len());
        println!("{mode}\tcircuit_rows\t{}\t{elapsed:.3}",run["evidence"]["circuit"].as_array().unwrap().len());
        println!("{mode}\tcell_states\t{}\t{elapsed:.3}",run["evidence"]["multiplier_cell"]["states"].as_array().unwrap().len());
    } else {
        for key in ["variables","sc","tc","log_z","e_min","ground_degeneracy","count","distinct","stored_sparse_boundary_states"] {
            if !run["evidence"][key].is_null(){println!("{mode}\t{key}\t{}\t{elapsed:.3}",run["evidence"][key]);}
        }
    }
    Ok(())
}
