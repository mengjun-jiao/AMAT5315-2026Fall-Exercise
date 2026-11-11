mod linalg;
use anyhow::{ensure,Context,Result};
use linalg::{Backend,write_npy};
use rand::{SeedableRng,Rng};
use rand_chacha::ChaCha20Rng;
use rand_distr::StandardNormal;
use serde_json::json;
use std::{env,fs,path::PathBuf,time::Instant};

fn main()->Result<()> {
    let args:Vec<_>=env::args().skip(1).collect();
    ensure!(args.len()==2&&args[0]=="--out","usage: svd-evidence --out DIRECTORY");
    let out=PathBuf::from(&args[1]);fs::create_dir_all(&out)?;
    let backend=Backend::new()?;let mut rng=ChaCha20Rng::seed_from_u64(2026);
    let a:Vec<f64>=(0..512*512).map(|_|rng.sample(StandardNormal)).collect();
    write_npy(&out.join("matrix.npy"),&[512,512],&a)?;
    let t=Instant::now();let svd=backend.svd(&a,512,512).context("Gaussian matrix SVD")?;
    let seconds=t.elapsed().as_secs_f64();
    write_npy(&out.join("singular_values.npy"),&[512],&svd.s)?;
    for rank in [16,64,128,256,512] {
        let mut left=vec![0.;512*rank];
        for i in 0..512 {for q in 0..rank {left[i*rank+q]=svd.u[i*svd.rank+q]*svd.s[q];}}
        let mut right=vec![0.;rank*512];right.copy_from_slice(&svd.vt[..rank*512]);
        let recon=backend.multiply(&left,&right,512,rank,512)?;
        write_npy(&out.join(format!("reconstruction-{rank}.npy")),&[512,512],&recon)?;
    }
    fs::write(out.join("metadata.json"),serde_json::to_string_pretty(&json!({
        "seed":2026,"generator":"rand_chacha 0.3 ChaCha20Rng::seed_from_u64, rand_distr 0.4 StandardNormal",
        "shape":[512,512],"ranks":[16,64,128,256,512],
        "svd":"OpenBLAS LAPACKE dgesdd, f64, thin","svd_seconds":seconds
    }))?)?;
    println!("SVD and five reconstructions saved to {}; SVD {:.6}s",out.display(),seconds);
    Ok(())
}
