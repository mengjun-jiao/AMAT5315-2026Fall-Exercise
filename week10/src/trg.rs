use anyhow::{ensure,Result};
use serde::Serialize;
use std::path::Path;
use crate::{boundary::site,linalg::{Backend,write_npy}};

#[derive(Serialize)]
pub struct Step {
    pub step:usize,pub tensors_before:usize,pub scale:f64,pub input_dim:usize,pub retained_dim:usize,
    pub discarded_squared_weight_cut1:f64,pub discarded_squared_weight_cut2:f64,
    pub singular_values_cut1:Vec<f64>,pub singular_values_cut2:Vec<f64>,
}
pub struct TrgResult {pub log_z:f64,pub final_dim:usize,pub final_tensor:Vec<f64>,pub steps:Vec<Step>}

fn split(backend:&Backend,a:&[f64],d:usize,chi:usize,cut:usize)->Result<(Vec<f64>,Vec<f64>,Vec<f64>,usize)> {
    let dd=d*d;let mut matrix=vec![0.;dd*dd];
    for r in 0..d {for u in 0..d {for l in 0..d {for down in 0..d {
        let v=a[((r*d+u)*d+l)*d+down];
        let (i,j)=if cut==1 {(l*d+u,r*d+down)} else {(down*d+l,u*d+r)};
        matrix[i*dd+j]=v;
    }}}}
    let svd=backend.svd(&matrix,dd,dd)?;let q=chi.min(svd.rank);
    let mut x=vec![0.;dd*q];let mut y=vec![0.;q*dd];
    for i in 0..dd {for j in 0..q {x[i*q+j]=svd.u[i*svd.rank+j]*svd.s[j].sqrt();}}
    for j in 0..q {for i in 0..dd {y[j*dd+i]=svd.s[j].sqrt()*svd.vt[j*dd+i];}}
    Ok((x,y,svd.s,q))
}

// Equivalent to S1[w,a,r] S2[a,b,u] S3[l,b,g] S4[d,g,w].
// Three matrix products preserve the exact specified index pairing.
fn regroup(backend:&Backend,s1:&[f64],s2:&[f64],s3:&[f64],s4:&[f64],d:usize,q:usize)->Result<Vec<f64>>{
    let mut left=vec![0.;d*q*d];let mut right=vec![0.;d*d*q];
    for w in 0..d {for r in 0..q {for a in 0..d {left[(w*q+r)*d+a]=s1[(w*d+a)*q+r];}}}
    for a in 0..d {for b in 0..d {for u in 0..q {right[a*(d*q)+b*q+u]=s2[(a*d+b)*q+u];}}}
    let p=backend.multiply(&left,&right,d*q,d,d*q)?;
    let mut x=vec![0.;q*d*d];let mut y=vec![0.;d*q*d];
    for l in 0..q {for b in 0..d {for g in 0..d {x[(l*d+b)*d+g]=s3[(l*d+b)*d+g];}}}
    for g in 0..d {for down in 0..q {for w in 0..d {y[g*(q*d)+down*d+w]=s4[(down*d+g)*d+w];}}}
    let z=backend.multiply(&x,&y,q*d,d,q*d)?;
    let mut p2=vec![0.;q*q*d*d];let mut z2=vec![0.;d*d*q*q];
    for r in 0..q {for u in 0..q {for w in 0..d {for b in 0..d {
        p2[(r*q+u)*(d*d)+w*d+b]=p[(w*q+r)*(d*q)+b*q+u];
    }}}}
    for w in 0..d {for b in 0..d {for l in 0..q {for down in 0..q {
        z2[(w*d+b)*(q*q)+l*q+down]=z[(l*d+b)*(q*d)+down*d+w];
    }}}}
    backend.multiply(&p2,&z2,q*q,d*d,q*q)
}

pub fn solve(backend:&Backend,l:usize,k:f64,chi:usize,full:bool,out:&Path)->Result<TrgResult>{
    ensure!(l>=2&&l.is_power_of_two()&&k.is_finite()&&k>=0.&&chi>=1,"TRG requires power-of-two L >= 2, K >= 0 and chi >= 1");
    let steps=2*l.ilog2() as usize;let mut count=l*l;let mut a=site(k).to_vec();let mut d=2usize;
    let mut log_scale=0.;let mut records=Vec::new();
    for step in 1..=steps {
        let scale=a.iter().fold(0f64,|x,y|x.max(y.abs()));
        ensure!(scale>0.&&scale.is_finite(),"invalid TRG scale");
        for x in &mut a {*x/=scale} log_scale+=count as f64*scale.ln();
        let (s1,s3,vals1,q1)=split(backend,&a,d,chi,1)?;
        let (s2,s4,vals2,q2)=split(backend,&a,d,chi,2)?;
        ensure!(q1==q2,"incompatible split dimensions");let q=q1;
        if full {
            for (name,shape,data) in [
                ("s1",vec![d,d,q],s1.as_slice()),("s2",vec![d,d,q],s2.as_slice()),
                ("s3",vec![q,d,d],s3.as_slice()),("s4",vec![q,d,d],s4.as_slice()),
                ("singular-1",vec![vals1.len()],vals1.as_slice()),("singular-2",vec![vals2.len()],vals2.as_slice())
            ] {write_npy(&out.join(format!("step-{step:02}-{name}.npy")),&shape,data)?}
        }
        let next=regroup(backend,&s1,&s2,&s3,&s4,d,q)?;
        records.push(Step{step,tensors_before:count,scale,input_dim:d,retained_dim:q,
            discarded_squared_weight_cut1:vals1[q..].iter().map(|x|x*x).sum(),
            discarded_squared_weight_cut2:vals2[q..].iter().map(|x|x*x).sum(),
            singular_values_cut1:if full{vals1}else{Vec::new()},singular_values_cut2:if full{vals2}else{Vec::new()}});
        count/=2;a=next;d=q;
    }
    ensure!(count==1,"invalid TRG tensor count");
    let mut z=0.;for i in 0..d {for j in 0..d {z+=a[((i*d+j)*d+i)*d+j];}}
    ensure!(z>0.&&z.is_finite(),"invalid TRG closure");
    Ok(TrgResult{log_z:log_scale+z.ln(),final_dim:d,final_tensor:a,steps:records})
}
