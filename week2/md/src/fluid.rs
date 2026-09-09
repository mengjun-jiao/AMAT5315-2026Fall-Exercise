use crate::System;
#[derive(Clone,Debug)]
pub struct RunConfig { pub n:usize,pub rho:f64,pub temperature:f64,pub dt:f64,pub eq_steps:usize,pub steps:usize,pub sample_every:usize,pub seed:u64,pub integrator:String }
impl Default for RunConfig { fn default()->Self { Self {n:100,rho:0.8,temperature:0.5,dt:0.01,eq_steps:2000,steps:10000,sample_every:50,seed:2026,integrator:"velocity-verlet".into()} } }
impl RunConfig { pub fn validate(&self)->Result<(),String> { unimplemented!("Part 4 config") } }
pub fn lattice(_n:usize,_rho:f64)->Result<(Vec<[f64;2]>,[f64;2]),String> { unimplemented!("Part 4 lattice") }
pub fn initialize(_c:&RunConfig)->Result<System,String> { unimplemented!("Part 4 initial state") }
