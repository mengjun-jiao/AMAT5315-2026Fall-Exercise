use serde::{Serialize,Deserialize};
use crate::fluid::RunConfig;
use std::path::Path;
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct RunMetadata { #[serde(flatten)] pub config:RunConfig, #[serde(rename="box")] pub box_size:[f64;2] }
impl std::ops::Deref for RunMetadata { type Target=RunConfig;fn deref(&self)->&RunConfig { &self.config } }
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Frame { pub step:usize,pub t:f64,pub pos:Vec<[f64;2]>,pub vel:Vec<[f64;2]>,#[serde(rename="E_pot")] pub e_pot:f64,#[serde(rename="E_kin")] pub e_kin:f64 }
pub fn read_trajectory(_dir:&Path)->Result<(RunMetadata,Vec<Frame>),String> { unimplemented!("Part 4 read trajectory") }
pub fn run_to_directory(_c:&RunConfig,_dir:&Path)->Result<(),String> { unimplemented!("Part 4 write trajectory") }
