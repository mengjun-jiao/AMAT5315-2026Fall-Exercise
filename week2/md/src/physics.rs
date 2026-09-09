#[derive(Clone, Copy, Debug)]
pub enum PhysicalModel {
    OpenLennardJones,
    PeriodicShiftedLennardJones { box_size: [f64; 2], rc: f64 },
}
impl PhysicalModel {
    pub fn validate(&self) -> Result<(), String> { unimplemented!("Part 4 model") }
    pub fn displacement(&self, _a:[f64;2], _b:[f64;2])->[f64;2] { unimplemented!("Part 4 displacement") }
    pub fn wrap(&self, _p:[f64;2])->[f64;2] { unimplemented!("Part 4 wrap") }
    pub fn pair(&self, _r:f64)->Result<(f64,f64),String> { unimplemented!("Part 4 pair") }
}
pub fn accelerations(_pos:&[[f64;2]],_model:&PhysicalModel)->Result<Vec<[f64;2]>,String> { unimplemented!("Part 4 forces") }
pub fn energies(_pos:&[[f64;2]],_vel:&[[f64;2]],_model:&PhysicalModel)->Result<(f64,f64),String> { unimplemented!("Part 4 energies") }
