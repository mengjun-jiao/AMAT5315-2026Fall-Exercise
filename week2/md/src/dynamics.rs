pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
    #[cfg(test)]
    force_evaluations: usize,
}
impl System {
    pub fn new(_positions: Vec<[f64; 2]>, _velocities: Vec<[f64; 2]>) -> Self {
        todo!("System::new: validation and initial acceleration")
    }
    pub fn positions(&self) -> &[[f64; 2]] { &self.positions }
    pub fn velocities(&self) -> &[[f64; 2]] { &self.velocities }
    pub fn accelerations(&self) -> &[[f64; 2]] { &self.accelerations }
    pub fn kinetic_energy(&self) -> f64 { todo!("System::kinetic_energy") }
    pub fn potential_energy(&self) -> f64 { todo!("System::potential_energy") }
    pub fn total_energy(&self) -> f64 { todo!("System::total_energy") }
}
pub trait Integrator { fn step(&self, system: &mut System, dt: f64); }
pub struct Euler;
pub struct VelocityVerlet;
impl Integrator for Euler {
    fn step(&self, _: &mut System, _: f64) { todo!("Euler::step") }
}
impl Integrator for VelocityVerlet {
    fn step(&self, _: &mut System, _: f64) { todo!("VelocityVerlet::step") }
}
pub struct Sample {
    pub step: usize, pub time: f64, pub kinetic: f64,
    pub potential: f64, pub total: f64,
}
pub fn simulate<I: Integrator>(_: &I, _: &mut System, _: f64, _: usize) -> Vec<Sample> {
    todo!("simulate: complete-step sampling")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verlet_refreshes_once_at_initialization_and_once_per_step() {
        let mut s = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]);
        assert_eq!(s.force_evaluations, 1);
        let integrator = VelocityVerlet;
        for completed in 1..=5 {
            integrator.step(&mut s, 0.01);
            assert_eq!(s.force_evaluations, 1 + completed);
        }
    }
}
