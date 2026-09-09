pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
    #[cfg(test)]
    force_evaluations: usize,
}
impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert_eq!(positions.len(), velocities.len(),
                   "positions and velocities must have equal lengths");
        assert!(positions.len() >= 2, "at least two atoms");
        assert!(positions.iter().chain(&velocities).flatten().all(|v| v.is_finite()),
                "finite state");
        let accelerations = vec![[0.0; 2]; positions.len()];
        let mut s = Self { positions, velocities, accelerations,
                          #[cfg(test)] force_evaluations: 0 };
        s.refresh_accelerations();
        s
    }
    pub fn positions(&self) -> &[[f64; 2]] { &self.positions }
    pub fn velocities(&self) -> &[[f64; 2]] { &self.velocities }
    pub fn accelerations(&self) -> &[[f64; 2]] { &self.accelerations }
    fn refresh_accelerations(&mut self) {
        #[cfg(test)] { self.force_evaluations += 1; }
        self.accelerations.fill([0.0; 2]);
        for i in 0..self.positions.len() {
            for j in i+1..self.positions.len() {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                let r = dx.hypot(dy);
                assert!(r > 0.0 && r.is_finite(), "nonzero finite separation");
                let radial = crate::force(r);
                for (axis, d) in [dx, dy].into_iter().enumerate() {
                    let f = radial * d / r;
                    assert!(f.is_finite(), "finite force");
                    self.accelerations[i][axis] += f;
                    self.accelerations[j][axis] -= f;
                }
            }
        }
    }
    pub fn kinetic_energy(&self) -> f64 {
        self.velocities.iter().map(|v| 0.5*(v[0]*v[0] + v[1]*v[1])).sum()
    }
    pub fn potential_energy(&self) -> f64 {
        let mut u = 0.0;
        for i in 0..self.positions.len() {
            for j in i+1..self.positions.len() {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                u += crate::energy(dx.hypot(dy));
            }
        }
        u
    }
    pub fn total_energy(&self) -> f64 { self.kinetic_energy() + self.potential_energy() }
}
pub trait Integrator { fn step(&self, system: &mut System, dt: f64); }
pub struct Euler;
pub struct VelocityVerlet;
impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        assert!(dt.is_finite() && dt > 0.0, "positive finite dt");
        for i in 0..system.positions.len() {
            for axis in 0..2 {
                system.positions[i][axis] += system.velocities[i][axis] * dt;
                system.velocities[i][axis] += system.accelerations[i][axis] * dt;
            }
        }
        system.refresh_accelerations();
    }
}
impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        assert!(dt.is_finite() && dt > 0.0, "positive finite dt");
        for i in 0..system.positions.len() {
            for axis in 0..2 {
                system.velocities[i][axis] += 0.5*dt*system.accelerations[i][axis];
                system.positions[i][axis] += dt*system.velocities[i][axis];
            }
        }
        system.refresh_accelerations();
        for i in 0..system.positions.len() {
            for axis in 0..2 {
                system.velocities[i][axis] += 0.5*system.accelerations[i][axis]*dt;
            }
        }
    }
}
pub struct Sample {
    pub step: usize, pub time: f64, pub kinetic: f64,
    pub potential: f64, pub total: f64,
}
pub fn simulate<I: Integrator>(integrator: &I, system: &mut System,
                              dt: f64, steps: usize) -> Vec<Sample> {
    assert!(dt.is_finite() && dt > 0.0, "positive finite dt");
    let mut rows = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        if step > 0 { integrator.step(system, dt); }
        let kinetic = system.kinetic_energy();
        let potential = system.potential_energy();
        rows.push(Sample { step, time: step as f64 * dt,
                           kinetic, potential, total: kinetic + potential });
    }
    rows
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
