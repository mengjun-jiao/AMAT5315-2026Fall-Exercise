use crate::System;
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    #[serde(default)]
    pub ramp_to: Option<f64>,
}
impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
            integrator: "velocity-verlet".into(),
            ramp_to: None,
        }
    }
}
impl RunConfig {
    pub fn validate(&self) -> Result<(), String> {
        if [self.rho, self.temperature, self.dt]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err("rho, temperature and dt must be finite and positive".into());
        }
        if self.sample_every == 0
            || self.steps < self.sample_every
            || !(self.steps as f64 * self.dt).is_finite()
        {
            return Err("steps/sample_every must produce at least one finite-time frame".into());
        }
        if self.integrator != "velocity-verlet" && self.integrator != "euler" {
            return Err("unknown integrator".into());
        }
        if let Some(target) = self.ramp_to {
            if !target.is_finite() || target <= 0.0 {
                return Err("ramp_to must be finite and positive".into());
            }
        }
        let (_, box_size) = lattice(self.n, self.rho)?;
        crate::physics::PhysicalModel::PeriodicShiftedLennardJones { box_size, rc: 2.5 }.validate()
    }
}
pub fn lattice(n: usize, rho: f64) -> Result<(Vec<[f64; 2]>, [f64; 2]), String> {
    let m = n.isqrt();
    if m < 2 || m.checked_mul(m) != Some(n) || m % 2 != 0 {
        return Err("n must be a square with an even number of rows".into());
    }
    if !rho.is_finite() || rho <= 0. {
        return Err("positive finite rho required".into());
    }
    let a = (2. / (3_f64.sqrt() * rho)).sqrt();
    let h = 3_f64.sqrt() * a / 2.;
    let box_size = [m as f64 * a, m as f64 * h];
    crate::physics::PhysicalModel::PeriodicShiftedLennardJones { box_size, rc: 2.5 }.validate()?;
    let mut p = Vec::with_capacity(n);
    for j in 0..m {
        for i in 0..m {
            p.push([(i as f64 + 0.5 * (j % 2) as f64) * a, j as f64 * h]);
        }
    }
    Ok((p, box_size))
}
pub fn initialize(c: &RunConfig) -> Result<System, String> {
    initialize_with_force(c, crate::neighbors::ForceMethod::Naive)
}
pub fn initialize_with_force(
    c: &RunConfig,
    force_method: crate::neighbors::ForceMethod,
) -> Result<System, String> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    c.validate()?;
    let (pos, box_size) = lattice(c.n, c.rho)?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(c.seed);
    let mut vel = vec![[0.; 2]; c.n];
    for v in &mut vel {
        for x in v {
            let z: f64 = StandardNormal.sample(&mut rng);
            *x = z * c.temperature.sqrt();
        }
    }
    for axis in 0..2 {
        let mean = vel.iter().map(|v| v[axis]).sum::<f64>() / c.n as f64;
        for v in &mut vel {
            v[axis] -= mean;
        }
    }
    let mut s = System::with_model_and_force(
        pos,
        vel,
        crate::physics::PhysicalModel::PeriodicShiftedLennardJones { box_size, rc: 2.5 },
        force_method,
    )?;
    s.rescale_temperature(c.temperature)?;
    Ok(s)
}
pub fn evolve<I: crate::Integrator>(
    i: &I,
    s: &mut System,
    c: &RunConfig,
    mut emit: impl FnMut(crate::trajectory::Frame) -> Result<(), String>,
) -> Result<(), String> {
    c.validate()?;
    if s.positions().len() != c.n {
        return Err("system/config n mismatch".into());
    }
    for step in 1..=c.eq_steps {
        i.step(s, c.dt);
        if step % 50 == 0 {
            s.rescale_temperature(c.temperature)?;
        }
    }
    for step in 1..=c.steps {
        i.step(s, c.dt);
        if let Some(end) = c.ramp_to {
            if step % 50 == 0 {
                let target = c.temperature
                    + (end - c.temperature) * step as f64 / c.steps as f64;
                s.rescale_temperature(target)?;
            }
        }
        if step % c.sample_every == 0 {
            let e_pot = s.potential_energy();
            let e_kin = s.kinetic_energy();
            if !e_pot.is_finite() || !e_kin.is_finite() {
                return Err("nonfinite state energy".into());
            }
            emit(crate::trajectory::Frame {
                step,
                t: step as f64 * c.dt,
                pos: s.positions().to_vec(),
                vel: s.velocities().to_vec(),
                e_pot,
                e_kin,
            })?;
        }
    }
    Ok(())
}
