use crate::trajectory::{Frame, RunMetadata};
use std::path::Path;
#[derive(Debug)]
pub struct CheckReport {
    pub drift: f64,
    pub t_speed: f64,
    pub mb_score: f64,
    pub passed: bool,
}
pub fn evaluate(meta: &RunMetadata, frames: &[Frame]) -> Result<CheckReport, String> {
    let model = crate::physics::PhysicalModel::PeriodicShiftedLennardJones {
        box_size: meta.box_size,
        rc: 2.5,
    };
    let mut totals = Vec::with_capacity(frames.len());
    let mut speeds = Vec::new();
    for (index, f) in frames.iter().enumerate() {
        let (u, k) = crate::physics::energies(&f.pos, &f.vel, &model)?;
        for (name, saved, recomputed) in [("E_pot", f.e_pot, u), ("E_kin", f.e_kin, k)] {
            if !crate::trajectory::close(saved, recomputed) {
                return Err(format!(
                    "traj.jsonl line {}: {name} mismatch: saved={saved:.15e}, recomputed={recomputed:.15e}",
                    index + 1
                ));
            }
        }
        totals.push(u + k);
        speeds.extend(f.vel.iter().map(|v| v[0].hypot(v[1])));
    }
    metrics(&totals, &speeds)
}
pub fn check_directory(dir: &Path) -> Result<CheckReport, String> {
    let (meta, frames) = crate::trajectory::read_trajectory(dir)?;
    evaluate(&meta, &frames)
}
fn metrics(energies: &[f64], speeds: &[f64]) -> Result<CheckReport, String> {
    if energies.is_empty()
        || speeds.is_empty()
        || energies[0] == 0.
        || !energies.iter().all(|e| e.is_finite())
        || !speeds.iter().all(|v| v.is_finite() && *v >= 0.)
    {
        return Err("nonempty finite energy/speed samples and nonzero E0 required".into());
    }
    let k = (energies.len() / 10).max(1);
    let first = energies[..k].iter().sum::<f64>() / k as f64;
    let last = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    let drift = (last - first).abs() / energies[0].abs();
    let t_speed = speeds.iter().map(|v| v * v).sum::<f64>() / (2. * speeds.len() as f64);
    if !t_speed.is_finite() || t_speed <= 0. || !drift.is_finite() {
        return Err("nonpositive/nonfinite T_speed or nonfinite drift".into());
    }
    let edges: Vec<f64> = (1..24)
        .map(|k| (-2. * t_speed * (1. - k as f64 / 24.).ln()).sqrt())
        .collect();
    let mut counts = [0usize; 24];
    for &v in speeds {
        counts[edges.partition_point(|&b| v >= b)] += 1;
    }
    let expected = speeds.len() as f64 / 24.;
    let mb_score = counts
        .iter()
        .map(|&o| (o as f64 - expected).powi(2) / expected)
        .sum::<f64>()
        / 22.;
    let passed = drift < 2e-3 && (t_speed - 0.5).abs() < 0.05 && mb_score < 2.;
    Ok(CheckReport {
        drift,
        t_speed,
        mb_score,
        passed,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_probability_fixture_and_first_saved_energy() {
        let speeds: Vec<f64> = (0..24)
            .flat_map(|j| {
                let v = (-(1.0 - (j as f64 + 0.5) / 24.0).ln()).sqrt();
                std::iter::repeat_n(v, 100)
            })
            .collect();
        let mut e = vec![-100.; 200];
        let report = metrics(&e, &speeds).unwrap();
        assert!(report.passed);
        assert_eq!(report.drift, 0.);
        e[180..].fill(-99.);
        assert!((metrics(&e, &speeds).unwrap().drift - 0.01).abs() < 1e-12);
        assert!(metrics(&[0.], &speeds).is_err());
        assert!(metrics(&[-1.], &[0.]).is_err());
    }
}
