use crate::fluid::RunConfig;
use serde::{Deserialize, Serialize};
use std::path::Path;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    #[serde(flatten)]
    pub config: RunConfig,
    #[serde(rename = "box")]
    pub box_size: [f64; 2],
    #[serde(default)]
    pub force_method: crate::neighbors::ForceMethod,
}
impl std::ops::Deref for RunMetadata {
    type Target = RunConfig;
    fn deref(&self) -> &RunConfig {
        &self.config
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}
pub fn read_trajectory(dir: &Path) -> Result<(RunMetadata, Vec<Frame>), String> {
    use std::io::{BufRead, BufReader};
    let path = dir.join("run.json");
    let meta: RunMetadata = serde_json::from_reader(
        std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?,
    )
    .map_err(|e| format!("run.json: {e}"))?;
    meta.config
        .validate()
        .map_err(|e| format!("run.json: {e}"))?;
    let (_, expected_box) = crate::fluid::lattice(meta.n, meta.rho)?;
    for axis in 0..2 {
        if !close(meta.box_size[axis], expected_box[axis]) {
            return Err("run.json: box/rho/n mismatch".into());
        }
    }
    let model = crate::physics::PhysicalModel::PeriodicShiftedLennardJones {
        box_size: meta.box_size,
        rc: 2.5,
    };
    model.validate()?;
    let path = dir.join("traj.jsonl");
    let input =
        BufReader::new(std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?);
    let mut frames = Vec::new();
    let expected_count = meta.steps / meta.sample_every;
    for (index, line) in input.lines().enumerate() {
        let context = format!("traj.jsonl line {}", index + 1);
        let line = line.map_err(|e| format!("{context}: {e}"))?;
        let f: Frame = serde_json::from_str(&line).map_err(|e| format!("{context}: {e}"))?;
        if index >= expected_count {
            return Err(format!("{context}: excess frame"));
        }
        let expected_step = (index + 1)
            .checked_mul(meta.sample_every)
            .ok_or("step overflow")?;
        if f.step != expected_step || !close(f.t, expected_step as f64 * meta.dt) {
            return Err(format!("{context}: step/time mismatch"));
        }
        if f.pos.len() != meta.n || f.vel.len() != meta.n {
            return Err(format!("{context}: pos/vel length mismatch"));
        }
        if !f.pos.iter().chain(&f.vel).flatten().all(|x| x.is_finite())
            || !f.e_pot.is_finite()
            || !f.e_kin.is_finite()
        {
            return Err(format!("{context}: nonfinite state or energy"));
        }
        for p in &f.pos {
            for axis in 0..2 {
                if p[axis] < 0. || p[axis] >= meta.box_size[axis] {
                    return Err(format!("{context}: pos outside box"));
                }
            }
        }
        crate::physics::energies(&f.pos, &f.vel, &model).map_err(|e| format!("{context}: {e}"))?;
        frames.push(f);
    }
    if frames.len() != expected_count {
        return Err(format!(
            "traj.jsonl: frame count {}, expected {expected_count}",
            frames.len()
        ));
    }
    Ok((meta, frames))
}
pub(crate) fn close(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 1e-10 * expected.abs().max(1.)
}
pub fn run_to_directory(c: &RunConfig, dir: &Path) -> Result<(), String> {
    use std::io::{BufWriter, Write};
    c.validate()?;
    let mut s = crate::fluid::initialize(c)?;
    let (_, box_size) = crate::fluid::lattice(c.n, c.rho)?;
    let meta = RunMetadata {
        config: c.clone(),
        box_size,
        force_method: crate::neighbors::ForceMethod::Naive,
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut meta_file = tempfile::NamedTempFile::new_in(dir).map_err(|e| e.to_string())?;
    let mut traj_file = tempfile::NamedTempFile::new_in(dir).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut meta_file, &meta).map_err(|e| e.to_string())?;
    meta_file.flush().map_err(|e| e.to_string())?;
    {
        let mut writer = BufWriter::new(&mut traj_file);
        let emit = |f: Frame| -> Result<(), String> {
            serde_json::to_writer(&mut writer, &f).map_err(|e| e.to_string())?;
            writeln!(writer).map_err(|e| e.to_string())
        };
        match c.integrator.as_str() {
            "velocity-verlet" => crate::fluid::evolve(&crate::VelocityVerlet, &mut s, c, emit)?,
            "euler" => crate::fluid::evolve(&crate::Euler, &mut s, c, emit)?,
            _ => return Err("unknown integrator".into()),
        }
        writer.flush().map_err(|e| e.to_string())?;
    }
    traj_file
        .persist(dir.join("traj.jsonl"))
        .map_err(|e| e.to_string())?;
    meta_file
        .persist(dir.join("run.json"))
        .map_err(|e| e.to_string())?;
    Ok(())
}
