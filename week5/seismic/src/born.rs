use std::{fs, path::Path};

use anyhow::{Context, Result};
use ndarray::{Array2, Array3, Axis};
use ndarray_npy::write_npy;
use serde_json::{json, Value};

use crate::{
    enzyme,
    experiment::{Experiment, GridPoint},
    field::State,
    forward,
    physics::{gaussian_footprint, ricker_pulse, sponge_damping},
    timestep::sample_receivers,
};

#[derive(Debug)]
pub struct BornResult {
    pub data: Array3<f64>,
    pub primal_traces: Array3<f64>,
}

#[derive(Debug)]
struct BornShot {
    data: Array2<f64>,
    primal_traces: Array2<f64>,
}

fn flat(field: &Array2<f64>) -> Result<&[f64]> {
    field
        .as_slice()
        .context("Born timestep requires contiguous row-major fields")
}

fn simulate_born_shot(
    experiment: &Experiment,
    sponge: &Array2<f64>,
    shot: GridPoint,
) -> Result<BornShot> {
    let footprint = gaussian_footprint(experiment.nz, experiment.nx, shot)?;
    let mut primal = State::zeros(experiment.nz, experiment.nx);
    let mut tangent = State::zeros(experiment.nz, experiment.nx);
    let mut data = Array2::zeros((experiment.steps, experiment.receivers.len()));
    let mut primal_traces = Array2::zeros((experiment.steps, experiment.receivers.len()));

    for n in 0..experiment.steps {
        let source_scale = experiment.source_amplitude
            * ricker_pulse(
                experiment.source_frequency,
                experiment.source_peak_time,
                n as f64 * experiment.dt,
            );
        let source = &footprint * source_scale;
        let (next_flat, dnext_flat) = enzyme::timestep_jvp(
            flat(&primal.previous)?,
            flat(&tangent.previous)?,
            flat(&primal.current)?,
            flat(&tangent.current)?,
            flat(&experiment.background)?,
            flat(&experiment.perturbation)?,
            flat(sponge)?,
            flat(&source)?,
            experiment.nx,
            experiment.nz,
            experiment.dx,
            experiment.dt,
        )?;
        let next = Array2::from_shape_vec((experiment.nz, experiment.nx), next_flat)
            .context("Enzyme primal output had an invalid grid shape")?;
        let dnext = Array2::from_shape_vec((experiment.nz, experiment.nx), dnext_flat)
            .context("Enzyme tangent output had an invalid grid shape")?;
        let primal_samples = sample_receivers(&next, &experiment.receivers)?;
        let born_samples = sample_receivers(&dnext, &experiment.receivers)?;
        for (receiver_index, value) in primal_samples.into_iter().enumerate() {
            primal_traces[(n, receiver_index)] = value;
        }
        for (receiver_index, value) in born_samples.into_iter().enumerate() {
            data[(n, receiver_index)] = value;
        }
        primal.advance(next)?;
        tangent.advance(dnext)?;
    }

    Ok(BornShot {
        data,
        primal_traces,
    })
}

pub fn simulate_born_with_primal(experiment: &Experiment) -> Result<BornResult> {
    let sponge = sponge_damping(
        experiment.nz,
        experiment.nx,
        experiment.sponge_width,
        experiment.sponge_strength,
    )?;
    let mut data = Array3::zeros((
        experiment.shots.len(),
        experiment.steps,
        experiment.receivers.len(),
    ));
    let mut primal_traces = Array3::zeros(data.raw_dim());
    for (shot_index, shot) in experiment.shots.iter().copied().enumerate() {
        let result = simulate_born_shot(experiment, &sponge, shot)?;
        data.index_axis_mut(Axis(0), shot_index)
            .assign(&result.data);
        primal_traces
            .index_axis_mut(Axis(0), shot_index)
            .assign(&result.primal_traces);
    }
    Ok(BornResult {
        data,
        primal_traces,
    })
}

pub fn simulate_born(experiment: &Experiment) -> Result<Array3<f64>> {
    Ok(simulate_born_with_primal(experiment)?.data)
}

fn coordinates(points: &[GridPoint]) -> Vec<[usize; 2]> {
    points.iter().map(|point| [point.x, point.z]).collect()
}

pub fn result_metadata(experiment: &Experiment) -> Value {
    json!({
        "mode": "born",
        "nx": experiment.nx,
        "nz": experiment.nz,
        "dx": experiment.dx,
        "dt": experiment.dt,
        "steps": experiment.steps,
        "shots": coordinates(&experiment.shots),
        "receivers": coordinates(&experiment.receivers)
    })
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let contents = serde_json::to_string_pretty(value).context("failed to serialize JSON")?;
    fs::write(path, format!("{contents}\n"))
        .with_context(|| format!("failed to write {}", path.display()))
}

pub fn write_born_outputs(
    experiment_file: &str,
    experiment: &Experiment,
    data: &Array3<f64>,
    out: &Path,
) -> Result<()> {
    fs::create_dir_all(out)
        .with_context(|| format!("failed to create output directory {}", out.display()))?;
    write_json(
        &out.join("run.json"),
        &forward::run_metadata(experiment_file, experiment),
    )?;
    write_json(&out.join("result.json"), &result_metadata(experiment))?;
    write_npy(out.join("born_data.npy"), data)
        .with_context(|| format!("failed to write {}/born_data.npy", out.display()))?;
    Ok(())
}

pub fn run_born(experiment_file: &str, experiment: &Experiment, out: &Path) -> Result<Array3<f64>> {
    let data = simulate_born(experiment)?;
    write_born_outputs(experiment_file, experiment, &data, out)?;
    Ok(data)
}
