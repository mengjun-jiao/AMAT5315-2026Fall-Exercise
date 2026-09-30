use std::{fs, path::Path};

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3, ArrayView2, Axis};
use ndarray_npy::{read_npy, write_npy};
use serde_json::{json, Value};

use crate::{
    enzyme,
    experiment::{Experiment, GridPoint},
    field::State,
    forward,
    physics::{gaussian_footprint, ricker_pulse, sponge_damping},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShotStatistics {
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjointStatistics {
    pub storage: String,
    pub checkpoints: Option<usize>,
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    pub peak_saved_bytes: usize,
    pub per_shot: Vec<ShotStatistics>,
}

#[derive(Debug)]
pub struct AdjointResult {
    pub image: Array2<f64>,
    pub statistics: AdjointStatistics,
}

fn flat(field: &Array2<f64>) -> Result<&[f64]> {
    field
        .as_slice()
        .context("adjoint timestep requires contiguous row-major fields")
}

fn source_for_step(experiment: &Experiment, footprint: &Array2<f64>, step: usize) -> Array2<f64> {
    let scale = experiment.source_amplitude
        * ricker_pulse(
            experiment.source_frequency,
            experiment.source_peak_time,
            step as f64 * experiment.dt,
        );
    footprint * scale
}

fn forward_history(
    experiment: &Experiment,
    sponge: &Array2<f64>,
    shot: GridPoint,
) -> Result<Vec<State>> {
    let footprint = gaussian_footprint(experiment.nz, experiment.nx, shot)?;
    let mut state = State::zeros(experiment.nz, experiment.nx);
    let mut history = Vec::with_capacity(experiment.steps + 1);
    history.push(state.clone());
    for step in 0..experiment.steps {
        let source = source_for_step(experiment, &footprint, step);
        let next_flat = enzyme::timestep_primal(
            flat(&state.previous)?,
            flat(&state.current)?,
            flat(&experiment.background)?,
            flat(sponge)?,
            flat(&source)?,
            experiment.nx,
            experiment.nz,
            experiment.dx,
            experiment.dt,
        )?;
        let next = Array2::from_shape_vec((experiment.nz, experiment.nx), next_flat)
            .context("Enzyme primal output had an invalid history state shape")?;
        state.advance(next)?;
        history.push(state.clone());
    }
    Ok(history)
}

fn inject_receivers(
    adjoint: &mut [f64],
    receivers: &[GridPoint],
    weights: &ArrayView2<'_, f64>,
    step: usize,
    nx: usize,
) {
    for (receiver_index, receiver) in receivers.iter().enumerate() {
        let index = receiver.z * nx + receiver.x;
        adjoint[index] += weights[(step, receiver_index)];
    }
}

fn simulate_adjoint_shot(
    experiment: &Experiment,
    sponge: &Array2<f64>,
    shot: GridPoint,
    weights: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, ShotStatistics)> {
    let history = forward_history(experiment, sponge, shot)?;
    let length = experiment.nx * experiment.nz;
    let footprint = gaussian_footprint(experiment.nz, experiment.nx, shot)?;
    let mut bar_out_previous = vec![0.0; length];
    let mut bar_out_current = vec![0.0; length];
    let mut image = vec![0.0; length];
    let mut reverse_calls = 0;

    for step in (0..experiment.steps).rev() {
        inject_receivers(
            &mut bar_out_current,
            &experiment.receivers,
            &weights,
            step,
            experiment.nx,
        );
        let source = source_for_step(experiment, &footprint, step);
        let (bar_prev, bar_curr, bar_speed) = enzyme::timestep_vjp(
            flat(&history[step].previous)?,
            flat(&history[step].current)?,
            flat(&experiment.background)?,
            flat(sponge)?,
            flat(&source)?,
            &bar_out_current,
            experiment.nx,
            experiment.nz,
            experiment.dx,
            experiment.dt,
        )?;
        for (image_value, speed_value) in image.iter_mut().zip(bar_speed) {
            *image_value += speed_value;
        }
        let next_bar_current = bar_curr
            .into_iter()
            .zip(bar_out_previous)
            .map(|(from_timestep, from_state_shift)| from_timestep + from_state_shift)
            .collect();
        bar_out_previous = bar_prev;
        bar_out_current = next_bar_current;
        reverse_calls += 1;
    }

    let image = Array2::from_shape_vec((experiment.nz, experiment.nx), image)
        .context("adjoint image had an invalid grid shape")?;
    Ok((
        image,
        ShotStatistics {
            reverse_calls,
            scheduler_forward_calls: experiment.steps,
            peak_saved_states: history.len(),
        },
    ))
}

fn validate_weights(experiment: &Experiment, weights: &Array3<f64>) -> Result<()> {
    let expected = (
        experiment.shots.len(),
        experiment.steps,
        experiment.receivers.len(),
    );
    if weights.dim() != expected {
        bail!(
            "adjoint data shape {:?} does not match required shape {:?}",
            weights.dim(),
            expected
        );
    }
    if !weights.iter().all(|value| value.is_finite()) {
        bail!("adjoint data contains non-finite values");
    }
    Ok(())
}

pub fn read_adjoint_data(path: &Path, experiment: &Experiment) -> Result<Array3<f64>> {
    let weights: Array3<f64> = read_npy(path)
        .with_context(|| format!("failed to read float64 adjoint data {}", path.display()))?;
    validate_weights(experiment, &weights)?;
    Ok(weights)
}

pub fn simulate_adjoint(experiment: &Experiment, weights: &Array3<f64>) -> Result<AdjointResult> {
    validate_weights(experiment, weights)?;
    let sponge = sponge_damping(
        experiment.nz,
        experiment.nx,
        experiment.sponge_width,
        experiment.sponge_strength,
    )?;
    let mut image = Array2::zeros((experiment.nz, experiment.nx));
    let mut per_shot = Vec::with_capacity(experiment.shots.len());
    for (shot_index, shot) in experiment.shots.iter().copied().enumerate() {
        let (shot_image, statistics) = simulate_adjoint_shot(
            experiment,
            &sponge,
            shot,
            weights.index_axis(Axis(0), shot_index),
        )?;
        image += &shot_image;
        per_shot.push(statistics);
    }
    let peak_saved_states = experiment.steps + 1;
    let peak_saved_bytes = peak_saved_states * 2 * experiment.nx * experiment.nz * 8;
    Ok(AdjointResult {
        image,
        statistics: AdjointStatistics {
            storage: "full".to_owned(),
            checkpoints: None,
            reverse_calls: experiment.shots.len() * experiment.steps,
            scheduler_forward_calls: experiment.shots.len() * experiment.steps,
            peak_saved_states,
            peak_saved_bytes,
            per_shot,
        },
    })
}

fn coordinates(points: &[GridPoint]) -> Vec<[usize; 2]> {
    points.iter().map(|point| [point.x, point.z]).collect()
}

fn statistics_value(statistics: &AdjointStatistics) -> Value {
    json!({
        "storage": statistics.storage,
        "checkpoints": statistics.checkpoints,
        "reverse_calls": statistics.reverse_calls,
        "scheduler_forward_calls": statistics.scheduler_forward_calls,
        "peak_saved_states": statistics.peak_saved_states,
        "peak_saved_bytes": statistics.peak_saved_bytes,
        "per_shot": statistics.per_shot.iter().map(|shot| json!({
            "reverse_calls": shot.reverse_calls,
            "scheduler_forward_calls": shot.scheduler_forward_calls,
            "peak_saved_states": shot.peak_saved_states
        })).collect::<Vec<_>>()
    })
}

pub fn result_metadata(experiment: &Experiment, statistics: &AdjointStatistics) -> Value {
    json!({
        "mode": "adjoint",
        "nx": experiment.nx,
        "nz": experiment.nz,
        "dx": experiment.dx,
        "dt": experiment.dt,
        "steps": experiment.steps,
        "shots": coordinates(&experiment.shots),
        "receivers": coordinates(&experiment.receivers),
        "statistics": statistics_value(statistics)
    })
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let contents = serde_json::to_string_pretty(value).context("failed to serialize JSON")?;
    fs::write(path, format!("{contents}\n"))
        .with_context(|| format!("failed to write {}", path.display()))
}

pub fn write_adjoint_outputs(
    experiment_file: &str,
    experiment: &Experiment,
    image: &Array2<f64>,
    statistics: &AdjointStatistics,
    out: &Path,
) -> Result<()> {
    fs::create_dir_all(out)
        .with_context(|| format!("failed to create output directory {}", out.display()))?;
    write_json(
        &out.join("run.json"),
        &forward::run_metadata(experiment_file, experiment),
    )?;
    write_json(
        &out.join("result.json"),
        &result_metadata(experiment, statistics),
    )?;
    write_npy(out.join("image.npy"), image)
        .with_context(|| format!("failed to write {}/image.npy", out.display()))?;
    Ok(())
}

pub fn run_adjoint(
    experiment_file: &str,
    experiment: &Experiment,
    weights: &Array3<f64>,
    out: &Path,
) -> Result<AdjointResult> {
    let result = simulate_adjoint(experiment, weights)?;
    write_adjoint_outputs(
        experiment_file,
        experiment,
        &result.image,
        &result.statistics,
        out,
    )?;
    Ok(result)
}
