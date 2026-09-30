use std::{fs, path::Path};

use anyhow::{Context, Result};
use ndarray::{Array2, Array3, Axis};
use ndarray_npy::write_npy;
use serde_json::{json, Value};

use crate::{
    experiment::{Experiment, GridPoint},
    field::State,
    physics::{gaussian_footprint, ricker_pulse, sponge_damping},
    timestep::{compute_next, sample_receivers},
};

/// Simulate one fresh shot and return traces with shape [step, receiver].
pub fn simulate_shot(
    experiment: &Experiment,
    sponge: &Array2<f64>,
    shot: GridPoint,
) -> Result<Array2<f64>> {
    let footprint = gaussian_footprint(experiment.nz, experiment.nx, shot)?;
    let mut state = State::zeros(experiment.nz, experiment.nx);
    let mut traces = Array2::zeros((experiment.steps, experiment.receivers.len()));

    for n in 0..experiment.steps {
        let source_scale = experiment.source_amplitude
            * ricker_pulse(
                experiment.source_frequency,
                experiment.source_peak_time,
                n as f64 * experiment.dt,
            );
        let source = &footprint * source_scale;
        let next = compute_next(
            &state.previous,
            &state.current,
            &experiment.background,
            sponge,
            &source,
            experiment.dx,
            experiment.dt,
        )?;
        let samples = sample_receivers(&next, &experiment.receivers)?;
        for (receiver_index, value) in samples.into_iter().enumerate() {
            traces[(n, receiver_index)] = value;
        }
        state.advance(next)?;
    }

    Ok(traces)
}

/// Simulate every shot in experiment order and return [shot, step, receiver].
pub fn simulate_forward(experiment: &Experiment) -> Result<Array3<f64>> {
    let sponge = sponge_damping(
        experiment.nz,
        experiment.nx,
        experiment.sponge_width,
        experiment.sponge_strength,
    )?;
    let mut traces = Array3::zeros((
        experiment.shots.len(),
        experiment.steps,
        experiment.receivers.len(),
    ));
    for (shot_index, shot) in experiment.shots.iter().copied().enumerate() {
        let shot_traces = simulate_shot(experiment, &sponge, shot)?;
        traces
            .index_axis_mut(Axis(0), shot_index)
            .assign(&shot_traces);
    }
    Ok(traces)
}

fn coordinates(points: &[GridPoint]) -> Vec<[usize; 2]> {
    points.iter().map(|point| [point.x, point.z]).collect()
}

/// Build the non-recording run metadata required by the interface contract.
pub fn run_metadata(experiment_file: &str, experiment: &Experiment) -> Value {
    json!({
        "experiment_file": experiment_file,
        "experiment": {
            "nx": experiment.nx,
            "nz": experiment.nz,
            "dx": experiment.dx,
            "dt": experiment.dt,
            "steps": experiment.steps,
            "source_frequency": experiment.source_frequency,
            "source_peak_time": experiment.source_peak_time,
            "source_amplitude": experiment.source_amplitude,
            "shots": coordinates(&experiment.shots),
            "receivers": coordinates(&experiment.receivers),
            "sponge_width": experiment.sponge_width,
            "sponge_strength": experiment.sponge_strength,
            "length_unit_m": experiment.length_unit_m,
            "time_unit_s": experiment.time_unit_s
        }
    })
}

/// Build the result metadata required by the interface contract.
pub fn result_metadata(experiment: &Experiment) -> Value {
    json!({
        "mode": "forward",
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

/// Write the non-recording forward outputs into the requested directory.
pub fn write_forward_outputs(
    experiment_file: &str,
    experiment: &Experiment,
    traces: &Array3<f64>,
    out: &Path,
) -> Result<()> {
    fs::create_dir_all(out)
        .with_context(|| format!("failed to create output directory {}", out.display()))?;
    write_json(
        &out.join("run.json"),
        &run_metadata(experiment_file, experiment),
    )?;
    write_json(&out.join("result.json"), &result_metadata(experiment))?;
    write_npy(out.join("traces.npy"), traces)
        .with_context(|| format!("failed to write {}/traces.npy", out.display()))?;
    Ok(())
}

/// Run the complete forward calculation and write its non-recording outputs.
pub fn run_forward(
    experiment_file: &str,
    experiment: &Experiment,
    out: &Path,
) -> Result<Array3<f64>> {
    let traces = simulate_forward(experiment)?;
    write_forward_outputs(experiment_file, experiment, &traces, out)?;
    Ok(traces)
}

pub fn shot_l2_norms(traces: &Array3<f64>) -> Vec<f64> {
    traces
        .axis_iter(Axis(0))
        .map(|shot| shot.iter().map(|value| value * value).sum::<f64>().sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;
    use ndarray::{Array2, Array3};
    use ndarray_npy::read_npy;

    fn experiment(
        shots: Vec<GridPoint>,
        receivers: Vec<GridPoint>,
        steps: usize,
        background_value: f64,
        perturbation_value: f64,
    ) -> Experiment {
        Experiment {
            nx: 7,
            nz: 7,
            dx: 1.0,
            dt: 0.2,
            steps,
            source_frequency: 0.08,
            source_peak_time: 1.0,
            source_amplitude: 1.0,
            shots,
            receivers,
            sponge_width: 1,
            sponge_strength: 0.0,
            background: Array2::from_elem((7, 7), background_value),
            perturbation: Array2::from_elem((7, 7), perturbation_value),
            length_unit_m: 100.0,
            time_unit_s: 0.1,
        }
    }

    #[test]
    fn single_shot_trace_shape_is_steps_by_receivers() {
        let experiment = experiment(
            vec![GridPoint { x: 3, z: 3 }],
            vec![GridPoint { x: 2, z: 3 }, GridPoint { x: 4, z: 3 }],
            5,
            1.0,
            8.0,
        );
        let sponge = sponge_damping(7, 7, 1, 0.0).unwrap();
        let traces = simulate_shot(&experiment, &sponge, experiment.shots[0]).unwrap();
        assert_eq!(traces.shape(), &[5, 2]);
    }

    #[test]
    fn multiple_shots_preserve_shape_and_order() {
        let experiment = experiment(
            vec![GridPoint { x: 2, z: 3 }, GridPoint { x: 4, z: 3 }],
            vec![GridPoint { x: 2, z: 3 }],
            4,
            1.0,
            0.0,
        );
        let traces = simulate_forward(&experiment).unwrap();
        assert_eq!(traces.shape(), &[2, 4, 1]);
        assert_ne!(traces[[0, 0, 0]], traces[[1, 0, 0]]);
    }

    #[test]
    fn each_shot_starts_from_a_fresh_zero_state() {
        let experiment = experiment(
            vec![GridPoint { x: 3, z: 3 }, GridPoint { x: 3, z: 3 }],
            vec![GridPoint { x: 2, z: 3 }],
            4,
            1.0,
            99.0,
        );
        let traces = simulate_forward(&experiment).unwrap();
        assert_eq!(traces.index_axis(Axis(0), 0), traces.index_axis(Axis(0), 1));
    }

    #[test]
    fn first_trace_sample_comes_from_u_one_not_u_zero() {
        let experiment = experiment(
            vec![GridPoint { x: 3, z: 3 }],
            vec![GridPoint { x: 3, z: 3 }],
            1,
            1.0,
            0.0,
        );
        let traces = simulate_forward(&experiment).unwrap();
        assert_ne!(traces[[0, 0, 0]], 0.0);
    }

    #[test]
    fn normal_forward_uses_background_not_perturbation() {
        let first = experiment(
            vec![GridPoint { x: 3, z: 3 }],
            vec![GridPoint { x: 2, z: 3 }],
            5,
            1.0,
            0.0,
        );
        let mut second = first.clone();
        second.perturbation.fill(1000.0);
        assert_eq!(
            simulate_forward(&first).unwrap(),
            simulate_forward(&second).unwrap()
        );
    }

    #[test]
    fn metadata_has_contract_fields_but_no_model_arrays() {
        let experiment = experiment(
            vec![GridPoint { x: 3, z: 3 }],
            vec![GridPoint { x: 2, z: 3 }],
            4,
            1.0,
            2.0,
        );
        let run = run_metadata("inputs/test.json", &experiment);
        let run_experiment = &run["experiment"];
        assert_eq!(run["experiment_file"], "inputs/test.json");
        assert!(run_experiment.get("background").is_none());
        assert!(run_experiment.get("perturbation").is_none());
        assert_eq!(run_experiment["shots"], serde_json::json!([[3, 3]]));
        assert_eq!(run_experiment["receivers"], serde_json::json!([[2, 3]]));

        let result = result_metadata(&experiment);
        assert_eq!(result["mode"], "forward");
        for key in ["nx", "nz", "dx", "dt", "steps", "shots", "receivers"] {
            assert!(result.get(key).is_some(), "missing result field {key}");
        }
    }

    #[test]
    fn npy_round_trip_preserves_shape_and_values() {
        let path: PathBuf =
            std::env::temp_dir().join(format!("seismic-part2d-traces-{}.npy", std::process::id()));
        let traces = Array3::from_shape_fn((2, 3, 1), |(shot, step, _)| (shot * 10 + step) as f64);
        write_npy(&path, &traces).unwrap();
        let loaded: Array3<f64> = read_npy(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(loaded.shape(), &[2, 3, 1]);
        assert_eq!(loaded, traces);
    }

    #[test]
    fn shot_norms_are_computed_independently() {
        let traces = Array3::from_shape_vec((2, 1, 2), vec![3.0, 4.0, 5.0, 12.0]).unwrap();
        assert_eq!(shot_l2_norms(&traces), vec![5.0, 13.0]);
    }
}
