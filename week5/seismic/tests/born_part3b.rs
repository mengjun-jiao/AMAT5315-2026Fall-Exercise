use std::{fs, path::PathBuf};

use ndarray::{Array2, Array3, Axis};
use ndarray_npy::read_npy;

use seismic::{born, experiment::Experiment, experiment::GridPoint, forward};

fn experiment(
    perturbation: Array2<f64>,
    shots: Vec<GridPoint>,
    receivers: Vec<GridPoint>,
) -> Experiment {
    Experiment {
        nx: 9,
        nz: 9,
        dx: 1.0,
        dt: 0.1,
        steps: 6,
        source_frequency: 0.08,
        source_peak_time: 0.3,
        source_amplitude: 1.0,
        shots,
        receivers,
        sponge_width: 1,
        sponge_strength: 0.2,
        background: Array2::from_shape_fn((9, 9), |(z, x)| 1.2 + 0.02 * z as f64 + 0.01 * x as f64),
        perturbation,
        length_unit_m: 100.0,
        time_unit_s: 0.1,
    }
}

fn base_experiment() -> Experiment {
    let perturbation = Array2::from_shape_fn((9, 9), |(z, x)| {
        if (z + x) % 3 == 0 {
            0.01 + 0.001 * z as f64
        } else {
            0.0
        }
    });
    experiment(
        perturbation,
        vec![GridPoint { x: 4, z: 2 }, GridPoint { x: 2, z: 4 }],
        vec![
            GridPoint { x: 3, z: 2 },
            GridPoint { x: 4, z: 2 },
            GridPoint { x: 5, z: 2 },
        ],
    )
}

fn relative_l2(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
    let numerator = (left - right)
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let denominator = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    numerator / denominator.max(1.0e-30)
}

#[test]
fn born_has_expected_shape_and_zero_tangent_is_zero() {
    let mut experiment = base_experiment();
    experiment.perturbation.fill(0.0);
    let result = born::simulate_born_with_primal(&experiment).unwrap();
    assert_eq!(result.data.shape(), &[2, 6, 3]);
    assert_eq!(result.primal_traces.shape(), &[2, 6, 3]);
    assert!(result.data.iter().all(|value| *value == 0.0));
}

#[test]
fn born_trajectory_matches_centered_finite_difference() {
    let experiment = base_experiment();
    let eps = 1.0e-5;
    let mut plus = experiment.clone();
    plus.background = &plus.background + &(plus.perturbation.mapv(|value| eps * value));
    let mut minus = experiment.clone();
    minus.background = &minus.background - &(minus.perturbation.mapv(|value| eps * value));
    let finite_difference = (&forward::simulate_forward(&plus).unwrap()
        - &forward::simulate_forward(&minus).unwrap())
        / (2.0 * eps);
    let born = born::simulate_born(&experiment).unwrap();
    let relative = relative_l2(&born, &finite_difference);
    assert!(relative < 1.0e-6);
    let result = born::simulate_born_with_primal(&experiment).unwrap();
    let forward = forward::simulate_forward(&experiment).unwrap();
    assert!(relative_l2(&result.primal_traces, &forward) < 1.0e-15);
}

#[test]
fn born_is_linear_in_the_velocity_tangent() {
    let base = base_experiment();
    let m1 = Array2::from_shape_fn((9, 9), |(z, x)| 0.01 + 0.002 * z as f64 - 0.001 * x as f64);
    let m2 = Array2::from_shape_fn((9, 9), |(z, x)| -0.02 + 0.001 * z as f64 + 0.003 * x as f64);
    let mut first = base.clone();
    first.perturbation = m1.clone();
    let mut second = base.clone();
    second.perturbation = m2.clone();
    let mut combined = base;
    combined.perturbation = &m1 * 1.7 + &m2 * -0.4;
    let actual = born::simulate_born(&combined).unwrap();
    let expected =
        born::simulate_born(&first).unwrap() * 1.7 + born::simulate_born(&second).unwrap() * -0.4;
    assert!(relative_l2(&actual, &expected) < 1.0e-13);
}

#[test]
fn born_shots_are_independent_and_receiver_order_is_preserved() {
    let experiment = base_experiment();
    let all = born::simulate_born(&experiment).unwrap();
    for (shot_index, shot) in experiment.shots.iter().copied().enumerate() {
        let mut one_shot = experiment.clone();
        one_shot.shots = vec![shot];
        let individual = born::simulate_born(&one_shot).unwrap();
        assert_eq!(
            individual.index_axis(Axis(0), 0),
            all.index_axis(Axis(0), shot_index)
        );
    }

    let mut reversed = experiment.clone();
    reversed.receivers.reverse();
    let reversed_data = born::simulate_born(&reversed).unwrap();
    for receiver_index in 0..experiment.receivers.len() {
        assert_eq!(
            reversed_data.index_axis(Axis(2), receiver_index),
            all.index_axis(Axis(2), experiment.receivers.len() - 1 - receiver_index)
        );
    }
}

#[test]
fn born_outputs_follow_json_and_npy_contracts() {
    let experiment = base_experiment();
    let result = born::simulate_born_with_primal(&experiment).unwrap();
    let output = PathBuf::from(format!("/tmp/amat5315-born-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&output);
    born::write_born_outputs("inputs/test.json", &experiment, &result.data, &output).unwrap();
    let run: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output.join("run.json")).unwrap()).unwrap();
    let metadata: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output.join("result.json")).unwrap()).unwrap();
    let round_trip: Array3<f64> = read_npy(output.join("born_data.npy")).unwrap();
    assert_eq!(run["experiment_file"], "inputs/test.json");
    assert!(run["experiment"]["background"].is_null());
    assert!(run["experiment"]["perturbation"].is_null());
    assert_eq!(metadata["mode"], "born");
    assert!(metadata.get("statistics").is_none());
    assert_eq!(round_trip, result.data);
    assert!(!output.join("traces.npy").exists());
    let _ = fs::remove_dir_all(output);
}
