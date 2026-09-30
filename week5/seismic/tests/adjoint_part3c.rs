use std::{fs, path::PathBuf};

use ndarray::{Array2, Array3, Axis};
use ndarray_npy::write_npy;

use seismic::{adjoint, born, experiment::Experiment, experiment::GridPoint};

fn experiment(shots: Vec<GridPoint>, receivers: Vec<GridPoint>, steps: usize) -> Experiment {
    Experiment {
        nx: 7,
        nz: 7,
        dx: 1.0,
        dt: 0.1,
        steps,
        source_frequency: 0.08,
        source_peak_time: 0.2,
        source_amplitude: 1.0,
        shots,
        receivers,
        sponge_width: 1,
        sponge_strength: 0.15,
        background: Array2::from_shape_fn((7, 7), |(z, x)| 1.1 + 0.03 * z as f64 + 0.02 * x as f64),
        perturbation: Array2::from_shape_fn((7, 7), |(z, x)| {
            0.01 + 0.001 * z as f64 - 0.0005 * x as f64
        }),
        length_unit_m: 100.0,
        time_unit_s: 0.1,
    }
}

fn base_experiment() -> Experiment {
    experiment(
        vec![GridPoint { x: 3, z: 2 }, GridPoint { x: 2, z: 3 }],
        vec![GridPoint { x: 2, z: 2 }, GridPoint { x: 3, z: 2 }],
        5,
    )
}

fn relative(left: f64, right: f64) -> f64 {
    (left - right).abs() / left.abs().max(right.abs()).max(1.0e-30)
}

fn dot(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[test]
fn full_history_adjoint_transpose_identity_with_arbitrary_weights() {
    let experiment = base_experiment();
    let born_data = born::simulate_born(&experiment).unwrap();
    let weights = Array3::from_shape_fn(born_data.raw_dim(), |(shot, step, receiver)| {
        0.02 + 0.01 * shot as f64 - 0.03 * step as f64 + 0.04 * receiver as f64
    });
    let result = adjoint::simulate_adjoint(&experiment, &weights).unwrap();
    let left = dot(&born_data, &weights);
    let right: f64 = experiment
        .perturbation
        .iter()
        .zip(result.image.iter())
        .map(|(m, image)| m * image)
        .sum();
    assert!(relative(left, right) < 1.0e-10, "left={left} right={right}");
}

#[test]
fn state_shift_identity_is_used_when_future_state_is_adjointed() {
    // A weight at a later receiver sample must flow through u_curr -> output.previous.
    let experiment = experiment(
        vec![GridPoint { x: 3, z: 2 }],
        vec![GridPoint { x: 3, z: 2 }],
        4,
    );
    let mut weights = Array3::zeros((1, 4, 1));
    weights[(0, 3, 0)] = 0.7;
    let result = adjoint::simulate_adjoint(&experiment, &weights).unwrap();
    let eps = 1.0e-5;
    let mut plus = experiment.clone();
    plus.background = &plus.background + &(plus.perturbation.mapv(|value| eps * value));
    let mut minus = experiment.clone();
    minus.background = &minus.background - &(minus.perturbation.mapv(|value| eps * value));
    let plus_data = born::simulate_born_with_primal(&plus)
        .unwrap()
        .primal_traces;
    let minus_data = born::simulate_born_with_primal(&minus)
        .unwrap()
        .primal_traces;
    let finite_difference = (plus_data[(0, 3, 0)] - minus_data[(0, 3, 0)]) / (2.0 * eps);
    let directional: f64 = experiment
        .perturbation
        .iter()
        .zip(result.image.iter())
        .map(|(m, image)| m * image)
        .sum();
    assert!(relative(0.7 * finite_difference, directional) < 1.0e-6);
}

#[test]
fn zero_receiver_weights_produce_zero_image() {
    let experiment = base_experiment();
    let weights = Array3::zeros((2, 5, 2));
    let result = adjoint::simulate_adjoint(&experiment, &weights).unwrap();
    assert!(result.image.iter().all(|value| *value == 0.0));
}

#[test]
fn adjoint_shots_sum_independently() {
    let experiment = base_experiment();
    let weights = Array3::from_shape_fn((2, 5, 2), |(shot, step, receiver)| {
        0.01 + 0.02 * shot as f64 + 0.03 * step as f64 - 0.01 * receiver as f64
    });
    let all = adjoint::simulate_adjoint(&experiment, &weights).unwrap();
    let mut summed = Array2::zeros((7, 7));
    for shot_index in 0..2 {
        let mut one_shot = experiment.clone();
        one_shot.shots = vec![experiment.shots[shot_index]];
        let one_weights = weights
            .index_axis(Axis(0), shot_index)
            .to_owned()
            .insert_axis(Axis(0));
        summed += &adjoint::simulate_adjoint(&one_shot, &one_weights)
            .unwrap()
            .image;
    }
    assert_eq!(all.image, summed);
}

#[test]
fn repeated_receiver_weights_are_added_at_one_grid_cell() {
    let repeated = experiment(
        vec![GridPoint { x: 3, z: 2 }],
        vec![GridPoint { x: 2, z: 2 }, GridPoint { x: 2, z: 2 }],
        5,
    );
    let unique = experiment(
        vec![GridPoint { x: 3, z: 2 }],
        vec![GridPoint { x: 2, z: 2 }],
        5,
    );
    let repeated_weights = Array3::from_shape_fn((1, 5, 2), |(_, step, receiver)| {
        (step as f64 + 1.0) * (receiver as f64 + 1.0)
    });
    let unique_weights = repeated_weights.index_axis(Axis(2), 0).to_owned()
        + &repeated_weights.index_axis(Axis(2), 1).to_owned();
    let unique_weights = unique_weights.insert_axis(Axis(2));
    let repeated_image = adjoint::simulate_adjoint(&repeated, &repeated_weights)
        .unwrap()
        .image;
    let unique_image = adjoint::simulate_adjoint(&unique, &unique_weights)
        .unwrap()
        .image;
    assert_eq!(repeated_image, unique_image);
}

#[test]
fn scalar_gradient_matches_centered_finite_difference() {
    let experiment = base_experiment();
    let weights = Array3::from_shape_fn((2, 5, 2), |(shot, step, receiver)| {
        0.02 + 0.01 * shot as f64 - 0.01 * step as f64 + 0.03 * receiver as f64
    });
    let result = adjoint::simulate_adjoint(&experiment, &weights).unwrap();
    let directional: f64 = experiment
        .perturbation
        .iter()
        .zip(result.image.iter())
        .map(|(m, image)| m * image)
        .sum();
    for eps in [1.0e-4, 1.0e-5, 1.0e-6] {
        let mut plus = experiment.clone();
        plus.background = &plus.background + &(plus.perturbation.mapv(|value| eps * value));
        let mut minus = experiment.clone();
        minus.background = &minus.background - &(minus.perturbation.mapv(|value| eps * value));
        let plus_data = born::simulate_born_with_primal(&plus)
            .unwrap()
            .primal_traces;
        let minus_data = born::simulate_born_with_primal(&minus)
            .unwrap()
            .primal_traces;
        let finite_difference = dot(&(&plus_data - &minus_data), &weights) / (2.0 * eps);
        assert!(relative(finite_difference, directional) < 1.0e-6);
    }
}

#[test]
fn adjoint_statistics_and_image_round_trip_follow_contract() {
    let experiment = base_experiment();
    let weights = Array3::zeros((2, 5, 2));
    let result = adjoint::simulate_adjoint(&experiment, &weights).unwrap();
    assert_eq!(result.statistics.storage, "full");
    assert_eq!(result.statistics.checkpoints, None);
    assert_eq!(result.statistics.reverse_calls, 10);
    assert_eq!(result.statistics.scheduler_forward_calls, 10);
    assert_eq!(result.statistics.peak_saved_states, 6);
    assert_eq!(result.statistics.peak_saved_bytes, 6 * 2 * 7 * 7 * 8);
    assert_eq!(result.statistics.per_shot.len(), 2);
    assert!(result.statistics.per_shot.iter().all(|shot| {
        shot.reverse_calls == 5 && shot.scheduler_forward_calls == 5 && shot.peak_saved_states == 6
    }));

    let output = PathBuf::from(format!("/tmp/amat5315-adjoint-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&output);
    adjoint::write_adjoint_outputs(
        "inputs/test.json",
        &experiment,
        &result.image,
        &result.statistics,
        &output,
    )
    .unwrap();
    let image: Array2<f64> = ndarray_npy::read_npy(output.join("image.npy")).unwrap();
    let metadata: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output.join("result.json")).unwrap()).unwrap();
    assert_eq!(image, result.image);
    assert_eq!(metadata["mode"], "adjoint");
    assert_eq!(metadata["statistics"]["storage"], "full");
    assert_eq!(
        metadata["statistics"]["checkpoints"],
        serde_json::Value::Null
    );
    assert!(!output.join("born_data.npy").exists());
    let _ = fs::remove_dir_all(output);
}

#[test]
fn adjoint_data_reader_rejects_wrong_shape() {
    let experiment = base_experiment();
    let path = PathBuf::from(format!(
        "/tmp/amat5315-adjoint-data-{}-bad.npy",
        std::process::id()
    ));
    write_npy(&path, &Array3::<f64>::zeros((1, 5, 2))).unwrap();
    let error = adjoint::read_adjoint_data(&path, &experiment).unwrap_err();
    assert!(error.to_string().contains("shape"));
    let _ = fs::remove_file(path);
}
