use std::{fs, path::PathBuf};

use ndarray::{Array2, Array3};

use seismic::{
    adjoint, born,
    cli::{validate_checkpoint_options, Mode, Storage},
    experiment::{Experiment, GridPoint},
};

fn experiment() -> Experiment {
    Experiment {
        nx: 7,
        nz: 7,
        dx: 1.0,
        dt: 0.1,
        steps: 6,
        source_frequency: 0.08,
        source_peak_time: 0.2,
        source_amplitude: 1.0,
        shots: vec![GridPoint { x: 3, z: 2 }, GridPoint { x: 2, z: 3 }],
        receivers: vec![GridPoint { x: 2, z: 2 }, GridPoint { x: 3, z: 2 }],
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
fn treeverse_image_matches_full_history_and_records_exact_statistics() {
    let experiment = experiment();
    let weights = Array3::from_shape_fn((2, 6, 2), |(shot, step, receiver)| {
        0.02 + 0.01 * shot as f64 - 0.02 * step as f64 + 0.03 * receiver as f64
    });
    let full = adjoint::simulate_adjoint(&experiment, &weights).unwrap();
    let tree = adjoint::simulate_treeverse(&experiment, &weights, 2, None).unwrap();
    let difference = &tree.image - &full.image;
    let relative_error = difference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        / full
            .image
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
    assert!(
        relative_error < 1.0e-12,
        "relative image error={relative_error}"
    );
    assert_eq!(tree.statistics.reverse_calls, 12);
    assert_eq!(tree.statistics.scheduler_forward_calls, 16);
    assert_eq!(tree.statistics.peak_saved_states, 3);
    assert_eq!(tree.statistics.peak_saved_bytes, 3 * 2 * 7 * 7 * 8);
    assert!(tree.statistics.per_shot.iter().all(|shot| {
        shot.reverse_calls == 6
            && shot.scheduler_forward_calls == 8
            && shot.peak_saved_states == 3
            && shot.actions_file.is_some()
    }));
}

#[test]
fn treeverse_transpose_identity_and_zero_weights_hold() {
    let experiment = experiment();
    let born_data = born::simulate_born(&experiment).unwrap();
    let weights = Array3::from_shape_fn(born_data.raw_dim(), |(shot, step, receiver)| {
        0.01 + 0.02 * shot as f64 - 0.01 * step as f64 + 0.03 * receiver as f64
    });
    let tree = adjoint::simulate_treeverse(&experiment, &weights, 2, None).unwrap();
    let left = dot(&born_data, &weights);
    let right: f64 = experiment
        .perturbation
        .iter()
        .zip(tree.image.iter())
        .map(|(m, image)| m * image)
        .sum();
    assert!(relative(left, right) < 1.0e-10);

    let zero =
        adjoint::simulate_treeverse(&experiment, &Array3::zeros((2, 6, 2)), 2, None).unwrap();
    assert!(zero.image.iter().all(|value| *value == 0.0));
}

#[test]
fn treeverse_recording_matches_full_history_recording() {
    let experiment = experiment();
    let weights = Array3::from_shape_fn((2, 6, 2), |(shot, step, receiver)| {
        0.03 + 0.01 * shot as f64 + 0.02 * step as f64 - 0.01 * receiver as f64
    });
    let full = adjoint::simulate_adjoint_with_recording(&experiment, &weights, Some(2)).unwrap();
    let tree = adjoint::simulate_treeverse(&experiment, &weights, 2, Some(2)).unwrap();
    assert_eq!(
        full.recording.as_ref().unwrap().steps,
        tree.recording.as_ref().unwrap().steps
    );
    assert_eq!(
        full.recording.as_ref().unwrap().times,
        tree.recording.as_ref().unwrap().times
    );
    assert_eq!(
        full.recording.as_ref().unwrap().wavefield,
        tree.recording.as_ref().unwrap().wavefield
    );
}

#[test]
fn treeverse_action_files_are_written_with_runtime_counts() {
    let experiment = experiment();
    let weights = Array3::zeros((2, 6, 2));
    let result = adjoint::simulate_treeverse(&experiment, &weights, 2, None).unwrap();
    let output = PathBuf::from(format!(
        "/tmp/amat5315-treeverse-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&output);
    adjoint::write_adjoint_outputs("inputs/test.json", &experiment, &result, &output).unwrap();
    for shot in 0..2 {
        let path = output.join(format!("actions-{shot}.json"));
        let actions: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert!(actions.as_array().unwrap().iter().all(|action| {
            action
                .as_object()
                .unwrap()
                .keys()
                .all(|key| matches!(key.as_str(), "action" | "step" | "saved_states"))
        }));
    }
    assert_eq!(result.statistics.storage, "treeverse");
    assert_eq!(result.statistics.checkpoints, Some(2));
    assert_eq!(
        result.statistics.per_shot[0].actions_file.as_deref(),
        Some("actions-0.json")
    );
    let _ = fs::remove_dir_all(output);
}

#[test]
fn treeverse_action_log_grad_order_is_descending() {
    let experiment = experiment();
    let result =
        adjoint::simulate_treeverse(&experiment, &Array3::zeros((2, 6, 2)), 2, None).unwrap();
    let actions = result.action_logs.unwrap();
    for shot_actions in actions {
        let grads: Vec<_> = shot_actions
            .iter()
            .filter(|action| action.action == seismic::checkpoint::ActionKind::Grad)
            .map(|action| action.step)
            .collect();
        assert_eq!(grads, (0..6).rev().collect::<Vec<_>>());
    }
}

#[test]
fn cli_storage_and_checkpoint_validation_is_explicit() {
    assert!(validate_checkpoint_options(Mode::Adjoint, Storage::Full, None).is_ok());
    assert!(validate_checkpoint_options(Mode::Adjoint, Storage::Treeverse, Some(1)).is_ok());
    assert!(validate_checkpoint_options(Mode::Adjoint, Storage::Treeverse, None).is_err());
    assert!(validate_checkpoint_options(Mode::Adjoint, Storage::Treeverse, Some(0)).is_err());
    assert!(validate_checkpoint_options(Mode::Adjoint, Storage::Full, Some(1)).is_err());
    assert!(validate_checkpoint_options(Mode::Forward, Storage::Treeverse, Some(1)).is_err());
    assert!(validate_checkpoint_options(Mode::Born, Storage::Full, Some(1)).is_err());
}
