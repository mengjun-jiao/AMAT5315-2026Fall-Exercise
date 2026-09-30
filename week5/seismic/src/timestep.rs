use anyhow::{bail, Result};
use ndarray::Array2;

use crate::{
    experiment::{Experiment, GridPoint},
    field::State,
    physics::{laplacian, source_field},
};

fn require_same_shape(name: &str, field: &Array2<f64>, shape: (usize, usize)) -> Result<()> {
    if field.dim() != shape {
        bail!(
            "{name} shape {:?} does not match required shape {:?}",
            field.dim(),
            shape
        );
    }
    Ok(())
}

/// Compute one damped acoustic update without changing the input state.
pub fn compute_next(
    previous: &Array2<f64>,
    current: &Array2<f64>,
    wave_speed: &Array2<f64>,
    sponge: &Array2<f64>,
    source: &Array2<f64>,
    dx: f64,
    dt: f64,
) -> Result<Array2<f64>> {
    if !dt.is_finite() || dt <= 0.0 {
        bail!("dt must be finite and positive");
    }

    let shape = current.dim();
    require_same_shape("previous", previous, shape)?;
    require_same_shape("wave_speed", wave_speed, shape)?;
    require_same_shape("sponge", sponge, shape)?;
    require_same_shape("source", source, shape)?;

    let spatial_laplacian = laplacian(current, dx)?;
    let (nz, nx) = shape;
    let mut next = Array2::zeros(shape);
    for z in 1..(nz - 1) {
        for x in 1..(nx - 1) {
            let damping = sponge[(z, x)];
            let numerator = 2.0 * current[(z, x)] - (1.0 - damping * dt) * previous[(z, x)]
                + dt * dt
                    * (wave_speed[(z, x)].powi(2) * spatial_laplacian[(z, x)] + source[(z, x)]);
            next[(z, x)] = numerator / (1.0 + damping * dt);
        }
    }
    Ok(next)
}

/// Compute and install one state update: previous <- old current, current <- next.
pub fn advance(
    state: &mut State,
    wave_speed: &Array2<f64>,
    sponge: &Array2<f64>,
    source: &Array2<f64>,
    dx: f64,
    dt: f64,
) -> Result<()> {
    let next = compute_next(
        &state.previous,
        &state.current,
        wave_speed,
        sponge,
        source,
        dx,
        dt,
    )?;
    state.advance(next)
}

/// Build the source field for timestep `n`, using reduced-unit time n * dt.
pub fn source_for_timestep(
    experiment: &Experiment,
    n: usize,
    source: GridPoint,
) -> Result<Array2<f64>> {
    source_field(
        experiment.nz,
        experiment.nx,
        source,
        experiment.source_amplitude,
        experiment.source_frequency,
        experiment.source_peak_time,
        n as f64 * experiment.dt,
    )
}

/// Sample a field in the input receiver order using [x,z] coordinates.
pub fn sample_receivers(field: &Array2<f64>, receivers: &[GridPoint]) -> Result<Vec<f64>> {
    let (nz, nx) = field.dim();
    receivers
        .iter()
        .map(|receiver| {
            if receiver.x >= nx || receiver.z >= nz {
                bail!(
                    "receiver coordinate [{}, {}] is outside the field",
                    receiver.x,
                    receiver.z
                );
            }
            Ok(field[(receiver.z, receiver.x)])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{actual} != {expected}"
        );
    }

    fn fields(shape: (usize, usize), value: f64) -> Array2<f64> {
        Array2::from_elem(shape, value)
    }

    #[test]
    fn zero_state_and_source_stays_zero() {
        let shape = (5, 5);
        let result = compute_next(
            &fields(shape, 0.0),
            &fields(shape, 0.0),
            &fields(shape, 2.0),
            &fields(shape, 0.3),
            &fields(shape, 0.0),
            1.0,
            0.1,
        )
        .unwrap();
        assert!(result.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn source_only_update_matches_manual_interior_formula_and_zero_boundary() {
        let shape = (5, 5);
        let source =
            Array2::from_shape_fn(shape, |(z, x)| if (z, x) == (2, 2) { 3.0 } else { 7.0 });
        let result = compute_next(
            &fields(shape, 0.0),
            &fields(shape, 0.0),
            &fields(shape, 2.0),
            &fields(shape, 0.5),
            &source,
            1.0,
            0.2,
        )
        .unwrap();
        assert_close(result[(2, 2)], 0.04 * 3.0 / 1.1);
        assert_close(result[(2, 1)], 0.04 * 7.0 / 1.1);
        assert!(result.row(0).iter().all(|value| *value == 0.0));
        assert!(result.column(0).iter().all(|value| *value == 0.0));
        assert!(result.row(4).iter().all(|value| *value == 0.0));
        assert!(result.column(4).iter().all(|value| *value == 0.0));
    }

    #[test]
    fn no_damping_update_matches_manual_laplacian_formula() {
        let shape = (5, 5);
        let previous = Array2::from_elem(shape, 0.5);
        let current = Array2::from_shape_fn(shape, |(z, x)| (z * z + x * x) as f64);
        let source = Array2::from_elem(shape, 0.25);
        let result = compute_next(
            &previous,
            &current,
            &fields(shape, 2.0),
            &fields(shape, 0.0),
            &source,
            1.0,
            0.1,
        )
        .unwrap();
        let expected = 2.0 * current[(2, 2)] - previous[(2, 2)] + 0.01 * (4.0 * 4.0 + 0.25);
        assert_close(result[(2, 2)], expected);
    }

    #[test]
    fn damped_update_uses_local_speed_and_damping() {
        let shape = (5, 6);
        let previous = Array2::from_elem(shape, 0.7);
        let current = Array2::from_shape_fn(shape, |(z, x)| (z + 2 * x) as f64);
        let wave_speed =
            Array2::from_shape_fn(shape, |(z, x)| if (z, x) == (2, 2) { 2.0 } else { 3.0 });
        let sponge = Array2::from_elem(shape, 0.4);
        let source = Array2::from_elem(shape, 0.3);
        let result =
            compute_next(&previous, &current, &wave_speed, &sponge, &source, 1.0, 0.1).unwrap();
        let laplacian_value = 0.0;
        let expected = (2.0 * current[(2, 2)] - (1.0 - 0.4 * 0.1) * previous[(2, 2)]
            + 0.1_f64.powi(2) * (2.0_f64.powi(2) * laplacian_value + 0.3))
            / (1.0 + 0.4 * 0.1);
        assert_close(result[(2, 2)], expected);

        let other_expected = (2.0 * current[(2, 3)] - (1.0 - 0.4 * 0.1) * previous[(2, 3)]
            + 0.1_f64.powi(2) * (3.0_f64.powi(2) * 0.0 + 0.3))
            / (1.0 + 0.4 * 0.1);
        assert_close(result[(2, 3)], other_expected);
    }

    #[test]
    fn boundaries_are_zero_even_when_inputs_are_nonzero() {
        let shape = (5, 5);
        let result = compute_next(
            &fields(shape, 4.0),
            &fields(shape, 5.0),
            &fields(shape, 2.0),
            &fields(shape, 0.2),
            &fields(shape, 9.0),
            1.0,
            0.1,
        )
        .unwrap();
        for x in 0..5 {
            assert_eq!(result[(0, x)], 0.0);
            assert_eq!(result[(4, x)], 0.0);
        }
        for z in 0..5 {
            assert_eq!(result[(z, 0)], 0.0);
            assert_eq!(result[(z, 4)], 0.0);
        }
    }

    #[test]
    fn shape_and_timestep_validation_is_explicit() {
        let shape = (5, 5);
        let wrong_shape = fields((4, 5), 0.0);
        assert!(compute_next(
            &wrong_shape,
            &fields(shape, 0.0),
            &fields(shape, 1.0),
            &fields(shape, 0.0),
            &fields(shape, 0.0),
            1.0,
            0.1,
        )
        .is_err());
        assert!(compute_next(
            &fields(shape, 0.0),
            &fields(shape, 0.0),
            &fields(shape, 1.0),
            &fields(shape, 0.0),
            &fields(shape, 0.0),
            1.0,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn state_advance_moves_old_current_without_overwriting_next() {
        let mut state = State::zeros(3, 4);
        state.current[(1, 2)] = 7.0;
        let old_current = state.current.clone();
        let next = Array2::from_elem((3, 4), 11.0);
        state.advance(next.clone()).unwrap();
        assert_eq!(state.previous, old_current);
        assert_eq!(state.current, next);
    }

    #[test]
    fn one_step_advances_state_using_computed_next_field() {
        let shape = (5, 5);
        let mut state = State::zeros(5, 5);
        state.current[(2, 2)] = 1.0;
        let old_current = state.current.clone();
        let source = fields(shape, 0.0);
        let expected = compute_next(
            &state.previous,
            &state.current,
            &fields(shape, 1.0),
            &fields(shape, 0.0),
            &source,
            1.0,
            0.1,
        )
        .unwrap();
        advance(
            &mut state,
            &fields(shape, 1.0),
            &fields(shape, 0.0),
            &source,
            1.0,
            0.1,
        )
        .unwrap();
        assert_eq!(state.previous, old_current);
        assert_eq!(state.current, expected);
    }

    #[test]
    fn source_for_timestep_uses_n_times_dt() {
        let experiment = Experiment::from_json_str(
            r#"
            {
              "nx": 5, "nz": 5, "dx": 1.0, "dt": 0.5, "steps": 40,
              "source_frequency": 0.08, "source_peak_time": 15.0,
              "source_amplitude": 2.0, "sponge_width": 1,
              "sponge_strength": 0.4,
              "shots": [[2,1]], "receivers": [[2,1]],
              "background": [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
              "perturbation": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
              "length_unit_m": 100.0, "time_unit_s": 0.1
            }
            "#,
        )
        .unwrap();
        let source = GridPoint { x: 2, z: 1 };
        let at_zero = source_for_timestep(&experiment, 0, source).unwrap();
        assert_close(at_zero[(1, 2)], 2.0 * ricker_at(0.0, 0.08, 15.0));

        let at_peak = source_for_timestep(&experiment, 30, source).unwrap();
        assert_close(at_peak[(1, 2)], 2.0);
    }

    fn ricker_at(time: f64, frequency: f64, peak_time: f64) -> f64 {
        crate::physics::ricker_pulse(frequency, peak_time, time)
    }

    #[test]
    fn receiver_sampling_preserves_order_and_uses_zx_indexing() {
        let field = Array2::from_shape_fn((3, 5), |(z, x)| (10 * z + x) as f64);
        let receivers = vec![
            GridPoint { x: 3, z: 1 },
            GridPoint { x: 0, z: 2 },
            GridPoint { x: 3, z: 1 },
        ];
        assert_eq!(
            sample_receivers(&field, &receivers).unwrap(),
            vec![13.0, 20.0, 13.0]
        );
    }

    #[test]
    fn receiver_sampling_uses_newly_updated_field() {
        let old_current = Array2::from_elem((3, 5), 2.0);
        let next = Array2::from_elem((3, 5), 9.0);
        let receiver = [GridPoint { x: 3, z: 1 }];
        let sampled = sample_receivers(&next, &receiver).unwrap();
        assert_eq!(sampled, vec![9.0]);
        assert_ne!(sampled, sample_receivers(&old_current, &receiver).unwrap());
    }

    #[test]
    fn receiver_sampling_rejects_out_of_range_coordinate() {
        let field = Array2::zeros((3, 5));
        assert!(sample_receivers(&field, &[GridPoint { x: 5, z: 1 }]).is_err());
    }

    #[test]
    fn deterministic_timestep_diagnostic() {
        let shape = (5, 5);
        let previous = Array2::from_elem(shape, 0.7);
        let current = Array2::from_shape_fn(shape, |(z, x)| (z + 2 * x) as f64);
        let wave_speed = Array2::from_elem(shape, 2.0);
        let sponge = Array2::from_elem(shape, 0.4);
        let source = Array2::from_elem(shape, 0.3);
        let dx = 1.0;
        let dt = 0.1;
        let laplacian_value = laplacian(&current, dx).unwrap()[(2, 2)];
        let c = wave_speed[(2, 2)];
        let sigma = sponge[(2, 2)];
        let q = source[(2, 2)];
        let numerator = 2.0 * current[(2, 2)] - (1.0 - sigma * dt) * previous[(2, 2)]
            + dt * dt * (c * c * laplacian_value + q);
        let denominator = 1.0 + sigma * dt;
        let computed = compute_next(&previous, &current, &wave_speed, &sponge, &source, dx, dt)
            .unwrap()[(2, 2)];
        let expected = numerator / denominator;
        println!("u_prev={:.16}", previous[(2, 2)]);
        println!("u_curr={:.16}", current[(2, 2)]);
        println!("laplacian={laplacian_value:.16}");
        println!("c={c:.16}");
        println!("sigma={sigma:.16}");
        println!("q={q:.16}");
        println!("dt={dt:.16}");
        println!("numerator={numerator:.16}");
        println!("denominator={denominator:.16}");
        println!("computed_u_next={computed:.16}");
        println!("expected_u_next={expected:.16}");
        println!("absolute_difference={:.16}", (computed - expected).abs());
        assert_close(computed, expected);
    }
}
