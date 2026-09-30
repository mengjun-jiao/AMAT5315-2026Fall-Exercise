use ndarray::Array2;

use seismic::{enzyme, timestep};

fn flatten(field: &Array2<f64>) -> Vec<f64> {
    field.iter().copied().collect()
}

#[allow(clippy::type_complexity)]
fn fields() -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let shape = (5, 6);
    let previous = Array2::from_shape_fn(shape, |(z, x)| 0.2 + 0.03 * z as f64 - 0.02 * x as f64);
    let current = Array2::from_shape_fn(shape, |(z, x)| 0.4 + 0.05 * z as f64 + 0.01 * x as f64);
    let wave_speed = Array2::from_shape_fn(shape, |(z, x)| 1.5 + 0.04 * z as f64 + 0.03 * x as f64);
    let sigma = Array2::from_shape_fn(shape, |(z, x)| 0.1 + 0.01 * z as f64 + 0.02 * x as f64);
    let source = Array2::from_shape_fn(shape, |(z, x)| -0.2 + 0.07 * z as f64 - 0.03 * x as f64);
    (previous, current, wave_speed, sigma, source)
}

fn assert_primal_equivalence(
    previous: &Array2<f64>,
    current: &Array2<f64>,
    wave_speed: &Array2<f64>,
    sigma: &Array2<f64>,
    source: &Array2<f64>,
    dx: f64,
    dt: f64,
) -> f64 {
    let (nz, nx) = current.dim();
    let expected =
        timestep::compute_next(previous, current, wave_speed, sigma, source, dx, dt).unwrap();
    let actual = enzyme::timestep_primal(
        &flatten(previous),
        &flatten(current),
        &flatten(wave_speed),
        &flatten(sigma),
        &flatten(source),
        nx,
        nz,
        dx,
        dt,
    )
    .unwrap();
    let expected = flatten(&expected);
    for z in 0..nz {
        for x in 0..nx {
            if z == 0 || x == 0 || z + 1 == nz || x + 1 == nx {
                assert_eq!(actual[z * nx + x], 0.0, "boundary index {}", z * nx + x);
            }
        }
    }
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

#[test]
fn cube_smoke_has_primal_jvp_and_vjp_values() {
    let values = enzyme::cube_smoke(2.0).unwrap();
    assert_eq!(values, [8.0, 12.0, 12.0]);
}

#[test]
fn safe_wrappers_reject_wrong_flattened_lengths() {
    let error = enzyme::timestep_primal(
        &[0.0; 3], &[0.0; 4], &[1.0; 4], &[0.0; 4], &[0.0; 4], 2, 2, 1.0, 0.1,
    )
    .unwrap_err();
    assert!(error.to_string().contains("u_prev"));
}

#[test]
fn isolated_primal_matches_part2_timestep_and_keeps_boundary_zero() {
    let (previous, current, wave_speed, sigma, source) = fields();
    let max_error = assert_primal_equivalence(
        &previous,
        &current,
        &wave_speed,
        &sigma,
        &source,
        0.75,
        0.08,
    );
    println!("primal_max_error={max_error:.17e}");
    assert!(max_error < 1.0e-14, "max error {max_error}");
}

#[test]
fn isolated_primal_matches_zero_damping_and_source_variants() {
    let shape = (5, 6);
    let zero = Array2::zeros(shape);
    let constant_speed = Array2::from_elem(shape, 2.0);
    let zero_sigma = Array2::zeros(shape);
    let zero_error =
        assert_primal_equivalence(&zero, &zero, &constant_speed, &zero_sigma, &zero, 1.0, 0.1);
    assert_eq!(zero_error, 0.0);

    let source = Array2::from_shape_fn(shape, |(z, x)| 0.2 + 0.03 * z as f64 - 0.04 * x as f64);
    let varying_speed =
        Array2::from_shape_fn(shape, |(z, x)| 1.0 + 0.1 * z as f64 + 0.05 * x as f64);
    let damping = Array2::from_elem(shape, 0.3);
    let source_error =
        assert_primal_equivalence(&zero, &zero, &varying_speed, &damping, &source, 0.5, 0.2);
    assert!(source_error < 1.0e-14, "max error {source_error}");
}

#[test]
fn isolated_kernel_uses_row_major_zx_indexing() {
    let shape = (4, 5);
    let previous = Array2::zeros(shape);
    let current = Array2::from_elem(shape, 0.0);
    let wave_speed = Array2::from_elem(shape, 1.0);
    let sigma = Array2::from_elem(shape, 0.0);
    let source = Array2::from_shape_fn(shape, |(z, x)| if (z, x) == (2, 3) { 7.0 } else { 0.0 });
    let actual = enzyme::timestep_primal(
        &flatten(&previous),
        &flatten(&current),
        &flatten(&wave_speed),
        &flatten(&sigma),
        &flatten(&source),
        5,
        4,
        1.0,
        0.1,
    )
    .unwrap();
    assert_eq!(actual[2 * 5 + 3], 0.07);
    assert_eq!(actual[5 + 3], 0.0);
    assert_eq!(actual[2 * 5 + 2], 0.0);
}

#[test]
fn jvp_matches_centered_finite_difference() {
    let (previous, current, wave_speed, sigma, source) = fields();
    let du_previous = previous.mapv(|value| 0.2 * value - 0.1);
    let du_current = current.mapv(|value| -0.15 * value + 0.03);
    let dc = wave_speed.mapv(|value| 0.1 * value + 0.02);
    let previous = flatten(&previous);
    let current = flatten(&current);
    let wave_speed = flatten(&wave_speed);
    let sigma = flatten(&sigma);
    let source = flatten(&source);
    let du_previous = flatten(&du_previous);
    let du_current = flatten(&du_current);
    let dc = flatten(&dc);
    let (_, tangent) = enzyme::timestep_jvp(
        &previous,
        &du_previous,
        &current,
        &du_current,
        &wave_speed,
        &dc,
        &sigma,
        &source,
        6,
        5,
        0.75,
        0.08,
    )
    .unwrap();
    let eps = 1.0e-6;
    let plus = enzyme::timestep_primal(
        &previous
            .iter()
            .zip(&du_previous)
            .map(|(x, dx)| x + eps * dx)
            .collect::<Vec<_>>(),
        &current
            .iter()
            .zip(&du_current)
            .map(|(x, dx)| x + eps * dx)
            .collect::<Vec<_>>(),
        &wave_speed
            .iter()
            .zip(&dc)
            .map(|(x, dx)| x + eps * dx)
            .collect::<Vec<_>>(),
        &sigma,
        &source,
        6,
        5,
        0.75,
        0.08,
    )
    .unwrap();
    let minus = enzyme::timestep_primal(
        &previous
            .iter()
            .zip(&du_previous)
            .map(|(x, dx)| x - eps * dx)
            .collect::<Vec<_>>(),
        &current
            .iter()
            .zip(&du_current)
            .map(|(x, dx)| x - eps * dx)
            .collect::<Vec<_>>(),
        &wave_speed
            .iter()
            .zip(&dc)
            .map(|(x, dx)| x - eps * dx)
            .collect::<Vec<_>>(),
        &sigma,
        &source,
        6,
        5,
        0.75,
        0.08,
    )
    .unwrap();
    let max_error = tangent
        .iter()
        .zip(plus.iter().zip(minus.iter()))
        .map(|(actual, (plus, minus))| (actual - (plus - minus) / (2.0 * eps)).abs())
        .fold(0.0, f64::max);
    println!("jvp_finite_difference_max_error={max_error:.17e}");
    assert!(
        max_error < 1.0e-8,
        "max finite-difference error {max_error}"
    );
}

#[test]
fn one_step_jvp_and_vjp_satisfy_transpose_identity() {
    let (previous, current, wave_speed, sigma, source) = fields();
    let du_previous = previous.mapv(|value| 0.13 * value - 0.04);
    let du_current = current.mapv(|value| -0.11 * value + 0.02);
    let dc = wave_speed.mapv(|value| 0.09 * value + 0.01);
    let weight = Array2::from_shape_fn((5, 6), |(z, x)| 0.03 + 0.02 * z as f64 - 0.01 * x as f64);
    let previous = flatten(&previous);
    let current = flatten(&current);
    let wave_speed = flatten(&wave_speed);
    let sigma = flatten(&sigma);
    let source = flatten(&source);
    let du_previous = flatten(&du_previous);
    let du_current = flatten(&du_current);
    let dc = flatten(&dc);
    let weight = flatten(&weight);
    let (_, tangent) = enzyme::timestep_jvp(
        &previous,
        &du_previous,
        &current,
        &du_current,
        &wave_speed,
        &dc,
        &sigma,
        &source,
        6,
        5,
        0.75,
        0.08,
    )
    .unwrap();
    let (previous_bar, current_bar, wave_speed_bar) = enzyme::timestep_vjp(
        &previous,
        &current,
        &wave_speed,
        &sigma,
        &source,
        &weight,
        6,
        5,
        0.75,
        0.08,
    )
    .unwrap();
    let left: f64 = tangent.iter().zip(&weight).map(|(x, w)| x * w).sum();
    let right: f64 = du_previous
        .iter()
        .zip(&previous_bar)
        .map(|(x, b)| x * b)
        .sum::<f64>()
        + du_current
            .iter()
            .zip(&current_bar)
            .map(|(x, b)| x * b)
            .sum::<f64>()
        + dc.iter()
            .zip(&wave_speed_bar)
            .map(|(x, b)| x * b)
            .sum::<f64>();
    let relative = (left - right).abs() / left.abs().max(right.abs()).max(1.0e-30);
    println!("transpose_left={left:.17e} transpose_right={right:.17e} transpose_relative={relative:.17e}");
    assert!(
        relative < 1.0e-10,
        "left={left} right={right} relative={relative}"
    );
}
