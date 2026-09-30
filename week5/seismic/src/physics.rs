use std::f64::consts::PI;

use anyhow::{bail, Result};
use ndarray::Array2;

use crate::experiment::GridPoint;

/// Return the non-periodic five-point Laplacian, with zero values on the boundary.
pub fn laplacian(field: &Array2<f64>, dx: f64) -> Result<Array2<f64>> {
    if !dx.is_finite() || dx <= 0.0 {
        bail!("dx must be finite and positive");
    }

    let (nz, nx) = field.dim();
    if nz < 3 || nx < 3 {
        bail!("the field must have an interior grid");
    }

    let inverse_dx_squared = 1.0 / (dx * dx);
    let mut result = Array2::zeros((nz, nx));
    for z in 1..(nz - 1) {
        for x in 1..(nx - 1) {
            result[(z, x)] =
                (field[(z, x + 1)] + field[(z, x - 1)] + field[(z + 1, x)] + field[(z - 1, x)]
                    - 4.0 * field[(z, x)])
                    * inverse_dx_squared;
        }
    }
    Ok(result)
}

/// Construct the time-independent quadratic sponge damping field.
pub fn sponge_damping(
    nz: usize,
    nx: usize,
    sponge_width: usize,
    sponge_strength: f64,
) -> Result<Array2<f64>> {
    if nz < 3 || nx < 3 {
        bail!("the sponge grid must have an interior grid");
    }
    if sponge_width == 0 || sponge_width > nx.min(nz) / 2 {
        bail!("sponge_width must be positive and fit within half the grid");
    }
    if !sponge_strength.is_finite() || sponge_strength < 0.0 {
        bail!("sponge_strength must be finite and nonnegative");
    }

    let mut result = Array2::zeros((nz, nx));
    for z in 0..nz {
        for x in 0..nx {
            let distance = x.min(nx - 1 - x).min(z).min(nz - 1 - z);
            let normalized = (1.0 - distance as f64 / sponge_width as f64).max(0.0);
            result[(z, x)] = sponge_strength * normalized * normalized;
        }
    }
    Ok(result)
}

/// Evaluate the reduced-unit Ricker pulse at time `time`.
pub fn ricker_pulse(source_frequency: f64, source_peak_time: f64, time: f64) -> f64 {
    let theta = PI * source_frequency * (time - source_peak_time);
    (1.0 - 2.0 * theta * theta) * (-theta * theta).exp()
}

/// Construct a unit-peak Gaussian source footprint using [x,z] coordinates.
pub fn gaussian_footprint(nz: usize, nx: usize, source: GridPoint) -> Result<Array2<f64>> {
    if source.x >= nx || source.z >= nz {
        bail!("source coordinate is outside the grid");
    }

    let mut result = Array2::zeros((nz, nx));
    for z in 0..nz {
        for x in 0..nx {
            let dx = x as f64 - source.x as f64;
            let dz = z as f64 - source.z as f64;
            result[(z, x)] = (-(dx * dx + dz * dz) / 2.0).exp();
        }
    }
    Ok(result)
}

/// Combine a Ricker pulse and Gaussian footprint into a source field.
pub fn source_field(
    nz: usize,
    nx: usize,
    source: GridPoint,
    source_amplitude: f64,
    source_frequency: f64,
    source_peak_time: f64,
    time: f64,
) -> Result<Array2<f64>> {
    if !source_amplitude.is_finite() {
        bail!("source_amplitude must be finite");
    }
    if !source_frequency.is_finite() || source_frequency < 0.0 {
        bail!("source_frequency must be finite and nonnegative");
    }
    if !source_peak_time.is_finite() || !time.is_finite() {
        bail!("source_peak_time and time must be finite");
    }

    let footprint = gaussian_footprint(nz, nx, source)?;
    let scale = source_amplitude * ricker_pulse(source_frequency, source_peak_time, time);
    Ok(footprint * scale)
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

    #[test]
    fn laplacian_of_constant_field_is_zero_including_boundary() {
        let field = Array2::from_elem((5, 6), 3.5);
        let result = laplacian(&field, 2.0).unwrap();
        assert!(result.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn laplacian_of_quadratic_field_has_expected_interior_value() {
        let field = Array2::from_shape_fn((5, 6), |(z, x)| (z * z + x * x) as f64);
        let result = laplacian(&field, 1.0).unwrap();
        assert_close(result[(2, 3)], 4.0);
        assert!(result.row(0).iter().all(|value| *value == 0.0));
        assert!(result.column(0).iter().all(|value| *value == 0.0));
        assert!(result.row(4).iter().all(|value| *value == 0.0));
        assert!(result.column(5).iter().all(|value| *value == 0.0));
    }

    #[test]
    fn laplacian_of_center_impulse_has_five_point_pattern() {
        let mut field = Array2::zeros((5, 5));
        field[(2, 2)] = 1.0;
        let result = laplacian(&field, 1.0).unwrap();
        assert_close(result[(2, 2)], -4.0);
        assert_close(result[(2, 1)], 1.0);
        assert_close(result[(2, 3)], 1.0);
        assert_close(result[(1, 2)], 1.0);
        assert_close(result[(3, 2)], 1.0);
        assert!(result.row(0).iter().all(|value| *value == 0.0));
        assert!(result.column(0).iter().all(|value| *value == 0.0));
    }

    #[test]
    fn laplacian_rejects_invalid_dx() {
        let field = Array2::zeros((3, 3));
        assert!(laplacian(&field, 0.0).is_err());
        assert!(laplacian(&field, f64::NAN).is_err());
    }

    #[test]
    fn sponge_has_edge_maximum_and_zero_inside_width() {
        let result = sponge_damping(15, 15, 6, 0.4).unwrap();
        assert_close(result[(0, 7)], 0.4);
        assert_close(result[(1, 7)], 0.4 * (5.0_f64 / 6.0).powi(2));
        assert_eq!(result[(6, 7)], 0.0);
        assert_eq!(result[(7, 7)], 0.0);
        assert_close(result[(0, 0)], 0.4);
    }

    #[test]
    fn sponge_is_symmetric_about_all_grid_edges() {
        let result = sponge_damping(11, 13, 3, 0.7).unwrap();
        for z in 0..11 {
            for x in 0..13 {
                assert_eq!(result[(z, x)], result[(z, 12 - x)]);
                assert_eq!(result[(z, x)], result[(10 - z, x)]);
            }
        }
    }

    #[test]
    fn sponge_rejects_invalid_parameters() {
        assert!(sponge_damping(2, 5, 1, 0.4).is_err());
        assert!(sponge_damping(5, 5, 0, 0.4).is_err());
        assert!(sponge_damping(5, 5, 3, 0.4).is_err());
        assert!(sponge_damping(5, 5, 1, -0.1).is_err());
    }

    #[test]
    fn ricker_pulse_peaks_and_is_symmetric() {
        let peak = ricker_pulse(0.08, 15.0, 15.0);
        assert_close(peak, 1.0);
        assert_close(
            ricker_pulse(0.08, 15.0, 14.0),
            ricker_pulse(0.08, 15.0, 16.0),
        );
    }

    #[test]
    fn ricker_pulse_has_negative_lobe() {
        assert!(ricker_pulse(0.08, 15.0, 10.0) < 0.0);
    }

    #[test]
    fn gaussian_uses_xz_coordinates_and_unit_grid_width() {
        let source = GridPoint { x: 3, z: 1 };
        let result = gaussian_footprint(5, 7, source).unwrap();
        assert_close(result[(1, 3)], 1.0);
        assert_close(result[(1, 4)], (-0.5_f64).exp());
        assert_close(result[(2, 4)], (-1.0_f64).exp());
        assert_close(result[(1, 2)], result[(1, 4)]);
        assert_close(result[(0, 3)], result[(2, 3)]);
        assert!(result[(1, 3)] > result[(3, 1)]);
    }

    #[test]
    fn gaussian_rejects_out_of_range_source() {
        assert!(gaussian_footprint(5, 7, GridPoint { x: 7, z: 1 }).is_err());
    }

    #[test]
    fn combined_source_has_amplitude_at_peak_and_center() {
        let source = GridPoint { x: 3, z: 1 };
        let result = source_field(5, 7, source, 2.5, 0.08, 15.0, 15.0).unwrap();
        assert_close(result[(1, 3)], 2.5);
    }

    #[test]
    fn deterministic_physics_diagnostic() {
        let mut impulse = Array2::zeros((5, 5));
        impulse[(2, 2)] = 1.0;
        let laplacian_value = laplacian(&impulse, 1.0).unwrap()[(2, 2)];
        let sponge = sponge_damping(15, 15, 6, 0.4).unwrap();
        let source = GridPoint { x: 3, z: 1 };
        let gaussian = gaussian_footprint(5, 7, source).unwrap();

        println!("laplacian_center={laplacian_value:.16}");
        println!("sponge_distance_0={:.16}", sponge[(0, 7)]);
        println!("sponge_distance_1={:.16}", sponge[(1, 7)]);
        println!("sponge_distance_6={:.16}", sponge[(6, 7)]);
        println!("ricker_at_peak={:.16}", ricker_pulse(0.08, 15.0, 15.0));
        println!("gaussian_center={:.16}", gaussian[(1, 3)]);
        println!("gaussian_one_cell_neighbor={:.16}", gaussian[(1, 4)]);
    }
}
