use std::f64::consts::PI;
use week4::fluid::{
    centered_dx, centered_dxdy, centered_dxx, centered_laplacian, dx, dxdy, dxx, laplacian,
};

fn field(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let x = 2.0 * PI * (index % n) as f64 / n as f64;
            let y = 2.0 * PI * (index / n) as f64 / n as f64;
            (3.0 * x).sin() * (2.0 * y).cos()
        })
        .collect()
}

fn maximum_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn expected(n: usize, derivative: &str) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let x = 2.0 * PI * (index % n) as f64 / n as f64;
            let y = 2.0 * PI * (index / n) as f64 / n as f64;
            match derivative {
                "dx" => 3.0 * (3.0 * x).cos() * (2.0 * y).cos(),
                "dxx" => -9.0 * (3.0 * x).sin() * (2.0 * y).cos(),
                "dxdy" => -6.0 * (3.0 * x).cos() * (2.0 * y).sin(),
                "laplacian" => -13.0 * (3.0 * x).sin() * (2.0 * y).cos(),
                _ => unreachable!(),
            }
        })
        .collect()
}

fn main() {
    for derivative in ["dx", "dxx", "dxdy", "laplacian"] {
        let values32 = field(32);
        let values64 = field(64);
        let fd32 = match derivative {
            "dx" => centered_dx(&values32, 32),
            "dxx" => centered_dxx(&values32, 32),
            "dxdy" => centered_dxdy(&values32, 32),
            _ => centered_laplacian(&values32, 32),
        };
        let fd64 = match derivative {
            "dx" => centered_dx(&values64, 64),
            "dxx" => centered_dxx(&values64, 64),
            "dxdy" => centered_dxdy(&values64, 64),
            _ => centered_laplacian(&values64, 64),
        };
        let fourier = match derivative {
            "dx" => dx(&values32, 32),
            "dxx" => dxx(&values32, 32),
            "dxdy" => dxdy(&values32, 32),
            _ => laplacian(&values32, 32),
        };
        println!(
            "{derivative},{:.15},{:.15},{:.15}",
            maximum_error(&fd32, &expected(32, derivative)),
            maximum_error(&fd64, &expected(64, derivative)),
            maximum_error(&fourier, &expected(32, derivative))
        );
    }
}
