use std::f64::consts::PI;
use week4::{AdvectionDiffusion1D, Integrator, Rk4, SpatialDerivative};

fn rk4_growth(real: f64, imaginary: f64) -> f64 {
    let result = Rk4.step(&[1.0, 0.0], 0.0, 1.0, &|_, state| {
        vec![
            real * state[0] - imaginary * state[1],
            imaginary * state[0] + real * state[1],
        ]
    });
    result[0].hypot(result[1])
}

fn rk4_amplification(real: f64, imaginary: f64) -> f64 {
    let real_part = 1.0
        + real
        + (real * real - imaginary * imaginary) / 2.0
        + (real.powi(3) - 3.0 * real * imaginary * imaginary) / 6.0
        + (real.powi(4) - 6.0 * real * real * imaginary * imaginary + imaginary.powi(4)) / 24.0;
    let imaginary_part = imaginary
        + real * imaginary
        + (3.0 * real * real * imaginary - imaginary.powi(3)) / 6.0
        + (4.0 * real.powi(3) * imaginary - 4.0 * real * imaginary.powi(3)) / 24.0;
    real_part.hypot(imaginary_part)
}

fn spectral_point(k: isize, n: usize, c: f64, nu: f64) -> (f64, f64) {
    if n.is_multiple_of(2) && k == -(n as isize) / 2 {
        (-nu * (n as f64 / 2.0).powi(2), 0.0)
    } else {
        (-nu * (k as f64).powi(2), -c * k as f64)
    }
}

fn stable_dt(n: usize, c: f64, nu: f64) -> f64 {
    let stable = |dt: f64| {
        (-((n / 2) as isize)..=(n as isize / 2 - 1)).all(|k| {
            let (real, imaginary) = spectral_point(k, n, c, nu);
            rk4_amplification(real * dt, imaginary * dt) <= 1.0 + 1e-13
        })
    };
    let mut low = 0.0;
    let mut high = 0.1;
    while stable(high) {
        high *= 2.0;
    }
    for _ in 0..80 {
        let middle = 0.5 * (low + high);
        if stable(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn main() {
    let n = 64;
    let c = 1.0;
    let nu = 0.05;
    println!("CRITICAL,{:.15}", stable_dt(n, c, nu));
    for row in 0..=240 {
        let real = -4.0 + 5.0 * row as f64 / 240.0;
        for col in 0..=240 {
            let imaginary = -4.0 + 8.0 * col as f64 / 240.0;
            println!(
                "GRID,{real:.15},{imaginary:.15},{:.15}",
                rk4_growth(real, imaginary)
            );
        }
    }
    for &dt in &[0.045_f64, 0.056_f64] {
        for k in -32isize..32 {
            let (real, imaginary) = spectral_point(k, n, c, nu);
            println!(
                "SPECTRUM,{dt:.15},{k},{real:.15},{imaginary:.15},{:.15}",
                rk4_amplification(real * dt, imaginary * dt)
            );
        }
    }

    let solver = AdvectionDiffusion1D::new(n, c, nu, SpatialDerivative::Fourier);
    let grid = solver.grid();
    for &dt in &[0.045_f64, 0.056_f64] {
        let mut state: Vec<_> = grid
            .iter()
            .map(|&x| week4::periodic_gaussian(x, PI / 2.0, 0.35, 1.0))
            .collect();
        let mut time = 0.0;
        let integrator = Rk4;
        for (j, (&x, &value)) in grid.iter().zip(&state).enumerate() {
            println!("LINE,{dt:.15},{time:.15},{j},{x:.15},{value:.15}");
        }
        while time < 6.0 - 1e-14 {
            let step = dt.min(6.0 - time);
            state = solver.step(&integrator, &state, time, step);
            time += step;
            for (j, (&x, &value)) in grid.iter().zip(&state).enumerate() {
                println!("LINE,{dt:.15},{time:.15},{j},{x:.15},{value:.15}");
            }
        }
    }
}
