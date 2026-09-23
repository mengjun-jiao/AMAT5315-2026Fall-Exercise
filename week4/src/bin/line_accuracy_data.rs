use week4::{
    exact_periodic_gaussian, AdvectionDiffusion1D, EqualWeightFourStage, ForwardEuler, Integrator,
    Midpoint, Rk4, SpatialDerivative,
};

fn evolve<I: Integrator + ?Sized>(
    solver: &AdvectionDiffusion1D,
    integrator: &I,
    mut state: Vec<f64>,
    dt: f64,
    t_end: f64,
) -> Vec<f64> {
    let mut time = 0.0;
    while time < t_end - 1e-14 {
        let step = dt.min(t_end - time);
        state = integrator.step(&state, time, step, &|_, values| solver.rhs(values));
        time += step;
    }
    state
}

fn error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let n = 64;
    let c = 1.0;
    let nu = 0.002;
    let center = std::f64::consts::PI / 2.0;
    let sigma = 0.25;
    let grid: Vec<_> = (0..n)
        .map(|j| 2.0 * std::f64::consts::PI * j as f64 / n as f64)
        .collect();
    let initial: Vec<_> = grid
        .iter()
        .map(|&x| week4::periodic_gaussian(x, center, sigma, 1.0))
        .collect();
    let exact = exact_periodic_gaussian(&grid, 2.0 * std::f64::consts::PI, c, nu, center, sigma);
    let cases: [(&str, SpatialDerivative, &dyn Integrator, f64); 3] = [
        ("RK4 Fourier", SpatialDerivative::Fourier, &Rk4, 0.02),
        (
            "RK4 centered finite difference",
            SpatialDerivative::CenteredFiniteDifference,
            &Rk4,
            0.02,
        ),
        (
            "Euler Fourier",
            SpatialDerivative::Fourier,
            &ForwardEuler,
            0.005,
        ),
    ];
    for (label, derivative, integrator, dt) in cases {
        let solver = AdvectionDiffusion1D::new(n, c, nu, derivative);
        let final_state = evolve(
            &solver,
            integrator,
            initial.clone(),
            dt,
            2.0 * std::f64::consts::PI,
        );
        println!("PROFILE,{label},{:.15}", error(&final_state, &exact));
        for (j, (&x, &value)) in grid.iter().zip(&final_state).enumerate() {
            println!(
                "PROFILE_VALUE,{label},{j},{x:.15},{value:.15},{:.15}",
                exact[j]
            );
        }
    }

    let nu = 0.05;
    let sigma = 0.35;
    let center = std::f64::consts::PI / 2.0;
    let solver = AdvectionDiffusion1D::new(n, c, nu, SpatialDerivative::Fourier);
    let initial: Vec<_> = grid
        .iter()
        .map(|&x| week4::periodic_gaussian(x, center, sigma, 1.0))
        .collect();
    let exact = exact_periodic_gaussian(&grid, 1.0, c, nu, center, sigma);
    let methods: [(&str, &dyn Integrator); 4] = [
        ("Forward Euler", &ForwardEuler),
        ("Midpoint", &Midpoint),
        ("RK4", &Rk4),
        ("EqualWeightFourStage", &EqualWeightFourStage),
    ];
    for (name, integrator) in methods {
        for &dt in &[0.02, 0.01, 0.005, 0.0025] {
            let final_state = evolve(&solver, integrator, initial.clone(), dt, 1.0);
            println!(
                "CONVERGENCE,{name},{dt:.15},{:.15}",
                error(&final_state, &exact)
            );
        }
    }
}
