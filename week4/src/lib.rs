//! Reusable time integrators and a one-dimensional periodic advection-diffusion solver.

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

impl std::ops::MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        );
    }
}

/// A one-step explicit integrator for a vector-valued autonomous or non-autonomous ODE.
pub trait Integrator {
    /// Advance `state` from time `time` by `dt`, using `rate(time, state)`.
    fn step(
        &self,
        state: &[f64],
        time: f64,
        dt: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64>;
}

fn add_scaled(state: &[f64], scale: f64, increment: &[f64]) -> Vec<f64> {
    assert_eq!(
        state.len(),
        increment.len(),
        "rate vector has the wrong length"
    );
    state
        .iter()
        .zip(increment)
        .map(|(&value, &derivative)| value + scale * derivative)
        .collect()
}

/// Forward Euler time integration.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEuler;

impl Integrator for ForwardEuler {
    fn step(
        &self,
        state: &[f64],
        time: f64,
        dt: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        add_scaled(state, dt, &rate(time, state))
    }
}

/// Explicit midpoint, also known as second-order Runge-Kutta.
#[derive(Clone, Copy, Debug, Default)]
pub struct Midpoint;

impl Integrator for Midpoint {
    fn step(
        &self,
        state: &[f64],
        time: f64,
        dt: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let first = rate(time, state);
        let midpoint = add_scaled(state, 0.5 * dt, &first);
        add_scaled(state, dt, &rate(time + 0.5 * dt, &midpoint))
    }
}

/// Classical four-stage fourth-order Runge-Kutta integration.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rk4;

impl Integrator for Rk4 {
    fn step(
        &self,
        state: &[f64],
        time: f64,
        dt: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let k1 = rate(time, state);
        let y2 = add_scaled(state, 0.5 * dt, &k1);
        let k2 = rate(time + 0.5 * dt, &y2);
        let y3 = add_scaled(state, 0.5 * dt, &k2);
        let k3 = rate(time + 0.5 * dt, &y3);
        let y4 = add_scaled(state, dt, &k3);
        let k4 = rate(time + dt, &y4);
        state
            .iter()
            .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(&k4))))
            .map(|(&value, (&a, (&b, (&c, &d))))| value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0)
            .collect()
    }
}

/// Four-stage method with equal final weights; this is intentionally not classical RK4.
#[derive(Clone, Copy, Debug, Default)]
pub struct EqualWeightFourStage;

impl Integrator for EqualWeightFourStage {
    fn step(
        &self,
        state: &[f64],
        time: f64,
        dt: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let k1 = rate(time, state);
        let y2 = add_scaled(state, 0.5 * dt, &k1);
        let k2 = rate(time + 0.5 * dt, &y2);
        let y3 = add_scaled(state, 0.5 * dt, &k2);
        let k3 = rate(time + 0.5 * dt, &y3);
        let y4 = add_scaled(state, dt, &k3);
        let k4 = rate(time + dt, &y4);
        state
            .iter()
            .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(&k4))))
            .map(|(&value, (&a, (&b, (&c, &d))))| value + dt * (a + b + c + d) / 4.0)
            .collect()
    }
}

/// Spatial discretization used by [`AdvectionDiffusion1D`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpatialDerivative {
    Fourier,
    CenteredFiniteDifference,
}

/// A periodic grid and the linear equation `u_t + c u_x = nu u_xx`.
#[derive(Clone, Copy, Debug)]
pub struct AdvectionDiffusion1D {
    pub n: usize,
    pub length: f64,
    pub c: f64,
    pub nu: f64,
    pub derivative: SpatialDerivative,
}

impl AdvectionDiffusion1D {
    pub fn new(n: usize, c: f64, nu: f64, derivative: SpatialDerivative) -> Self {
        assert!(n >= 2, "the grid must contain at least two points");
        Self {
            n,
            length: 2.0 * PI,
            c,
            nu,
            derivative,
        }
    }

    pub fn grid(&self) -> Vec<f64> {
        (0..self.n)
            .map(|j| self.length * j as f64 / self.n as f64)
            .collect()
    }

    pub fn rhs(&self, state: &[f64]) -> Vec<f64> {
        assert_eq!(state.len(), self.n, "state length must equal grid size");
        let (first, second) = match self.derivative {
            SpatialDerivative::Fourier => (
                fourier_derivative(state, self.length, 1),
                fourier_derivative(state, self.length, 2),
            ),
            SpatialDerivative::CenteredFiniteDifference => (
                centered_derivative(state, self.length, 1),
                centered_derivative(state, self.length, 2),
            ),
        };
        first
            .iter()
            .zip(second.iter())
            .map(|(&u_x, &u_xx)| -self.c * u_x + self.nu * u_xx)
            .collect()
    }

    pub fn step<I: Integrator>(
        &self,
        integrator: &I,
        state: &[f64],
        time: f64,
        dt: f64,
    ) -> Vec<f64> {
        integrator.step(state, time, dt, &|_, values| self.rhs(values))
    }
}

/// The periodic Gaussian `amplitude * exp(-d(x, center)^2 / (2 * width^2))`.
pub fn periodic_gaussian(x: f64, center: f64, width: f64, amplitude: f64) -> f64 {
    assert!(width > 0.0, "Gaussian width must be positive");
    let mut distance = (x - center + PI).rem_euclid(2.0 * PI) - PI;
    if distance == -PI {
        distance = PI;
    }
    amplitude * (-distance * distance / (2.0 * width * width)).exp()
}

/// Exact continuum evolution of `cos(k x)` under the advection-diffusion equation.
pub fn exact_wave(x: f64, time: f64, c: f64, nu: f64, k: i32) -> f64 {
    let wavenumber = k as f64;
    (wavenumber * (x - c * time)).cos() * (-nu * wavenumber * wavenumber * time).exp()
}

/// Exact Fourier-truncated evolution of the sampled periodic Gaussian.
pub fn exact_periodic_gaussian(
    grid: &[f64],
    time: f64,
    c: f64,
    nu: f64,
    center: f64,
    sigma: f64,
) -> Vec<f64> {
    let mut spectrum = dft(&grid
        .iter()
        .map(|&x| periodic_gaussian(x, center, sigma, 1.0))
        .collect::<Vec<_>>());
    let n = grid.len();
    for (index, coefficient) in spectrum.iter_mut().enumerate() {
        let signed_index = if index <= n / 2 {
            index as isize
        } else {
            index as isize - n as isize
        };
        let k = signed_index as f64;
        let decay = (-nu * k * k * time).exp();
        let angle = -c * k * time;
        *coefficient *= Complex::new(decay * angle.cos(), decay * angle.sin());
    }
    inverse_dft(&spectrum)
}

fn centered_derivative(values: &[f64], length: f64, order: usize) -> Vec<f64> {
    let n = values.len();
    let dx = length / n as f64;
    (0..n)
        .map(|j| match order {
            1 => (values[(j + 1) % n] - values[(j + n - 1) % n]) / (2.0 * dx),
            2 => (values[(j + 1) % n] - 2.0 * values[j] + values[(j + n - 1) % n]) / (dx * dx),
            _ => panic!("only first and second derivatives are supported"),
        })
        .collect()
}

fn fourier_derivative(values: &[f64], length: f64, order: usize) -> Vec<f64> {
    let n = values.len();
    let mut spectrum = dft(values);
    for (index, coefficient) in spectrum.iter_mut().enumerate() {
        let signed_index = if index <= n / 2 {
            index as isize
        } else {
            index as isize - n as isize
        };
        let k = 2.0 * PI * signed_index as f64 / length;
        let multiplier = match order {
            1 if n.is_multiple_of(2) && index == n / 2 => Complex::new(0.0, 0.0),
            1 => Complex::new(0.0, k),
            2 => Complex::new(-k * k, 0.0),
            _ => panic!("only first and second derivatives are supported"),
        };
        *coefficient *= multiplier;
    }
    inverse_dft(&spectrum)
}

fn dft(values: &[f64]) -> Vec<Complex> {
    let n = values.len();
    (0..n)
        .map(|frequency| {
            (0..n).fold(Complex::new(0.0, 0.0), |sum, sample| {
                let angle = -2.0 * PI * frequency as f64 * sample as f64 / n as f64;
                let factor = Complex::new(angle.cos(), angle.sin());
                Complex::new(
                    sum.re + values[sample] * factor.re,
                    sum.im + values[sample] * factor.im,
                )
            })
        })
        .collect()
}

fn inverse_dft(spectrum: &[Complex]) -> Vec<f64> {
    let n = spectrum.len();
    (0..n)
        .map(|sample| {
            spectrum
                .iter()
                .enumerate()
                .map(|(frequency, coefficient)| {
                    let angle = 2.0 * PI * frequency as f64 * sample as f64 / n as f64;
                    coefficient.re * angle.cos() - coefficient.im * angle.sin()
                })
                .sum::<f64>()
                / n as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn advance<I: Integrator + ?Sized>(
        integrator: &I,
        mut state: Vec<f64>,
        dt: f64,
        steps: usize,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let mut time = 0.0;
        for _ in 0..steps {
            state = integrator.step(&state, time, dt, rate);
            time += dt;
        }
        state
    }

    #[test]
    fn euler_solves_known_ode() {
        let value = ForwardEuler.step(&[1.0], 0.0, 0.1, &|_, y| vec![y[0]]);
        assert!((value[0] - 1.1).abs() < 1e-14);
    }

    #[test]
    fn midpoint_solves_known_ode() {
        let value = Midpoint.step(&[1.0], 0.0, 0.1, &|_, y| vec![y[0]]);
        assert!((value[0] - 1.105).abs() < 1e-14);
    }

    #[test]
    fn rk4_solves_known_ode() {
        let value = Rk4.step(&[1.0], 0.0, 0.1, &|_, y| vec![y[0]]);
        assert!((value[0] - 1.1051708333333333).abs() < 1e-14);
    }

    #[test]
    fn fourier_derivative_represents_waves() {
        let n = 32;
        let grid: Vec<_> = (0..n).map(|j| 2.0 * PI * j as f64 / n as f64).collect();
        let values: Vec<_> = grid.iter().map(|&x| (3.0 * x).sin()).collect();
        let derivative = fourier_derivative(&values, 2.0 * PI, 1);
        assert!(derivative
            .iter()
            .zip(grid)
            .all(|(&actual, x)| (actual - 3.0 * (3.0 * x).cos()).abs() < 1e-12));
    }

    #[test]
    fn nyquist_first_derivative_is_zero() {
        let n = 16;
        let values: Vec<_> = (0..n)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!(fourier_derivative(&values, 2.0 * PI, 1)
            .iter()
            .all(|value| value.abs() < 1e-12));
        assert!(fourier_derivative(&values, 2.0 * PI, 2)
            .iter()
            .zip(&values)
            .all(|(value, &sample)| { (value + (n as f64 / 2.0).powi(2) * sample).abs() < 1e-10 }));
    }

    #[test]
    fn centered_difference_wraps_periodically() {
        let values = [0.0, 1.0, 4.0, 9.0];
        let first = centered_derivative(&values, 4.0, 1);
        assert!((first[0] - (1.0 - 9.0) / 2.0).abs() < 1e-14);
        let second = centered_derivative(&values, 4.0, 2);
        assert!((second[0] - (1.0 - 0.0 + 9.0) / 1.0).abs() < 1e-14);
    }

    #[test]
    fn integrators_match_exact_single_wave_evolution() {
        let solver = AdvectionDiffusion1D::new(32, 0.7, 0.08, SpatialDerivative::Fourier);
        let k = 3;
        let initial: Vec<_> = solver
            .grid()
            .into_iter()
            .map(|x| exact_wave(x, 0.0, solver.c, solver.nu, k))
            .collect();
        let rate = |_: f64, state: &[f64]| solver.rhs(state);
        for (integrator, tolerance) in [
            (&ForwardEuler as &dyn Integrator, 2e-4),
            (&Midpoint as &dyn Integrator, 2e-7),
            (&Rk4 as &dyn Integrator, 2e-11),
        ] {
            let final_state = advance(integrator, initial.clone(), 0.0005, 200, &rate);
            let expected: Vec<_> = solver
                .grid()
                .into_iter()
                .map(|x| exact_wave(x, 0.1, solver.c, solver.nu, k))
                .collect();
            let error = final_state
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                error < tolerance,
                "error {error} too large for tolerance {tolerance}"
            );
        }
    }
}
