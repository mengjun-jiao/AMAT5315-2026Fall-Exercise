//! Reusable time integrators and a one-dimensional periodic advection-diffusion solver.

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
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

/// Two-dimensional Fourier operators and the pseudospectral vorticity solver.
pub mod fluid {
    use super::{Complex, Integrator, PI};

    /// A periodic square spectral grid for the two-dimensional vorticity equation.
    #[derive(Clone, Copy, Debug)]
    pub struct VorticitySolver {
        pub n: usize,
        pub nu: f64,
    }

    impl VorticitySolver {
        pub fn new(n: usize, nu: f64) -> Self {
            assert!(
                n >= 4 && n.is_power_of_two(),
                "n must be a power of two at least 4"
            );
            Self { n, nu }
        }

        /// Evaluate the vorticity right-hand side, rebuilding velocity at each call.
        pub fn rate(&self, omega: &[f64]) -> Vec<f64> {
            assert_eq!(omega.len(), self.n * self.n, "field length must equal n*n");
            let mut omega_hat = real_to_spectrum(omega, self.n);
            two_thirds_mask(&mut omega_hat, self.n);
            let (u_hat, v_hat) = velocity_from_spectrum(&omega_hat, self.n);
            let omega_x = spectrum_to_real(
                &multiply_by_wavenumber(&omega_hat, self.n, true, false),
                self.n,
            );
            let omega_y = spectrum_to_real(
                &multiply_by_wavenumber(&omega_hat, self.n, false, true),
                self.n,
            );
            let u = spectrum_to_real(&u_hat, self.n);
            let v = spectrum_to_real(&v_hat, self.n);
            let advection: Vec<_> = u
                .iter()
                .zip(&omega_x)
                .zip(v.iter().zip(&omega_y))
                .map(|((&u, &omega_x), (&v, &omega_y))| u * omega_x + v * omega_y)
                .collect();
            let mut advection_hat = real_to_spectrum(&advection, self.n);
            two_thirds_mask(&mut advection_hat, self.n);
            for (index, coefficient) in omega_hat.iter().enumerate() {
                let (kx, ky) = mode(index, self.n);
                advection_hat[index].re =
                    -advection_hat[index].re - self.nu * (kx * kx + ky * ky) * coefficient.re;
                advection_hat[index].im =
                    -advection_hat[index].im - self.nu * (kx * kx + ky * ky) * coefficient.im;
            }
            spectrum_to_real(&advection_hat, self.n)
        }

        pub fn step<I: Integrator + ?Sized>(
            &self,
            integrator: &I,
            omega: &[f64],
            time: f64,
            dt: f64,
        ) -> Vec<f64> {
            let next = integrator.step(omega, time, dt, &|_, state| self.rate(state));
            project_real(&next, self.n)
        }
    }

    pub fn grid(n: usize) -> Vec<f64> {
        (0..n * n)
            .map(|index| 2.0 * PI * (index % n) as f64 / n as f64)
            .collect()
    }

    pub fn mode(index: usize, n: usize) -> (f64, f64) {
        let kx = index % n;
        let ky = index / n;
        let signed = |k: usize| {
            if k <= n / 2 {
                k as isize
            } else {
                k as isize - n as isize
            }
        };
        (signed(kx) as f64, signed(ky) as f64)
    }

    pub fn cutoff(n: usize) -> usize {
        n / 3
    }

    pub fn two_thirds_mask(spectrum: &mut [Complex], n: usize) {
        assert_eq!(spectrum.len(), n * n);
        let cut = cutoff(n) as f64;
        for (index, coefficient) in spectrum.iter_mut().enumerate() {
            let (kx, ky) = mode(index, n);
            if kx.abs() > cut || ky.abs() > cut {
                *coefficient = Complex::new(0.0, 0.0);
            }
        }
    }

    pub fn real_to_spectrum(values: &[f64], n: usize) -> Vec<Complex> {
        assert_eq!(values.len(), n * n);
        let mut spectrum: Vec<_> = values
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        fft2(&mut spectrum, n, false);
        spectrum
    }

    pub fn spectrum_to_real(spectrum: &[Complex], n: usize) -> Vec<f64> {
        assert_eq!(spectrum.len(), n * n);
        let mut values = spectrum.to_vec();
        fft2(&mut values, n, true);
        values.into_iter().map(|value| value.re).collect()
    }

    pub fn dx(values: &[f64], n: usize) -> Vec<f64> {
        derivative(values, n, true, false)
    }
    pub fn dy(values: &[f64], n: usize) -> Vec<f64> {
        derivative(values, n, false, true)
    }
    pub fn dxx(values: &[f64], n: usize) -> Vec<f64> {
        second_derivative(values, n, true)
    }
    pub fn dyy(values: &[f64], n: usize) -> Vec<f64> {
        second_derivative(values, n, false)
    }

    pub fn dxdy(values: &[f64], n: usize) -> Vec<f64> {
        let mut spectrum = real_to_spectrum(values, n);
        for (index, coefficient) in spectrum.iter_mut().enumerate() {
            let (kx, ky) = mode(index, n);
            *coefficient *= Complex::new(-kx * ky, 0.0);
        }
        spectrum_to_real(&spectrum, n)
    }

    pub fn laplacian(values: &[f64], n: usize) -> Vec<f64> {
        let mut spectrum = real_to_spectrum(values, n);
        for (index, coefficient) in spectrum.iter_mut().enumerate() {
            let (kx, ky) = mode(index, n);
            *coefficient *= Complex::new(-(kx * kx + ky * ky), 0.0);
        }
        spectrum_to_real(&spectrum, n)
    }

    pub fn poisson_streamfunction(omega: &[f64], n: usize) -> Vec<f64> {
        let mut spectrum = real_to_spectrum(omega, n);
        for (index, coefficient) in spectrum.iter_mut().enumerate() {
            let (kx, ky) = mode(index, n);
            let denominator = kx * kx + ky * ky;
            if denominator == 0.0 {
                *coefficient = Complex::new(0.0, 0.0);
            } else {
                coefficient.re /= denominator;
                coefficient.im /= denominator;
            }
        }
        spectrum_to_real(&spectrum, n)
    }

    pub fn vorticity_to_velocity(omega: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut omega_hat = real_to_spectrum(omega, n);
        two_thirds_mask(&mut omega_hat, n);
        let (u_hat, v_hat) = velocity_from_spectrum(&omega_hat, n);
        (spectrum_to_real(&u_hat, n), spectrum_to_real(&v_hat, n))
    }

    pub fn velocity_to_vorticity(u: &[f64], v: &[f64], n: usize) -> Vec<f64> {
        assert_eq!(u.len(), n * n);
        assert_eq!(v.len(), n * n);
        let mut u_hat = real_to_spectrum(u, n);
        let v_hat = real_to_spectrum(v, n);
        for (index, coefficient) in u_hat.iter_mut().enumerate() {
            let (kx, ky) = mode(index, n);
            let dv_dx = Complex::new(-kx * v_hat[index].im, kx * v_hat[index].re);
            let du_dy = Complex::new(-ky * coefficient.im, ky * coefficient.re);
            *coefficient = Complex::new(dv_dx.re - du_dy.re, dv_dx.im - du_dy.im);
        }
        spectrum_to_real(&u_hat, n)
    }

    pub fn divergence(u: &[f64], v: &[f64], n: usize) -> Vec<f64> {
        let mut u_hat = real_to_spectrum(u, n);
        let v_hat = real_to_spectrum(v, n);
        for (index, coefficient) in u_hat.iter_mut().enumerate() {
            let (kx, ky) = mode(index, n);
            let du_dx = Complex::new(-kx * coefficient.im, kx * coefficient.re);
            let dv_dy = Complex::new(-ky * v_hat[index].im, ky * v_hat[index].re);
            *coefficient = Complex::new(du_dx.re + dv_dy.re, du_dx.im + dv_dy.im);
        }
        spectrum_to_real(&u_hat, n)
    }

    pub fn energy(u: &[f64], v: &[f64]) -> f64 {
        0.5 * u.iter().zip(v).map(|(&u, &v)| u * u + v * v).sum::<f64>() / u.len() as f64
    }

    pub fn enstrophy(omega: &[f64]) -> f64 {
        0.5 * omega.iter().map(|&value| value * value).sum::<f64>() / omega.len() as f64
    }

    pub fn project_real(values: &[f64], n: usize) -> Vec<f64> {
        let mut spectrum = real_to_spectrum(values, n);
        two_thirds_mask(&mut spectrum, n);
        spectrum_to_real(&spectrum, n)
    }

    pub fn centered_dx(values: &[f64], n: usize) -> Vec<f64> {
        centered_derivative(values, n, 1)
    }

    pub fn centered_dy(values: &[f64], n: usize) -> Vec<f64> {
        centered_derivative(values, n, 2)
    }

    pub fn centered_dxx(values: &[f64], n: usize) -> Vec<f64> {
        centered_derivative(values, n, 3)
    }

    pub fn centered_dyy(values: &[f64], n: usize) -> Vec<f64> {
        centered_derivative(values, n, 4)
    }

    pub fn centered_dxdy(values: &[f64], n: usize) -> Vec<f64> {
        let dx = centered_dx(values, n);
        centered_dy(&dx, n)
    }

    pub fn centered_laplacian(values: &[f64], n: usize) -> Vec<f64> {
        let dxx = centered_dxx(values, n);
        let dyy = centered_dyy(values, n);
        dxx.iter().zip(dyy).map(|(&x, y)| x + y).collect()
    }

    pub fn random_velocity(
        n: usize,
        seed: u64,
        k_min: usize,
        k_max: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        assert!(k_min > 0 && k_min <= k_max && k_max <= n / 3);
        let mut state = seed;
        let mut omega_hat = vec![Complex::new(0.0, 0.0); n * n];
        for kx in -(k_max as isize)..=(k_max as isize) {
            for ky in -(k_max as isize)..=(k_max as isize) {
                let radius = ((kx * kx + ky * ky) as f64).sqrt();
                if radius < k_min as f64
                    || radius > k_max as f64
                    || (kx == 0 && ky == 0)
                    || kx < 0
                    || (kx == 0 && ky < 0)
                {
                    continue;
                }
                let phase = 2.0 * PI * next_unit(&mut state);
                let index =
                    (ky.rem_euclid(n as isize) as usize) * n + kx.rem_euclid(n as isize) as usize;
                let partner = ((-ky).rem_euclid(n as isize) as usize) * n
                    + (-kx).rem_euclid(n as isize) as usize;
                omega_hat[index] = Complex::new(phase.cos(), phase.sin());
                omega_hat[partner] = Complex::new(phase.cos(), -phase.sin());
            }
        }
        let omega = spectrum_to_real(&omega_hat, n);
        let (mut u, mut v) = vorticity_to_velocity(&omega, n);
        let scale = (0.5 / energy(&u, &v)).sqrt();
        for value in &mut u {
            *value *= scale;
        }
        for value in &mut v {
            *value *= scale;
        }
        (u, v)
    }

    fn derivative(values: &[f64], n: usize, x: bool, y: bool) -> Vec<f64> {
        spectrum_to_real(
            &multiply_by_wavenumber(&real_to_spectrum(values, n), n, x, y),
            n,
        )
    }

    fn second_derivative(values: &[f64], n: usize, x: bool) -> Vec<f64> {
        let mut spectrum = real_to_spectrum(values, n);
        for (index, coefficient) in spectrum.iter_mut().enumerate() {
            let (kx, ky) = mode(index, n);
            let k = if x { kx } else { ky };
            *coefficient *= Complex::new(-k * k, 0.0);
        }
        spectrum_to_real(&spectrum, n)
    }

    fn multiply_by_wavenumber(spectrum: &[Complex], n: usize, x: bool, y: bool) -> Vec<Complex> {
        spectrum
            .iter()
            .enumerate()
            .map(|(index, coefficient)| {
                let (kx, ky) = mode(index, n);
                let k = if x { kx } else { ky };
                let _ = y;
                Complex::new(-k * coefficient.im, k * coefficient.re)
            })
            .collect()
    }

    fn centered_derivative(values: &[f64], n: usize, order: usize) -> Vec<f64> {
        assert_eq!(values.len(), n * n);
        let dx = 2.0 * PI / n as f64;
        let at = |row: usize, col: usize| values[(row % n) * n + col % n];
        (0..n * n)
            .map(|index| {
                let row = index / n;
                let col = index % n;
                match order {
                    1 => (at(row, col + 1) - at(row, col + n - 1)) / (2.0 * dx),
                    2 => (at(row + 1, col) - at(row + n - 1, col)) / (2.0 * dx),
                    3 => (at(row, col + 1) - 2.0 * at(row, col) + at(row, col + n - 1)) / (dx * dx),
                    4 => (at(row + 1, col) - 2.0 * at(row, col) + at(row + n - 1, col)) / (dx * dx),
                    _ => unreachable!(),
                }
            })
            .collect()
    }

    fn next_unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn velocity_from_spectrum(omega_hat: &[Complex], n: usize) -> (Vec<Complex>, Vec<Complex>) {
        let mut u_hat = vec![Complex::new(0.0, 0.0); n * n];
        let mut v_hat = u_hat.clone();
        for (index, omega) in omega_hat.iter().enumerate() {
            let (kx, ky) = mode(index, n);
            let denominator = kx * kx + ky * ky;
            if denominator != 0.0 {
                let psi = Complex::new(omega.re / denominator, omega.im / denominator);
                u_hat[index] = Complex::new(-ky * psi.im, ky * psi.re);
                v_hat[index] = Complex::new(kx * psi.im, -kx * psi.re);
            }
        }
        (u_hat, v_hat)
    }

    fn fft2(values: &mut [Complex], n: usize, inverse: bool) {
        for row in 0..n {
            fft1d(&mut values[row * n..(row + 1) * n], inverse);
        }
        let mut column = vec![Complex::new(0.0, 0.0); n];
        for col in 0..n {
            for row in 0..n {
                column[row] = values[row * n + col];
            }
            fft1d(&mut column, inverse);
            for row in 0..n {
                values[row * n + col] = column[row];
            }
        }
    }

    fn fft1d(values: &mut [Complex], inverse: bool) {
        let n = values.len();
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                values.swap(i, j);
            }
        }
        let mut length = 2;
        while length <= n {
            let sign = if inverse { 1.0 } else { -1.0 };
            let angle = sign * 2.0 * PI / length as f64;
            let base = Complex::new(angle.cos(), angle.sin());
            for start in (0..n).step_by(length) {
                let mut factor = Complex::new(1.0, 0.0);
                for offset in 0..length / 2 {
                    let even = values[start + offset];
                    let odd = values[start + offset + length / 2];
                    let product = Complex::new(
                        odd.re * factor.re - odd.im * factor.im,
                        odd.re * factor.im + odd.im * factor.re,
                    );
                    values[start + offset] =
                        Complex::new(even.re + product.re, even.im + product.im);
                    values[start + offset + length / 2] =
                        Complex::new(even.re - product.re, even.im - product.im);
                    factor *= base;
                }
            }
            length <<= 1;
        }
        if inverse {
            for value in values {
                value.re /= n as f64;
                value.im /= n as f64;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn max_error(actual: &[f64], expected: impl Fn(usize) -> f64) -> f64 {
            actual
                .iter()
                .enumerate()
                .map(|(i, &value)| (value - expected(i)).abs())
                .fold(0.0, f64::max)
        }

        #[test]
        fn derivatives_match_two_dimensional_wave() {
            let n = 32;
            let values: Vec<_> = (0..n * n)
                .map(|index| {
                    let x = 2.0 * PI * (index % n) as f64 / n as f64;
                    let y = 2.0 * PI * (index / n) as f64 / n as f64;
                    (3.0 * x).sin() * (2.0 * y).cos()
                })
                .collect();
            let dx_error = max_error(&dx(&values, n), |i| {
                let x = 2.0 * PI * (i % n) as f64 / n as f64;
                let y = 2.0 * PI * (i / n) as f64 / n as f64;
                3.0 * (3.0 * x).cos() * (2.0 * y).cos()
            });
            let dxx_error = max_error(&dxx(&values, n), |i| -9.0 * values[i]);
            let dxdy_error = max_error(&dxdy(&values, n), |i| {
                let x = 2.0 * PI * (i % n) as f64 / n as f64;
                let y = 2.0 * PI * (i / n) as f64 / n as f64;
                -6.0 * (3.0 * x).cos() * (2.0 * y).sin()
            });
            let lap_error = max_error(&laplacian(&values, n), |i| -13.0 * values[i]);
            println!("2D Fourier errors: dx={dx_error:.3e}, dxx={dxx_error:.3e}, dxdy={dxdy_error:.3e}, laplacian={lap_error:.3e}");
            assert!(
                dx_error < 1e-11 && dxx_error < 1e-10 && dxdy_error < 1e-10 && lap_error < 1e-10
            );
        }

        #[test]
        fn recovery_is_divergence_free_and_has_correct_vorticity() {
            let n = 32;
            let omega: Vec<_> = (0..n * n)
                .map(|index| {
                    let x = 2.0 * PI * (index % n) as f64 / n as f64;
                    let y = 2.0 * PI * (index / n) as f64 / n as f64;
                    (2.0 * x).sin() * (3.0 * y).cos()
                })
                .collect();
            let (u, v) = vorticity_to_velocity(&omega, n);
            let divergence_error = divergence(&u, &v, n)
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            let reconstructed = velocity_to_vorticity(&u, &v, n);
            let vorticity_error = reconstructed
                .iter()
                .zip(&omega)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            println!("Recovery errors: divergence={divergence_error:.3e}, vorticity={vorticity_error:.3e}");
            assert!(divergence_error < 1e-11 && vorticity_error < 1e-11);
            assert!(u
                .iter()
                .zip(&v)
                .any(|(&u, &v)| u.abs() > 1e-8 && v.abs() > 1e-8));
        }

        #[test]
        fn mask_removes_all_modes_outside_two_thirds_square() {
            let n = 32;
            let mut spectrum = vec![Complex::new(1.0, -2.0); n * n];
            two_thirds_mask(&mut spectrum, n);
            for (index, coefficient) in spectrum.iter().enumerate() {
                let (kx, ky) = mode(index, n);
                if kx.abs() > cutoff(n) as f64 || ky.abs() > cutoff(n) as f64 {
                    assert_eq!((coefficient.re, coefficient.im), (0.0, 0.0));
                }
            }
            assert_eq!(spectrum[n / 2].re, 0.0);
        }

        #[test]
        fn taylor_green_has_expected_invariants_and_zero_advection() {
            let n = 32;
            let values: Vec<_> = (0..n * n)
                .map(|index| {
                    let x = 2.0 * PI * (index % n) as f64 / n as f64;
                    let y = 2.0 * PI * (index / n) as f64 / n as f64;
                    (x.cos() * y.sin(), -x.sin() * y.cos())
                })
                .collect();
            let u: Vec<_> = values.iter().map(|pair| pair.0).collect();
            let v: Vec<_> = values.iter().map(|pair| pair.1).collect();
            let omega = velocity_to_vorticity(&u, &v, n);
            let (recovered_u, recovered_v) = vorticity_to_velocity(&omega, n);
            let advection = {
                let ox = dx(&omega, n);
                let oy = dy(&omega, n);
                recovered_u
                    .iter()
                    .zip(ox)
                    .zip(recovered_v.iter().zip(oy))
                    .map(|((&u, ox), (&v, oy))| u * ox + v * oy)
                    .fold(0.0, f64::max)
            };
            println!(
                "Taylor-Green diagnostics: E={:.15}, Z={:.15}, max advection={advection:.3e}",
                energy(&recovered_u, &recovered_v),
                enstrophy(&omega)
            );
            assert!((energy(&recovered_u, &recovered_v) - 0.25).abs() < 1e-12);
            assert!((enstrophy(&omega) - 0.5).abs() < 1e-12);
            assert!(advection < 1e-11);
        }

        #[test]
        fn random_field_is_independent_of_grid_resolution() {
            let (u64, v64) = random_velocity(64, 2026, 2, 6);
            let (u128, v128) = random_velocity(128, 2026, 2, 6);
            let omega64 = velocity_to_vorticity(&u64, &v64, 64);
            let omega128 = velocity_to_vorticity(&u128, &v128, 128);
            let mut u_error: f64 = 0.0;
            let mut v_error: f64 = 0.0;
            let mut omega_error: f64 = 0.0;
            for row in 0..64 {
                for col in 0..64 {
                    let i64 = row * 64 + col;
                    let i128 = (2 * row) * 128 + 2 * col;
                    u_error = u_error.max((u64[i64] - u128[i128]).abs());
                    v_error = v_error.max((v64[i64] - v128[i128]).abs());
                    omega_error = omega_error.max((omega64[i64] - omega128[i128]).abs());
                }
            }
            println!("Random cross-resolution errors: u={u_error:.3e}, v={v_error:.3e}, omega={omega_error:.3e}");
            assert!(u_error < 1e-13 && v_error < 1e-13 && omega_error < 1e-12);
        }
    }
}

/// Minimal JSON interchange for the field and fluid command-line binaries.
pub mod field_io {
    #[derive(Debug)]
    pub struct FieldData {
        pub case: String,
        pub n: usize,
        pub seed: u64,
        pub k_band: String,
        pub u: Vec<f64>,
        pub v: Vec<f64>,
    }

    pub fn json_number_array(values: &[f64], decimals: Option<usize>) -> String {
        let body = values
            .iter()
            .map(|value| match decimals {
                Some(decimals) => format!("{value:.decimals$}"),
                None => format!("{value:.17}"),
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("[{body}]")
    }

    pub fn field_json(
        case: &str,
        n: usize,
        seed: u64,
        k_band: &str,
        u: &[f64],
        v: &[f64],
    ) -> String {
        format!("{{\"case\":\"{case}\",\"n\":{n},\"seed\":{seed},\"k_band\":{k_band},\"u\":{},\"v\":{}}}", json_number_array(u, None), json_number_array(v, None))
    }

    pub fn parse_field_json(input: &str) -> Result<FieldData, String> {
        let case = raw_value(input, "case")?.trim_matches('"').to_owned();
        let n = raw_value(input, "n")?
            .parse::<usize>()
            .map_err(|_| "n must be an integer")?;
        let seed = raw_value(input, "seed")?
            .parse::<u64>()
            .map_err(|_| "seed must be an integer")?;
        let k_band = raw_value(input, "k_band")?.to_owned();
        let u = parse_array(raw_value(input, "u")?)?;
        let v = parse_array(raw_value(input, "v")?)?;
        if u.len() != n * n || v.len() != n * n {
            return Err("u and v must each contain n*n values".to_owned());
        }
        Ok(FieldData {
            case,
            n,
            seed,
            k_band,
            u,
            v,
        })
    }

    fn raw_value<'a>(input: &'a str, key: &str) -> Result<&'a str, String> {
        let marker = format!("\"{key}\"");
        let start = input
            .find(&marker)
            .ok_or_else(|| format!("missing JSON field {key}"))?
            + marker.len();
        let after_colon = input[start..]
            .find(':')
            .ok_or_else(|| format!("missing colon after {key}"))?
            + start
            + 1;
        let absolute =
            after_colon + (input[after_colon..].len() - input[after_colon..].trim_start().len());
        let bytes = input.as_bytes();
        if bytes[absolute] == b'"' {
            let end = input[absolute + 1..]
                .find('"')
                .ok_or_else(|| format!("unterminated string field {key}"))?
                + absolute
                + 2;
            Ok(&input[absolute..end])
        } else if bytes[absolute] == b'[' || bytes[absolute] == b'{' {
            let open = bytes[absolute];
            let close = if open == b'[' { b']' } else { b'}' };
            let mut depth = 0;
            for (index, byte) in bytes[absolute..].iter().enumerate() {
                if *byte == open {
                    depth += 1;
                }
                if *byte == close {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(&input[absolute..absolute + index + 1]);
                    }
                }
            }
            Err(format!("unterminated structured field {key}"))
        } else {
            let end = input[absolute..]
                .find([',', '}'])
                .unwrap_or(input.len() - absolute)
                + absolute;
            Ok(input[absolute..end].trim())
        }
    }

    fn parse_array(raw: &str) -> Result<Vec<f64>, String> {
        let body = raw
            .trim()
            .strip_prefix('[')
            .and_then(|value| value.strip_suffix(']'))
            .ok_or_else(|| "JSON field must be an array".to_owned())?;
        if body.trim().is_empty() {
            return Ok(Vec::new());
        }
        body.split(',')
            .map(|value| {
                value
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| "array contains a non-number".to_owned())
            })
            .collect()
    }
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
