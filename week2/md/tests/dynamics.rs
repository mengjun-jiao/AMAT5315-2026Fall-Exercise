use md::{Euler, Integrator, System, VelocityVerlet, simulate};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}");
}

fn initial() -> System {
    System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2])
}

#[test]
fn euler_uses_old_position_velocity_and_acceleration() {
    let mut s = System::new(vec![[0.0, 0.0], [1.0, 0.0]],
                            vec![[0.1, 0.2], [-0.1, -0.2]]);
    let integrator = Euler; // Deliberately not mut: exercises &self.
    integrator.step(&mut s, 0.01);
    for (actual, expected) in s.positions().iter().flatten()
        .zip([0.001, 0.002, 0.999, -0.002]) {
        close(*actual, expected, 1e-12);
    }
    for (actual, expected) in s.velocities().iter().flatten()
        .zip([-0.14, 0.2, 0.14, -0.2]) {
        close(*actual, expected, 1e-12);
    }
}

#[test]
#[should_panic(expected = "positions and velocities must have equal lengths")]
fn rejects_mismatched_lengths() {
    System::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![[0.0; 2]]);
}

#[test]
#[should_panic(expected = "at least two atoms")]
fn rejects_empty_system() { System::new(vec![], vec![]); }

#[test]
#[should_panic(expected = "finite state")]
fn rejects_nonfinite_state() {
    System::new(vec![[f64::NAN, 0.0], [1.0, 0.0]], vec![[0.0; 2]; 2]);
}

#[test]
#[should_panic(expected = "nonzero finite separation")]
fn rejects_coincident_atoms() {
    System::new(vec![[0.0; 2]; 2], vec![[0.0; 2]; 2]);
}

#[test]
fn constructor_initializes_vector_force_and_counts_pair_energy_once() {
    let s = System::new(vec![[0.0, 0.0], [0.6, 0.8]],
                        vec![[1.0, 2.0], [-1.0, -2.0]]);
    for (actual, expected) in s.accelerations().iter().flatten()
        .zip([-14.4, -19.2, 14.4, 19.2]) {
        close(*actual, expected, 1e-10);
    }
    close(s.kinetic_energy(), 5.0, 1e-12);
    close(s.potential_energy(), 0.0, 1e-10);
    let well = System::new(vec![[0.0, 0.0], [2_f64.powf(1.0/6.0), 0.0]],
                           vec![[0.0; 2]; 2]);
    close(well.total_energy(), -1.0, 1e-12);
}

#[test]
fn verlet_uses_initial_cache_and_finishes_velocity_update() {
    let mut s = System::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![[0.0; 2]; 2]);
    let integrator = VelocityVerlet;
    integrator.step(&mut s, 0.01);
    close(s.positions()[0][0], -0.0012, 1e-12);
    close(s.positions()[1][0], 1.0012, 1e-12);
    // Independent check of the new force via U's derivative, test only.
    let r = 1.0024;
    let h = 1e-6;
    let f = -(md::energy(r+h) - md::energy(r-h)) / (2.0*h);
    close(s.accelerations()[0][0], -f, 1e-7);
    close(s.velocities()[0][0], -0.12 - 0.005*f, 1e-9);
    close(s.velocities()[1][0], 0.12 + 0.005*f, 1e-9);
}

#[test]
fn samples_are_recorded_after_complete_steps() {
    let mut s = System::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![[0.0; 2]; 2]);
    let rows = simulate(&Euler, &mut s, 0.01, 1);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].step, 0);
    assert_eq!(rows[1].step, 1);
    close(rows[0].total, 0.0, 1e-12);
    close(rows[1].time, 0.01, 1e-12);
    close(rows[1].kinetic, 0.0576, 1e-12);
    close(rows[1].potential, 0.0, 1e-12);
    close(rows[1].total, 0.0576, 1e-12);
}

#[test]
#[should_panic(expected = "positive finite dt")]
fn euler_rejects_zero_dt() {
    Euler.step(&mut initial(), 0.0);
}

#[test]
#[should_panic(expected = "positive finite dt")]
fn verlet_rejects_nan_dt() {
    VelocityVerlet.step(&mut initial(), f64::NAN);
}

#[test]
#[should_panic(expected = "positive finite dt")]
fn driver_rejects_negative_dt_even_for_zero_steps() {
    simulate(&Euler, &mut initial(), -0.01, 0);
}

#[test]
fn fixed_500_step_energy_acceptance() {
    fn deltas<I: Integrator>(integrator: &I) -> Vec<f64> {
        let mut s = initial();
        let rows = simulate(integrator, &mut s, 0.01, 500);
        assert_eq!(rows.len(), 501);
        close(rows[500].time, 5.0, 1e-12);
        for component in s.positions().iter().chain(s.velocities())
            .chain(s.accelerations()).flatten() { assert!(component.is_finite()); }
        for axis in 0..2 {
            close(s.velocities()[0][axis] + s.velocities()[1][axis], 0.0, 1e-12);
        }
        let e0 = md::energy(1.2);
        close(rows[0].total, e0, 1e-12);
        rows.iter().map(|row| {
            assert!(row.total.is_finite());
            (row.total - e0) / e0.abs()
        }).collect()
    }
    let verlet = deltas(&VelocityVerlet);
    let euler = deltas(&Euler);
    let verlet_max = verlet.iter().map(|d| d.abs()).fold(0.0, f64::max);
    let euler_final = euler[500];
    assert!(verlet_max < 1e-3, "Verlet max(abs(delta))={verlet_max}");
    assert!(euler_final > 0.5, "Euler final delta={euler_final}");
}

