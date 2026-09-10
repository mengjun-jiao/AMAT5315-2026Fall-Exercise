use md::fluid::{RunConfig, evolve, initialize, lattice};
use md::physics::PhysicalModel;
use md::{Euler, Integrator, System, VelocityVerlet};
#[test]
fn even_square_lattice_sizes_and_invalid_counts() {
    for n in [100, 400, 1600] {
        let (p, b) = lattice(n, 0.8).unwrap();
        assert_eq!(p.len(), n);
        assert!((b[0] * b[1] - n as f64 / 0.8).abs() < 1e-9);
        let a = (2. / (3_f64.sqrt() * 0.8)).sqrt();
        let m = (n as f64).sqrt() as usize;
        assert!((p[m][0] - 0.5 * a).abs() < 1e-12);
        assert!(
            p.iter()
                .all(|x| x[0] >= 0. && x[0] < b[0] && x[1] >= 0. && x[1] < b[1])
        );
    }
    for n in [0, 99, 121, 399] {
        assert!(lattice(n, 0.8).is_err());
    }
}
#[test]
fn seeded_velocities_have_target_temperature_and_zero_momentum() {
    let c = RunConfig::default();
    let s = initialize(&c).unwrap();
    let again = initialize(&c).unwrap();
    assert_eq!(s.velocities(), again.velocities());
    assert!((2. * s.kinetic_energy() / (2 * c.n - 2) as f64 - c.temperature).abs() < 1e-12);
    for axis in 0..2 {
        assert!(s.velocities().iter().map(|v| v[axis]).sum::<f64>().abs() < 1e-10);
    }
    let mut different = c.clone();
    different.seed += 1;
    assert_ne!(s.velocities(), initialize(&different).unwrap().velocities());
}
#[test]
fn wrapping_at_full_step_preserves_velocity_for_both_integrators() {
    fn check<I: Integrator>(i: I) {
        let mut s = System::with_model(
            vec![[7.99, 1.], [3.99, 1.]],
            vec![[2., 0.]; 2],
            PhysicalModel::PeriodicShiftedLennardJones {
                box_size: [8., 6.],
                rc: 2.5,
            },
        )
        .unwrap();
        i.step(&mut s, 0.01);
        assert!((s.positions()[0][0] - 0.01).abs() < 1e-12);
        assert_eq!(s.velocities(), &[[2., 0.]; 2]);
    }
    check(Euler);
    check(VelocityVerlet);
}
#[test]
fn config_and_rescaling_reject_invalid_values() {
    let mut c = RunConfig::default();
    c.sample_every = 0;
    assert!(c.validate().is_err());
    c = RunConfig::default();
    c.temperature = f64::NAN;
    assert!(c.validate().is_err());
    c = RunConfig::default();
    c.rho = 100.;
    assert!(c.validate().is_err());
    let mut s = System::new(vec![[0., 0.], [1.2, 0.]], vec![[0.; 2]; 2]);
    assert!(s.rescale_temperature(0.5).is_err());
}

fn thermo(s: &System) -> f64 {
    2.0 * s.kinetic_energy() / (2 * s.positions().len() - 2) as f64
}

#[test]
fn ramp_targets_are_linear_and_equal_start_temperature_still_rescales() {
    let c = RunConfig {
        eq_steps: 0,
        steps: 100,
        sample_every: 50,
        ramp_to: Some(1.0),
        ..Default::default()
    };
    let mut actual = initialize(&c).unwrap();
    evolve(&VelocityVerlet, &mut actual, &c, |_| Ok(())).unwrap();
    assert!((thermo(&actual) - 1.0).abs() < 1e-12);

    let equal = RunConfig {
        steps: 50,
        sample_every: 50,
        ramp_to: Some(0.5),
        ..c.clone()
    };
    let mut equal_actual = initialize(&equal).unwrap();
    let mut equal_expected = initialize(&equal).unwrap();
    for step in 1..=50 {
        VelocityVerlet.step(&mut equal_expected, equal.dt);
        if step == 50 {
            equal_expected.rescale_temperature(equal.temperature).unwrap();
        }
    }
    evolve(&VelocityVerlet, &mut equal_actual, &equal, |_| Ok(())).unwrap();
    assert_eq!(equal_actual.velocities(), equal_expected.velocities());
}

#[test]
fn ramp_sampling_is_after_scaling_and_final_non_multiple_step_is_not_scaled() {
    let c = RunConfig {
        eq_steps: 0,
        steps: 51,
        sample_every: 50,
        ramp_to: Some(1.0),
        ..Default::default()
    };
    let mut actual = initialize(&c).unwrap();
    let mut frames = Vec::new();
    evolve(&VelocityVerlet, &mut actual, &c, |frame| {
        frames.push(frame);
        Ok(())
    })
    .unwrap();
    assert_eq!(frames.len(), 1);

    let mut expected_at_50 = initialize(&c).unwrap();
    for step in 1..=50 {
        VelocityVerlet.step(&mut expected_at_50, c.dt);
        if step == 50 {
            expected_at_50
                .rescale_temperature(0.5 + 0.5 * 50.0 / 51.0)
                .unwrap();
        }
    }
    assert!((frames[0].e_kin - expected_at_50.kinetic_energy()).abs() < 1e-12);

    let mut expected_at_51 = expected_at_50;
    VelocityVerlet.step(&mut expected_at_51, c.dt);
    assert_eq!(actual.velocities(), expected_at_51.velocities());
}

#[test]
fn no_ramp_formal_steps_match_independent_integrator_steps() {
    let c = RunConfig {
        eq_steps: 0,
        steps: 51,
        sample_every: 50,
        ramp_to: None,
        ..Default::default()
    };
    let mut actual = initialize(&c).unwrap();
    evolve(&VelocityVerlet, &mut actual, &c, |_| Ok(())).unwrap();
    let mut expected = initialize(&c).unwrap();
    for _ in 0..51 {
        VelocityVerlet.step(&mut expected, c.dt);
    }
    assert_eq!(actual.positions(), expected.positions());
    assert_eq!(actual.velocities(), expected.velocities());
}
