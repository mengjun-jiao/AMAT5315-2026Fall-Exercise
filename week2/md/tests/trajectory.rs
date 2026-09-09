use md::fluid::{RunConfig, evolve, initialize};
use md::{Integrator, VelocityVerlet};
#[test]
fn equilibration_rescales_at_50_and_production_does_not() {
    let c = RunConfig {
        eq_steps: 50,
        steps: 1,
        sample_every: 1,
        ..Default::default()
    };
    let mut expected = initialize(&c).unwrap();
    for _ in 0..50 {
        VelocityVerlet.step(&mut expected, c.dt);
    }
    expected.rescale_temperature(c.temperature).unwrap();
    VelocityVerlet.step(&mut expected, c.dt);
    let mut actual = initialize(&c).unwrap();
    let mut frames = Vec::new();
    evolve(&VelocityVerlet, &mut actual, &c, |f| {
        frames.push(f);
        Ok(())
    })
    .unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].step, 1);
    assert_eq!(frames[0].pos, expected.positions());
    assert_eq!(frames[0].vel, expected.velocities());
    let c = RunConfig { eq_steps: 0, ..c };
    let mut actual = initialize(&c).unwrap();
    actual.rescale_temperature(0.2).unwrap();
    let mut expected = initialize(&c).unwrap();
    expected.rescale_temperature(0.2).unwrap();
    VelocityVerlet.step(&mut expected, c.dt);
    evolve(&VelocityVerlet, &mut actual, &c, |_| Ok(())).unwrap();
    assert_eq!(actual.velocities(), expected.velocities());
}
#[test]
fn trajectory_roundtrip_rejects_damage() {
    let tmp = tempfile::tempdir().unwrap();
    let c = RunConfig {
        eq_steps: 0,
        steps: 2,
        sample_every: 1,
        ..Default::default()
    };
    md::trajectory::run_to_directory(&c, tmp.path()).unwrap();
    let (_, f) = md::trajectory::read_trajectory(tmp.path()).unwrap();
    assert_eq!(f.len(), 2);
    std::fs::write(tmp.path().join("traj.jsonl"), "{broken}\n").unwrap();
    assert!(
        md::trajectory::read_trajectory(tmp.path())
            .unwrap_err()
            .contains("line 1")
    );
}
