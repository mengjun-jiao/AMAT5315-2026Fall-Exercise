use md::physics::{PhysicalModel, accelerations, energies};
fn model() -> PhysicalModel {
    PhysicalModel::PeriodicShiftedLennardJones {
        box_size: [8., 6.],
        rc: 2.5,
    }
}
#[test]
fn cutoff_is_continuous_from_inside_without_force_shift() {
    let m = model();
    let (u, f) = m.pair(2.5 - 1e-8).unwrap();
    assert!(u.abs() < 1e-9);
    assert!((f - md::force(2.5 - 1e-8)).abs() < 1e-12);
    assert!(f.abs() > 0.03);
    assert_eq!(m.pair(2.5).unwrap(), (0., 0.));
    assert_eq!(m.pair(2.6).unwrap(), (0., 0.));
    assert!(PhysicalModel::OpenLennardJones.pair(3.).unwrap().1.abs() > 0.);
}
#[test]
fn rectangular_minimum_image_and_multiple_wraps() {
    assert_eq!(model().displacement([7.5, 5.5], [0.5, 0.5]), [-1., -1.]);
    assert_eq!(model().wrap([-16.5, 18.5]), [7.5, 0.5]);
    assert_eq!(model().wrap([8., 6.]), [0., 0.]);
    assert_eq!(PhysicalModel::OpenLennardJones.wrap([-1., 8.]), [-1., 8.]);
}
#[test]
fn pair_energy_counted_once_and_total_internal_force_zero() {
    let (u, k) = energies(&[[0., 0.], [7., 0.]], &[[1., 2.], [-1., -2.]], &model()).unwrap();
    assert!((u + md::energy(2.5)).abs() < 1e-12);
    assert_eq!(k, 5.);
    let a = accelerations(&[[0., 0.], [1.2, 0.], [0.4, 1.3]], &model()).unwrap();
    for axis in 0..2 {
        assert!(a.iter().map(|v| v[axis]).sum::<f64>().abs() < 1e-12);
    }
}
#[test]
fn rejects_invalid_models_states_and_periodic_coincidence() {
    assert!(
        PhysicalModel::PeriodicShiftedLennardJones {
            box_size: [5., 6.],
            rc: 2.5
        }
        .validate()
        .is_err()
    );
    assert!(accelerations(&[[0., 0.], [8., 0.]], &model()).is_err());
    assert!(energies(&[[0., 0.], [1., 0.]], &[[0.; 2]], &model()).is_err());
    assert!(model().pair(0.).is_err());
    assert!(model().pair(f64::NAN).is_err());
}
