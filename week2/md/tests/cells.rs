use md::neighbors::ForceMethod;
use md::physics::{accelerations_with_method, energies_with_method, PhysicalModel};
use md::{Integrator, System, VelocityVerlet};

fn compare(pos: &[[f64; 2]], box_size: [f64; 2]) {
    let m = PhysicalModel::PeriodicShiftedLennardJones {
        box_size,
        rc: 2.5,
    };
    let v = vec![[0.2, -0.1]; pos.len()];
    let a = accelerations_with_method(pos, &m, ForceMethod::Naive).unwrap();
    let b = accelerations_with_method(pos, &m, ForceMethod::Cells).unwrap();
    assert_eq!(a.len(), pos.len());
    assert_eq!(b.len(), pos.len());
    for (&x, &y) in a.iter().flatten().zip(b.iter().flatten()) {
        assert!(x.is_finite() && y.is_finite());
        assert!((x - y).abs() <= 1e-10 * x.abs().max(1.));
    }
    let (u, k) = energies_with_method(pos, &v, &m, ForceMethod::Naive).unwrap();
    let (w, l) = energies_with_method(pos, &v, &m, ForceMethod::Cells).unwrap();
    assert!(u.is_finite() && w.is_finite());
    assert!((u - w).abs() <= 1e-10 * u.abs().max(1.));
    assert_eq!(k, l);
    for axis in 0..2 {
        let residual = b.iter().map(|x| x[axis]).sum::<f64>().abs();
        let scale = b.iter().map(|x| x[axis].abs()).sum::<f64>().max(1.);
        assert!(residual <= 1e-12 * scale);
    }
}

#[test]
fn perturbed_lattice_and_periodic_pairs_match() {
    let (mut p, b) = md::fluid::lattice(100, 0.8).unwrap();
    for (i, x) in p.iter_mut().enumerate() {
        x[0] += 0.04 * (i as f64 * 1.7).sin();
        x[1] += 0.03 * (i as f64 * 0.9).cos();
    }
    compare(&p, b);
    compare(&[[0.1, 1.], [5.9, 1.], [3., 4.]], [6., 6.]);
    compare(&[[-0.1, 1.], [0.1, 1.], [3., 6.]], [6., 9.]);
}

#[test]
fn cutoff_and_two_cell_boxes_match() {
    for r in [2.5 - 1e-8, 2.5, 2.5 + 1e-8] {
        compare(&[[0., 0.], [r, 0.]], [6., 6.]);
        compare(&[[0., 0.], [r, 0.]], [6., 9.]);
    }
}

#[test]
fn rebuilt_membership_tracks_current_positions() {
    let mut p = [[1., 1.], [3.01, 1.], [10., 8.]];
    compare(&p, [12., 12.]);
    p[1] = [2.99, 1.];
    compare(&p, [12., 12.]);
    p[0] = [-0.1, 1.];
    p[1] = [0.1, 1.];
    compare(&p, [12., 12.]);
}

#[test]
fn open_cells_and_periodic_overlap_are_rejected() {
    assert!(accelerations_with_method(
        &[[0., 0.], [1.2, 0.]],
        &PhysicalModel::OpenLennardJones,
        ForceMethod::Cells
    )
    .is_err());
    let m = PhysicalModel::PeriodicShiftedLennardJones {
        box_size: [6., 6.],
        rc: 2.5,
    };
    assert!(accelerations_with_method(&[[0., 0.], [6., 0.]], &m, ForceMethod::Cells).is_err());
}

#[test]
fn systems_with_naive_and_cells_follow_same_short_trajectory() {
    let model = PhysicalModel::PeriodicShiftedLennardJones {
        box_size: [8., 6.],
        rc: 2.5,
    };
    let positions = vec![[0.1, 1.], [7.9, 1.], [3.99, 4.]];
    let velocities = vec![[0.2, 0.], [-0.1, 0.], [0., 0.1]];
    let mut naive = System::with_model_and_force(
        positions.clone(),
        velocities.clone(),
        model,
        ForceMethod::Naive,
    )
    .unwrap();
    let mut cells = System::with_model_and_force(positions, velocities, model, ForceMethod::Cells)
        .unwrap();
    for _ in 0..10 {
        VelocityVerlet.step(&mut naive, 0.001);
        VelocityVerlet.step(&mut cells, 0.001);
    }
    for (&x, &y) in naive
        .positions()
        .iter()
        .flatten()
        .zip(cells.positions().iter().flatten())
    {
        assert!((x - y).abs() <= 1e-10 * x.abs().max(1.));
    }
    for (&x, &y) in naive
        .velocities()
        .iter()
        .flatten()
        .zip(cells.velocities().iter().flatten())
    {
        assert!((x - y).abs() <= 1e-10 * x.abs().max(1.));
    }
}
