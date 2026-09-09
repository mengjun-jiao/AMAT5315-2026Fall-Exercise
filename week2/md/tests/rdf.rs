use md::rdf::RdfWindow;
#[test]
fn density_normalization_counts_beyond_force_cutoff() {
    let mut w = RdfWindow::new(2, [10., 10.], 5, 20).unwrap();
    let s = w.push(&[[0., 0.], [3., 0.]]).unwrap();
    let expected = 2. / (2. * (2. / 100.) * std::f64::consts::PI * (16. - 9.));
    assert!((s.values[3] - expected).abs() < 1e-12);
    assert_eq!(s.frames, 1);
    assert_eq!(s.centers, vec![0.5, 1.5, 2.5, 3.5, 4.5]);
    let again = w.push(&[[0., 0.], [3., 0.]]).unwrap();
    assert_eq!(again.values, s.values);
}
#[test]
fn twenty_first_frame_evicts_first() {
    let mut w = RdfWindow::new(2, [10., 10.], 5, 20).unwrap();
    w.push(&[[0., 0.], [1.5, 0.]]).unwrap();
    for _ in 0..19 {
        w.push(&[[0., 0.], [3.5, 0.]]).unwrap();
    }
    let last = w.push(&[[0., 0.], [3.5, 0.]]).unwrap();
    assert_eq!(last.frames, 20);
    assert_eq!(last.values[1], 0.);
    let expected = 2. / (2. * (2. / 100.) * std::f64::consts::PI * 7.);
    assert!((last.values[3] - expected).abs() < 1e-12);
}
#[test]
fn periodic_distance_and_maximum_radius_endpoint() {
    let mut w = RdfWindow::new(2, [12., 10.], 5, 1).unwrap();
    assert!(w.push(&[[0., 0.], [11., 0.]]).unwrap().values[1] > 0.);
    assert!(w.push(&[[0., 0.], [5., 0.]]).unwrap().values[4] > 0.);
    assert!(
        w.push(&[[0., 0.], [6., 0.]])
            .unwrap()
            .values
            .iter()
            .all(|g| *g == 0.)
    );
}
