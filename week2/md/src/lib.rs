pub fn greeting() -> &'static str {
    "Hello, world!"
}

pub fn energy(r: f64) -> f64 {
    4.0 * (r.powi(-12) - r.powi(-6))
}

pub fn force(_r: f64) -> f64 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::{energy, force, greeting};

    #[test]
    fn greeting_returns_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn energy_has_unit_well_depth() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn force_matches_negative_energy_derivative() {
        let h = 1e-5;
        for r in [0.95, 1.05, 1.2, 1.5, 2.0] {
            let radial_force = force(r);
            let estimated_force = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let tolerance = 1e-6 * radial_force.abs().max(1.0);
            assert!(
                (estimated_force - radial_force).abs() < tolerance,
                "r={r}: force={radial_force}, estimated force={estimated_force}, tolerance={tolerance}"
            );
        }
    }
}
