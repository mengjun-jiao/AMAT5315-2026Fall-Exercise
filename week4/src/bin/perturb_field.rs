//! Add the prescribed initial-vorticity ripple and reconstruct a valid field.

use std::io::{self, Read};
use std::process::exit;
use week4::field_io::{field_json, parse_field_json};
use week4::fluid::{velocity_to_vorticity, vorticity_to_velocity};

fn main() {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .unwrap_or_else(|_| fail("could not read field JSON"));
    let field = parse_field_json(&input)
        .unwrap_or_else(|message| fail(&format!("invalid field JSON: {message}")));
    let n = field.n;
    let mut omega = velocity_to_vorticity(&field.u, &field.v, n);
    let component_max = field
        .u
        .iter()
        .chain(&field.v)
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    for row in 0..n {
        for col in 0..n {
            let x = 2.0 * std::f64::consts::PI * col as f64 / n as f64;
            let y = 2.0 * std::f64::consts::PI * row as f64 / n as f64;
            omega[row * n + col] += -7e-5 * component_max * (3.0 * x).cos() * (4.0 * y).cos();
        }
    }
    let (u, v) = vorticity_to_velocity(&omega, n);
    println!(
        "{}",
        field_json(&field.case, field.n, field.seed, &field.k_band, &u, &v)
    );
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    exit(2);
}
