//! Export a 2D field from the crate's analytic Lennard-Jones functions.
use std::io::{self, BufWriter, Write};

fn main() -> io::Result<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    let points = 321;
    let extent = 3.0;
    let min_radius = 0.85;
    writeln!(out, "ix,iy,x,y,energy,fx,fy")?;
    for iy in 0..points {
        let y = -extent + 2.0 * extent * iy as f64 / (points - 1) as f64;
        for ix in 0..points {
            let x = -extent + 2.0 * extent * ix as f64 / (points - 1) as f64;
            let r = x.hypot(y);
            if r < min_radius {
                continue;
            }
            let energy = md::energy(r);
            let force = md::force(r);
            let fx = force * x / r;
            let fy = force * y / r;
            writeln!(out, "{ix},{iy},{x},{y},{energy},{fx},{fy}")?;
        }
    }
    Ok(())
}
