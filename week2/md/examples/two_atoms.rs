use md::{Euler, Integrator, System, VelocityVerlet, simulate};
use std::io::{self, BufWriter, Write};

fn export<I: Integrator>(out: &mut impl Write, name: &str,
                         integrator: &I, steps: usize) -> io::Result<()> {
    let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]);
    let e0 = md::energy(1.2);
    for s in simulate(integrator, &mut system, 0.01, steps) {
        let delta = (s.total - e0) / e0.abs();
        writeln!(out, "{name},{},{},{},{},{},{e0},{delta}",
                 s.step, s.time, s.kinetic, s.potential, s.total)?;
    }
    Ok(())
}
fn main() -> io::Result<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(out, "method,step,time,kinetic,potential,total,e0,delta")?;
    export(&mut out, "Euler", &Euler, 500)?;
    export(&mut out, "VelocityVerlet", &VelocityVerlet, 500)?;
    export(&mut out, "VelocityVerletLong", &VelocityVerlet, 5000)?;
    out.flush()
}
