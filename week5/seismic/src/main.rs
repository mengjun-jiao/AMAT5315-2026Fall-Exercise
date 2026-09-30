use anyhow::{bail, Result};
use clap::Parser;

use seismic::{
    cli::{Cli, Mode},
    experiment::Experiment,
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let experiment = Experiment::from_path(&cli.experiment)?;

    match cli.mode {
        Mode::Forward => {
            let _ = (&cli.out, cli.every);
            println!(
                "validated experiment: grid={}x{}, steps={}, shots={}, receivers={}",
                experiment.nx,
                experiment.nz,
                experiment.steps,
                experiment.shots.len(),
                experiment.receivers.len()
            );
            bail!("forward solver is not implemented yet");
        }
        Mode::Born => bail!("unsupported mode: born"),
        Mode::Adjoint => bail!("unsupported mode: adjoint"),
    }
}
