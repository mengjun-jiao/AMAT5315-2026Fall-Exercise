use anyhow::{bail, Result};
use clap::Parser;

use seismic::{
    born::run_born,
    cli::{Cli, Mode},
    experiment::Experiment,
    forward::{run_forward, run_forward_recording, shot_l2_norms},
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let experiment = Experiment::from_path(&cli.experiment)?;

    match cli.mode {
        Mode::Forward => {
            let experiment_file = cli.experiment.to_string_lossy();
            let traces = match cli.every {
                Some(every) => {
                    run_forward_recording(&experiment_file, &experiment, every, &cli.out)?.traces
                }
                None => run_forward(&experiment_file, &experiment, &cli.out)?,
            };
            println!("shot\tmode\tdata_l2_norm");
            for (shot, norm) in shot_l2_norms(&traces).into_iter().enumerate() {
                println!("{shot}\tforward\t{norm:.16e}");
            }
            Ok(())
        }
        Mode::Born => {
            if cli.every.is_some() {
                bail!("--every is not supported for born mode");
            }
            let experiment_file = cli.experiment.to_string_lossy();
            let data = run_born(&experiment_file, &experiment, &cli.out)?;
            println!("shot\tmode\tdata_l2_norm");
            for (shot, norm) in shot_l2_norms(&data).into_iter().enumerate() {
                println!("{shot}\tborn\t{norm:.16e}");
            }
            Ok(())
        }
        Mode::Adjoint => bail!("unsupported mode: adjoint"),
    }
}
