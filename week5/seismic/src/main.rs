use anyhow::{bail, Result};
use clap::Parser;

use seismic::{
    adjoint::{read_adjoint_data, run_adjoint},
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
            if cli.data.is_some() {
                bail!("--data is only supported for adjoint mode");
            }
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
            if cli.data.is_some() {
                bail!("--data is only supported for adjoint mode");
            }
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
        Mode::Adjoint => {
            if cli.every.is_some() {
                bail!("--every is not supported for adjoint mode");
            }
            let data_path = cli
                .data
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--data is required for adjoint mode"))?;
            let weights = read_adjoint_data(data_path, &experiment)?;
            let experiment_file = cli.experiment.to_string_lossy();
            let result = run_adjoint(&experiment_file, &experiment, &weights, &cli.out)?;
            println!("shot\tmode\tdata_l2_norm");
            for (shot, norm) in shot_l2_norms(&weights).into_iter().enumerate() {
                println!("{shot}\tadjoint\t{norm:.16e}");
            }
            let _ = result;
            Ok(())
        }
    }
}
