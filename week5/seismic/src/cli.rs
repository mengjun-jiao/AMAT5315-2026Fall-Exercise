use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    Forward,
    Born,
    Adjoint,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Storage {
    #[default]
    Full,
    Treeverse,
}

#[derive(Debug, Parser)]
#[command(name = "seismic")]
#[command(about = "Simulate acoustic waves and their derivatives")]
pub struct Cli {
    #[arg(long)]
    pub experiment: PathBuf,

    #[arg(long, value_enum)]
    pub mode: Mode,

    #[arg(long)]
    pub out: PathBuf,

    #[arg(long)]
    pub every: Option<usize>,

    #[arg(long)]
    pub data: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = Storage::Full)]
    pub storage: Storage,

    #[arg(long)]
    pub checkpoints: Option<usize>,
}

pub fn validate_checkpoint_options(
    mode: Mode,
    storage: Storage,
    checkpoints: Option<usize>,
) -> Result<()> {
    match mode {
        Mode::Forward | Mode::Born => {
            if storage != Storage::Full || checkpoints.is_some() {
                bail!("--storage and --checkpoints are only supported for adjoint mode");
            }
        }
        Mode::Adjoint => match storage {
            Storage::Full => {
                if checkpoints.is_some() {
                    bail!("--checkpoints requires --storage treeverse");
                }
            }
            Storage::Treeverse => {
                let delta = checkpoints.ok_or_else(|| {
                    anyhow::anyhow!("--checkpoints is required for treeverse storage")
                })?;
                if delta == 0 {
                    bail!("--checkpoints must be at least 1 for treeverse storage");
                }
            }
        },
    }
    Ok(())
}
