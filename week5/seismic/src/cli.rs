use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Mode {
    Forward,
    Born,
    Adjoint,
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
}
