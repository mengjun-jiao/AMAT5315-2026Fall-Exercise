use clap::{Parser,Subcommand,Args};
use std::path::PathBuf;
#[derive(Parser)]
#[command(about="AMAT5315 molecular dynamics")]
struct Cli { #[command(subcommand)] command:Option<Commands> }
#[derive(Subcommand)]
enum Commands {
    Run(RunArgs),
    Check { directory:PathBuf },
    Video { directory:PathBuf, #[arg(long)] out:PathBuf },
}
#[derive(Args)]
struct RunArgs {
    #[arg(long,default_value_t=100)] n:usize,
    #[arg(long,default_value_t=0.8)] rho:f64,
    #[arg(long,default_value_t=0.5)] temperature:f64,
    #[arg(long,default_value_t=0.01)] dt:f64,
    #[arg(long,default_value_t=2000)] eq_steps:usize,
    #[arg(long,default_value_t=10000)] steps:usize,
    #[arg(long,default_value_t=50)] sample_every:usize,
    #[arg(long,default_value_t=2026)] seed:u64,
    #[arg(long,default_value="velocity-verlet")] integrator:String,
    #[arg(long,default_value="artifacts")] out:PathBuf,
}
fn execute(cli:Cli)->Result<(),String> {
    match cli.command {
        None=>println!("{}",md::greeting()),
        Some(Commands::Run(a))=> {
            let c=md::fluid::RunConfig{n:a.n,rho:a.rho,temperature:a.temperature,dt:a.dt,eq_steps:a.eq_steps,steps:a.steps,sample_every:a.sample_every,seed:a.seed,integrator:a.integrator};
            md::trajectory::run_to_directory(&c,&a.out)?;
            println!("Saved {} frames to {}",c.steps/c.sample_every,a.out.display());
        }
        Some(Commands::Check{..})=>return Err("Part 4 check not implemented".into()),
        Some(Commands::Video{..})=>return Err("Part 4 video not implemented".into()),
    }
    Ok(())
}
fn main() {
    if let Err(e)=execute(Cli::parse()) { eprintln!("FAIL: {e}");std::process::exit(1); }
}
