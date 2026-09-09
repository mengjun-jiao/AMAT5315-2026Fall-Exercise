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
        Some(Commands::Check{directory})=> {
            let r=md::check::check_directory(&directory)?;
            let label=|passed:bool|if passed {"PASS"} else {"FAIL"};
            println!("energy_drift={:.15e} < 2e-3 {}",r.drift,label(r.drift<2e-3));
            println!("T_speed={:.15e}; abs(T_speed-0.5)={:.15e} < 0.05 {}",r.t_speed,(r.t_speed-0.5).abs(),label((r.t_speed-0.5).abs()<0.05));
            println!("chi2/22={:.15e} < 2 {} (teaching tolerance, not a significance test)",r.mb_score,label(r.mb_score<2.));
            if !r.passed { return Err("physical acceptance failed".into()); }
            println!("PASS");
        }
        Some(Commands::Video{directory,out})=>md::video::render_video(&directory,&out)?,
    }
    Ok(())
}
fn main() {
    if let Err(e)=execute(Cli::parse()) { eprintln!("FAIL: {e}");std::process::exit(1); }
}
