use clap::Parser;
use ising::{energy, magnetization, metropolis_sweep, wolff_cluster_flip};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ising",
    about = "sample the Ising model along a temperature ramp; no unstated defaults"
)]
struct Args {
    #[arg(long)]
    update: String,
    #[arg(long)]
    l: usize,
    #[arg(long = "t-from")]
    t_from: f64,
    #[arg(long = "t-to")]
    t_to: f64,
    #[arg(long = "t-step")]
    t_step: f64,
    #[arg(long)]
    discard: usize,
    #[arg(long)]
    measure: usize,
    #[arg(long, default_value_t = 0)]
    every: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Serialize)]
struct Run<'a> {
    #[serde(rename = "L")]
    l: usize,
    update: &'a str,
    t_grid: Vec<f64>,
    discard: usize,
    measure: usize,
    seed: u64,
    sample_every: usize,
    time_unit: &'a str,
}

fn invalid(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.update != "metropolis" && args.update != "wolff" {
        invalid("update must be metropolis or wolff")
    }
    if args.l < 2 {
        invalid("l must be at least 2")
    }
    if !args.t_from.is_finite()
        || !args.t_to.is_finite()
        || !args.t_step.is_finite()
        || args.t_from <= 0.0
        || args.t_to < args.t_from
        || args.t_step <= 0.0
    {
        invalid("temperatures must be finite, positive, ordered, with t-step > 0")
    }
    if args.measure == 0 {
        invalid("measure must be positive")
    }
    let mut t_grid = Vec::new();
    for i in 0..=((args.t_to - args.t_from) / args.t_step).floor() as usize {
        t_grid.push(args.t_from + i as f64 * args.t_step);
    }
    create_dir_all(&args.out)?;
    let run = Run {
        l: args.l,
        update: &args.update,
        t_grid: t_grid.clone(),
        discard: args.discard,
        measure: args.measure,
        seed: args.seed,
        sample_every: 1,
        time_unit: if args.update == "metropolis" {
            "sweep"
        } else {
            "cluster_flip"
        },
    };
    let mut run_file = BufWriter::new(File::create(args.out.join("run.json"))?);
    serde_json::to_writer_pretty(&mut run_file, &run)?;
    writeln!(run_file)?;
    let mut series = BufWriter::new(File::create(args.out.join("series.jsonl"))?);
    let mut frames = if args.every > 0 {
        Some(BufWriter::new(File::create(args.out.join("spins.jsonl"))?))
    } else {
        None
    };
    let mut spins = vec![1i8; args.l * args.l];
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    if args.update == "metropolis" {
        println!("T\tmean_abs_M\tacceptance_rate");
    } else {
        println!("T\tmean_abs_M\tmean_cluster_size");
    }
    let mut cumulative_sweep = 0usize;
    for &temperature in &t_grid {
        let mut accepted = 0usize;
        for _ in 0..args.discard {
            if args.update == "metropolis" {
                accepted += metropolis_sweep(&mut spins, args.l, temperature, &mut rng);
            } else {
                wolff_cluster_flip(&mut spins, args.l, temperature, &mut rng);
            }
            cumulative_sweep += 1;
        }
        let mut abs_m_sum = 0.0;
        let mut cluster_size_sum = 0usize;
        for local_sweep in 1..=args.measure {
            let cluster_size = if args.update == "metropolis" {
                accepted += metropolis_sweep(&mut spins, args.l, temperature, &mut rng);
                0
            } else {
                wolff_cluster_flip(&mut spins, args.l, temperature, &mut rng)
            };
            cluster_size_sum += cluster_size;
            cumulative_sweep += 1;
            let m = magnetization(&spins);
            let e = energy(&spins, args.l) / (args.l * args.l) as f64;
            abs_m_sum += m.abs();
            if args.update == "metropolis" {
                writeln!(
                    series,
                    "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"M\":{:.6},\"E\":{:.6}}}",
                    args.l, temperature, local_sweep, m, e
                )?;
            } else {
                writeln!(series, "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"M\":{:.6},\"E\":{:.6},\"cluster_size\":{}}}", args.l, temperature, local_sweep, m, e, cluster_size)?;
            }
            if let Some(output) = frames.as_mut() {
                if local_sweep % args.every == 0 {
                    write!(
                        output,
                        "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"m\":{:.6},\"spins\":[",
                        args.l, temperature, cumulative_sweep, m
                    )?;
                    for (i, s) in spins.iter().enumerate() {
                        if i > 0 {
                            write!(output, ",")?;
                        }
                        write!(output, "{s}")?;
                    }
                    writeln!(output, "]}}")?;
                }
            }
        }
        if args.update == "metropolis" {
            let proposals = (args.discard + args.measure) * args.l * args.l;
            println!(
                "{temperature:.6}\t{:.6}\t{:.6}",
                abs_m_sum / args.measure as f64,
                accepted as f64 / proposals as f64
            );
        } else {
            println!(
                "{temperature:.6}\t{:.6}\t{:.6}",
                abs_m_sum / args.measure as f64,
                cluster_size_sum as f64 / args.measure as f64
            );
        }
    }
    Ok(())
}
