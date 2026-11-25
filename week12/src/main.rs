mod glass;
mod network;
mod search;

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use glass::{EPS, Formulation, Glass, diet, rounded};
use network::{NAMES, Network};
use search::{Policy, millis, search};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Copy, Clone, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Gap,
    Bnb,
    Eval,
}
#[derive(Parser, Serialize)]
#[command(version, about = "Exact spin-glass optimization and learned branching")]
struct Args {
    #[arg(long)]
    mode: Mode,
    #[arg(long, default_value_t = 22)]
    spins: usize,
    #[arg(long, default_value_t = 52)]
    edges: usize,
    #[arg(long, default_value_t = 16)]
    count: usize,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[arg(long, default_value = "most-frac")]
    policy: Policy,
    #[arg(long)]
    export_labels: bool,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
}
fn save(path: &Path, record: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&tmp, serde_json::to_vec(record)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}
#[derive(Serialize)]
struct Gap {
    lp: f64,
    lp_x: Vec<f64>,
    lp_z: Vec<f64>,
    exact: f64,
    exact_bits: Vec<u8>,
    gap: f64,
    relative_gap: Option<f64>,
    fractional: usize,
    rounded_energy: f64,
    rounded_bits: Vec<u8>,
    excess: f64,
}
fn gap(g: &Glass) -> Result<Gap> {
    let p = Formulation::new(g, false).root()?;
    let oracle = Formulation::new(g, true).root()?;
    let exact_bits = rounded(&oracle.x);
    ensure!(
        (oracle.energy - g.energy(&exact_bits)).abs() < EPS,
        "MILP witness energy mismatch"
    );
    let rounded_bits = rounded(&p.x);
    let rounded_energy = g.energy(&rounded_bits);
    let fractional = p.x.iter().filter(|&&x| x > EPS && x < 1. - EPS).count();
    let delta = oracle.energy - p.energy;
    Ok(Gap {
        lp: p.energy,
        lp_x: p.x,
        lp_z: p.bonds,
        exact: oracle.energy,
        exact_bits,
        gap: delta,
        relative_gap: if oracle.energy != 0. {
            Some(delta / oracle.energy.abs())
        } else {
            None
        },
        fractional,
        rounded_energy,
        rounded_bits,
        excess: rounded_energy - oracle.energy,
    })
}
fn run_gap(a: &Args) -> Result<(Value, Value)> {
    let square = gap(&Glass::square())?;
    let (food, cost) = diet()?;
    let mut instances = Vec::new();
    let mut samples = Vec::new();
    for id in 0..a.count {
        let g = Glass::seeded(a.spins, a.edges, a.seed, id)?;
        let row = gap(&g)?;
        instances.push(json!({"id":id,"graph":g,"result":row}));
        samples.push(row);
    }
    let mean = |f: fn(&Gap) -> f64| samples.iter().map(f).sum::<f64>() / samples.len() as f64;
    let rel: Vec<_> = samples.iter().filter_map(|r| r.relative_gap).collect();
    let relative = if rel.is_empty() {
        None
    } else {
        Some(rel.iter().sum::<f64>() / rel.len() as f64)
    };
    let summary = json!({"mean_lp":mean(|s|s.lp),"mean_exact":mean(|s|s.exact),"mean_gap":mean(|s|s.gap),
        "mean_relative_gap":relative,"mean_fractional":mean(|s|s.fractional as f64),
        "mean_rounding_excess":mean(|s|s.excess),"suboptimal_roundings":samples.iter().filter(|r|r.excess>EPS).count(),
        "square_lp":square.lp,"square_exact":square.exact,"square_rounded":square.rounded_energy,"diet_cost":cost});
    Ok((
        summary,
        json!({"square":square,"diet":{"x":food,"cost":cost},"instances":instances}),
    ))
}
#[derive(Default)]
struct Totals {
    nodes: usize,
    lps: usize,
    wall: f64,
    lp: f64,
    inference: f64,
    max_error: f64,
    equal: bool,
}
fn run_searches(a: &Args) -> Result<(Value, Value)> {
    let is_eval = matches!(a.mode, Mode::Eval);
    // The worksheet checks Part 2 against a tighter oracle tolerance than Parts 3–4.
    // The LP pruning and fractionality tolerance remains EPS for every mode.
    let oracle_tol = if is_eval { 1e-6 } else { 1e-9 };
    ensure!(
        !a.export_labels || (!is_eval && a.policy == Policy::Strong && a.count <= 48),
        "export labels from strong bnb on instance IDs 0..47"
    );
    let rules = if is_eval {
        vec![
            Policy::MostFrac,
            Policy::Neural,
            Policy::RootNeural,
            Policy::Strong,
        ]
    } else {
        vec![a.policy]
    };
    let net: Option<Network> = a
        .model
        .as_ref()
        .map(|path| -> Result<_> {
            Ok(serde_json::from_slice(
                &fs::read(path).context("read model")?,
            )?)
        })
        .transpose()?;
    ensure!(
        !rules
            .iter()
            .any(|r| matches!(r, Policy::Neural | Policy::RootNeural))
            || net.is_some(),
        "learned policies require --model"
    );
    let parity = net.as_ref().map(Network::parity).transpose()?;
    let tetra = Glass::tetrahedron();
    let trace = search(&tetra, Policy::MostFrac, None, 0, false, true)?.0;
    let mut all_labels = Vec::new();
    let mut instances = Vec::new();
    let mut totals: BTreeMap<&str, Totals> = BTreeMap::new();
    let mut best_saving = i64::MIN;
    let mut example = None;
    for id in 0..a.count {
        let g = Glass::seeded(a.spins, a.edges, a.seed, id)?;
        let oracle = Formulation::new(&g, true).root()?;
        let oracle_bits = rounded(&oracle.x);
        ensure!(
            (g.energy(&oracle_bits) - oracle.energy).abs() < EPS,
            "incorrect MILP witness"
        );
        let mut policies = BTreeMap::new();
        for &rule in &rules {
            let (res, labels) = search(&g, rule, net.as_ref(), id, a.export_labels, false)?;
            let error = (res.energy - oracle.energy).abs();
            ensure!(
                error <= oracle_tol,
                "instance {id}, {} differs from HiGHS by {error}",
                rule.name()
            );
            let s = totals.entry(rule.name()).or_insert(Totals {
                equal: true,
                ..Default::default()
            });
            s.nodes += res.nodes;
            s.lps += res.lp_solves;
            s.wall += res.wall_ms;
            s.lp += res.lp_ms;
            s.inference += res.inference_ms;
            s.equal &= res.nodes == res.lp_solves;
            s.max_error = s.max_error.max(error);
            policies.insert(rule.name(), json!({"search":res,"error":error}));
            all_labels.extend(labels);
        }
        if is_eval {
            let most = &policies["most-frac"]["search"];
            let neural = &policies["neural"]["search"];
            let saved = most["nodes"].as_i64().unwrap() - neural["nodes"].as_i64().unwrap();
            if saved > best_saving {
                best_saving = saved;
                example = Some(
                    json!({"id":id,"saved":saved,"energy":oracle.energy,"most-frac":most,"neural":neural}),
                );
            }
        }
        instances.push(json!({"id":id,"graph":g,"oracle":oracle.energy,"oracle_bits":oracle_bits,"policies":policies}));
    }
    if a.export_labels {
        save(
            &a.out.join("dataset.json"),
            &json!({"schema":1,"seed":a.seed,"spins":a.spins,"edges":a.edges,
            "feature_names":NAMES,"nodes":all_labels}),
        )?;
    }
    let n = a.count as f64;
    let summary: BTreeMap<_, _> = totals
        .iter()
        .map(|(k, t)| {
            (
                k,
                json!({"mean_nodes":t.nodes as f64/n,"mean_lp_solves":t.lps as f64/n,
        "mean_wall_ms":t.wall/n,"mean_lp_ms":t.lp/n,"mean_inference_ms":t.inference/n,
        "max_error":t.max_error,"lp_solves_equal_nodes":t.equal}),
            )
        })
        .collect();
    Ok((
        json!(summary),
        json!({"instances":instances,"tetrahedron":trace,"parity":parity,"largest_saving":example}),
    ))
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.count > 0, "--count must be positive");
    ensure!(
        !matches!(a.mode, Mode::Gap) || !a.export_labels,
        "--export-labels requires bnb"
    );
    fs::create_dir_all(&a.out)?;
    let t = Instant::now();
    let (summary, evidence) = match a.mode {
        Mode::Gap => run_gap(&a)?,
        _ => run_searches(&a)?,
    };
    let ms = millis(t.elapsed());
    let mode = match a.mode {
        Mode::Gap => "gap",
        Mode::Bnb => "bnb",
        Mode::Eval => "eval",
    };
    let run = json!({"schema":1,"mode":mode,"seed":a.seed,"parameters":a,"runtime_ms":ms,
        "summary":summary,"evidence":evidence});
    if matches!(a.mode, Mode::Bnb) && a.policy == Policy::Strong && !a.export_labels {
        // The label-export command later reuses this output directory.
        save(&a.out.join("run_no_labels.json"), &run)?;
    }
    save(&a.out.join("run.json"), &run)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for (key, value) in summary.as_object().unwrap() {
        if let Some(fields) = value.as_object() {
            for (field, v) in fields {
                println!("{mode}\t{key}.{field}\t{v}\t{ms:.3}");
            }
        } else {
            println!("{mode}\t{key}\t{value}\t{ms:.3}");
        }
    }
    Ok(())
}
