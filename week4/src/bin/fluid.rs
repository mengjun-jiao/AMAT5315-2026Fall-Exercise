use std::fs::{create_dir_all, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::exit;
use week4::field_io::{json_number_array, parse_field_json, FieldData};
use week4::fluid::{
    energy, enstrophy, velocity_to_vorticity, vorticity_to_velocity, VorticitySolver,
};
use week4::{ForwardEuler, Integrator, Midpoint, Rk4};

struct Arguments {
    method: String,
    nu: f64,
    dt: f64,
    t_end: f64,
    every: f64,
    out: PathBuf,
}

fn error(message: &str) -> ! {
    eprintln!("error: {message}");
    exit(2);
}

fn main() {
    let args = parse_args();
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .unwrap_or_else(|_| error("could not read the field from stdin"));
    let field = parse_field_json(&input)
        .unwrap_or_else(|message| error(&format!("invalid field JSON: {message}")));
    let solver = VorticitySolver::new(field.n, args.nu);
    let omega = velocity_to_vorticity(&field.u, &field.v, field.n);
    let snapshot_every = (args.every / args.dt).round() as usize;
    if snapshot_every == 0 {
        error("every/dt must round to at least one step");
    }
    create_dir_all(&args.out).unwrap_or_else(|_| error("could not create output folder"));
    let mut fields = File::create(args.out.join("fields.jsonl"))
        .unwrap_or_else(|_| error("could not create fields.jsonl"));
    let run_json = format!("{{\"case\":\"{}\",\"n\":{},\"seed\":{},\"k_band\":{},\"method\":\"{}\",\"nu\":{},\"dt\":{},\"t_end\":{},\"snapshot_every\":{}}}", field.case, field.n, field.seed, field.k_band, args.method, args.nu, args.dt, args.t_end, snapshot_every);
    std::fs::write(args.out.join("run.json"), run_json)
        .unwrap_or_else(|_| error("could not write run.json"));
    println!("t\tE\tZ");
    write_snapshot(&mut fields, &omega, 0, 0.0, field.n)
        .unwrap_or_else(|_| error("could not write initial snapshot"));
    let (u0, v0) = vorticity_to_velocity(&omega, field.n);
    println!(
        "{:.15}\t{:.15}\t{:.15}",
        0.0,
        energy(&u0, &v0),
        enstrophy(&omega)
    );
    run(&args, &solver, &field, omega, &mut fields, snapshot_every);
}

fn run(
    args: &Arguments,
    solver: &VorticitySolver,
    field: &FieldData,
    mut omega: Vec<f64>,
    fields: &mut File,
    snapshot_every: usize,
) -> ! {
    let integrator: &dyn Integrator = match args.method.as_str() {
        "euler" => &ForwardEuler,
        "rk2" => &Midpoint,
        "rk4" => &Rk4,
        _ => error("method must be euler, rk2, or rk4"),
    };
    let mut step = 0usize;
    let mut time = 0.0;
    while time < args.t_end - 1e-14 {
        omega = solver.step(integrator, &omega, time, args.dt);
        step += 1;
        time += args.dt;
        let (u, v) = vorticity_to_velocity(&omega, field.n);
        let e = energy(&u, &v);
        let z = enstrophy(&omega);
        if !e.is_finite() {
            println!("{time:.15}\t{e:.15}\t{z:.15}");
            exit(1);
        }
        if step.is_multiple_of(snapshot_every) {
            write_snapshot(fields, &omega, step, time, field.n)
                .unwrap_or_else(|_| error("could not write snapshot"));
            println!("{time:.15}\t{e:.15}\t{z:.15}");
        }
    }
    exit(0);
}

fn write_snapshot(
    fields: &mut File,
    omega: &[f64],
    step: usize,
    time: f64,
    n: usize,
) -> io::Result<()> {
    let (u, v) = vorticity_to_velocity(omega, n);
    writeln!(
        fields,
        "{{\"t\":{time:.15},\"step\":{step},\"u\":{},\"v\":{},\"omega\":{}}}",
        json_number_array(&u, Some(6)),
        json_number_array(&v, Some(6)),
        json_number_array(omega, Some(6))
    )
}

fn parse_args() -> Arguments {
    let args: Vec<_> = std::env::args().collect();
    let value = |name: &str| {
        args.windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| pair[1].clone())
            .unwrap_or_else(|| error(&format!("missing required option {name}")))
    };
    let method = value("--method");
    let nu: f64 = value("--nu")
        .parse()
        .unwrap_or_else(|_| error("invalid value for --nu"));
    let dt: f64 = value("--dt")
        .parse()
        .unwrap_or_else(|_| error("invalid value for --dt"));
    let t_end: f64 = value("--t-end")
        .parse()
        .unwrap_or_else(|_| error("invalid value for --t-end"));
    let every: f64 = value("--every")
        .parse()
        .unwrap_or_else(|_| error("invalid value for --every"));
    let out = PathBuf::from(value("--out"));
    if !nu.is_finite()
        || nu < 0.0
        || !dt.is_finite()
        || dt <= 0.0
        || !t_end.is_finite()
        || t_end < 0.0
        || !every.is_finite()
        || every <= 0.0
    {
        error("nu, dt, t-end, and every must be finite with positive time values");
    }
    Arguments {
        method,
        nu,
        dt,
        t_end,
        every,
        out,
    }
}
