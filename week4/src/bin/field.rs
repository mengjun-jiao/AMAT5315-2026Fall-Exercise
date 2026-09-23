use std::f64::consts::PI;
use std::process::exit;
use week4::field_io::field_json;
use week4::fluid::random_velocity;

fn usage_error(message: &str) -> ! {
    eprintln!("error: {message}");
    exit(2);
}

fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn required<T: std::str::FromStr>(args: &[String], name: &str) -> T {
    option(args, name)
        .unwrap_or_else(|| usage_error(&format!("missing required option {name}")))
        .parse()
        .unwrap_or_else(|_| usage_error(&format!("invalid value for {name}")))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        usage_error("usage is field taylor-green ... or field random ...");
    }
    let case = args[1].as_str();
    let n: usize = required(&args, "--n");
    if n < 4 || !n.is_power_of_two() {
        usage_error("n must be a power of two at least 4");
    }
    let (u, v, seed, k_band, case_name) = match case {
        "taylor-green" => {
            let time: f64 = option(&args, "--t")
                .map(|value| {
                    value
                        .parse()
                        .unwrap_or_else(|_| usage_error("invalid value for --t"))
                })
                .unwrap_or(0.0);
            let nu: f64 = option(&args, "--nu")
                .map(|value| {
                    value
                        .parse()
                        .unwrap_or_else(|_| usage_error("invalid value for --nu"))
                })
                .unwrap_or(0.0);
            if time > 0.0 && option(&args, "--nu").is_none() {
                usage_error("--nu is required when --t is positive");
            }
            if !time.is_finite() || !nu.is_finite() || time < 0.0 {
                usage_error("t and nu must be finite and non-negative");
            }
            let decay = (-2.0 * nu * time).exp();
            let mut u = Vec::with_capacity(n * n);
            let mut v = Vec::with_capacity(n * n);
            for l in 0..n {
                for j in 0..n {
                    let x = 2.0 * PI * j as f64 / n as f64;
                    let y = 2.0 * PI * l as f64 / n as f64;
                    u.push(x.cos() * y.sin() * decay);
                    v.push(-x.sin() * y.cos() * decay);
                }
            }
            (u, v, 0, "null".to_owned(), "taylor-green".to_owned())
        }
        "random" => {
            let seed: u64 = required(&args, "--seed");
            let k_min: usize = required(&args, "--k-min");
            let k_max: usize = required(&args, "--k-max");
            if k_min == 0 || k_min > k_max || k_max > n / 3 {
                usage_error("require 0 < k-min <= k-max <= n/3");
            }
            let (u, v) = random_velocity(n, seed, k_min, k_max);
            let k_band = format!("{{\"k_min\":{k_min},\"k_max\":{k_max}}}");
            (u, v, seed, k_band, "random".to_owned())
        }
        _ => usage_error("case must be taylor-green or random"),
    };
    println!("{}", field_json(&case_name, n, seed, &k_band, &u, &v));
}
