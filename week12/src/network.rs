use crate::glass::{Glass, Point};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub type Features = [f64; 12];
pub const NAMES: [&str; 12] = [
    "x_val",
    "frac_dist",
    "field_h",
    "abs_field_h",
    "deg_norm",
    "bond_bias",
    "eff_field",
    "abs_eff_field",
    "mean_bond_dual",
    "mean_bond_gap",
    "mean_nbr_frac",
    "gap_ratio",
];

pub fn frac(x: f64) -> f64 {
    let x = x.clamp(0., 1.);
    2. * x.min(1. - x)
}
pub struct Neighbors {
    around: Vec<Vec<(usize, usize, f64)>>,
    max_degree: f64,
}
impl Neighbors {
    pub fn new(g: &Glass) -> Self {
        let mut around = vec![Vec::new(); g.h.len()];
        for (e, &(i, j, w)) in g.edges.iter().enumerate() {
            around[i].push((j, e, w));
            around[j].push((i, e, w));
        }
        let max_degree = around.iter().map(Vec::len).max().unwrap_or(0).max(1) as f64;
        Self { around, max_degree }
    }
    pub fn features(
        &self,
        g: &Glass,
        p: &Point,
        fixed: &[i8],
        best: f64,
        ids: &[usize],
    ) -> Vec<Features> {
        let dual_scale = p.duals.iter().map(|d| d.abs()).fold(1e-9, f64::max);
        let gap = if best.is_finite() {
            ((best - p.energy) / best.abs().max(1.)).clamp(0., 2.)
        } else {
            1.
        };
        ids.iter()
            .map(|&i| {
                let x = p.x[i].clamp(0., 1.);
                let spin = 1. - 2. * x;
                let h = g.h[i];
                let degree = self.around[i].len() as f64;
                let denom = degree.max(1.);
                let (mut bias, mut field, mut dual, mut residual, mut neighbor_frac) =
                    (0., h, 0., 0., 0.);
                for &(j, e, w) in &self.around[i] {
                    let s = 1. - 2. * p.x[j].clamp(0., 1.);
                    if w > 0. {
                        bias += 1.;
                    } else if w < 0. {
                        bias -= 1.;
                    }
                    field += w * s;
                    dual += p.duals[4 * e..4 * e + 4]
                        .iter()
                        .map(|v| v.abs())
                        .sum::<f64>()
                        / dual_scale;
                    residual += (p.bonds[e] - 0.5 * (1. - spin * s)).abs();
                    if fixed[j] < 0 {
                        neighbor_frac += frac(p.x[j]);
                    }
                }
                field /= degree + 1.;
                [
                    x,
                    frac(x),
                    h,
                    h.abs(),
                    degree / self.max_degree,
                    bias / denom,
                    field,
                    field.abs(),
                    dual / denom,
                    residual / denom,
                    neighbor_frac / denom,
                    gap,
                ]
            })
            .collect()
    }
}
#[derive(Deserialize)]
pub struct Probe {
    pub features: Vec<Features>,
    pub logits: Vec<f64>,
    pub probabilities: Vec<f64>,
}
#[derive(Deserialize)]
pub struct Network {
    feature_names: Vec<String>,
    mean: Features,
    std: Features,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    probes: Vec<Probe>,
}
#[derive(Serialize)]
pub struct Parity {
    pub probability_errors: Vec<f64>,
    pub logit_errors: Vec<f64>,
    pub max_probability_error: f64,
}
impl Network {
    pub fn score(&self, f: &Features) -> f64 {
        let mut a = [0.; 48];
        a.copy_from_slice(&self.b1);
        for (k, w) in self.w1.iter().enumerate() {
            let x = (f[k] - self.mean[k]) / self.std[k].max(1e-6);
            for (val, weight) in a.iter_mut().zip(w) {
                *val += x * weight;
            }
        }
        a.iter()
            .zip(&self.w2)
            .fold(self.b2, |s, (val, w)| s + val.tanh() * w)
    }
    pub fn parity(&self) -> Result<Parity> {
        ensure!(
            self.feature_names == NAMES
                && self.w1.len() == 12
                && self.w1.iter().all(|r| r.len() == 48)
                && self.b1.len() == 48
                && self.w2.len() == 48
                && self.probes.len() == 6,
            "invalid network or probes"
        );
        let mut probability_errors = Vec::new();
        let mut logit_errors = Vec::new();
        for p in &self.probes {
            ensure!(
                !p.features.is_empty()
                    && p.features.len() == p.logits.len()
                    && p.features.len() == p.probabilities.len(),
                "invalid probe"
            );
            let logits: Vec<_> = p.features.iter().map(|f| self.score(f)).collect();
            let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<_> = logits.iter().map(|v| (v - peak).exp()).collect();
            let sum: f64 = exps.iter().sum();
            probability_errors.push(
                exps.iter()
                    .zip(&p.probabilities)
                    .map(|(a, b)| (a / sum - b).abs())
                    .fold(0., f64::max),
            );
            logit_errors.push(
                logits
                    .iter()
                    .zip(&p.logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max),
            );
        }
        let max_probability_error = probability_errors.iter().copied().fold(0., f64::max);
        ensure!(
            max_probability_error < 1e-4,
            "JAX/Rust probe probability mismatch"
        );
        Ok(Parity {
            probability_errors,
            logit_errors,
            max_probability_error,
        })
    }
}
