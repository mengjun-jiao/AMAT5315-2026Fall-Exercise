use crate::glass::{EPS, Formulation, Glass, Point, rounded};
use crate::network::{Features, Neighbors, Network, frac};
use anyhow::{Result, ensure};
use clap::ValueEnum;
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Policy {
    MostFrac,
    Strong,
    Neural,
    RootNeural,
}
impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Self::MostFrac => "most-frac",
            Self::Strong => "strong",
            Self::Neural => "neural",
            Self::RootNeural => "root-neural",
        }
    }
}
#[derive(Serialize)]
pub struct Candidate {
    spin: usize,
    gains: [f64; 2],
    score: f64,
    pruned: usize,
    features: Features,
}
#[derive(Serialize)]
pub struct Label {
    pub instance: usize,
    pub node: usize,
    pub split: &'static str,
    pub chosen: usize,
    pub candidates: Vec<Candidate>,
    fixed: Vec<i8>,
    lower_bound: f64,
    incumbent: Option<f64>,
    x: Vec<f64>,
    z: Vec<f64>,
    row_duals: Vec<f64>,
}
#[derive(Serialize)]
pub struct Visit {
    path: Vec<(usize, i8)>,
    status: &'static str,
    bound: Option<f64>,
    x: Option<Vec<f64>>,
    split: Option<usize>,
}
#[derive(Serialize)]
pub struct Search {
    pub energy: f64,
    pub bits: Vec<u8>,
    pub nodes: usize,
    pub lp_solves: usize,
    pub wall_ms: f64,
    pub lp_ms: f64,
    pub inference_ms: f64,
    pub first_splits: Vec<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub trace: Vec<Visit>,
}
pub fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.
}
struct Counter {
    count: usize,
    time: Duration,
}
impl Counter {
    fn solve(&mut self, lp: &Formulation, fixed: &[i8]) -> Result<Option<Point>> {
        self.count += 1;
        let tic = Instant::now();
        let p = lp.solve(fixed)?;
        self.time += tic.elapsed();
        Ok(p)
    }
}
fn argmax(scores: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    best
}
fn standardized(scores: &mut [f64]) {
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let sd = (scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-6);
    for x in scores {
        *x = (*x - mean) / sd;
    }
}
struct Trial {
    gains: [f64; 2],
    score: f64,
    pruned: usize,
}
fn probe(
    lp: &Formulation,
    counter: &mut Counter,
    fixed: &mut [i8],
    i: usize,
    parent: f64,
    best: f64,
) -> Result<Trial> {
    let mut gains = [0.; 2];
    let mut pruned = 0;
    for (side, gain) in gains.iter_mut().enumerate() {
        fixed[i] = side as i8;
        let child = counter.solve(lp, fixed)?;
        *gain = match child {
            Some(c) if c.energy < best - EPS => c.energy - parent,
            _ => {
                pruned += 1;
                1e6
            }
        };
    }
    fixed[i] = -1;
    Ok(Trial {
        gains,
        score: gains[0].max(1e-6) * gains[1].max(1e-6),
        pruned,
    })
}
pub fn search(
    g: &Glass,
    rule: Policy,
    model: Option<&Network>,
    id: usize,
    export: bool,
    trace: bool,
) -> Result<(Search, Vec<Label>)> {
    let clock = Instant::now();
    let lp = Formulation::new(g, false);
    let neighbors = Neighbors::new(g);
    let n = g.h.len();
    let mut stack = vec![(vec![-1i8; n], Vec::new())];
    let mut best = f64::INFINITY;
    let mut bits = Vec::new();
    let mut count = Counter {
        count: 0,
        time: Duration::ZERO,
    };
    let mut infer = Duration::ZERO;
    let mut nodes = 0;
    let mut first_splits = Vec::new();
    let mut visits = Vec::new();
    let mut labels = Vec::new();
    let mut root_scores = Vec::new();
    while let Some((mut fixed, path)) = stack.pop() {
        nodes += 1;
        let point = count.solve(&lp, &fixed)?;
        let mut split = None;
        let status = match &point {
            None => "infeasible",
            Some(p) if p.energy >= best - EPS => "pruned by bound",
            Some(p) => {
                let ids: Vec<_> = (0..n)
                    .filter(|&i| fixed[i] < 0 && p.x[i] > EPS && p.x[i] < 1. - EPS)
                    .collect();
                if ids.is_empty() {
                    bits = rounded(&p.x);
                    best = g.energy(&bits);
                    ensure!(
                        (best - p.energy).abs() < EPS,
                        "integral LP and direct spin energy disagree"
                    );
                    "incumbent"
                } else {
                    let chosen = match rule {
                        Policy::MostFrac => {
                            ids[argmax(&ids.iter().map(|&i| frac(p.x[i])).collect::<Vec<_>>())]
                        }
                        Policy::Strong => {
                            let trials: Vec<_> = ids
                                .iter()
                                .map(|&i| probe(&lp, &mut count, &mut fixed, i, p.energy, best))
                                .collect::<Result<_>>()?;
                            let pos = argmax(&trials.iter().map(|t| t.score).collect::<Vec<_>>());
                            if export && ids.len() >= 2 && labels.len() < 80 {
                                let features = neighbors.features(g, p, &fixed, best, &ids);
                                let candidates = ids
                                    .iter()
                                    .zip(trials)
                                    .zip(features)
                                    .map(|((&spin, t), f)| Candidate {
                                        spin,
                                        gains: t.gains,
                                        pruned: t.pruned,
                                        score: t.score,
                                        features: f,
                                    })
                                    .collect();
                                labels.push(Label {
                                    instance: id,
                                    node: nodes,
                                    split: if id < 36 { "train" } else { "val" },
                                    chosen: ids[pos],
                                    candidates,
                                    fixed: fixed.clone(),
                                    lower_bound: p.energy,
                                    incumbent: best.is_finite().then_some(best),
                                    x: p.x.clone(),
                                    z: p.bonds.clone(),
                                    row_duals: p.duals.clone(),
                                });
                            }
                            ids[pos]
                        }
                        Policy::Neural => {
                            let t = Instant::now();
                            let net = model.expect("model validated at entry");
                            let features = neighbors.features(g, p, &fixed, best, &ids);
                            let logits: Vec<_> = features.iter().map(|f| net.score(f)).collect();
                            infer += t.elapsed();
                            ids[argmax(&logits)]
                        }
                        Policy::RootNeural => {
                            if root_scores.is_empty() {
                                let t = Instant::now();
                                let net = model.expect("model validated at entry");
                                let all: Vec<_> = (0..n).collect();
                                root_scores = neighbors
                                    .features(g, p, &fixed, best, &all)
                                    .iter()
                                    .map(|f| net.score(f))
                                    .collect();
                                standardized(&mut root_scores);
                                infer += t.elapsed();
                            }
                            ids[argmax(
                                &ids.iter()
                                    .map(|&i| root_scores[i] + 3. * frac(p.x[i]))
                                    .collect::<Vec<_>>(),
                            )]
                        }
                    };
                    split = Some(chosen);
                    if first_splits.len() < 2 {
                        first_splits.push(chosen);
                    }
                    let near = i8::from(p.x[chosen] >= 0.5);
                    for side in [1 - near, near] {
                        let mut child = fixed.clone();
                        child[chosen] = side;
                        let mut child_path = path.clone();
                        if trace {
                            child_path.push((chosen, side));
                        }
                        stack.push((child, child_path));
                    }
                    "branch"
                }
            }
        };
        if trace {
            visits.push(Visit {
                path,
                status,
                bound: point.as_ref().map(|p| p.energy),
                x: point.map(|p| p.x),
                split,
            });
        }
    }
    ensure!(best.is_finite(), "no feasible integral solution");
    let result = Search {
        energy: best,
        bits,
        nodes,
        lp_solves: count.count,
        wall_ms: millis(clock.elapsed()),
        lp_ms: millis(count.time),
        inference_ms: millis(infer),
        first_splits,
        trace: visits,
    };
    Ok((result, labels))
}
