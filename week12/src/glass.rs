use anyhow::{Result, ensure};
use highs::{Col, ColProblem, HighsModelStatus, RowProblem, Sense};
use serde::Serialize;
use std::collections::BTreeSet;

pub const EPS: f64 = 1e-6;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn index(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Serialize)]
pub struct Glass {
    pub h: Vec<f64>,
    pub edges: Vec<(usize, usize, f64)>,
}
impl Glass {
    pub fn seeded(n: usize, m: usize, seed: u64, id: usize) -> Result<Self> {
        ensure!(
            n > 0 && m <= n.saturating_mul(n - 1) / 2,
            "invalid size or too many edges"
        );
        let initial = seed ^ (id as u64).wrapping_mul(0xD1B54A32D192ED03);
        let mut rng = SplitMix64(SplitMix64(initial).next());
        let mut pairs = BTreeSet::new();
        let mut degrees = vec![0; n];
        for i in 0..n {
            while degrees[i] < 2.min(n - 1) {
                let j = rng.index(n);
                if i != j && pairs.insert((i.min(j), i.max(j))) {
                    degrees[i] += 1;
                    degrees[j] += 1;
                }
            }
        }
        ensure!(
            pairs.len() <= m,
            "requested edge count is below the seeded minimum-degree construction"
        );
        while pairs.len() < m {
            let i = rng.index(n);
            let j = rng.index(n);
            if i != j {
                pairs.insert((i.min(j), i.max(j)));
            }
        }
        let edges = pairs
            .into_iter()
            .map(|(i, j)| (i, j, if rng.next() & 1 == 0 { 1. } else { -1. }))
            .collect();
        let h = (0..n).map(|_| rng.index(3) as f64 - 1.).collect();
        Ok(Self { h, edges })
    }
    pub fn square() -> Self {
        Self {
            h: vec![0.; 4],
            edges: vec![(0, 1, 1.), (1, 2, 1.), (2, 3, 1.), (3, 0, -1.)],
        }
    }
    pub fn tetrahedron() -> Self {
        Self {
            h: vec![1., 0., 0., 0.],
            edges: (0..4)
                .flat_map(|i| (i + 1..4).map(move |j| (i, j, 1.)))
                .collect(),
        }
    }
    pub fn energy(&self, bits: &[u8]) -> f64 {
        let spin = |i: usize| 1. - 2. * f64::from(bits[i]);
        self.h
            .iter()
            .enumerate()
            .map(|(i, h)| h * spin(i))
            .sum::<f64>()
            + self
                .edges
                .iter()
                .map(|&(u, v, j)| j * spin(u) * spin(v))
                .sum::<f64>()
    }
}
pub fn rounded(x: &[f64]) -> Vec<u8> {
    x.iter().map(|&v| u8::from(v >= 0.5)).collect()
}

pub struct Point {
    pub energy: f64,
    pub x: Vec<f64>,
    pub bonds: Vec<f64>,
    pub duals: Vec<f64>,
}
pub struct Formulation {
    template: ColProblem,
    columns: Vec<Col>,
    offset: f64,
}
fn solve_model(problem: ColProblem) -> Result<highs::SolvedModel> {
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1);
    model.set_option("presolve", "off");
    model.set_option("mip_rel_gap", 0.);
    model.set_option("mip_abs_gap", 0.);
    model
        .try_solve()
        .map_err(|e| anyhow::anyhow!("HiGHS solve: {e:?}"))
}
impl Formulation {
    pub fn new(g: &Glass, integral: bool) -> Self {
        let mut p = RowProblem::new();
        let columns: Vec<_> =
            g.h.iter()
                .map(|h| {
                    if integral {
                        p.add_integer_column(-2. * h, 0..=1)
                    } else {
                        p.add_column(-2. * h, 0..=1)
                    }
                })
                .collect();
        for &(u, v, j) in &g.edges {
            let z = p.add_column(-2. * j, 0..=1);
            let (a, b) = (columns[u], columns[v]);
            p.add_row(..=0., [(z, 1.), (a, -1.), (b, -1.)]);
            p.add_row(..=2., [(z, 1.), (a, 1.), (b, 1.)]);
            p.add_row(..=0., [(a, 1.), (b, -1.), (z, -1.)]);
            p.add_row(..=0., [(a, -1.), (b, 1.), (z, -1.)]);
        }
        Self {
            template: p.into(),
            columns,
            offset: g.h.iter().sum::<f64>() + g.edges.iter().map(|e| e.2).sum::<f64>(),
        }
    }
    pub fn solve(&self, fixed: &[i8]) -> Result<Option<Point>> {
        let mut p = self.template.clone();
        for (i, &v) in fixed.iter().enumerate().filter(|(_, v)| **v >= 0) {
            p.change_column_bounds(self.columns[i], f64::from(v)..=f64::from(v));
        }
        let solved = solve_model(p)?;
        if solved.status() == HighsModelStatus::Infeasible {
            return Ok(None);
        }
        ensure!(
            solved.status() == HighsModelStatus::Optimal,
            "unexpected HiGHS status {:?}",
            solved.status()
        );
        let sol = solved.get_solution();
        let (x, z) = sol.columns().split_at(self.columns.len());
        Ok(Some(Point {
            energy: solved.objective_value() + self.offset,
            x: x.to_vec(),
            bonds: z.to_vec(),
            duals: sol.dual_rows().to_vec(),
        }))
    }
    pub fn root(&self) -> Result<Point> {
        self.solve(&[])?
            .ok_or_else(|| anyhow::anyhow!("empty spin domain"))
    }
}
pub fn diet() -> Result<([f64; 2], f64)> {
    let mut p = RowProblem::new();
    let a = p.add_column(6., 0..);
    let b = p.add_column(3.5, 0..);
    p.add_row(8.., [(b, 7.)]);
    p.add_row(15.., [(a, 4.), (b, 2.)]);
    p.add_row(3.., [(a, 2.), (b, 1.)]);
    let solved = solve_model(p.into())?;
    ensure!(
        solved.status() == HighsModelStatus::Optimal,
        "diet LP failed"
    );
    let s = solved.get_solution();
    Ok(([s.columns()[0], s.columns()[1]], solved.objective_value()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_bond_is_xor_for_all_four_spin_pairs() -> Result<()> {
        let g = Glass {
            h: vec![0.25, -0.75],
            edges: vec![(0, 1, -1.5)],
        };
        let lp = Formulation::new(&g, false);
        for a in 0..=1 {
            for b in 0..=1 {
                let point = lp.solve(&[a, b])?.unwrap();
                assert!((point.bonds[0] - f64::from(a ^ b)).abs() < 1e-9);
                assert!((point.energy - g.energy(&[a as u8, b as u8])).abs() < 1e-9);
            }
        }
        Ok(())
    }

    #[test]
    fn dominant_fields_can_make_the_relaxation_integral() -> Result<()> {
        let g = Glass {
            h: vec![5.5, -4.25, 3.75, -2.5],
            edges: vec![(0, 1, 0.75), (1, 2, -1.25), (0, 2, 0.375)],
        };
        let known = (0..16)
            .map(|k| {
                let bits: Vec<_> = (0..4).map(|i| ((k >> i) & 1) as u8).collect();
                g.energy(&bits)
            })
            .fold(f64::INFINITY, f64::min);
        for integral in [false, true] {
            let optimum = Formulation::new(&g, integral).root()?;
            assert!((optimum.energy - known).abs() < 1e-9);
            assert!(
                optimum
                    .x
                    .iter()
                    .all(|v| v.abs() < 1e-9 || (1. - v).abs() < 1e-9)
            );
        }
        Ok(())
    }
}
