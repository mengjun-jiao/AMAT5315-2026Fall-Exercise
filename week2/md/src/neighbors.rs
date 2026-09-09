use crate::physics::PhysicalModel;

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum ForceMethod {
    #[default]
    Naive,
    Cells,
}

pub(crate) fn visit_candidates(
    pos: &[[f64; 2]],
    model: &PhysicalModel,
    method: ForceMethod,
    mut visit: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    model.validate()?;
    if pos.len() < 2 || !pos.iter().flatten().all(|x| x.is_finite()) {
        return Err("at least two finite positions required".into());
    }
    if method == ForceMethod::Naive {
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                visit(i, j)?;
            }
        }
        return Ok(());
    }
    let PhysicalModel::PeriodicShiftedLennardJones { box_size, rc } = model else {
        return Err("cell lists require periodic model".into());
    };
    let nx = (box_size[0] / rc).floor() as usize;
    let ny = (box_size[1] / rc).floor() as usize;
    if nx < 1 || ny < 1 {
        return Err("periodic box must contain at least one cell per axis".into());
    }
    let wx = box_size[0] / nx as f64;
    let wy = box_size[1] / ny as f64;
    let count = nx
        .checked_mul(ny)
        .ok_or_else(|| "grid size overflow".to_string())?;
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    buckets
        .try_reserve_exact(count)
        .map_err(|e| format!("grid allocation: {e}"))?;
    buckets.resize_with(count, Vec::new);
    let mut atom_cells = Vec::with_capacity(pos.len());
    for (i, &p) in pos.iter().enumerate() {
        let wrapped = model.wrap(p);
        let cx = ((wrapped[0] / wx).floor() as usize).min(nx - 1);
        let cy = ((wrapped[1] / wy).floor() as usize).min(ny - 1);
        atom_cells.push((cx, cy));
        buckets[cy * nx + cx].push(i);
    }
    for (i, &(cx, cy)) in atom_cells.iter().enumerate() {
        let xs = [(cx + nx - 1) % nx, cx, (cx + 1) % nx];
        let ys = [(cy + ny - 1) % ny, cy, (cy + 1) % ny];
        let mut ids = Vec::with_capacity(9);
        for y in ys {
            for x in xs {
                let id = y * nx + x;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        for id in ids {
            for &j in &buckets[id] {
                if j > i {
                    visit(i, j)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(p: &[[f64; 2]], b: [f64; 2], method: ForceMethod) -> Vec<(usize, usize)> {
        let m = PhysicalModel::PeriodicShiftedLennardJones { box_size: b, rc: 2.5 };
        let mut result = Vec::new();
        visit_candidates(p, &m, method, |i, j| {
            result.push((i, j));
            Ok(())
        })
        .unwrap();
        result
    }

    #[test]
    fn two_by_two_bins_do_not_repeat_wrapped_neighbors() {
        let p = [[0.1, 0.1], [1.1, 0.1], [3.1, 0.1], [3.1, 3.1]];
        let mut got = pairs(&p, [6., 6.], ForceMethod::Cells);
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|&(i, j)| i < j));
        got.sort_unstable();
        got.dedup();
        assert_eq!(got.len(), 6);
    }

    #[test]
    fn cells_reduce_candidates_without_losing_near_pairs() {
        let p = [[1., 1.], [2., 1.], [15., 15.], [16., 15.]];
        let mut got = pairs(&p, [30., 30.], ForceMethod::Cells);
        got.sort_unstable();
        assert_eq!(got, vec![(0, 1), (2, 3)]);
        assert_eq!(pairs(&p, [30., 30.], ForceMethod::Naive).len(), 6);
    }

    #[test]
    fn all_interacting_pairs_are_present_without_duplicates() {
        let (mut p, b) = crate::fluid::lattice(100, 0.8).unwrap();
        for (i, x) in p.iter_mut().enumerate() {
            x[0] += 0.03 * (i as f64).sin();
        }
        let got = pairs(&p, b, ForceMethod::Cells);
        let unique: std::collections::BTreeSet<_> = got.iter().copied().collect();
        assert_eq!(unique.len(), got.len());
        let m = PhysicalModel::PeriodicShiftedLennardJones { box_size: b, rc: 2.5 };
        for i in 0..p.len() {
            for j in i + 1..p.len() {
                let d = m.displacement(p[i], p[j]);
                if d[0].hypot(d[1]) < 2.5 {
                    assert!(unique.contains(&(i, j)));
                }
            }
        }
    }

    #[test]
    fn wrapped_boundary_pairs_are_candidates() {
        let got = pairs(
            &[[-0.1, 1.], [0.1, 1.], [12.1, 5.]],
            [12., 9.],
            ForceMethod::Cells,
        );
        assert!(got.contains(&(0, 1)));
    }
}
