use crate::physics::PhysicalModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ForceMethod {
    #[default]
    Naive,
    Cells,
}

pub(crate) fn visit_candidates(
    _pos: &[[f64; 2]],
    _model: &PhysicalModel,
    _method: ForceMethod,
    _visit: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    Err("cell candidate enumeration not implemented".into())
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
