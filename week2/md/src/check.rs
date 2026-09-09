use crate::trajectory::{RunMetadata,Frame};
use std::path::Path;
#[derive(Debug)]
pub struct CheckReport { pub drift:f64,pub t_speed:f64,pub mb_score:f64,pub passed:bool }
pub fn evaluate(_meta:&RunMetadata,_frames:&[Frame])->Result<CheckReport,String> { unimplemented!("Part 4 recompute") }
pub fn check_directory(_dir:&Path)->Result<CheckReport,String> { unimplemented!("Part 4 check directory") }
#[cfg(test)]
fn metrics(_energies:&[f64],_speeds:&[f64])->Result<CheckReport,String> { unimplemented!("Part 4 metrics") }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_probability_fixture_and_first_saved_energy() {
        let speeds:Vec<f64>=(0..24).flat_map(|j| {
            let v=(-(1.0-(j as f64+0.5)/24.0).ln()).sqrt();std::iter::repeat_n(v,100)
        }).collect();
        let mut e=vec![-100.;200];let report=metrics(&e,&speeds).unwrap();
        assert!(report.passed);assert_eq!(report.drift,0.);
        e[180..].fill(-99.);assert!((metrics(&e,&speeds).unwrap().drift-0.01).abs()<1e-12);
        assert!(metrics(&[0.],&speeds).is_err());assert!(metrics(&[-1.],&[0.]).is_err());
    }
}
