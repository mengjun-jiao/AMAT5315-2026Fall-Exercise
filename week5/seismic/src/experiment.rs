use std::{collections::HashMap, fs, path::Path};

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridPoint {
    pub x: usize,
    pub z: usize,
}

#[derive(Debug, Deserialize)]
pub struct RawExperiment {
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub shots: Vec<[f64; 2]>,
    pub receivers: Vec<[f64; 2]>,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub background: Vec<Vec<f64>>,
    pub perturbation: Vec<Vec<f64>>,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct Experiment {
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub shots: Vec<GridPoint>,
    pub receivers: Vec<GridPoint>,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub background: Array2<f64>,
    pub perturbation: Array2<f64>,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
}

impl Experiment {
    pub fn from_path(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read experiment file {}", path.display()))?;
        Self::from_json_str(&contents)
            .with_context(|| format!("failed to parse experiment file {}", path.display()))
    }

    pub fn from_json_str(contents: &str) -> Result<Self> {
        let raw: RawExperiment =
            serde_json::from_str(contents).context("invalid experiment JSON")?;
        raw.validate()
    }
}

impl RawExperiment {
    fn validate(self) -> Result<Experiment> {
        if self.nx < 3 || self.nz < 3 {
            bail!("nx and nz must each be at least 3 to provide an interior grid");
        }
        if !self.dx.is_finite() || self.dx <= 0.0 {
            bail!("dx must be finite and positive");
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            bail!("dt must be finite and positive");
        }
        if self.steps == 0 {
            bail!("steps must be positive");
        }
        if !self.source_frequency.is_finite() || self.source_frequency < 0.0 {
            bail!("source_frequency must be finite and nonnegative");
        }
        if !self.source_peak_time.is_finite() {
            bail!("source_peak_time must be finite");
        }
        if !self.source_amplitude.is_finite() {
            bail!("source_amplitude must be finite");
        }
        if self.sponge_width == 0 || self.sponge_width > self.nx.min(self.nz) / 2 {
            bail!("sponge_width must be positive and fit within half the grid");
        }
        if !self.sponge_strength.is_finite() || self.sponge_strength < 0.0 {
            bail!("sponge_strength must be finite and nonnegative");
        }
        if !self.length_unit_m.is_finite() || self.length_unit_m <= 0.0 {
            bail!("length_unit_m must be finite and positive");
        }
        if !self.time_unit_s.is_finite() || self.time_unit_s <= 0.0 {
            bail!("time_unit_s must be finite and positive");
        }

        let shots = validate_coordinates("shots", self.shots, self.nx, self.nz)?;
        let receivers = validate_coordinates("receivers", self.receivers, self.nx, self.nz)?;
        let background = array_from_rows("background", self.background, self.nz, self.nx)?;
        let perturbation = array_from_rows("perturbation", self.perturbation, self.nz, self.nx)?;

        Ok(Experiment {
            nx: self.nx,
            nz: self.nz,
            dx: self.dx,
            dt: self.dt,
            steps: self.steps,
            source_frequency: self.source_frequency,
            source_peak_time: self.source_peak_time,
            source_amplitude: self.source_amplitude,
            shots,
            receivers,
            sponge_width: self.sponge_width,
            sponge_strength: self.sponge_strength,
            background,
            perturbation,
            length_unit_m: self.length_unit_m,
            time_unit_s: self.time_unit_s,
        })
    }
}

fn validate_coordinates(
    name: &str,
    coordinates: Vec<[f64; 2]>,
    nx: usize,
    nz: usize,
) -> Result<Vec<GridPoint>> {
    coordinates
        .into_iter()
        .enumerate()
        .map(|(index, [x, z])| {
            for (axis, value) in [("x", x), ("z", z)] {
                if !value.is_finite() {
                    bail!("{name}[{index}] {axis} coordinate must be finite");
                }
                if value.fract() != 0.0 {
                    bail!("{name}[{index}] {axis} coordinate must be integral");
                }
            }
            if x < 0.0 || x >= nx as f64 || z < 0.0 || z >= nz as f64 {
                bail!("{name}[{index}] coordinate [{x}, {z}] is outside the grid");
            }
            Ok(GridPoint {
                x: x as usize,
                z: z as usize,
            })
        })
        .collect()
}

fn array_from_rows(name: &str, rows: Vec<Vec<f64>>, nz: usize, nx: usize) -> Result<Array2<f64>> {
    if rows.len() != nz {
        bail!("{name} must have {nz} rows, got {}", rows.len());
    }
    let mut flat = Vec::with_capacity(nz * nx);
    for (z, row) in rows.into_iter().enumerate() {
        if row.len() != nx {
            bail!("{name} row {z} must have {nx} values, got {}", row.len());
        }
        for (x, value) in row.into_iter().enumerate() {
            if !value.is_finite() {
                bail!("{name}[{z}][{x}] must be finite");
            }
            flat.push(value);
        }
    }
    Array2::from_shape_vec((nz, nx), flat).context("failed to construct experiment array")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> String {
        r#"
        {
          "nx": 5, "nz": 4, "dx": 1.0, "dt": 0.1, "steps": 2,
          "source_frequency": 0.5, "source_peak_time": 1.0,
          "source_amplitude": 1.0, "sponge_width": 1,
          "sponge_strength": 0.4,
          "shots": [[1.0, 2.0]], "receivers": [[0, 2]],
          "background": [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
          "perturbation": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
          "length_unit_m": 100.0, "time_unit_s": 0.1
        }
        "#
        .to_owned()
    }

    #[test]
    fn synthetic_valid_json_parses() {
        let experiment = Experiment::from_json_str(&valid_json()).unwrap();
        assert_eq!((experiment.nz, experiment.nx), (4, 5));
        assert_eq!(experiment.shots, vec![GridPoint { x: 1, z: 2 }]);
        assert_eq!(experiment.background.shape(), &[4, 5]);
    }

    #[test]
    fn coordinate_validation_rejects_nonintegral_and_out_of_range_values() {
        let nonintegral = valid_json().replace("[1.0, 2.0]", "[1.5, 2.0]");
        assert!(Experiment::from_json_str(&nonintegral)
            .unwrap_err()
            .to_string()
            .contains("integral"));

        let negative = valid_json().replace("[1.0, 2.0]", "[-1.0, 2.0]");
        assert!(Experiment::from_json_str(&negative)
            .unwrap_err()
            .to_string()
            .contains("outside"));

        let too_large = valid_json().replace("[1.0, 2.0]", "[5.0, 2.0]");
        assert!(Experiment::from_json_str(&too_large)
            .unwrap_err()
            .to_string()
            .contains("outside"));
    }

    #[test]
    fn array_shape_validation_rejects_wrong_rows_and_columns() {
        let wrong_rows = valid_json().replace(
            "[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]",
            "[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]",
        );
        assert!(Experiment::from_json_str(&wrong_rows)
            .unwrap_err()
            .to_string()
            .contains("rows"));

        let wrong_columns = valid_json().replace("[1,1,1,1,1]", "[1,1,1,1]");
        assert!(Experiment::from_json_str(&wrong_columns)
            .unwrap_err()
            .to_string()
            .contains("row 0"));
    }

    #[test]
    fn representative_invalid_scalars_are_rejected() {
        for (needle, replacement) in [
            ("\"dx\": 1.0", "\"dx\": 0.0"),
            ("\"dt\": 0.1", "\"dt\": -0.1"),
            ("\"steps\": 2", "\"steps\": 0"),
            ("\"source_frequency\": 0.5", "\"source_frequency\": -0.5"),
            ("\"sponge_width\": 1", "\"sponge_width\": 0"),
            ("\"sponge_strength\": 0.4", "\"sponge_strength\": -0.4"),
            ("\"length_unit_m\": 100.0", "\"length_unit_m\": 0.0"),
            ("\"time_unit_s\": 0.1", "\"time_unit_s\": -0.1"),
        ] {
            let invalid = valid_json().replace(needle, replacement);
            assert!(
                Experiment::from_json_str(&invalid).is_err(),
                "{replacement}"
            );
        }
    }

    #[test]
    fn official_reflector_parses_when_available() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inputs/reflector.json");
        if !path.exists() {
            return;
        }
        let experiment = Experiment::from_path(&path).unwrap();
        assert_eq!((experiment.nx, experiment.nz), (41, 41));
        assert_eq!(experiment.background.shape(), &[41, 41]);
        assert_eq!(experiment.perturbation.shape(), &[41, 41]);
        assert_eq!(experiment.shots.len(), 3);
        assert_eq!(experiment.receivers.len(), 14);
        assert_eq!(experiment.shots[0], GridPoint { x: 10, z: 8 });
    }
}
