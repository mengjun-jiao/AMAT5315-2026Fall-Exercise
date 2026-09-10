use md::fluid::{RunConfig, lattice};
use md::trajectory::{Frame, RunMetadata};
use md::neighbors::ForceMethod;
#[test]
fn check_recomputes_known_state_without_simulation() {
    let c = RunConfig {
        steps: 1,
        sample_every: 1,
        eq_steps: 0,
        ..Default::default()
    };
    let (p, b) = lattice(c.n, c.rho).unwrap();
    let mut u = 0.;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            let mut d = [p[i][0] - p[j][0], p[i][1] - p[j][1]];
            for axis in 0..2 {
                d[axis] -= b[axis] * (d[axis] / b[axis]).round();
            }
            let r = d[0].hypot(d[1]);
            if r < 2.5 {
                u += 4. * (r.powi(-12) - r.powi(-6)) - md::energy(2.5);
            }
        }
    }
    let f = Frame {
        step: 1,
        t: 0.01,
        pos: p,
        vel: vec![[1., 0.]; 100],
        e_pot: u,
        e_kin: 50.,
    };
    let m = RunMetadata {
        config: c,
        box_size: b,
        force_method: ForceMethod::Naive,
    };
    let report = md::check::evaluate(&m, &[f.clone()]).unwrap();
    assert_eq!(report.t_speed, 0.5);
    assert_eq!(report.drift, 0.);
    for potential in [true, false] {
        let mut bad = f.clone();
        if potential {
            bad.e_pot += 1.;
        } else {
            bad.e_kin += 1.;
        }
        assert!(
            md::check::evaluate(&m, &[bad])
                .unwrap_err()
                .contains(if potential { "E_pot" } else { "E_kin" })
        );
    }
}
#[test]
fn real_check_binary_rejects_tampered_and_malformed_files() {
    use std::process::Command;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let exe = env!("CARGO_BIN_EXE_md");
    assert!(
        Command::new(exe)
            .args(["run", "--eq-steps", "0", "--steps", "100", "--out"])
            .arg(dir)
            .output()
            .unwrap()
            .status
            .success()
    );
    let original = std::fs::read_to_string(dir.join("traj.jsonl")).unwrap();
    let rows: Vec<serde_json::Value> = original
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let cases = [
        ("E_pot", "E_pot"),
        ("E_kin", "E_kin"),
        ("vel", "E_kin"),
        ("pos", "E_pot"),
        ("step", "step/time"),
        ("t", "step/time"),
        ("short", "length"),
        ("outside", "outside"),
        ("coincident", "separation"),
        ("missing", "missing field"),
    ];
    for (change, reason) in cases {
        let mut r = rows.clone();
        let f = &mut r[0];
        match change {
            "E_pot" | "E_kin" => f[change] = serde_json::json!(f[change].as_f64().unwrap() + 1.),
            "vel" => f["vel"][0][0] = serde_json::json!(f["vel"][0][0].as_f64().unwrap() + 1.),
            "pos" => {
                f["pos"][0][0] = serde_json::json!((f["pos"][0][0].as_f64().unwrap() + 0.1) % 10.)
            }
            "step" => f["step"] = serde_json::json!(51),
            "t" => f["t"] = serde_json::json!(5.),
            "short" => {
                f["pos"].as_array_mut().unwrap().pop();
            }
            "outside" => f["pos"][0][0] = serde_json::json!(-1.),
            "coincident" => f["pos"][0] = f["pos"][1].clone(),
            "missing" => {
                f.as_object_mut().unwrap().remove("vel");
            }
            _ => unreachable!(),
        }
        std::fs::write(
            dir.join("traj.jsonl"),
            r.iter().map(|v| format!("{v}\n")).collect::<String>(),
        )
        .unwrap();
        let out = Command::new(exe).arg("check").arg(dir).output().unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success());
        assert!(err.contains(reason), "{change}: {err}");
    }
    for (content, reason) in [
        ("{bad}\n".to_string(), "line 1"),
        (format!("{}\n", rows[0]), "frame count"),
    ] {
        std::fs::write(dir.join("traj.jsonl"), content).unwrap();
        let out = Command::new(exe).arg("check").arg(dir).output().unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains(reason));
    }
}

#[test]
fn ramp_check_passes_integrity_and_reports_skips() {
    use std::process::Command;
    let tmp = tempfile::tempdir().unwrap();
    let exe = env!("CARGO_BIN_EXE_md");
    let out = Command::new(exe)
        .args([
            "run", "--ramp-to", "1.0", "--eq-steps", "0", "--steps", "100",
            "--sample-every", "50", "--out",
        ])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = Command::new(exe)
        .args(["check"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("SKIP"));
    assert!(stdout.contains("不适用于启用 ramp_to 的轨迹"));
    assert!(stdout.contains("加热轨迹完整性检查通过"));
}

#[test]
fn ramp_check_rejects_tampered_energy_and_state() {
    use std::process::Command;
    let tmp = tempfile::tempdir().unwrap();
    let exe = env!("CARGO_BIN_EXE_md");
    assert!(
        Command::new(exe)
            .args(["run", "--ramp-to", "1.0", "--eq-steps", "0", "--steps", "100", "--sample-every", "50", "--out"])
            .arg(tmp.path())
            .output()
            .unwrap()
            .status
            .success()
    );
    let original = std::fs::read_to_string(tmp.path().join("traj.jsonl")).unwrap();
    let rows: Vec<serde_json::Value> = original.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
    for (field, reason) in [("E_pot", "E_pot"), ("vel", "E_kin")] {
        let mut damaged = rows.clone();
        if field == "E_pot" {
            damaged[0][field] = serde_json::json!(damaged[0][field].as_f64().unwrap() + 1.0);
        } else {
            damaged[0]["vel"][0][0] = serde_json::json!(damaged[0]["vel"][0][0].as_f64().unwrap() + 1.0);
        }
        std::fs::write(
            tmp.path().join("traj.jsonl"),
            damaged.iter().map(|v| format!("{v}\n")).collect::<String>(),
        )
        .unwrap();
        let out = Command::new(exe).arg("check").arg(tmp.path()).output().unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains(reason));
    }
}

#[test]
fn ramp_check_skips_metrics_even_when_speed_is_zero() {
    let c = RunConfig {
        eq_steps: 0,
        steps: 1,
        sample_every: 1,
        ramp_to: Some(1.0),
        ..Default::default()
    };
    let (pos, box_size) = lattice(c.n, c.rho).unwrap();
    let model = md::physics::PhysicalModel::PeriodicShiftedLennardJones { box_size, rc: 2.5 };
    let (e_pot, e_kin) = md::physics::energies(&pos, &vec![[0.; 2]; 100], &model).unwrap();
    let meta = RunMetadata { config: c, box_size, force_method: ForceMethod::Naive };
    let frame = Frame {
        step: 1,
        t: 0.01,
        pos,
        vel: vec![[0.; 2]; 100],
        e_pot,
        e_kin,
    };
    let report = md::check::evaluate(&meta, &[frame]).unwrap();
    assert!(!report.acceptance_applicable);
    assert!(report.passed);
}
