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
