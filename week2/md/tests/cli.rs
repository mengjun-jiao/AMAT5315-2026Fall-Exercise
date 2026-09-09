use std::process::Command;
#[test]
fn run_binary_writes_exact_frames_and_required_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run",
            "--eq-steps",
            "0",
            "--steps",
            "101",
            "--sample-every",
            "50",
            "--out",
        ])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        tmp.path().join("run.json").exists(),
        "run must write metadata"
    );
    let (meta, frames) = md::trajectory::read_trajectory(tmp.path()).unwrap();
    assert_eq!(meta.n, 100);
    assert_eq!(meta.seed, 2026);
    assert_eq!(frames.len(), 2);
    assert_eq!(frames.iter().map(|f| f.step).collect::<Vec<_>>(), [50, 100]);
    assert_eq!(frames[0].t, 0.5);
    assert_eq!(frames[1].pos.len(), 100);
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(json.as_object().unwrap().len(), 10);
    assert!(json["box"].is_array());
}
#[test]
fn invalid_n_and_unknown_command_fail() {
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run", "--n", "121", "--out"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(!tmp.path().join("traj.jsonl").exists());
    assert!(
        !Command::new(env!("CARGO_BIN_EXE_md"))
            .arg("nonsense")
            .output()
            .unwrap()
            .status
            .success()
    );
}
