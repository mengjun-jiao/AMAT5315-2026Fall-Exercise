use std::process::Command;

#[test]
fn cli_defaults_to_cells_and_records_force_method() {
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run", "--eq-steps", "0", "--steps", "1", "--sample-every", "1", "--out"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(json["force_method"], "cells");
}

#[test]
fn cli_can_select_naive_and_records_it() {
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run",
            "--eq-steps",
            "0",
            "--steps",
            "1",
            "--sample-every",
            "1",
            "--force",
            "naive",
            "--out",
        ])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(json["force_method"], "naive");
}

#[test]
fn metadata_without_force_method_reads_as_naive() {
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run", "--eq-steps", "0", "--steps", "1", "--sample-every", "1", "--force", "naive", "--out"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("run.json")).unwrap())
            .unwrap();
    json.as_object_mut().unwrap().remove("force_method");
    std::fs::write(tmp.path().join("run.json"), serde_json::to_vec(&json).unwrap()).unwrap();
    let (meta, _) = md::trajectory::read_trajectory(tmp.path()).unwrap();
    assert_eq!(meta.force_method, md::neighbors::ForceMethod::Naive);
}

#[test]
fn metadata_with_unknown_force_method_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run", "--eq-steps", "0", "--steps", "1", "--sample-every", "1", "--out"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("run.json")).unwrap())
            .unwrap();
    json["force_method"] = serde_json::Value::String("bogus".into());
    std::fs::write(tmp.path().join("run.json"), serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(md::trajectory::read_trajectory(tmp.path()).is_err());
}
