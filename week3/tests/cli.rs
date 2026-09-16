use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

fn run(out: &std::path::Path, seed: u64) -> assert_cmd::assert::Assert {
    let mut cmd = Command::cargo_bin("ising").unwrap();
    cmd.args([
        "--update",
        "metropolis",
        "--l",
        "2",
        "--t-from",
        "1",
        "--t-to",
        "2",
        "--t-step",
        "0.5",
        "--discard",
        "2",
        "--measure",
        "3",
        "--every",
        "2",
        "--seed",
        &seed.to_string(),
        "--out",
    ])
    .arg(out)
    .assert()
}

fn run_wolff(out: &std::path::Path) -> assert_cmd::assert::Assert {
    let mut cmd = Command::cargo_bin("ising").unwrap();
    cmd.args([
        "--update",
        "wolff",
        "--l",
        "2",
        "--t-from",
        "1",
        "--t-to",
        "2",
        "--t-step",
        "0.5",
        "--discard",
        "2",
        "--measure",
        "3",
        "--every",
        "2",
        "--seed",
        "2026",
        "--out",
    ])
    .arg(out)
    .assert()
}

#[test]
fn grid_outputs_and_frames_have_contract_counts() {
    let dir = tempdir().unwrap();
    run(dir.path(), 7).success();
    let lines: Vec<_> = fs::read_to_string(dir.path().join("series.jsonl"))
        .unwrap()
        .lines()
        .map(|x| serde_json::from_str::<Value>(x).unwrap())
        .collect();
    assert_eq!(lines.len(), 9);
    let frames_text = fs::read_to_string(dir.path().join("spins.jsonl")).unwrap();
    let frames: Vec<_> = frames_text.lines().collect();
    assert_eq!(frames.len(), 3);
    assert!(frames[0].contains("\"sweep\":4"));
    assert_eq!(
        serde_json::from_str::<Value>(&fs::read_to_string(dir.path().join("run.json")).unwrap())
            .unwrap()["t_grid"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn same_seed_same_outputs_and_invalid_update_is_explicit() {
    let a = tempdir().unwrap();
    let b = tempdir().unwrap();
    run(a.path(), 9).success();
    run(b.path(), 9).success();
    assert_eq!(
        fs::read(a.path().join("series.jsonl")).unwrap(),
        fs::read(b.path().join("series.jsonl")).unwrap()
    );
    let mut cmd = Command::cargo_bin("ising").unwrap();
    cmd.args([
        "--update",
        "invalid",
        "--l",
        "2",
        "--t-from",
        "1",
        "--t-to",
        "1",
        "--t-step",
        "1",
        "--discard",
        "0",
        "--measure",
        "1",
        "--seed",
        "1",
        "--out",
    ])
    .arg(a.path())
    .assert()
    .code(2)
    .stderr(predicates::str::contains("metropolis or wolff"));

    let mut cmd = Command::cargo_bin("ising").unwrap();
    cmd.args([
        "--update",
        "metropolis",
        "--l",
        "1",
        "--t-from",
        "1",
        "--t-to",
        "1",
        "--t-step",
        "1",
        "--discard",
        "0",
        "--measure",
        "1",
        "--seed",
        "1",
        "--out",
    ])
    .arg(a.path())
    .assert()
    .code(2)
    .stderr(predicates::str::contains("at least 2"));

    let mut cmd = Command::cargo_bin("ising").unwrap();
    cmd.args([
        "--update",
        "metropolis",
        "--l",
        "2",
        "--t-from",
        "2",
        "--t-to",
        "1",
        "--t-step",
        "1",
        "--discard",
        "0",
        "--measure",
        "1",
        "--seed",
        "1",
        "--out",
    ])
    .arg(a.path())
    .assert()
    .code(2)
    .stderr(predicates::str::contains("ordered"));
}

#[test]
fn wolff_cli_writes_metadata_rows_and_cluster_sizes() {
    let dir = tempdir().unwrap();
    run_wolff(dir.path())
        .success()
        .stdout(predicates::str::contains("mean_cluster_size"));
    let run: Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(run["update"], "wolff");
    assert_eq!(run["time_unit"], "cluster_flip");
    assert_eq!(run["sample_every"], 1);
    let rows: Vec<Value> = fs::read_to_string(dir.path().join("series.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 9);
    assert!(rows.iter().all(|row| row["cluster_size"]
        .as_u64()
        .is_some_and(|size| (1..=4).contains(&size))));
    let frames = fs::read_to_string(dir.path().join("spins.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 3);
}
