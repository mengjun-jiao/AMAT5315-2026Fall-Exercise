use std::process::Command;
#[test]
fn default_run_passes_all_three_physical_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let exe = env!("CARGO_BIN_EXE_md");
    let out = Command::new(exe)
        .args(["run", "--out"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (_, frames) = md::trajectory::read_trajectory(tmp.path()).unwrap();
    assert_eq!(frames.len(), 200);
    assert_eq!(frames[0].step, 50);
    assert_eq!(frames[199].step, 10000);
    let r = md::check::check_directory(tmp.path()).unwrap();
    eprintln!(
        "drift={:.15e}, T_speed={:.15e}, abs(T-0.5)={:.15e}, chi2/22={:.15e}",
        r.drift,
        r.t_speed,
        (r.t_speed - 0.5).abs(),
        r.mb_score
    );
    assert!(r.drift < 2e-3);
    assert!((r.t_speed - 0.5).abs() < 0.05);
    assert!(r.mb_score < 2.);
    assert!(
        Command::new(exe)
            .arg("check")
            .arg(tmp.path())
            .output()
            .unwrap()
            .status
            .success()
    );
}
#[test]
fn make_reproduce_runs_release_without_video_dependencies() {
    let tmp = tempfile::tempdir().unwrap();
    let week = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    // Isolate outputs while using the real crate. No source-text assertions.
    std::os::unix::fs::symlink(week.join("md"), tmp.path().join("md")).unwrap();
    let makefile = week.join("Makefile");
    assert!(makefile.exists(), "make reproduce must be provided");
    std::fs::copy(makefile, tmp.path().join("Makefile")).unwrap();
    let out = Command::new("make")
        .arg("reproduce")
        .current_dir(tmp.path())
        .env("MD_PYTHON", "/nonexistent/python")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (_, frames) = md::trajectory::read_trajectory(&tmp.path().join("artifacts")).unwrap();
    assert_eq!(frames.len(), 200);
    assert!(!tmp.path().join("fluid.mp4").exists());
    assert!(String::from_utf8_lossy(&out.stderr).contains("release"));
}
