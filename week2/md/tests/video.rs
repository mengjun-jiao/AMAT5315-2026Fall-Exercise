use std::process::Command;
#[test]
fn video_binary_encodes_every_saved_frame_and_checks_dependencies() {
    let tmp=tempfile::tempdir().unwrap();let exe=env!("CARGO_BIN_EXE_md");let mp4=tmp.path().join("fluid.mp4");
    assert!(Command::new(exe).args(["run","--eq-steps","0","--steps","100","--out"]).arg(tmp.path()).output().unwrap().status.success());
    let out=Command::new(exe).arg("video").arg(tmp.path()).arg("--out").arg(&mp4).output().unwrap();
    assert!(out.status.success(),"{}",String::from_utf8_lossy(&out.stderr));
    assert!(std::fs::metadata(&mp4).unwrap().len()<2_000_000);
    let probe=Command::new("ffprobe").args(["-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0"]).arg(&mp4).output().unwrap();
    assert!(probe.status.success());assert_eq!(String::from_utf8(probe.stdout).unwrap().trim(),"2");
    let out=Command::new(exe).env("MD_PYTHON",tmp.path().join("missing-python")).arg("video").arg(tmp.path()).arg("--out").arg(&mp4).output().unwrap();
    assert!(!out.status.success());assert!(String::from_utf8_lossy(&out.stderr).contains("Python"));
}
