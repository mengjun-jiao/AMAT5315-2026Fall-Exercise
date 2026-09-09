//! Export Rust-computed RDF and invoke the repository's rendering script.
use std::{path::Path,process::Command,io::Write};
fn checked(command:&mut Command,label:&str)->Result<std::process::Output,String> {
    let out=command.output().map_err(|e|format!("{label}: {e}"))?;
    if !out.status.success() { return Err(format!("{label}: {}",String::from_utf8_lossy(&out.stderr))); }
    Ok(out)
}
pub fn render_video(dir:&Path,out:&Path)->Result<(),String> {
    let python=std::env::var_os("MD_PYTHON").unwrap_or_else(||"python3".into());
    let temp=tempfile::tempdir().map_err(|e|e.to_string())?;
    let cache=temp.path().join("matplotlib");
    checked(Command::new(&python).env("MPLCONFIGDIR",&cache).args(["-c","import numpy, matplotlib"]),"Python video dependencies (set MD_PYTHON)")?;
    let encoders=checked(Command::new("ffmpeg").args(["-hide_banner","-encoders"]),"ffmpeg")?;
    if !String::from_utf8_lossy(&encoders.stdout).contains("libx264") { return Err("ffmpeg requires libx264".into()); }
    checked(Command::new("ffprobe").arg("-version"),"ffprobe")?;
    let (meta,frames)=crate::trajectory::read_trajectory(dir)?;
    let mut window=crate::rdf::RdfWindow::new(meta.n,meta.box_size,80,20)?;
    let data=temp.path().join("render.jsonl");
    let mut writer=std::io::BufWriter::new(std::fs::File::create(&data).map_err(|e|e.to_string())?);
    for f in &frames {
        let rdf=window.push(&f.pos)?;
        let row=serde_json::json!({"step":f.step,"t":f.t,"box":meta.box_size,"pos":f.pos,
            "rdf_r":rdf.centers,"rdf_g":rdf.values,"window_frames":rdf.frames,"window_limit":20});
        serde_json::to_writer(&mut writer,&row).map_err(|e|e.to_string())?;
        writeln!(writer).map_err(|e|e.to_string())?;
    }
    writer.flush().map_err(|e|e.to_string())?;
    let parent=out.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e|e.to_string())?;
    let movie=tempfile::Builder::new().suffix(".mp4").tempfile_in(parent).map_err(|e|e.to_string())?;
    let script=Path::new(env!("CARGO_MANIFEST_DIR")).join("../render_fluid.py");
    checked(Command::new(&python).env("MPLCONFIGDIR",&cache).arg(script).arg(&data).arg(movie.path()),"video rendering")?;
    let probe=checked(Command::new("ffprobe").args(["-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0"]).arg(movie.path()),"video frame verification")?;
    let count=String::from_utf8_lossy(&probe.stdout).trim().parse::<usize>().map_err(|e|format!("invalid video frame count: {e}"))?;
    let bytes=movie.as_file().metadata().map_err(|e|e.to_string())?.len();
    if count!=frames.len() || bytes>=2_000_000 || bytes==0 { return Err(format!("video validation failed: frames={count}, bytes={bytes}")); }
    movie.persist(out).map_err(|e|e.to_string())?;
    println!("Saved {}: frames={count}, bytes={bytes}",out.display());
    Ok(())
}
