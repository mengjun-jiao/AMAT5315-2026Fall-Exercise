use anyhow::{bail, Result};

unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
    fn enzyme_timestep_primal(
        u_prev: *const f64,
        u_curr: *const f64,
        wave_speed: *const f64,
        sigma: *const f64,
        source: *const f64,
        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,
        u_next: *mut f64,
    );
    fn enzyme_timestep_jvp(
        u_prev: *const f64,
        du_prev: *const f64,
        u_curr: *const f64,
        du_curr: *const f64,
        wave_speed: *const f64,
        dc: *const f64,
        sigma: *const f64,
        source: *const f64,
        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,
        u_next: *mut f64,
        du_next: *mut f64,
    );
    fn enzyme_timestep_vjp(
        u_prev: *const f64,
        u_prev_bar: *mut f64,
        u_curr: *const f64,
        u_curr_bar: *mut f64,
        wave_speed: *const f64,
        wave_speed_bar: *mut f64,
        sigma: *const f64,
        source: *const f64,
        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,
        u_next: *mut f64,
        u_next_bar: *mut f64,
    );
}

fn expected_len(nx: usize, nz: usize) -> Result<usize> {
    nx.checked_mul(nz)
        .ok_or_else(|| anyhow::anyhow!("grid dimensions overflow"))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        bail!("{name} length {actual} does not match required length {expected}");
    }
    Ok(())
}

fn require_grid(nx: usize, nz: usize, fields: &[(&str, usize)]) -> Result<usize> {
    let length = expected_len(nx, nz)?;
    for (name, actual) in fields {
        require_len(name, *actual, length)?;
    }
    Ok(length)
}

pub fn cube_smoke(x: f64) -> Result<[f64; 3]> {
    let mut output = [0.0; 3];
    unsafe { enzyme_cube(x, output.as_mut_ptr()) };
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn timestep_primal(
    u_prev: &[f64],
    u_curr: &[f64],
    wave_speed: &[f64],
    sigma: &[f64],
    source: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) -> Result<Vec<f64>> {
    let length = require_grid(
        nx,
        nz,
        &[
            ("u_prev", u_prev.len()),
            ("u_curr", u_curr.len()),
            ("wave_speed", wave_speed.len()),
            ("sigma", sigma.len()),
            ("source", source.len()),
        ],
    )?;
    let mut u_next = vec![0.0; length];
    unsafe {
        enzyme_timestep_primal(
            u_prev.as_ptr(),
            u_curr.as_ptr(),
            wave_speed.as_ptr(),
            sigma.as_ptr(),
            source.as_ptr(),
            nx,
            nz,
            dx,
            dt,
            u_next.as_mut_ptr(),
        );
    }
    Ok(u_next)
}

#[allow(clippy::too_many_arguments)]
pub fn timestep_jvp(
    u_prev: &[f64],
    du_prev: &[f64],
    u_curr: &[f64],
    du_curr: &[f64],
    wave_speed: &[f64],
    dc: &[f64],
    sigma: &[f64],
    source: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let length = require_grid(
        nx,
        nz,
        &[
            ("u_prev", u_prev.len()),
            ("du_prev", du_prev.len()),
            ("u_curr", u_curr.len()),
            ("du_curr", du_curr.len()),
            ("wave_speed", wave_speed.len()),
            ("dc", dc.len()),
            ("sigma", sigma.len()),
            ("source", source.len()),
        ],
    )?;
    let mut u_next = vec![0.0; length];
    let mut du_next = vec![0.0; length];
    unsafe {
        enzyme_timestep_jvp(
            u_prev.as_ptr(),
            du_prev.as_ptr(),
            u_curr.as_ptr(),
            du_curr.as_ptr(),
            wave_speed.as_ptr(),
            dc.as_ptr(),
            sigma.as_ptr(),
            source.as_ptr(),
            nx,
            nz,
            dx,
            dt,
            u_next.as_mut_ptr(),
            du_next.as_mut_ptr(),
        );
    }
    Ok((u_next, du_next))
}

#[allow(clippy::too_many_arguments)]
pub fn timestep_vjp(
    u_prev: &[f64],
    u_curr: &[f64],
    wave_speed: &[f64],
    sigma: &[f64],
    source: &[f64],
    u_next_bar: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let length = require_grid(
        nx,
        nz,
        &[
            ("u_prev", u_prev.len()),
            ("u_curr", u_curr.len()),
            ("wave_speed", wave_speed.len()),
            ("sigma", sigma.len()),
            ("source", source.len()),
            ("u_next_bar", u_next_bar.len()),
        ],
    )?;
    let mut u_next = vec![0.0; length];
    let mut u_next_bar = u_next_bar.to_vec();
    let mut u_prev_bar = vec![0.0; length];
    let mut u_curr_bar = vec![0.0; length];
    let mut wave_speed_bar = vec![0.0; length];
    unsafe {
        enzyme_timestep_primal(
            u_prev.as_ptr(),
            u_curr.as_ptr(),
            wave_speed.as_ptr(),
            sigma.as_ptr(),
            source.as_ptr(),
            nx,
            nz,
            dx,
            dt,
            u_next.as_mut_ptr(),
        );
        enzyme_timestep_vjp(
            u_prev.as_ptr(),
            u_prev_bar.as_mut_ptr(),
            u_curr.as_ptr(),
            u_curr_bar.as_mut_ptr(),
            wave_speed.as_ptr(),
            wave_speed_bar.as_mut_ptr(),
            sigma.as_ptr(),
            source.as_ptr(),
            nx,
            nz,
            dx,
            dt,
            u_next.as_mut_ptr(),
            u_next_bar.as_mut_ptr(),
        );
    }
    Ok((u_prev_bar, u_curr_bar, wave_speed_bar))
}
