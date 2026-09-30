#![no_std]
#![feature(autodiff)]

use core::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_forward(timestep_jvp, Dual, Dual, Dual, Const, Const, Const, Const, Const, Const, Dual)]
#[autodiff_reverse(timestep_vjp, Duplicated, Duplicated, Duplicated, Const, Const, Const, Const, Const, Const, Duplicated)]
fn timestep(
    u_prev: &[f64],
    u_curr: &[f64],
    wave_speed: &[f64],
    sigma: &[f64],
    source: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    u_next: &mut [f64],
) {
    let inverse_dx_squared = 1.0 / (dx * dx);
    for value in u_next.iter_mut() {
        *value = 0.0;
    }
    for z in 1..(nz - 1) {
        for x in 1..(nx - 1) {
            let index = z * nx + x;
            let laplacian = (u_curr[index + 1]
                + u_curr[index - 1]
                + u_curr[index + nx]
                + u_curr[index - nx]
                - 4.0 * u_curr[index])
                * inverse_dx_squared;
            let damping = sigma[index];
            let c2 = wave_speed[index] * wave_speed[index];
            let numerator = 2.0 * u_curr[index]
                - (1.0 - damping * dt) * u_prev[index]
                + dt * dt * (c2 * laplacian + source[index]);
            u_next[index] = numerator / (1.0 + damping * dt);
        }
    }
}

#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 {
    x * x * x
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (value, tangent) = cube_forward(x, 1.0);
    let (_, adjoint) = cube_reverse(x, 1.0);
    unsafe {
        *out = value;
        *out.add(1) = tangent;
        *out.add(2) = adjoint;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_primal(
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
) {
    let length = nx * nz;
    let u_prev = unsafe { core::slice::from_raw_parts(u_prev, length) };
    let u_curr = unsafe { core::slice::from_raw_parts(u_curr, length) };
    let wave_speed = unsafe { core::slice::from_raw_parts(wave_speed, length) };
    let sigma = unsafe { core::slice::from_raw_parts(sigma, length) };
    let source = unsafe { core::slice::from_raw_parts(source, length) };
    let u_next = unsafe { core::slice::from_raw_parts_mut(u_next, length) };
    timestep(u_prev, u_curr, wave_speed, sigma, source, nx, nz, dx, dt, u_next);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_jvp(
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
) {
    let length = nx * nz;
    let u_prev = unsafe { core::slice::from_raw_parts(u_prev, length) };
    let du_prev = unsafe { core::slice::from_raw_parts(du_prev, length) };
    let u_curr = unsafe { core::slice::from_raw_parts(u_curr, length) };
    let du_curr = unsafe { core::slice::from_raw_parts(du_curr, length) };
    let wave_speed = unsafe { core::slice::from_raw_parts(wave_speed, length) };
    let dc = unsafe { core::slice::from_raw_parts(dc, length) };
    let sigma = unsafe { core::slice::from_raw_parts(sigma, length) };
    let source = unsafe { core::slice::from_raw_parts(source, length) };
    let u_next = unsafe { core::slice::from_raw_parts_mut(u_next, length) };
    let du_next = unsafe { core::slice::from_raw_parts_mut(du_next, length) };
    timestep_jvp(
        u_prev, du_prev, u_curr, du_curr, wave_speed, dc, sigma, source, nx, nz, dx, dt, u_next,
        du_next,
    );
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_vjp(
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
) {
    let length = nx * nz;
    let u_prev = unsafe { core::slice::from_raw_parts(u_prev, length) };
    let u_prev_bar = unsafe { core::slice::from_raw_parts_mut(u_prev_bar, length) };
    let u_curr = unsafe { core::slice::from_raw_parts(u_curr, length) };
    let u_curr_bar = unsafe { core::slice::from_raw_parts_mut(u_curr_bar, length) };
    let wave_speed = unsafe { core::slice::from_raw_parts(wave_speed, length) };
    let wave_speed_bar = unsafe { core::slice::from_raw_parts_mut(wave_speed_bar, length) };
    let sigma = unsafe { core::slice::from_raw_parts(sigma, length) };
    let source = unsafe { core::slice::from_raw_parts(source, length) };
    let u_next = unsafe { core::slice::from_raw_parts_mut(u_next, length) };
    let u_next_bar = unsafe { core::slice::from_raw_parts_mut(u_next_bar, length) };
    for value in u_prev_bar
        .iter_mut()
        .chain(u_curr_bar.iter_mut())
        .chain(wave_speed_bar.iter_mut())
    {
        *value = 0.0;
    }
    timestep_vjp(
        u_prev,
        u_prev_bar,
        u_curr,
        u_curr_bar,
        wave_speed,
        wave_speed_bar,
        sigma,
        source,
        nx,
        nz,
        dx,
        dt,
        u_next,
        u_next_bar,
    );
}

unsafe extern "C" {
    fn abort() -> !;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
