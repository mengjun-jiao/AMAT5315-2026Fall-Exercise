# Part 3 Two-Atom Implementation Plan

> **执行状态：仅供审核，不执行。** 按用户要求，不选择或调用尚未安装的执行流程技能。
> 下列复选框均为未来步骤，代码块是拟议文件内容，并未写入生产代码或运行。

**Goal:** 用同一个泛型驱动比较两个原子的显式 Euler 与 Velocity Verlet，保留 red/green 提交证据。

**Architecture:** 私有动态向量由 System 持有，包含当前位置的加速度缓存。
积分器无状态且只借用 &self。Rust 负责积分、力和能量，Python 负责呈现。

**Tech Stack:** 现有 Rust 2024 crate（不新增 Rust 依赖），Python 3、NumPy、Matplotlib；旧测试需要 pytest。

**Spec:** `docs/superpowers/specs/2026-09-09-part3-two-atom-design.md`

## Global Constraints

- 本轮实验严格只用两个二维原子，质量均为 1，采用约化单位。
- positions、velocities、accelerations 均为私有 Vec<[f64; 2]>，加速度缓存在 System 中。
- 构造时检查输入位置和速度长度一致，加速度向量内部按相同长度分配并初始化。
- Integrator 接口为 step(&self, system: &mut System, dt: f64)，积分器无可变状态。
- 同一个泛型驱动运行 Euler 和 Velocity Verlet，每种算法使用独立且相同的初态。
- Euler 的位置与速度都用旧状态更新；Verlet 初始化后每步只进行一次新的全系统力计算。
- 完整时间步结束后记录能量；另保留 t=0 的初态能量作为误差基准。
- 两种积分器各运行 500 步验收；Verlet 另从同一初态运行到 5000 步观察，不增加长期数值阈值。
- 保留全部旧测试，先提交已验证失败的 red，再提交实现及通过验证的 green。
- Rust 调用现有 energy 和 force 产生数据，Python 只读取数据和绘图。
- 开放边界，使用原始 Lennard-Jones 势；无截断、无温控、无周期边界或邻居表，不运行多原子实验。
- 不 push；实施计划须经用户审核后才能执行。

**学习单固定要求：** 初始位置 `(0,0)`、`(1.2,0)`；初速度均为零；质量均为 1；
`dt=0.01`。Euler 与 Verlet 各运行 500 步，Verlet 另从同一初态运行到 5000 步。
`E0=U(1.2)`，`delta=(E-E0)/abs(E0)`。
Verlet 前 500 步 `max(abs(delta)) < 1e-3`；Euler 第 500 步最终 `delta > 0.5`。
Verlet 5000 步只观察长期误差，不另加数值阈值。以上均为要求，不是待定建议。

## Task 1：建立行为失败并提交 red

**Files:**

- 新增 `week2/md/src/dynamics.rs`：本阶段只放可编译骨架。
- 修改 `week2/md/src/lib.rs`：增加模块与公开重导出，保留所有旧函数、测试。
- 新增 `week2/md/tests/dynamics.rs`：以下完整行为测试。
- 新增 `week2/part3-validation.md`：实际 red 命令、退出码及失败原因；后续追加 green 结果。

**Interfaces:** 消费已有 crate；产生 System、Euler、VelocityVerlet、Integrator、Sample、simulate 的编译接口。

- [ ] 在仓库根目录运行基线，记录实际结果；任何旧测试失败先调查，不进入新实现：

```bash
cargo test --manifest-path week2/md/Cargo.toml --all-targets
cargo test --manifest-path week2/md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest week1/
```

若临时 Python 环境不存在，先按 `week2/README.md` 恢复；不得把缺依赖记录为 red。

- [ ] 写入以下集成测试。首个测试用手算 `r=1`、径向力 `24`，初速度非零，
  能检查错误力方向以及误用新速度更新位置的半隐式 Euler。
  这些独立单步夹具不改变学习单实验初态。

```rust
use md::{Euler, Integrator, System, VelocityVerlet, simulate};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}");
}

fn initial() -> System {
    System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2])
}

#[test]
fn euler_uses_old_position_velocity_and_acceleration() {
    let mut s = System::new(vec![[0.0, 0.0], [1.0, 0.0]],
                            vec![[0.1, 0.2], [-0.1, -0.2]]);
    let integrator = Euler; // Deliberately not mut: exercises &self.
    integrator.step(&mut s, 0.01);
    for (actual, expected) in s.positions().iter().flatten()
        .zip([0.001, 0.002, 0.999, -0.002]) {
        close(*actual, expected, 1e-12);
    }
    for (actual, expected) in s.velocities().iter().flatten()
        .zip([-0.14, 0.2, 0.14, -0.2]) {
        close(*actual, expected, 1e-12);
    }
}

#[test]
#[should_panic(expected = "positions and velocities must have equal lengths")]
fn rejects_mismatched_lengths() {
    System::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![[0.0; 2]]);
}

#[test]
#[should_panic(expected = "at least two atoms")]
fn rejects_empty_system() { System::new(vec![], vec![]); }

#[test]
#[should_panic(expected = "finite state")]
fn rejects_nonfinite_state() {
    System::new(vec![[f64::NAN, 0.0], [1.0, 0.0]], vec![[0.0; 2]; 2]);
}

#[test]
#[should_panic(expected = "nonzero finite separation")]
fn rejects_coincident_atoms() {
    System::new(vec![[0.0; 2]; 2], vec![[0.0; 2]; 2]);
}

#[test]
fn constructor_initializes_vector_force_and_counts_pair_energy_once() {
    let s = System::new(vec![[0.0, 0.0], [0.6, 0.8]],
                        vec![[1.0, 2.0], [-1.0, -2.0]]);
    for (actual, expected) in s.accelerations().iter().flatten()
        .zip([-14.4, -19.2, 14.4, 19.2]) {
        close(*actual, expected, 1e-10);
    }
    close(s.kinetic_energy(), 5.0, 1e-12);
    close(s.potential_energy(), 0.0, 1e-10);
    let well = System::new(vec![[0.0, 0.0], [2_f64.powf(1.0/6.0), 0.0]],
                           vec![[0.0; 2]; 2]);
    close(well.total_energy(), -1.0, 1e-12);
}

#[test]
fn verlet_uses_initial_cache_and_finishes_velocity_update() {
    let mut s = System::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![[0.0; 2]; 2]);
    let integrator = VelocityVerlet;
    integrator.step(&mut s, 0.01);
    close(s.positions()[0][0], -0.0012, 1e-12);
    close(s.positions()[1][0], 1.0012, 1e-12);
    // Independent check of the new force via U's derivative, test only.
    let r = 1.0024;
    let h = 1e-6;
    let f = -(md::energy(r+h) - md::energy(r-h)) / (2.0*h);
    close(s.accelerations()[0][0], -f, 1e-7);
    close(s.velocities()[0][0], -0.12 - 0.005*f, 1e-9);
    close(s.velocities()[1][0], 0.12 + 0.005*f, 1e-9);
}

#[test]
fn samples_are_recorded_after_complete_steps() {
    let mut s = System::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![[0.0; 2]; 2]);
    let rows = simulate(&Euler, &mut s, 0.01, 1);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].step, 0);
    assert_eq!(rows[1].step, 1);
    close(rows[0].total, 0.0, 1e-12);
    close(rows[1].time, 0.01, 1e-12);
    close(rows[1].kinetic, 0.0576, 1e-12);
    close(rows[1].potential, 0.0, 1e-12);
    close(rows[1].total, 0.0576, 1e-12);
}

#[test]
#[should_panic(expected = "positive finite dt")]
fn euler_rejects_zero_dt() {
    Euler.step(&mut initial(), 0.0);
}

#[test]
#[should_panic(expected = "positive finite dt")]
fn verlet_rejects_nan_dt() {
    VelocityVerlet.step(&mut initial(), f64::NAN);
}

#[test]
#[should_panic(expected = "positive finite dt")]
fn driver_rejects_negative_dt_even_for_zero_steps() {
    simulate(&Euler, &mut initial(), -0.01, 0);
}

#[test]
fn fixed_500_step_energy_acceptance() {
    fn deltas<I: Integrator>(integrator: &I) -> Vec<f64> {
        let mut s = initial();
        let rows = simulate(integrator, &mut s, 0.01, 500);
        assert_eq!(rows.len(), 501);
        close(rows[500].time, 5.0, 1e-12);
        for component in s.positions().iter().chain(s.velocities())
            .chain(s.accelerations()).flatten() { assert!(component.is_finite()); }
        for axis in 0..2 {
            close(s.velocities()[0][axis] + s.velocities()[1][axis], 0.0, 1e-12);
        }
        let e0 = md::energy(1.2);
        close(rows[0].total, e0, 1e-12);
        rows.iter().map(|row| {
            assert!(row.total.is_finite());
            (row.total - e0) / e0.abs()
        }).collect()
    }
    let verlet = deltas(&VelocityVerlet);
    let euler = deltas(&Euler);
    let verlet_max = verlet.iter().map(|d| d.abs()).fold(0.0, f64::max);
    let euler_final = euler[500];
    assert!(verlet_max < 1e-3, "Verlet max(abs(delta))={verlet_max}");
    assert!(euler_final > 0.5, "Euler final delta={euler_final}");
}

```

- [ ] 在 `lib.rs` 添加以下两行，其他内容保留：

```rust
mod dynamics;
pub use dynamics::{Euler, Integrator, Sample, System, VelocityVerlet, simulate};
```

- [ ] 添加可编译的接口及 `todo!()` 占位，不添加空 step、恒零能量或其他错误实现。
  只读访问器可直接返回字段；需要实现的行为明确标注未实现。

```rust
pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
    #[cfg(test)]
    force_evaluations: usize,
}
impl System {
    pub fn new(_positions: Vec<[f64; 2]>, _velocities: Vec<[f64; 2]>) -> Self {
        todo!("System::new: validation and initial acceleration")
    }
    pub fn positions(&self) -> &[[f64; 2]] { &self.positions }
    pub fn velocities(&self) -> &[[f64; 2]] { &self.velocities }
    pub fn accelerations(&self) -> &[[f64; 2]] { &self.accelerations }
    pub fn kinetic_energy(&self) -> f64 { todo!("System::kinetic_energy") }
    pub fn potential_energy(&self) -> f64 { todo!("System::potential_energy") }
    pub fn total_energy(&self) -> f64 { todo!("System::total_energy") }
}
pub trait Integrator { fn step(&self, system: &mut System, dt: f64); }
pub struct Euler;
pub struct VelocityVerlet;
impl Integrator for Euler {
    fn step(&self, _: &mut System, _: f64) { todo!("Euler::step") }
}
impl Integrator for VelocityVerlet {
    fn step(&self, _: &mut System, _: f64) { todo!("VelocityVerlet::step") }
}
pub struct Sample {
    pub step: usize, pub time: f64, pub kinetic: f64,
    pub potential: f64, pub total: f64,
}
pub fn simulate<I: Integrator>(_: &I, _: &mut System, _: f64, _: usize) -> Vec<Sample> {
    todo!("simulate: complete-step sampling")
}
```

- [ ] 在同一模块加入 Task 2 末尾的缓存计数测试，随首次 red 一起提交。
  骨架已有 `cfg(test)` 计数字段；当前测试会在构造占位处失败，不伪造计数错误。

- [ ] 运行首次失败测试并检查退出码：

```bash
cargo test --manifest-path week2/md/Cargo.toml --test dynamics euler_uses_old_position_velocity_and_acceleration -- --exact --nocapture
echo $?
```

预期：成功编译、恰好运行 1 个测试，在 `System::new` 的 `todo!()` 处因未实现而失败，
通常退出码为 101。它不是断言失败，也不要求改造成断言失败。
以上仅是预期，记录时以实测为准。若编译错误、零测试、路径或环境错误，先解决这些问题；
确认失败输出指向待实现功能后才算有效 red。不要把带 `#[should_panic]` 的测试因任意 panic
通过当作验证，所以异常测试均明确匹配对应校验错误信息。

- [ ] 运行全部新测试，并单独运行缓存计数测试。旧 Rust 测试通过名字过滤单独核验：

```bash
cargo test --manifest-path week2/md/Cargo.toml --test dynamics
cargo test --manifest-path week2/md/Cargo.toml --lib dynamics::tests::verlet_refreshes_once_at_initialization_and_once_per_step -- --exact --nocapture
cargo test --manifest-path week2/md/Cargo.toml --lib tests::greeting_returns_hello_world -- --exact
cargo test --manifest-path week2/md/Cargo.toml --lib tests::energy_has_unit_well_depth -- --exact
cargo test --manifest-path week2/md/Cargo.toml --lib tests::force_matches_negative_energy_derivative -- --exact
/tmp/amat5315-field-venv/bin/python -m pytest week1/
```

每个旧 Rust 命令应实际运行 1 个测试并通过；旧 Python 测试也应通过。
带有新缓存测试的整个 `--lib` 目标在 red 阶段允许失败，不把它误报成旧测试回归。

- [ ] 在 `week2/part3-validation.md` 记录实际基线、首次失败命令、测试名、退出码、
  `todo!()` 失败位置和原因，以及旧测试通过结果；不预填实测数据。
- [ ] 明确暂存以下文件，检查并提交 red（失败测试不应用 `&&` 与提交命令串联）：

```bash
git add week2/md/src/lib.rs week2/md/src/dynamics.rs week2/md/tests/dynamics.rs week2/part3-validation.md
git diff --cached --check
git commit -m "test: record failing Part 3 dynamics behavior (red)"
git rev-parse HEAD
```

记下实际 red SHA，后续 green 保留它，不 squash 或 amend 掉 red。

## Task 2：实现 System、两种算法与泛型驱动

**Files:** 用下述内容替换 `week2/md/src/dynamics.rs`；保留 red 阶段已加入的缓存计数测试。
**Interfaces:** 与 Task 1 的公开接口完全一致；生产代码没有测试计数 API。

- [ ] 确认 Task 1 的 red 提交已完成，包含集成测试、缓存计数测试和真实失败记录。
  实现过程中若某测试先被构造占位阻塞，应在构造实现后再次运行对应测试，
  确认余下失败来自尚未实现的步骤。无需人为制造错误断言或额外错误实现。

- [ ] 实现以下代码；重导出和集成测试保持 Task 1 的接口与要求。

```rust
pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
    #[cfg(test)]
    force_evaluations: usize,
}
impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert_eq!(positions.len(), velocities.len(),
                   "positions and velocities must have equal lengths");
        assert!(positions.len() >= 2, "at least two atoms");
        assert!(positions.iter().chain(&velocities).flatten().all(|v| v.is_finite()),
                "finite state");
        let accelerations = vec![[0.0; 2]; positions.len()];
        let mut s = Self { positions, velocities, accelerations,
                          #[cfg(test)] force_evaluations: 0 };
        s.refresh_accelerations();
        s
    }
    pub fn positions(&self) -> &[[f64; 2]] { &self.positions }
    pub fn velocities(&self) -> &[[f64; 2]] { &self.velocities }
    pub fn accelerations(&self) -> &[[f64; 2]] { &self.accelerations }
    fn refresh_accelerations(&mut self) {
        #[cfg(test)] { self.force_evaluations += 1; }
        self.accelerations.fill([0.0; 2]);
        for i in 0..self.positions.len() {
            for j in i+1..self.positions.len() {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                let r = dx.hypot(dy);
                assert!(r > 0.0 && r.is_finite(), "nonzero finite separation");
                let radial = crate::force(r);
                for (axis, d) in [dx, dy].into_iter().enumerate() {
                    let f = radial * d / r;
                    assert!(f.is_finite(), "finite force");
                    self.accelerations[i][axis] += f;
                    self.accelerations[j][axis] -= f;
                }
            }
        }
    }
    pub fn kinetic_energy(&self) -> f64 {
        self.velocities.iter().map(|v| 0.5*(v[0]*v[0] + v[1]*v[1])).sum()
    }
    pub fn potential_energy(&self) -> f64 {
        let mut u = 0.0;
        for i in 0..self.positions.len() {
            for j in i+1..self.positions.len() {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                u += crate::energy(dx.hypot(dy));
            }
        }
        u
    }
    pub fn total_energy(&self) -> f64 { self.kinetic_energy() + self.potential_energy() }
}
pub trait Integrator { fn step(&self, system: &mut System, dt: f64); }
pub struct Euler;
pub struct VelocityVerlet;
impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        assert!(dt.is_finite() && dt > 0.0, "positive finite dt");
        for i in 0..system.positions.len() {
            for axis in 0..2 {
                system.positions[i][axis] += system.velocities[i][axis] * dt;
                system.velocities[i][axis] += system.accelerations[i][axis] * dt;
            }
        }
        system.refresh_accelerations();
    }
}
impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        assert!(dt.is_finite() && dt > 0.0, "positive finite dt");
        for i in 0..system.positions.len() {
            for axis in 0..2 {
                system.positions[i][axis] += system.velocities[i][axis]*dt
                    + 0.5*system.accelerations[i][axis]*dt*dt;
                system.velocities[i][axis] += 0.5*system.accelerations[i][axis]*dt;
            }
        }
        system.refresh_accelerations();
        for i in 0..system.positions.len() {
            for axis in 0..2 {
                system.velocities[i][axis] += 0.5*system.accelerations[i][axis]*dt;
            }
        }
    }
}
pub struct Sample {
    pub step: usize, pub time: f64, pub kinetic: f64,
    pub potential: f64, pub total: f64,
}
pub fn simulate<I: Integrator>(integrator: &I, system: &mut System,
                              dt: f64, steps: usize) -> Vec<Sample> {
    assert!(dt.is_finite() && dt > 0.0, "positive finite dt");
    let mut rows = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        if step > 0 { integrator.step(system, dt); }
        let kinetic = system.kinetic_energy();
        let potential = system.potential_energy();
        rows.push(Sample { step, time: step as f64 * dt,
                           kinetic, potential, total: kinetic + potential });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verlet_refreshes_once_at_initialization_and_once_per_step() {
        let mut s = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]);
        assert_eq!(s.force_evaluations, 1);
        let integrator = VelocityVerlet;
        for completed in 1..=5 {
            integrator.step(&mut s, 0.01);
            assert_eq!(s.force_evaluations, 1 + completed);
        }
    }
}
```

- [ ] 运行集成测试及全部 Rust 目标、文档测试和旧 Python 测试，命令同 Task 1。
  验证旧测试仍存在且通过；检查 500 步测试实际误差，失败时调查实现与要求的差异，
  不更改步数/阈值以获得绿灯。若发生实现改动，只重跑受影响与最终必要检查。
- [ ] 将实际 green 结果、red SHA 和通过测试数量补入验证记录，并提交：

```bash
git add week2/md/src/dynamics.rs week2/part3-validation.md
git diff --cached --check
git commit -m "feat: implement two-atom Euler and Verlet dynamics (green)"
```

## Task 3：Rust 导出、Python 绘图与复现说明

**Files:** 新增 `week2/md/examples/two_atoms.rs`、`week2/plot_two_atoms.py`、
`week2/two_atoms_energy.png`；更新 `week2/README.md`、`week2/part3-validation.md`。
**Interfaces:** 消费 Task 2 的 simulate/Sample；CSV 列固定如下；不修改 Part 2 的图片和脚本。

- [ ] 添加 Rust example，分别以同一初态实例化 System；两个算法调用同一个泛型函数。

```rust
use md::{Euler, Integrator, System, VelocityVerlet, simulate};
use std::io::{self, BufWriter, Write};

fn export<I: Integrator>(out: &mut impl Write, name: &str,
                         integrator: &I, steps: usize) -> io::Result<()> {
    let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]);
    let e0 = md::energy(1.2);
    for s in simulate(integrator, &mut system, 0.01, steps) {
        let delta = (s.total - e0) / e0.abs();
        writeln!(out, "{name},{},{},{},{},{},{e0},{delta}",
                 s.step, s.time, s.kinetic, s.potential, s.total)?;
    }
    Ok(())
}
fn main() -> io::Result<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(out, "method,step,time,kinetic,potential,total,e0,delta")?;
    export(&mut out, "Euler", &Euler, 500)?;
    export(&mut out, "VelocityVerlet", &VelocityVerlet, 500)?;
    export(&mut out, "VelocityVerletLong", &VelocityVerlet, 5000)?;
    out.flush()
}

```

- [ ] 添加 Python 绘图脚本，直接验证并消费真实 Rust 输出。下列检查能发现缺失的初态、
  少跑/多跑一步、不同初态、非有限数据或 CSV 能量列不一致。
  delta 由 Rust 导出，Python 不重算物理公式；长期数据只检查完整性，不添加误差阈值。

```python
import io
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    week = Path(__file__).resolve().parent
    result = subprocess.run([
        "cargo", "run", "--quiet", "--manifest-path", str(week / "md/Cargo.toml"),
        "--example", "two_atoms",
    ], check=True, capture_output=True, text=True)
    data = np.genfromtxt(io.StringIO(result.stdout), delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    runs = {"Euler": 500, "VelocityVerlet": 500, "VelocityVerletLong": 5000}
    assert set(data["method"]) == set(runs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    initial_energies = []
    for method, steps in runs.items():
        rows = data[data["method"] == method]
        assert len(rows) == steps + 1
        np.testing.assert_array_equal(rows["step"], np.arange(steps + 1))
        np.testing.assert_allclose(rows["time"], np.arange(steps + 1) * 0.01, atol=1e-12)
        for field in ("time", "kinetic", "potential", "total", "e0", "delta"):
            assert np.isfinite(rows[field]).all()
        np.testing.assert_allclose(rows["total"], rows["kinetic"] + rows["potential"])
        np.testing.assert_allclose(rows["e0"], rows["total"][0], rtol=0, atol=1e-12)
        initial_energies.append(rows["e0"][0])
        col = 1 if steps == 5000 else 0
        label = "VelocityVerlet" if col == 1 else method
        axes[0, col].plot(rows["time"], rows["total"], label=label)
        axes[1, col].plot(rows["time"], rows["delta"], label=label)
        print(f"{method}: max(abs(delta))={max(abs(rows['delta'])):.8g}, "
              f"final delta={rows['delta'][-1]:.8g}")
    np.testing.assert_allclose(initial_energies, initial_energies[0], rtol=0, atol=0)
    short = data[data["method"] == "VelocityVerlet"]
    long = data[data["method"] == "VelocityVerletLong"]
    for field in ("time", "kinetic", "potential", "total", "e0", "delta"):
        np.testing.assert_allclose(short[field], long[field][:501], rtol=0, atol=1e-12)
    for col, title in enumerate(("500 steps: Euler and Verlet", "5000 steps: Verlet observation")):
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Total energy [reduced units]")
        axes[1, col].set_ylabel("delta = (E - E0) / |E0|")
        for ax in axes[:, col]:
            ax.set_xlabel("Time [reduced units]")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle("Two atoms: (0,0), (1.2,0); v=0; mass=1; dt=0.01; open boundaries")
    fig.tight_layout()
    fig.savefig(week / "two_atoms_energy.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
```

- [ ] 从 `week2/` 执行以下准确命令，保留标准输出的实际最大偏差到验证记录。
  若缺依赖，使用现有 README 的环境安装步骤；不在本次文档编写中执行：

```bash
MPLCONFIGDIR=/tmp/amat5315-matplotlib /tmp/amat5315-field-venv/bin/python plot_two_atoms.py
cargo test --manifest-path md/Cargo.toml --all-targets
cargo test --manifest-path md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest ../week1/
```

- [ ] 查看实际图片，检查轴标、图例、两种算法、500/5000 步面板、完整步能量记录。
  若曲线重叠影响阅读，可以增加误差局部图或科学计数格式；不可裁掉错误数据或改变验收阈值。
- [ ] 将上述重建命令、依赖、CSV 数据来源、批准的参数、验收与观察的区别，
  以及 `![两原子能量比较](two_atoms_energy.png)` 追加到 `week2/README.md`。
  `part3-validation.md` 记录实测结果与异常处理，不保存整份 CSV 或构建日志。
- [ ] 在仓库根目录明确暂存相关文件，核查没有缓存、CSV 或构建文件，然后提交：

```bash
git add week2/md/examples/two_atoms.rs week2/plot_two_atoms.py week2/two_atoms_energy.png week2/README.md week2/part3-validation.md
git diff --cached --check
git diff --cached --stat
git commit -m "feat: plot Rust-generated two-atom energy comparison"
git status --short
```

## 本计划审核检查

已逐项核对：动态向量、私有字段、构造一致性、&self 接口、旧状态 Euler、
Verlet 一次刷新、完整步记录、同一泛型驱动、固定 500/5000 步、旧测试保留、
首次行为失败与 red 提交、green 验证、Rust 数据与 Python 呈现、文件及提交范围。
实现代码与测试接口一致。额外技能执行流程未选择。本轮未运行上面的任何实验或测试。

等待用户审核修订后的完整计划，不自动进入执行阶段。学习单参数与阈值已经明确，不再请求确认这些数值。
