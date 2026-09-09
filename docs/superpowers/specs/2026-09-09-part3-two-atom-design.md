# Part 3：两原子积分实验设计（修订版）

## 状态与依据

用户已认可整体结构，并明确要求本版本使用动态向量与无状态积分器。
实验参数与验收标准按用户明确转述的学习单要求固定如下，不是建议值。
用户已批准按修订计划在当前会话执行 Part 3，不启动子代理。

## 范围

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

## System

在 `week2/md/src/dynamics.rs` 定义 System，由 `lib.rs` 重导出。
`System::new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self`
检查长度相等且至少有两个原子、分量有限、两原子不重合。
加速度不作为构造参数，避免调用者传入不一致或过期的缓存。
构造函数分配 `vec![[0.0; 2]; positions.len()]` 并计算初始加速度，保证三者长度一致。
只公开只读切片访问器，不暴露可变位置/速度/加速度引用。

`refresh_accelerations(&mut self)` 是模块内私有操作。对每个 `i < j`：
令 `d = positions[i] - positions[j]`，`r = |d|`，
`f = crate::force(r) * d / r`，分别累加 `a[i] += f`、`a[j] -= f`。
只调用已有解析力，质量为 1，成对更新确保相互作用反对称。
循环按向量长度编写，为后续课程保留数据结构兼容性，但本轮不增加多原子实验。

`kinetic_energy()` 返回 `sum(|v|² / 2)`；`potential_energy()` 对 `i < j`
调用现有 `crate::energy(r)`，每对只计一次；`total_energy()` 返回两者之和。

## 积分器

```rust
pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}
pub struct Euler;
pub struct VelocityVerlet;
```

步长须为有限正数。每一步结束后加速度缓存对应当前的位置。

Euler 对每个原子执行 `x_new = x_old + v_old * dt`，
`v_new = v_old + a_old * dt`；全部更新后刷新加速度。
必须先使用旧速度更新位置，不能误写为半隐式 Euler。

Velocity Verlet 使用构造时已缓存的 `a_old`：

1. `v += 0.5 * dt * a_old`。
2. `x += dt * v`。
3. 对新位置调用一次 `refresh_accelerations()`，得到 `a_new`。
4. `v += 0.5 * dt * a_new`。

积分器只借用 `&self`，不存储步数、时间或加速度。无需复制整个 System 来保存旧状态。

## 泛型驱动与输出

在同一模块提供 `simulate<I: Integrator>(integrator: &I, system: &mut System,
dt: f64, steps: usize) -> Vec<Sample>`。
`Sample` 含 `step: usize`、`time`、`kinetic`、`potential`、`total: f64`。
先记录第 0 步，再完成每次 step 后记录；500 步对应 501 条记录，5000 步对应 5001 条。
时间由 `step as f64 * dt` 计算，不逐步累加浮点时间。

`week2/md/examples/two_atoms.rs` 用相同初态分别运行两个积分器，输出 CSV：
`method,step,time,kinetic,potential,total,e0,delta`。
Euler 与 Verlet 各运行 500 步，Verlet 另从同一初态运行到 5000 步。
三组数据分别标为 Euler、VelocityVerlet、VelocityVerletLong；不运行 Euler 5000 步。
Rust 用 `E0 = crate::energy(1.2)` 计算 `delta = (E-E0)/abs(E0)`，一并导出。
Python 通过子进程读取 CSV 到内存，不保存中间数据、不重写力或势能公式。
输出 `week2/dimer.png`，左右两个面板：左图显示两算法 500 步的 delta；右图显示 Verlet 5000 步的 1000*delta，纵轴明确标注相对能量误差 ×1000。不增加总能量面板。

## 学习单规定的实验与验收

| 项目 | 固定要求 |
| --- | --- |
| 初始位置 | `(0,0)`、`(1.2,0)`，使用 `vec![[0.0, 0.0], [1.2, 0.0]]` |
| 初始速度 | 均为 `(0,0)` |
| 质量 | 均为 1 |
| 边界与势 | 开放边界，原始 Lennard-Jones 势，无截断、无温控 |
| dt | `0.01` |
| 500 步实验 | Euler、Verlet 各 500 步，终止时间为 5 |
| 长期观察 | Verlet 另运行到 5000 步，终止时间为 50 |
| 初始能量 | `E0 = U(1.2)` |
| 相对误差 | `delta = (E-E0)/abs(E0)`，保留正负号 |
| Verlet 验收 | 前 500 步的 `max(abs(delta)) < 1e-3` |
| Euler 验收 | 第 500 步的最终 `delta > 0.5` |
| 长期观察标准 | 展示 Verlet 5000 步误差，不加额外数值阈值 |

不得用 Euler 最大误差代替最终有符号误差，不得修改初态、步长或阈值使测试通过。
测试用的独立手算单步案例可采用其他两原子位置/速度，它们不是学习单实验。

## 验证与提交边界

以手算的一步 Euler 用例区分显式/半隐式更新；构造、力方向、成对计能、缓存与
一步 Verlet 的位置由独立预期检查。用测试专用计数器验证初始化 1 次、n 步后 1+n 次
加速度刷新，不在生产 API 暴露计数器，不 mock 掉真实力。
500 步实验检查有限状态、动量守恒和能量偏差；5000 步曲线用于观察，不作为另一套阈值。

red 提交允许使用接口及 `todo!()` 占位，使新测试因待实现功能而失败，不强制断言失败。
不得编造错误计算、空积分步骤、恒零能量等假实现来制造失败。
先确认成功编译、测试实际运行，并从失败输出定位到待实现功能；拼写、路径、环境错误、
零测试或忽略测试均不算有效 red。旧测试必须继续通过。
记录实际命令、退出码、失败位置及原因，不预填未经实测的结果。
green 提交实现相同接口，运行原有 Rust/Python 测试、新测试、数据导出及绘图。
仅提交代码、测试、说明和最终图片，不提交 target、CSV、缓存或临时日志。
