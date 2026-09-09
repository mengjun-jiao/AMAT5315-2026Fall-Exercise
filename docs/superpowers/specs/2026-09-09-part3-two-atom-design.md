# Part 3：两原子积分实验设计（修订版）

## 状态与依据

用户已认可整体结构，并明确要求本版本使用动态向量与无状态积分器。
本文保存当前对话可确认的设计；仓库没有更早的 Part 3 设计或学习单副本。
下方实验数值是待审核建议，不冒充此前已批准的参数。本轮只保存文档，不执行。

## 范围

- 本轮实验严格只用两个二维原子，质量均为 1，采用约化单位。
- positions、velocities、accelerations 均为私有 Vec<[f64; 2]>，加速度缓存在 System 中。
- 构造时检查输入位置和速度长度一致，加速度向量内部按相同长度分配并初始化。
- Integrator 接口为 step(&self, system: &mut System, dt: f64)，积分器无可变状态。
- 同一个泛型驱动运行 Euler 和 Velocity Verlet，每种算法使用独立且相同的初态。
- Euler 的位置与速度都用旧状态更新；Verlet 初始化后每步只进行一次新的全系统力计算。
- 完整时间步结束后记录能量；另保留 t=0 的初态能量作为误差基准。
- 固定 500 步验收，5000 步仅作观察，不自动延长验收或调整阈值以通过测试。
- 保留全部旧测试，先提交已验证失败的 red，再提交实现及通过验证的 green。
- Rust 调用现有 energy 和 force 产生数据，Python 只读取数据和绘图。
- 不加入周期边界、截断、温控、邻居表，也不运行 100、400、1600 原子的实验。
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

1. `x_new = x_old + v_old * dt + 0.5 * a_old * dt²`。
2. `v_half = v_old + 0.5 * a_old * dt`。
3. 对新位置调用一次 `refresh_accelerations()`，得到 `a_new`。
4. `v_new = v_half + 0.5 * a_new * dt`。

积分器只借用 `&self`，不存储步数、时间或加速度。无需复制整个 System 来保存旧状态。

## 泛型驱动与输出

在同一模块提供 `simulate<I: Integrator>(integrator: &I, system: &mut System,
dt: f64, steps: usize) -> Vec<Sample>`。
`Sample` 含 `step: usize`、`time`、`kinetic`、`potential`、`total: f64`。
先记录第 0 步，再完成每次 step 后记录；500 步对应 501 条记录，5000 步对应 5001 条。
时间由 `step as f64 * dt` 计算，不逐步累加浮点时间。

`week2/md/examples/two_atoms.rs` 用相同初态分别运行两个积分器，输出 CSV：
`method,step,time,kinetic,potential,total`。
每种算法执行 5000 步；500 步验收由测试独立运行，图中展示前 500 步及全部 5000 步。
Python 通过子进程读取 CSV 到内存，不保存中间数据、不重写力或势能公式。
输出 `week2/two_atoms_energy.png`，显示两算法的总能量及相对初态能量偏差。

## 待审核的数值建议

尚未从可见材料确认初态、步长和验收数值。为使实施计划具体可审查，建议：

| 参数 | 建议值 |
| --- | --- |
| 初始位置 | `[[-0.6, 0.0], [0.6, 0.0]]`（构造时使用 vec!） |
| 初始速度 | 两者均 `[0.0, 0.0]` |
| dt | `0.001` |
| 验收时长 | 500 步，即 0.5 |
| 观察时长 | 5000 步，即 5.0 |
| 误差度量 | `max_n |E_n - E_0| / |E_0|`，包含完整步后的全部样本 |
| Verlet 建议验收上限 | `< 1e-4` |
| 算法对比建议 | Euler 最大相对偏差 `> 10 × Verlet 最大相对偏差` |

这些建议未运行验证，需要用户审核。若学习单另有规定，以学习单为准，先同步本文和计划，
不可运行后悄悄放宽阈值。初始能量非零，故上述相对误差分母有定义。

## 验证与提交边界

以手算的一步 Euler 用例区分显式/半隐式更新；构造、力方向、成对计能、缓存与
一步 Verlet 的位置由独立预期检查。用测试专用计数器验证初始化 1 次、n 步后 1+n 次
加速度刷新，不在生产 API 暴露计数器，不 mock 掉真实力。
500 步实验检查有限状态、动量守恒和能量偏差；5000 步曲线用于观察，不作为另一套阈值。

red 提交使用可编译的最小 API 骨架与真实断言失败；不能把编译错误、测试未发现、
环境错误或 `#[ignore]` 当成行为失败。保留实际命令、退出码和失败断言证据。
green 提交实现相同接口，运行原有 Rust/Python 测试、新测试、数据导出及绘图。
仅提交代码、测试、说明和最终图片，不提交 target、CSV、缓存或临时日志。
