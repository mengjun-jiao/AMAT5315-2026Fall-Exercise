# Part 5 `--ramp-to` 加热设计

## 目标

在现有周期 LJ 流体运行流程中加入可选的正式阶段线性升温/降温，同时保留旧的两原子模型、cell list、默认流体行为、轨迹格式和测试。

## 配置与元数据

`RunConfig` 增加唯一的配置来源：

```rust
pub ramp_to: Option<f64>
```

默认值为 `None`。CLI 增加 `md run --ramp-to <TEMPERATURE>`；提供该参数时要求有限且严格为正，等于起始温度也算启用 ramp。`RunConfig::validate` 负责校验该字段，运行驱动不再接受独立的 ramp 温度参数。

`run.json` 通过扁平化 `RunConfig` 写出 `ramp_to`，数值为目标温度，未启用时为 `null`。读取旧文件时使用 `#[serde(default)]`，缺字段解释为 `None`；非法负数、零、非数字或结构错误拒绝读取。其余字段和含义不变。

## 正式运行顺序

平衡阶段完全保持现有行为：初始化并去除质心速度后，以 `config.temperature` 为目标，每完成 50 个平衡步缩放一次。

正式运行每一步严格执行：

1. 完成一个完整的 Euler 或 velocity-Verlet 积分步。
2. 若 `config.ramp_to.is_some()` 且 `step % 50 == 0`，按当前正式步号计算目标温度并缩放速度。
3. 若 `step % sample_every == 0`，保存此时的位置、速度和由此速度重算的动能。

目标温度为：

```text
T_target(step) = temperature
    + (ramp_to - temperature) * step / steps
```

缩放仍使用 `T_thermo = 2K/(2N-2)` 和现有 `rescale_temperature`。目标温度连续随步号变化，但实际瞬时温度只在每 50 步缩放时被调整；两次缩放之间由动力学自然演化。若 `steps` 不是 50 的倍数，最后一步不额外缩放。未提供 `ramp_to` 时正式阶段没有速度缩放；`ramp_to == temperature` 仍在每 50 个正式步执行缩放，因此与 `None` 不同。

驱动函数保留现有 `evolve(integrator, system, config, emit)` 签名，直接读取 `config.ramp_to`，避免重复配置。只有在需要保留旧测试调用方式时，内部可以增加不暴露独立温度参数的私有辅助函数。

## `check` 语义

`md check <directory>` 始终从每帧位置和速度用 naive 物理路径重算势能、动能，并核对保存的 `E_pot`、`E_kin`；同时执行结构、长度、有限值、步号、时间和周期盒坐标检查。任何完整性错误返回非零。

`CheckReport` 保留旧无 ramp 调用者使用的数值字段，并增加能表示“验收是否适用”的状态。无 ramp 时保持原三项阈值和输出：能量漂移、固定 `T=0.5`、二维 Maxwell-Boltzmann 分箱，三项全部通过才总结为原验收 PASS。启用 ramp 时完整性检查成功返回 0，CLI 明确输出：

```text
结构与数值合法性：PASS
保存能量与独立重算一致性：PASS
能量漂移：SKIP（不适用于启用 ramp_to 的轨迹）
固定温度 T=0.5：SKIP（不适用于启用 ramp_to 的轨迹）
平衡速率分布：SKIP（不适用于启用 ramp_to 的轨迹）
加热轨迹完整性检查通过
```

这里的 SKIP 不能被格式化成原三项验收 PASS；若完整性检查失败，仍输出具体 FAIL 并返回非零。加热轨迹不使用能量守恒或固定平衡温度作为失败条件。

## viewer 兼容性

不修改用户的 `week2/week2-viewer.html`。源码检查确认其对额外 `run.json` 字段是容忍的，并且已有 `ramp_to` 读取逻辑可绘制目标温度参考线；后续可用独立加热轨迹手动加载验证，但在用户检查前不声称浏览器验证通过。

## 测试范围

- `ramp_to=None` 的默认运行与旧路径保持一致；保留真实积分器独立步进对照，验证正式阶段没有速度缩放。
- 线性目标温度在起点、中点和终点的数值正确。
- 缩放只发生在正式步 50、100 等调度点；非 50 倍数的最后一步不缩放。
- 同一步缩放和采样时，保存速度及 `E_kin` 反映缩放后的状态。
- `ramp_to == temperature` 仍执行正式阶段缩放。
- `run.json` 新旧格式往返、缺字段兼容、非法参数拒绝。
- 加热轨迹完整性成功返回 0 并显示明确 SKIP；篡改能量或状态仍被拒绝。
- 保留全部 Part 1–5 旧测试，不生成 cold/hot 视频，不发布网页。
