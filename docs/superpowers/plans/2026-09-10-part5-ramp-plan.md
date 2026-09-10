# Part 5 `--ramp-to` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Each task uses a real red test before production behavior, then a green implementation and commit.

**Goal:** 为周期 LJ 流体加入由 `RunConfig.ramp_to` 驱动的正式阶段线性温度 ramp，并让 `md check` 对加热轨迹执行完整性检查而明确跳过不适用的平衡验收。

**Architecture:** `RunConfig` 是 ramp 温度的唯一来源；现有 `evolve` 保持签名并在完整积分步后读取该字段。轨迹元数据沿用扁平化配置，`check` 保留旧报告字段并增加验收适用状态，CLI 根据该状态输出 PASS/FAIL/SKIP。

**Tech Stack:** 现有 Rust 2024 crate、clap、serde/serde_json、tempfile；不增加依赖，不修改 viewer，不生成视频。

**Spec:** `docs/superpowers/specs/2026-09-10-part5-ramp-design.md`

## Global Constraints

- `ramp_to: Option<f64>` 是唯一 ramp 配置来源；默认 `None`，有限且严格为正的值才有效。
- 顺序固定为完整积分步 → 每 50 个正式步按目标温度缩放 → 采样。
- `steps` 不是 50 的倍数时，最后一步不额外缩放；`ramp_to == temperature` 仍启用缩放。
- 平衡阶段固定使用 `temperature`；无 ramp 的正式阶段完全关闭温控。
- `md check` 始终 naive 重算；加热轨迹完整性成功返回 0，但三项旧验收必须显示 SKIP，不能显示为 PASS。
- 保留旧两原子模型、cell list、所有已有测试、默认参数和阈值；不运行正式加热实验、不生成视频、不 push。
- 每个功能批次先实际取得 red，再实现 green；编译错误先修正，不计为有效 red，不人为制造失败。

## 文件职责与接口

| 文件 | 责任 |
| --- | --- |
| `week2/md/src/fluid.rs` | `RunConfig.ramp_to`、参数校验、正式步调度和缩放顺序 |
| `week2/md/src/trajectory.rs` | 新字段序列化、旧元数据缺字段兼容 |
| `week2/md/src/main.rs` | `--ramp-to` CLI 参数及加热 check 输出 |
| `week2/md/src/check.rs` | 完整性结果与验收适用状态，保持 naive 重算 |
| `week2/md/tests/fluid.rs` | 调度、目标温度、无 ramp 独立步进回归 |
| `week2/md/tests/trajectory.rs` | 元数据往返及旧字段兼容 |
| `week2/md/tests/check.rs` | 加热完整性 PASS/SKIP、篡改拒绝 |
| `week2/md/tests/cli.rs`、`force_cli.rs` | 真实二进制参数和输出 |
| `week2/README.md`、`week2/part5-ramp-validation.md` | 使用说明、测试证据、viewer 手动检查边界 |

约定新增状态字段：

```rust
pub struct CheckReport {
    pub drift: f64,
    pub t_speed: f64,
    pub mb_score: f64,
    pub passed: bool,
    pub acceptance_applicable: bool,
}
```

`passed` 在无 ramp 时表示三项验收通过；启用 ramp 时表示完整性检查通过，`acceptance_applicable` 为 `false`，CLI 据此打印 SKIP。现有数值字段保留以避免破坏旧测试和调用者。

---

### Task 1: 配置、CLI 参数与元数据字段

**Files:**
- Modify: `week2/md/src/fluid.rs`
- Modify: `week2/md/src/main.rs`
- Modify: `week2/md/src/trajectory.rs`
- Test: `week2/md/tests/force_cli.rs`, `week2/md/tests/trajectory.rs`

**Interfaces:**
- `RunConfig` 增加 `pub ramp_to: Option<f64>`，`Default` 为 `None`。
- `RunConfig::validate` 拒绝 `Some(x)` 中非有限或 `x <= 0.0`。
- `RunMetadata` 通过 flatten 自动携带 `ramp_to`；缺字段按 `None` 读取。
- `RunArgs` 增加 `#[arg(long)] ramp_to: Option<f64>` 并写入 `RunConfig`。

- [ ] **Step 1: 写失败测试。** 在真实 CLI 测试中运行 `md run --ramp-to 1.0 --eq-steps 0 --steps 2 --sample-every 1`，断言 `run.json.ramp_to == 1.0`；构造缺少 `ramp_to` 的旧 metadata，断言读为 `None`；对 `0`、`-1` 和非数字值断言拒绝。先补齐测试所需结构字段，不能用错误导入制造 red。
- [ ] **Step 2: 运行测试确认真实失败。**

  Run: `cargo test --manifest-path week2/md/Cargo.toml --test force_cli --test trajectory`

  Expected: ramp CLI 参数无法解析或结构字段缺失；修正编译问题后，测试应因字段未写出/非法值未拒绝而失败。
- [ ] **Step 3: 实现最小配置和 CLI 支持。** 给 `RunConfig` 添加 `#[serde(default)] pub ramp_to: Option<f64>`，在 `validate` 中校验；在 CLI 构造配置时传入 `a.ramp_to`。`RunMetadata` 不复制第二份字段，继续由 flatten 写出。
- [ ] **Step 4: 运行测试确认 green。**

  Run: `cargo test --manifest-path week2/md/Cargo.toml --test force_cli --test trajectory`

  Expected: 新字段写出、旧字段缺失兼容和非法 CLI 参数测试通过，旧轨迹测试保持通过。
- [ ] **Step 5: 提交。**

  ```bash
  git add week2/md/src/fluid.rs week2/md/src/main.rs week2/md/src/trajectory.rs week2/md/tests/force_cli.rs week2/md/tests/trajectory.rs
  git diff --cached --check
  git commit -m "test: add ramp configuration and metadata (red)"
  git commit -m "feat: add ramp configuration and metadata"
  ```

  第二条提交必须在第一条 red 实际失败后创建；若测试批次没有有效失败，不伪造 red，记录原因并只提交 green。

### Task 2: 正式阶段 ramp 调度与采样顺序

**Files:**
- Modify: `week2/md/src/fluid.rs`
- Test: `week2/md/tests/fluid.rs`

**Interfaces:**
- 保持 `evolve<I: Integrator>(integrator: &I, system: &mut System, config: &RunConfig, emit: impl FnMut(Frame) -> Result<(), String>)` 签名。
- 正式循环直接读取 `config.ramp_to`；不增加同时接受 config 和独立温度的公开函数。

- [ ] **Step 1: 写失败测试。** 在 `tests/fluid.rs` 增加真实行为测试；测试辅助函数只读取公开的 `System::velocities` 和位置，不修改生产状态：

  ```rust
  fn thermo(s: &System) -> f64 {
      let kinetic = s.velocities().iter()
          .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
          .sum::<f64>();
      2.0 * kinetic / (2 * s.positions().len() - 2) as f64
  }

  #[test]
  fn ramp_target_is_linear_and_equal_temperature_still_rescales() {
      let mut c = RunConfig { eq_steps: 0, steps: 100, sample_every: 50,
          ramp_to: Some(1.0), ..Default::default() };
      assert_eq!(c.ramp_to, Some(1.0));
      let mut s = initialize(&c).unwrap();
      let mut frames = Vec::new();
      evolve(&VelocityVerlet, &mut s, &c, |f| { frames.push(f); Ok(()) }).unwrap();
      assert_eq!(frames.len(), 2);
      assert!((thermo(&s) - 1.0).abs() < 1e-12);
      c.ramp_to = Some(c.temperature);
      let mut equal = initialize(&c).unwrap();
      let before = equal.velocities().to_vec();
      evolve(&VelocityVerlet, &mut equal, &c, |_| Ok(())).unwrap();
      assert!(before != equal.velocities());
      assert!((thermo(&equal) - c.temperature).abs() < 1e-12);
  }

  #[test]
  fn ramp_sampling_occurs_after_scaling_and_non_multiple_steps_do_not_scale_at_end() {
      let c = RunConfig { eq_steps: 0, steps: 51, sample_every: 50,
          ramp_to: Some(1.0), ..Default::default() };
      let mut actual = initialize(&c).unwrap();
      let mut frames = Vec::new();
      evolve(&VelocityVerlet, &mut actual, &c, |f| { frames.push(f); Ok(()) }).unwrap();
      assert_eq!(frames.len(), 1);
      assert!((frames[0].e_kin - actual.kinetic_energy()).abs() > 0.0);
      assert!((frames[0].e_kin * 2.0 / (2 * c.n - 2) as f64 -
          0.5).abs() < 0.05);
      let mut expected = initialize(&c).unwrap();
      for step in 1..=51 {
          VelocityVerlet.step(&mut expected, c.dt);
          if step == 50 { expected.rescale_temperature(0.5 + 0.5 * 50.0 / 51.0).unwrap(); }
      }
      assert_eq!(actual.velocities(), expected.velocities());
  }

  #[test]
  fn no_ramp_keeps_formal_steps_unthermostatted() {
      let c = RunConfig { eq_steps: 0, steps: 51, sample_every: 50,
          ramp_to: None, ..Default::default() };
      let mut actual = initialize(&c).unwrap();
      evolve(&VelocityVerlet, &mut actual, &c, |_| Ok(())).unwrap();
      let mut expected = initialize(&c).unwrap();
      for _ in 0..51 { VelocityVerlet.step(&mut expected, c.dt); }
      assert_eq!(actual.positions(), expected.positions());
      assert_eq!(actual.velocities(), expected.velocities());
  }
  ```

  测试通过可观察的速度、`E_kin` 和目标温度计算验证行为；无 ramp 对照必须使用独立的真实积分器步进，不能只比较两个委托入口。
- [ ] **Step 2: 运行目标测试确认失败。**

  Run: `cargo test --manifest-path week2/md/Cargo.toml --test fluid -- --nocapture`

  Expected: ramp 目标/缩放/采样测试失败；无 ramp 独立步进回归应保留现有行为，不把已有通过测试改造成失败。
- [ ] **Step 3: 实现正式调度。** 每步先调用 `i.step`；仅当 `config.ramp_to.is_some()` 且 `step % 50 == 0` 时计算 `temperature + (end-temperature) * step as f64 / steps as f64` 并调用 `rescale_temperature`；之后才计算能量和 emit。不要在最后一步添加特殊缩放。
- [ ] **Step 4: 运行测试确认 green。**

  Run: `cargo test --manifest-path week2/md/Cargo.toml --test fluid --test dynamics`

  Expected: 调度测试和旧 Euler/Verlet 测试全部通过。
- [ ] **Step 5: 提交。**

  ```bash
  git add week2/md/src/fluid.rs week2/md/tests/fluid.rs
  git diff --cached --check
  git commit -m "test: cover ramp scheduling and sampling (red)"
  git commit -m "feat: apply formal temperature ramp"
  ```

### Task 3: 加热轨迹 check 语义和 CLI 输出

**Files:**
- Modify: `week2/md/src/check.rs`
- Modify: `week2/md/src/main.rs`
- Test: `week2/md/tests/check.rs`, `week2/md/tests/cli.rs`

**Interfaces:**
- `check::evaluate` 继续重算每帧能量并核对保存值。
- 无 ramp 返回 `acceptance_applicable=true`，沿用原 `passed` 和三项阈值。
- 有 ramp 返回 `acceptance_applicable=false`；完整性成功时 `passed=true`，三项数值不作为失败条件。

- [ ] **Step 1: 写失败测试。** 真实二进制生成短 ramp 轨迹，运行 `md check <dir>`，断言退出码为 0、输出包含三项 `SKIP` 和“加热轨迹完整性检查通过”；分别篡改 `E_pot`、`E_kin`、速度、位置和 JSON 结构，断言仍返回非零并指出完整性错误。
- [ ] **Step 2: 运行测试确认失败。**

  Run: `cargo test --manifest-path week2/md/Cargo.toml --test check --test cli`

  Expected: 当前 CLI 仍按固定温度验收或输出旧 PASS 语义，目标测试失败；若夹具编译失败先补齐 `ramp_to` 字段。
- [ ] **Step 3: 实现状态和输出。** 在 `CheckReport` 增加 `acceptance_applicable`；`evaluate` 从 `meta.ramp_to.is_some()` 判断模式。保留所有结构/有限值/保存能量交叉核对；CLI 分支打印结构数值、能量一致性 PASS/FAIL，并在 ramp 模式打印三项带“不适用于启用 ramp_to 的轨迹”的 SKIP，成功总结为“加热轨迹完整性检查通过”。
- [ ] **Step 4: 运行测试确认 green。**

  Run: `cargo test --release --manifest-path week2/md/Cargo.toml --test check --test cli --test trajectory`

  Expected: ramp 完整性 PASS/SKIP、篡改拒绝、旧 check 输出和旧阈值测试全部通过。
- [ ] **Step 5: 提交。**

  ```bash
  git add week2/md/src/check.rs week2/md/src/main.rs week2/md/tests/check.rs week2/md/tests/cli.rs
  git diff --cached --check
  git commit -m "test: define ramp trajectory check semantics (red)"
  git commit -m "feat: report ramp integrity checks with skips"
  ```

### Task 4: 回归、文档与 viewer 边界记录

**Files:**
- Modify: `week2/README.md`
- Create: `week2/part5-ramp-validation.md`
- Test: `week2/md/tests/acceptance.rs`, `week2/md/tests/trajectory.rs`

**Interfaces:** 使用前述 CLI 与 check 接口；不修改 `week2/week2-viewer.html`。

- [ ] **Step 1: 添加回归测试。** 保留无 ramp 默认完整验收；增加 CLI metadata 缺字段旧轨迹回归和 ramp 短轨迹帧数/时间检查。不要在本步改固定默认参数或运行正式加热实验。
- [ ] **Step 2: 运行回归。**

  ```bash
  cargo test --release --manifest-path week2/md/Cargo.toml --all-targets
  cargo test --manifest-path week2/md/Cargo.toml --doc
  git diff --check
  ```

  视频依赖不属于本功能；不生成 cold/hot 视频。若新增回归直接通过，不创建虚假的 red 提交。
- [ ] **Step 3: 检查 viewer 源码兼容性。** 确认额外 `ramp_to` 字段不会触发未知字段拒绝，且现有 ramp 温度绘图逻辑可读取数字；不修改用户文件，不声称浏览器手动检查通过。
- [ ] **Step 4: 更新文档和验证记录。** 记录参数格式、完整步/缩放/采样顺序、`md check` 的 PASS/SKIP 语义、旧文件兼容、测试命令和未来手动 viewer 检查边界。明确本轮不运行正式加热实验、不生成视频、不 push。
- [ ] **Step 5: 提交并自审。**

  ```bash
  git add week2/README.md week2/part5-ramp-validation.md week2/md/tests/acceptance.rs week2/md/tests/trajectory.rs
  git diff --cached --check
  git commit -m "docs: document ramp trajectory validation"
  ```

  最终检查 `git status` 不包含轨迹、缓存、视频或用户未跟踪 viewer；汇报所有 red/green SHA、测试结果和未执行的正式实验。
