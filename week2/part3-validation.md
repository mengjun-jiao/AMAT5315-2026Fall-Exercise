# Part 3 验证记录

## 基线与 red

基线：原有 3 个 Rust 测试及 1 个 Week 1 pytest 全部通过；Rust 文档测试 0 项。

首次 red 命令（仓库根目录）：

```bash
cargo test --manifest-path week2/md/Cargo.toml --test dynamics euler_uses_old_position_velocity_and_acceleration -- --exact --nocapture
```

成功编译，实际运行 1 个测试，退出码 101。失败输出：

```text
panicked at src/dynamics.rs:10:9:
not yet implemented: System::new: validation and initial acceleration
1 failed; 11 filtered out
```

`cargo test --manifest-path week2/md/Cargo.toml --test dynamics`：12 个新测试均失败于构造占位。
`cargo test --manifest-path week2/md/Cargo.toml --lib -- --nocapture`：原有 3 项通过，
新增缓存计数测试失败于相同占位，退出码 101。
确认失败来自尚未实现的功能，而非编译、拼写、路径或环境错误。
异常输入测试匹配指定错误消息，没有把任意占位 panic 当作成功。
red 只含接口、todo!()、测试和本记录，未编造错误实现。

## Green

red 提交：`fd78873`，保留在历史中。
实现 System 后再次运行测试：5 个集成测试通过，7 个仍在 Euler::step、
VelocityVerlet::step 或 simulate 的 todo!() 处失败；缓存测试进入 Verlet::step 后失败。
随后实现两种积分器与共同驱动，没有修改测试阈值或实验参数。

```bash
cargo test --manifest-path week2/md/Cargo.toml --all-targets
cargo test --manifest-path week2/md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest week1/
```

以上全部退出码 0：Rust 4 个库测试（包含原有 3 个）及 12 个集成测试通过，
文档测试 0 项；Week 1 pytest 1 项通过。原有 field example 仍编译成功。
缓存测试实测初始化 1 次，之后每步恰好增加 1 次真实加速度刷新。
Verlet 使用半步速度、位置、刷新加速度、半步速度的教学顺序。

## Rust 输出、绘图和最终验证

Green 提交：`53135d0`。计划修订提交：`aef1f92`。

Rust example 输出三组数据：Euler 501 行、VelocityVerlet 501 行、
VelocityVerletLong 5001 行，均包含第 0 步。
固定初态 (0,0)、(1.2,0)，零速度，质量 1，dt=0.01，原始势和开放边界。
初始能量由 Rust `md::energy(1.2)` 计算。

实际绘图输出（原始 delta，未乘 1000）：

```text
Euler: max(abs(delta))=1.9556247, final delta=1.9317763
VelocityVerlet: max(abs(delta))=0.00032504759, final delta=-9.414611e-06
VelocityVerletLong: max(abs(delta))=0.0003250492, final delta=-2.0154829e-05
```

Verlet 500 步最大绝对相对误差 < 1e-3；Euler 500 步最终有符号 delta > 0.5。
两项验收通过，未改变参数、步数或阈值。长期观察没有新增数值误差阈值。
图中右面板实际把原始 delta 乘 1000；在 t=0..50 内误差重复振荡，
最大幅度与短程结果接近，没有明显持续漂移。不据此推断无限时间的行为。

绘图脚本验证：列中数值有限、步骤与时间完整、总能量与分项一致、三组初始能量一致，
长期 Verlet 的前 501 条与短程相符。CSV 只经过内存，未提交临时数据。
已人工查看 dimer.png：仅左右两面板，左为两算法 delta，右为 Verlet 1000*delta，
轴标、倍率与图例完整。

从 week2/ 最终验证：

```bash
MPLCONFIGDIR=/tmp/amat5315-matplotlib /tmp/amat5315-field-venv/bin/python plot_two_atoms.py
cargo test --manifest-path md/Cargo.toml --all-targets
cargo test --manifest-path md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest ../week1/
```

全部成功。Rust 16 项测试、Python 1 项测试通过，Rust 文档测试 0 项；
field 与 two_atoms examples 编译成功。原有功能及测试保留。
