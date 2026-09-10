# Part 5 cell list validation

本记录覆盖已实现的邻居搜索优化；实现验收阶段没有加热或网页发布。后续 profiling
与规模测速记录在 README 及 `scaling-results.csv` 中。

## Red/green 提交

- `d8dea04` — `test: cell candidate enumeration (red)`；4 个候选枚举测试实际因接口未实现而失败。
- `bfe00bb` — `feat: cell candidate enumeration (green)`；周期格子去重、回绕和候选完整性测试通过。
- `fc1b69e` — `test: cell force and energy equivalence (red)`；等价性测试实际因带策略物理接口未实现而失败。
- `6fc3980` — `feat: share physics between naive and cell forces (green)`；5 个 cells 测试通过。
- `9c98638` — `test: force method CLI and metadata compatibility (red)`；默认值、显式选择和旧元数据测试实际失败。
- `7d0f4af` — `feat: default CLI force method to cells`；CLI、元数据兼容和旧 schema 回归通过。

Task 4 的补充回归测试直接通过，因此没有人为制造额外的 red 提交。

## 验证命令

从 `week2/` 执行：

```bash
PATH=/tmp/amat5315-ffmpeg:$PATH \
MD_PYTHON=/tmp/amat5315-field-venv/bin/python \
MPLCONFIGDIR=/tmp/amat5315-matplotlib \
cargo test --release --manifest-path md/Cargo.toml --all-targets
cargo test --manifest-path md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest ../week1/
```

默认和显式 naive 的完整运行均保存 200 帧，并通过 check 的能量漂移、速率温度和
Maxwell-Boltzmann 教学阈值；两条路径使用同一初始参数和驱动，但不要求长轨迹逐位一致。

本次从 `week2/` 用重新安装的 PATH 上 `md` 实际运行默认参数，得到：

| Force method | Frames | Energy drift | T_speed | abs(T_speed-0.5) | chi2/22 |
| --- | ---: | ---: | ---: | ---: | ---: |
| naive | 200 | 2.389356121140325e-4 | 4.933536579154182e-1 | 6.646342084581758e-3 | 9.240727272727269e-1 |
| cells | 200 | 6.277620068410122e-5 | 5.084462822705208e-1 | 8.446282270520800e-3 | 9.838545454545456e-1 |
cells viewer 数据由项目积分器生成：

```bash
cargo run --release --locked --manifest-path md/Cargo.toml -- \
  run --force cells --out artifacts/cells-viewer
md/target/release/md check artifacts/cells-viewer
```

课程 viewer 的浏览器手动加载尚未在本记录中声称通过。
