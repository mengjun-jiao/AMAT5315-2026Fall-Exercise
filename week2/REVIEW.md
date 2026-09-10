# Week 2 最终代码审查

日期：2026-09-10
审查方式：新的审查会话；按 `requesting-code-review` 清单由当前会话独立复核，未启动子代理。

## 范围与依据

审查了仓库 `README.md`、`AGENTS.md`，本地可见的 Week 2 学习/验证材料（`week2/README.md`、
`part3-validation.md`、`part4-validation.md`、`part5-cell-list-validation.md`、
`part5-ramp-validation.md`、`video-path-validation.md`），以及 `docs/superpowers/specs/`
和 `docs/superpowers/plans/` 中 Part 3、Part 4、cell list、ramp 的设计与计划。
仓库中未发现单独的本地学习单文件；相关学习单要求已由上述设计、README 和验证记录转述。

实际对照了 `week2/md/src/`、全部 Rust 测试、`Makefile`、视频渲染脚本、CLI 和交付文件。
未修改或提交未跟踪的 `week2/week2-viewer.html`。未重新测速、profiling 或运行正式加热实验，
也未进行干净克隆复现；因此本报告不把这些事项写成已完成证据。

## 审查结论

发现并修复两项真实缺陷；修复后本轮允许的完整回归通过。除下述问题外，未发现与本次审查范围
相关的未修复实现缺陷。

## 发现与修复

### 1. ramp 轨迹会错误进入固定温度验收（已修复）

问题：`check::evaluate` 先调用固定温度/速率分布 metrics，再根据 `ramp_to` 标记验收不可用。
因此一个结构和逐帧能量都合法、但 `T_speed=0` 的 ramp 轨迹会在应全部 `SKIP` 时因
`T_speed <= 0` 返回失败。

影响：ramp 的“完整性 PASS、三项教学验收 SKIP”语义不完整，合法加热轨迹可能被拒绝。

证据：新增测试 `ramp_check_skips_metrics_even_when_speed_is_zero` 在旧实现上实际失败，错误为
`nonpositive/nonfinite T_speed or nonfinite drift`。测试先提交 red：`19fa5fb`。

修复：先逐帧用 naive 物理路径核对保存能量，再在 `ramp_to.is_some()` 时直接返回
`passed=true, acceptance_applicable=false`；无 ramp 路径保持原三项 metrics。修复提交：
`7f22d4c`。该测试及完整 release 回归均通过。

### 2. `md video --out` 不能覆盖已存在的目标文件（已修复）

问题：视频生成完成后使用 `NamedTempFile::persist(out)`，目标文件已存在时会失败。

影响：README 给出的重复生成命令，以及对已有 `fluid.mp4`/其他输出的正常重跑，会在渲染成功后
报告失败；这不符合交付工具应能更新指定输出路径的行为。

证据：新增真实视频测试在第一次编码后再次使用同一 `--out` 路径；旧实现会在第二次调用时
失败。首次尝试运行该测试时系统 `python3` 缺少 NumPy，只到达依赖预检，不能作为代码证据；
随后使用既有 `/home/mengjun/.venvs/amat5315/bin/python` 和可用 ffmpeg/ffprobe 运行，修复后的
覆盖测试通过。测试随 red 提交 `19fa5fb`，修复提交 `7f22d4c`。

修复：在同一输出目录生成临时 MP4，验证帧数和大小后用 `std::fs::rename` 替换目标，保留
验证失败不报告成功的行为。

## 已核对且未发现问题的部分

- 积分：Euler 使用旧位置/速度/加速度；Velocity-Verlet 使用旧加速度半步速度、位置、刷新
  加速度、再半步速度；每一步只刷新一次新加速度，完整步后采样。力方向为
  `f * (x_i-x_j)/r`，成对反向，质量为 1。
- 周期物理：minimum image、位置回绕、`r < rc` 的势能平移截断、截断处力不平移、`i<j`
  单次计能均与设计一致；盒长和重合/非有限状态有校验。
- cell list：网格宽度不小于 `rc`，搜索自身及相邻 8 格；周期格子编号先去重，随后以 `j>i`
  去重原子对。已有测试覆盖双格折叠、边界候选、扰动晶格完整性以及 naive/cells 轨迹一致性。
- 初始化/阶段控制：固定 ChaCha8、去质心、自由度修正温度缩放；平衡阶段每 50 步温控，正式
  无 ramp 时关闭温控；ramp 在完整积分后按正式步号调度，采样在缩放之后，非 50 倍末步不额外缩放。
- check：不推进模拟，使用保存位置/速度以 naive 路径独立重算；保存能量、状态、步号、时间、
  结构、边界和周期重合错误均可拒绝。ramp 修复后完整性与 PASS/SKIP 语义符合设计。
- RDF/视频：Rust 计算 80 箱 RDF，使用 minimum image、二维环面积和最近 20 帧滑动窗口，
  不受力截断限制；Python 只绘制 Rust 输出。视频脚本通过 `include_bytes!` 嵌入，安装后路径
  测试通过；视频检查依赖、H.264、帧数和 `<2,000,000` 字节限制。
- CLI/元数据/交付：`run`、`check`、`video` 参数和默认 force method 符合设计；旧 metadata 缺少
  `force_method`/`ramp_to` 时有兼容默认值，未知 force method 拒绝；Makefile 仅执行 release 默认
  模拟；Cargo.lock、README、验证记录和已提交视频/网页数据均已检查。课程 viewer 文件保持不变。

## 本轮实际运行

以下命令均在现有工作树运行，未运行正式加热实验：

```text
cargo test --release --locked --manifest-path md/Cargo.toml --all-targets   PASS
cargo test --locked --manifest-path md/Cargo.toml --doc                      PASS (0 tests)
/home/mengjun/.venvs/amat5315/bin/python -m pytest ../week1/               PASS (1 passed)
```

针对性 red/green：

```text
ramp_check_skips_metrics_even_when_speed_is_zero（旧实现）              FAIL，预期 T_speed=0 拒绝
ramp_check_skips_metrics_even_when_speed_is_zero（修复后）              PASS
video_binary...（指定 NumPy 环境，含同路径覆盖）                       PASS
```

release all-targets 的最终结果为所有 Rust 单元、集成、RDF、cell list、check、ramp、CLI、轨迹和
视频测试通过；其中默认物理验收测试运行了默认非 ramp 模拟。没有重新运行 profiling、规模测速、
正式加热轨迹或公开 Pages 验证。

## 局限与后续

本审查未替代 clean clone 复现，也未重新确认历史视频、公开 Pages、正式 `n=400` 加热轨迹和
profiling/测速记录；这些是验证记录中已有的历史证据或明确的后续事项。本轮未发现需要因此
修改代码的问题。
