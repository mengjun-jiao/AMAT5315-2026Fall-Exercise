# Part 5 `--ramp-to` 验证记录

本轮实现了可选正式阶段温度 ramp；没有运行 400 原子正式加热实验，没有生成 cold/hot
视频，没有修改课程 viewer，也没有 push。

实现约束：`RunConfig.ramp_to` 是唯一来源；每步顺序为完整积分、每50步缩放、采样；
非50倍数末步不缩放；起止温度相同时仍执行正式阶段缩放。加热轨迹 check 只把结构、
数值和保存能量交叉核对作为完整性条件，三项默认平衡验收输出 SKIP。

## 提交证据

- `99163ff`：ramp 配置、CLI 参数和元数据字段（Task 1 green；字段测试在补齐结构后直接通过，未伪造编译错误为 red）。
- `6a32d8e`：真实 ramp 调度与采样顺序测试（red）。
- `bfef6f3`：正式阶段 ramp 实现（green）。
- `57967a3`：真实加热轨迹 check 语义测试（red）。
- `4e8fe31`：完整性 PASS、三项 SKIP 和篡改拒绝实现（green）。

## 测试

Task 2 的 7 个 fluid 测试全部通过，包含独立 Velocity-Verlet 步进的无 ramp 回归、
50 步缩放、采样顺序、非50倍数末步和相同起止温度仍缩放。Task 3 的 4 个 check 测试
全部通过，包含 ramp 完整性返回0、明确 SKIP、保存能量/状态篡改拒绝和旧 check 回归。
另有短 ramp CLI 帧数、时间和元数据回归。

短程命令及退出码（均为0）：

```bash
md run --ramp-to 1.0 --eq-steps 0 --steps 51 --sample-every 50 --out /tmp/md-ramp-short-final
md check /tmp/md-ramp-short-final
```

输出包含结构/数值 PASS、保存能量与独立重算 PASS、三项 `SKIP` 以及“加热轨迹完整性检查通过”。

最终回归命令：

```bash
PATH=/tmp/amat5315-ffmpeg:$PATH \
MD_PYTHON=/tmp/amat5315-scaling-venv/bin/python \
MPLCONFIGDIR=/tmp/amat5315-matplotlib \
cargo test --release --manifest-path md/Cargo.toml --all-targets
cargo test --manifest-path md/Cargo.toml --doc
```

正式加热实验和 cold/hot 视频留待后续，不在本轮记录结果。

带 NumPy 环境的 Rust release all-targets 回归中，除视频测试外均通过；视频测试未跳过，
实际失败原因为临时 ffmpeg/ffprobe 不存在。静态包下载两次均返回
`curl (35) TLS connect error: unexpected eof while reading`，`sudo apt-get install -y ffmpeg`
又因当前终端无法进行 sudo 认证而返回 `sudo: a terminal is required to authenticate`，
因此本轮不声称视频测试通过。最终非视频 Rust 回归为 9 个 integration test binaries、
47 个测试通过，doc tests 0 个，Week 1 pytest 1 个通过。
