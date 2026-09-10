# Part 5 `--ramp-to` 验证记录

本轮实现了可选正式阶段温度 ramp。随后用户运行并核对了 400 原子加热轨迹、cold/hot
视频，并在课程 viewer 中检查了轨迹首尾状态；本记录不重复运行这些实验，也没有修改
`week2/week2-viewer.html`、启用 Pages 或 push。

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
MD_PYTHON=$HOME/.venvs/amat5315/bin/python \
MPLCONFIGDIR=/tmp/amat5315-matplotlib \
cargo test --release --manifest-path md/Cargo.toml --all-targets
cargo test --manifest-path md/Cargo.toml --doc
```

## 用户核对的加热交付

加热运行参数为 `n=400`、`temperature=0.2`、`ramp_to=1.2`、`eq_steps=2000`、
`steps=20000`、`sample_every=100`、`seed=2026`、`integrator=velocity-verlet`、
`force_method=cells`，轨迹共200帧。用户执行 `md check artifacts/heating`，完整性检查
返回0；课程 viewer 的首尾状态由用户手动查看。加热轨迹不将三项固定温度/能量守恒验收
报告为 PASS，check 输出相应 `SKIP`。

已核对并交付的视频文件规格如下：

| 文件 | 帧数 | 帧率 | 时长 | 分辨率 | 字节数 |
| --- | ---: | ---: | ---: | --- | ---: |
| `week2/cold.mp4` | 200 | 20 fps | 10 s | 960×480 | 843516 |
| `week2/hot.mp4` | 200 | 20 fps | 10 s | 960×480 | 978380 |

网页入口和数据位于 `docs/index.html`、`docs/run.json`、`docs/traj.jsonl`。源码检查确认
页面从同目录 fetch `./run.json` 与 `./traj.jsonl`，并核对了 `n=400`、`steps=20000`、
`sample_every=100`、`temperature=0.2`、`ramp_to=1.2` 及200帧；这不等同于公开网站访问测试。

## Pages 与最终交付核对

公开入口为 <https://mengjun-jiao.github.io/AMAT5315-2026Fall-Exercise/> 。该页面已由用户
在浏览器中手动加载并确认：显示400个原子、200帧，四个图表正常显示。未将截图未证明的
播放或自动播放操作记录为已验证。

截至本记录，学习单要求中已完成 Part 1–5 的实现、测试、图表、视频、加热轨迹完整性
检查、profiling 和规模测速，且交付文件已经提交。检查仓库未发现 `REVIEW.md`，也没有
清洁克隆复现记录或演示录屏；这三项仍是剩余交付工作。本次只更新文档，未重新运行实验、
未修改实现、未修改课程 viewer、未 push。

复制前后 SHA-256 保持一致：

```text
run.json  fe76c7760b2d0b43a051f841de58fcd9db332fd405b2a046294d62dee7cc73a0
traj.jsonl 5dcbc44441800382aa8f2438e473baf3f9ef3a13310cb2762464e6d0a483e06e
```

带 NumPy 环境的 Rust release all-targets 回归中，除视频测试外均通过；视频测试未跳过，
实际失败原因为临时 ffmpeg/ffprobe 不存在。静态包下载两次均返回
`curl (35) TLS connect error: unexpected eof while reading`，`sudo apt-get install -y ffmpeg`
又因当前终端无法进行 sudo 认证而返回 `sudo: a terminal is required to authenticate`，
因此本轮不声称视频测试通过。最终非视频 Rust 回归为 9 个 integration test binaries、
47 个测试通过，doc tests 0 个，Week 1 pytest 1 个通过。
