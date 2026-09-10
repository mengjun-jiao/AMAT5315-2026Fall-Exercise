# Week 2 最终干净克隆复现

复现日期：2026-09-10
被验证提交：`6606357003b3734d3c7294d422de6cce6852b999`

## 克隆与环境

使用 `mktemp -d` 创建的新目录：
`/tmp/amat5315-week2-clean.rmPZLu`

该目录由以下 GitHub 仓库直接克隆，没有从原工作树复制 `target/`、`artifacts/` 或任何未跟踪文件：

```text
https://github.com/mengjun-jiao/AMAT5315-2026Fall-Exercise.git
```

克隆后的完整 `HEAD` SHA 与被验证提交一致：
`6606357003b3734d3c7294d422de6cce6852b999`

工具版本：

```text
rustc 1.98.1 (48a229cea 2026-09-01)
cargo 1.98.1 (797e8a9bc 2026-08-05)
Python 3.14.4 (/home/mengjun/.venvs/amat5315/bin/python)
GNU Make 4.4.1
git version 2.53.0
```

复现前执行 `unset CARGO_TARGET_DIR`，因此构建产物位于新克隆内的
`week2/md/target/`。最终实际使用的二进制为：

```text
/tmp/amat5315-week2-clean.rmPZLu/week2/md/target/release/md
sha256: 8225d417714ae18e0854ddd6a1258b4324b5bdbf8ced7ce36ac9580ba57ffc49
```

## 测试命令与结果

以下命令均从新克隆根目录执行；每条命令真实退出码均为 0：

```bash
MD_PYTHON="$HOME/.venvs/amat5315/bin/python" \
MPLCONFIGDIR=/tmp/amat5315-matplotlib \
cargo test --release --locked --manifest-path week2/md/Cargo.toml --all-targets
# exit 0
```

结果：11 个 library unit tests、51 个集成/测试目标中的测试通过，0 失败；main/example
测试目标各为 0 项，均成功完成。

```bash
cargo test --locked --manifest-path week2/md/Cargo.toml --doc
# exit 0
```

结果：0 个 doc tests，成功。

```bash
"$HOME/.venvs/amat5315/bin/python" -m pytest week1/
# exit 0
```

结果：`1 passed`。

## 默认复现与独立验收

以下命令从新克隆的 `week2/` 目录执行：

```bash
make reproduce
# exit 0
```

`make reproduce` 使用 `cargo run --release --locked --manifest-path md/Cargo.toml -- run --out artifacts`，
并生成新克隆内的 `artifacts/run.json`、`artifacts/traj.jsonl`。元数据确认默认配置为
`n=100`、`rho=0.8`、`temperature=0.5`、`eq_steps=2000`、`steps=10000`、
`sample_every=50`、`seed=2026`、`velocity-verlet`、`force_method=cells`、`ramp_to=null`。
轨迹行数为 200，首帧 step=50，末帧 step=10000。

```bash
./md/target/release/md check artifacts
# exit 0
```

这是新克隆构建的 release 二进制，输出的默认非 ramp 验收为：

```text
energy_drift=6.277620068410122e-5 < 2e-3 PASS
T_speed=5.084462822705208e-1; abs(T_speed-0.5)=8.446282270520800e-3 < 0.05 PASS
chi2/22=9.838545454545456e-1 < 2 PASS
PASS
```

因此默认轨迹的三项物理验收全部 PASS；本次没有用加热轨迹的 SKIP 语义代替默认验收。

## 实际问题与限制

本次克隆、构建、测试、默认模拟和默认 `check` 均无失败或环境阻塞。没有运行正式加热实验、
测速或 profiling，没有生成或提交轨迹、构建缓存或视频，也没有修改 viewer。新克隆中的
`target/` 和 `artifacts/` 仅为本次复现产生，未回写原仓库。
