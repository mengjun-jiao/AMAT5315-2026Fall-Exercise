# Week 2: 力场可视化

固定一个原子在二维原点，背景显示另一个原子的势能，箭头显示它受到的力。
采用约化单位（Lennard-Jones 参数 sigma = epsilon = 1）。

现有 `md/src/lib.rs` 中的函数为：

```rust
pub fn energy(r: f64) -> f64 {
    4.0 * (r.powi(-12) - r.powi(-6))
}

/// Returns the radial force: positive is repulsive, negative is attractive.
pub fn force(r: f64) -> f64 {
    24.0 / r * (2.0 * r.powi(-12) - r.powi(-6))
}
```

`force` 直接计算解析式 `-dU/dr`，没有使用数值差分。
已有测试中的中心差分仅用于验证解析力。

`md/examples/field.rs` 在 `[-3, 3] × [-3, 3]` 的 321 × 321 网格上调用
`md::energy(r)` 和 `md::force(r)`，并导出
`Fx = force(r) * x/r`、`Fy = force(r) * y/r`。
`plot_field.py` 自动执行此 Rust example，通过内存中的 CSV 读取结果，
不在 Python 中重新计算势能或力的公式，也不保存临时数据文件。

显示处理：

- Rust 在调用函数前跳过 `r < 0.85`，包括 `r = 0`；图中该区域为灰色。
- 势能色标范围为 `[-1, 1]`，所有 `U > 1` 使用最高颜色；原始数据不截断。
- 箭头每隔 16 个网格点采样。显示长度为
  `L = 0.28 |F|^(1/4) / (1 + |F|^(1/4))`，让远处较弱的力也清晰可见。
  两个分量同时乘以正数 `L / |F|`，方向不变；零力保持为零。
- 虚线圆标出 `r0 = 2^(1/6)`：圆内排斥、圆外吸引，圆上力为零。
  x、y 坐标保持等比例。

## 从 week2/ 重新生成

依赖：支持 Rust 2024 edition 的 Rust/Cargo（Rust 1.85 或更高版本），
Python 3、NumPy、Matplotlib。运行全部测试还需要 pytest。
Python 需要可用的 `venv` 和 `pip`（Ubuntu 可安装 `python3-venv`、`python3-pip`）。

在 `week2/` 目录执行以下命令，依赖安装到 `/tmp` 下的独立环境：

```bash
python3 -m venv /tmp/amat5315-field-venv
/tmp/amat5315-field-venv/bin/python -m pip install numpy matplotlib pytest
/tmp/amat5315-field-venv/bin/python plot_field.py
```

如果系统有 pip，但缺少 ensurepip，可用以下等价方式建立环境和安装依赖：

```bash
python3 -m venv --without-pip /tmp/amat5315-field-venv
python3 -m pip --python /tmp/amat5315-field-venv/bin/python install numpy matplotlib pytest
/tmp/amat5315-field-venv/bin/python plot_field.py
```

图片保存为本目录的 `field.png`。使用无界面的 Agg 后端，不需要图形桌面。
Rust 构建文件位于已被 Git 忽略的 `md/target/`。

## 运行全部测试

同样从 `week2/` 执行：

```bash
cargo test --manifest-path md/Cargo.toml --all-targets
cargo test --manifest-path md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest ../week1/
```

![二维势能与力场](field.png)
