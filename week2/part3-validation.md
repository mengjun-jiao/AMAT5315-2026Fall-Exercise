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
