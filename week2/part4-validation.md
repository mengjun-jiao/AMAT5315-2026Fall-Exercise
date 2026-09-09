# Part 4 验证记录

用户已批准实施。无子代理，不运行 Part 5，不 push。最终 fluid.mp4 提交；课程 viewer 尚未由用户手动检查。

基线：cargo test --manifest-path week2/md/Cargo.toml --all-targets：16 passed，退出0；doc tests退出0；/tmp/amat5315-field-venv/bin/python -m pytest week1/：1 passed，退出0。Python NumPy/Matplotlib 已存在；ffmpeg/ffprobe未找到，需配置后执行视频测试。

## Task 1
Red: `cargo test --manifest-path week2/md/Cargo.toml --test physics`，退出101，4个测试实际运行并在Part 4未实现接口失败；旧实现未改。
