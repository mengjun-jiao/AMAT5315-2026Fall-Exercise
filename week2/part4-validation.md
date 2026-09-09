# Part 4 验证记录

用户已批准实施。无子代理，不运行 Part 5，不 push。最终 fluid.mp4 提交；课程 viewer 尚未由用户手动检查。

基线：cargo test --manifest-path week2/md/Cargo.toml --all-targets：16 passed，退出0；doc tests退出0；/tmp/amat5315-field-venv/bin/python -m pytest week1/：1 passed，退出0。Python NumPy/Matplotlib 已存在；ffmpeg/ffprobe未找到，需配置后执行视频测试。

## Task 1
Red: `cargo test --manifest-path week2/md/Cargo.toml --test physics`，退出101，4个测试实际运行并在Part 4未实现接口失败；旧实现未改。
Green: `cargo test --manifest-path week2/md/Cargo.toml --all-targets`，退出0，20 passed，含原16测试。red SHA: 1d5629d。

## Task 2
Red: `cargo test --manifest-path week2/md/Cargo.toml --test fluid`，退出101；4测试因未实现config/lattice/initialize/with_model失败。Task1 green: bbaea0e。
Green: red SHA 0d3afbf；`cargo test --manifest-path week2/md/Cargo.toml --all-targets` 退出0，24 passed。随机数固定ChaCha8Rng(seed=2026)，rand_distr StandardNormal；100/400/1600仅晶格测试。依赖下载期间出现网络/未完成下载错误，未计为red；下载完成后上述检查通过。

## Task 3
Task2 green: 4384907。Red: `cargo test --manifest-path week2/md/Cargo.toml --test cli --test trajectory`退出101，CLI 2失败（未生成metadata、未拒绝非法n）；随后单独`--test trajectory`退出101，2失败于write/evolve未实现。不是编译或依赖失败。
Green: red SHA 8b88a25；all-targets退出0，28 passed。测试最初误数metadata字段为11，逐项核对学习单实际10字段，更正测试计数；输出字段未改变。正式阶段无温控与平衡第50步缩放的真实积分比较通过。

## Task 4
Task3 green: 556d099。Red: `cargo test --manifest-path week2/md/Cargo.toml --test check --lib`退出101：metrics未实现，旧lib 4通过。单独`--test check`：重算与真实CLI篡改拒绝测试在未实现接口/CLI失败，退出101。
Green: red SHA a2bf12d；all-targets退出0，31 passed。含独立已知状态重算、等概率速率夹具、能量/状态篡改及畸形文件真实CLI拒绝。check未调用积分器。

## Task 5
Task4 green: 27782d3。Red: `cargo test --manifest-path week2/md/Cargo.toml --test rdf`退出101，3测试实际运行于未实现RDF失败。
