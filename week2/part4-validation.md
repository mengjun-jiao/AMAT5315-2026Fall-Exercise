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
Green: red SHA 92c4024；`cargo test --manifest-path week2/md/Cargo.toml --test rdf`退出0，3 passed。覆盖N/A归一化、rc外计数、周期距离、r_max端点、第21帧移除第1帧；未额外缩放尾部。

## Task 6
Task5 green: 0d75e31。系统sudo安装无法非交互认证；独立ffmpeg/ffprobe 7.0.2已下载到/tmp/amat5315-ffmpeg，确认libx264；Python NumPy/Matplotlib导入成功。没有静默跳过视频测试。
Red: `PATH=/tmp/amat5315-ffmpeg:$PATH MD_PYTHON=/tmp/amat5315-field-venv/bin/python MPLCONFIGDIR=/tmp/amat5315-matplotlib cargo test --manifest-path week2/md/Cargo.toml --test video`退出101，真实CLI报告video未实现。
Green: red SHA 91db1ba；同一真实video命令测试退出0，1 passed，真实编码2帧并ffprobe核对，实际文件<2MB；缺失MD_PYTHON明确失败，不跳过。

## Task 7
Task6 green: 7ddc24c。Red: `cargo test --release --manifest-path week2/md/Cargo.toml --test acceptance -- --nocapture`退出101：默认完整物理验收已通过，只有Makefile未提供的真实入口测试失败。没有人为使物理检查失败。
默认seed=2026：200帧；drift=2.389356121140325e-4；T_speed=4.933536579154182e-1；abs(T_speed-0.5)=6.646342084581758e-3；chi2/22=9.240727272727269e-1。
Green: red SHA 26a138e；release acceptance测试退出0，2 passed。Makefile为真实Tab，默认run与make真实集成通过。
亲自在week2执行：`make reproduce`、`./md/target/release/md check artifacts`、`PATH=/tmp/amat5315-ffmpeg:$PATH MD_PYTHON=/tmp/amat5315-field-venv/bin/python ./md/target/release/md video artifacts --out fluid.mp4`，三者退出0。默认轨迹200帧；三个指标与首次测试相同；完整fluid.mp4真实编码200帧、929356字节。
