# Part 4：周期 LJ 流体、轨迹验收与视频设计

## 状态与范围

用户已认可主体设计，并批准本文件包含的五项修订；授权保存、提交设计及生成实施计划。
实施计划已获用户批准，在当前会话执行；不 push、不开始 Part 5。
后续在当前会话直接执行，无需安装或调用子代理执行技能。
继续扩展 `week2/md/`，保留 Part 1–3 的功能、测试、examples、图片与复现入口。
当前 Part 3 基线由用户确认：16 个 Rust 测试通过，dimer.png 符合要求。

## CLI 契约与默认配置

必须支持以下准确形式（md 指构建出的二进制）：

```bash
md run --out artifacts
md check artifacts
md video artifacts --out fluid.mp4
```

`run` 提供 `--n`、`--rho`、`--temperature`、`--dt`、`--eq-steps`、`--steps`、
`--sample-every`、`--seed`、`--integrator`、`--out`。
默认值为 n=100、rho=0.8、temperature=0.5、dt=0.01、eq_steps=2000、
steps=10000、sample_every=50、seed=2026、integrator="velocity-verlet"、out="artifacts"。
积分器名称支持 `velocity-verlet` 与 `euler`；物理验收只要求默认运行通过。
`check`、`video` 的输入目录是位置参数；video 的 out 是视频文件路径。
无子命令时保留 greeting；未知子命令、参数或非法输入返回非零并给出错误原因。

## 共享状态、物理模型与积分器

保留 `System` 的私有 `Vec<[f64; 2]>` 位置、速度、加速度缓存及现有访问器。
显式模型枚举提供 `OpenLennardJones` 和 `PeriodicShiftedLennardJones { box_size, rc }`。
`System::new` 仍使用开放边界、原始 LJ，无截断和温控；新增构造入口显式选择周期模型。
原有 `energy(r)`、`force(r)` 保持原义，周期模型调用它们，不更改原函数。

统一成对遍历 i<j；d=x_i-x_j，每轴周期位移 `d -= L*round(d/L)`。
周期模型 rc=2.5，严格要求 rc 小于两边长度的一半；r<rc 时势为 U(r)-U(rc)，
径向力为已有解析 force(r)；r>=rc 时二者为零。只平移势能，不做 force shift。
势能每对计一次，力等大反向，质量均为1。重合或非有限状态明确失败。
使用直接 O(N²) 配对，本轮不增加邻居表、加热流程或 Part 5 实验。

`Integrator::step(&self, system: &mut System, dt: f64)` 保持不变。
Euler 坚持旧位置、旧速度、旧加速度更新；Verlet 保留两次半步速度更新与一次新力计算。
每个完整步后位置通过欧几里得余数包回 [0,Lx)×[0,Ly)，速度不变。
新位置计算力时已使用 minimum image，随后包回不改变物理距离，加速度缓存仍有效。
读取状态和重算能量使用共享借用；积分和速度缩放使用可变借用。
原 `simulate` 的初态采样契约保持；新增 Part 4 驱动管理平衡和正式阶段。

## 初态、平衡与正式阶段

允许 N 为平方数且 sqrt(N) 为偶数，至少为2，并且计算出的盒子通过 rc 条件。
因此支持100、400、1600；非平方数、奇数行数、非正 N 明确报错。
本轮只运行默认100原子实验；400、1600仅检查参数和几何构造，不推进轨迹。
设 m=sqrt(N)，a=sqrt(2/(sqrt(3)*rho))，h=sqrt(3)*a/2，
Lx=m*a，Ly=m*h；i,j=0..m-1，位置=((i+0.5*(j mod 2))*a,j*h)。
rho、temperature、dt 必须为有限正数；steps、sample_every为正且至少能保存一帧；eq_steps可为0。

锁定有种子随机数算法及依赖锁文件，固定粒子遍历顺序。
独立高斯速度分量均值0、方差为目标温度；减去每轴平均速度，再按
T_thermo=2*K/(2*N-2) 缩放 sqrt(T_target/T_thermo)。
初始及每50个完整平衡步缩放，包括默认第2000步。零或非有限温度拒绝缩放。
正式运行重新从 step=0、t=0计时，完全关闭温控。
只在正式完整步 step%sample_every==0 时保存；不保存第0步或额外尾帧。
默认保存step=50,100,...,10000，共200帧；t=step*dt。

## 文件契约

`run.json` 必需字段：n, rho, box:[Lx,Ly], dt, temperature, eq_steps, steps,
sample_every, seed, integrator。rc=2.5和周期平移模型是本版文件契约的固定常量。
`traj.jsonl` 每行一帧，字段step,t,pos,vel,E_pot,E_kin；二维数组长度为n。
能量来自完整步真实状态；pos已包回，E_pot使用平移截断势，E_kin=sum(v²)/2。
流式缓冲写入，不必保留整条轨迹；完整JSON浮点精度，不人为四舍五入。
输出目录可创建，重跑可替换上述两个生成文件；使用临时文件完成写入后替换，失败不报告成功。

## md check

读取run.json及traj.jsonl，从每帧pos、vel直接重算，不调用Integrator、不推进或重建模拟。
重算函数为纯物理函数，run与check可共用；检查器不能以保存能量作为计算输入。
检查JSON结构、必需字段、有限数、数组长度、盒长/密度/N一致、rc合法、包回范围、
周期重合、步号顺序、t=step*dt、帧数floor(steps/sample_every)，拒绝空轨迹及损坏行。
浮点元数据和时间一致性使用 1e-10*max(1,abs(expected)) 的容差。
保存的每项能量与重算值须满足 abs(saved-recomputed)<=1e-10*max(1,abs(recomputed))。
任何错误输出具体文件/帧/字段原因，退出非零。

三个教学验收均基于重算数据：

1. E0=第一保存帧重算总能量，k=max(1,floor(frame_count/10))；
   abs(最后k帧平均能量-最前k帧平均能量)/abs(E0)<2e-3；E0为零或非有限则失败。
2. 汇集全部保存帧速率，T_speed=mean(v²)/2；abs(T_speed-0.5)<0.05。
   这是固定默认验收目标，不随元数据temperature改变；不使用T_thermo自由度修正。
3. 使用T_speed构造24个预测概率相等的箱，
   b_k=sqrt(-2*T_speed*ln(1-k/24)), k=0..23，b_24=正无穷；
   区间左闭右开，M为全部速率个数，Eb=M/24，sum((Ob-Eb)^2/Eb)/22<2。
   T_speed为零或非有限则失败；使用f(v)=v/T_speed*exp(-v²/(2*T_speed))。

逐项输出数值、阈值与PASS/FAIL，总体通过才退出0。第三项不是正式统计显著性检验。
不因验收失败改变种子、参数、势模型、采样方式或阈值；先定位实现原因并报告实测。

## RDF和视频

`md video` 读取轨迹并检查结构，不要求先通过默认物理验收，便于将来复用可视化。
Rust计算RDF，Python/Matplotlib仅绘制原子和提供的RDF，ffmpeg编码MP4。
只在video执行前检查Python、NumPy、Matplotlib、ffmpeg、ffprobe和H.264编码能力。
依赖缺失明确报错，不自动安装；run、check不依赖Python或ffmpeg。

每帧左图为当前包回原子位置，坐标等比例；右图为最近20个保存帧的滑动平均RDF。
不足20帧时使用已有帧，图中标注实际F、最大窗口20及当前正式时间。
80个等宽径向箱，r_max=min(Lx,Ly)/2；使用minimum image，无序对每帧计一次。
最后箱包含r_max端点；r>r_max不计，不受力截断rc限制。
rho=N/(Lx*Ly)，C_b为窗口内F帧的无序对累计计数：

g_b=2*C_b/[F*N*rho*pi*(r_outer²-r_inner²)]。

使用保存每帧直方图的队列，加入新帧，超过20则减去最旧一帧；不做从起点累计平均。
有限N下均匀体系期望(N-1)/N，不再次归一化尾部。
建议画布960×480、20fps，无音频，一条轨迹帧对应一视频帧；默认200帧/10秒。
编码后用ffprobe检查帧数，实际文件必须<2,000,000字节；超限降低码率重编码，
不丢帧、不改物理数据；若仍不能达标返回失败而不是声称完成。

## 复现和版本控制

新增week2/Makefile，从week2执行make reproduce仅以release模式运行默认模拟，
生成week2/artifacts/run.json和traj.jsonl；不绑定check、video、Python或ffmpeg。
README分别给出：

```bash
make reproduce
./md/target/release/md check artifacts
./md/target/release/md video artifacts --out fluid.mp4
```

README区分模拟/检查依赖与视频依赖，保留Part 1–3说明。
artifacts/、target/、视频临时帧和编码临时文件不进Git；最终week2/fluid.mp4必须提交；cold.mp4、hot.mp4也保持可提交，不全局忽略MP4。
提交Cargo.lock以固定Rust依赖，记录实际验证命令与结果，不提交轨迹或构建缓存。

## 测试和后续执行

每个行为批次先写可运行失败测试，实际验证预期失败并提交red，再实现并验证后提交green。
允许最小未实现接口使测试编译；不以编译错误、环境错误、零测试或忽略测试作为red证据。
旧测试在red中仍应通过。记录真实命令、退出码、失败原因和提交SHA，不预填结果。

必须覆盖总内力为零、成对计能、rc-ε处连续性及内侧原始力、minimum image和位置回绕；
初态几何、去质心及温度；完整步缓存和正式阶段不温控；真实CLI二进制与帧数；
check重算、拒绝畸形文件/篡改能量/改状态而保留能量；默认完整运行三项验收；
RDF滑动窗口移除第1帧、二维面积与指定rho归一化、rc以外计数；视频实际帧数和大小。
保留全部旧Rust测试及旧Python测试。400/1600只做构造/配置测试，不运行Part 5轨迹。
用户明确选择当前会话直接执行后续已批准流程，不安装或声称使用缺失的执行技能。

## 设计自审

已核对五项修订：CLI位置参数、Makefile仅run、20帧滑动窗口、N/A归一化、偶数行平方晶格。
已核对旧模型默认值、完整步能量、check独立计算、固定统计阈值及red/green边界。
本文件不声称默认100原子实验已经通过；验收结果由后续真实运行确定。
