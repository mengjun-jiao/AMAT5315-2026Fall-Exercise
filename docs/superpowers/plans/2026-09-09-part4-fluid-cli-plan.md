# Part 4 Fluid CLI Implementation Plan

> **状态：用户已批准执行。** 用户指定当前会话直接执行后续获批流程；此指令覆盖技能模板中的子代理/执行技能要求。执行时使用已安装的 test-driven-development，不安装或声称使用 subagent-driven-development、executing-plans、using-git-worktrees，不启动子代理。

**Goal:** 在现有crate中建立run、独立check和滑动窗口RDF视频，保留Part 1–3及真实red/green证据。

**Architecture:** System持有显式物理模型，沿用现有Integrator和加速度缓存。纯物理函数供模拟与轨迹验收共享，轨迹读取不推进模拟。Rust计算物理量和RDF，Python只渲染，ffmpeg编码。

**Tech Stack:** Rust 2024、Cargo、serde/serde_json、clap、rand/rand_chacha/rand_distr、测试用tempfile；视频使用Python 3、NumPy、Matplotlib、ffmpeg/ffprobe；旧Python测试使用pytest。新增依赖选择相互兼容版本并提交Cargo.lock，不预设未经验证的最新版本。

**Spec:** `docs/superpowers/specs/2026-09-09-part4-fluid-cli-design.md`（已提交81dcf1f）。

## Global Constraints

- 用户已批准实施计划，在当前会话执行下列步骤。
- 继续扩展week2/md/；保留旧System::new、Integrator、simulate、greeting、energy、force、examples和旧测试。
- Part 3保持两原子原始LJ、开放边界，无截断无温控；其simulate仍记录第0步。
- Part 4默认n=100,rho=0.8,temperature=0.5,dt=0.01,eq_steps=2000,steps=10000,sample_every=50,seed=2026,integrator="velocity-verlet",out="artifacts"。
- `md run --out artifacts`；`md check artifacts`；`md video artifacts --out fluid.mp4`。
- N为平方数且sqrt(N)为偶数，支持100/400/1600；本轮只推进默认100原子实验，400/1600只做参数及几何测试。
- a=sqrt(2/(sqrt(3)*rho)),h=sqrt(3)*a/2,Lx=sqrt(N)*a,Ly=sqrt(N)*h；交错行三角晶格。
- 质量1；minimum image每轴d-=L*round(d/L)；完整步后位置包回，不改速度。
- rc=2.5且rc<min(Lx,Ly)/2；内侧U(r)-U(rc)、原始解析力；外侧均零，i<j计对。
- 初始去平均速度并缩放；平衡每50个完整步缩放，T_thermo=2*K/(2*N-2)；正式阶段不温控。
- 正式第0步不保存，默认50..10000共200帧，t=step*dt；文件字段与设计一致。
- check只从pos、vel重算，能量交叉容差1e-10*max(1,abs(recomputed))，畸形或任一失败非零退出。
- 首尾k=max(1,floor(F/10))帧均值漂移/abs(第一保存帧E)<2e-3；abs(T_speed-0.5)<0.05；24等概率箱的sum((Ob-Eb)^2/Eb)/22<2。
- RDF最近20保存帧；g=2*C/[F*N*(N/A)*pi*(r_outer²-r_inner²)]；r_max=min(Lx,Ly)/2；不再缩放尾部。
- video每保存帧一帧，标注窗口，MP4<2,000,000字节；仅video依赖Python及ffmpeg。
- make reproduce仅release默认run，生成artifacts/run.json和traj.jsonl，不绑定check/video。
- 每个新增行为批次先实际运行失败测试并提交red，再实现并验证后提交green；旧测试保留。
- 不改变参数/种子/阈值以通过验收，不提前Part 5，不push。

## 文件与接口地图

所有新增Rust模块在lib.rs中公开模块名，旧顶层重导出保持。公共失败结果统一用`Result<T, String>`；CLI打印错误并退出非零，IO/JSON错误转成带路径或行号的String。

| 文件 | 职责 |
| --- | --- |
| md/src/physics.rs | PhysicalModel、位移/回绕、成对势和力、纯能量计算 |
| md/src/dynamics.rs | 现有System新增模型，复用积分器，速度缩放与回绕 |
| md/src/fluid.rs | RunConfig/RunMetadata、晶格、种子速度、平衡和正式驱动 |
| md/src/trajectory.rs | Frame、流式JSONL写入、结构校验读取 |
| md/src/check.rs | 重算交叉核对、三项指标和CheckReport |
| md/src/rdf.rs | 单帧直方图、20帧窗口、归一化 |
| md/src/video.rs | 依赖检查、Rust渲染数据导出、调用渲染脚本 |
| md/src/main.rs | clap解析、三个命令调度、错误退出及原greeting |
| md/tests/physics.rs, fluid.rs, trajectory.rs, check.rs, rdf.rs | 对应单元行为的外部接口测试 |
| md/tests/cli.rs, acceptance.rs, video.rs | 真二进制集成、默认完整验收、真实编码 |
| render_fluid.py | 消费Rust输出，用Agg绘制并经ffmpeg编码 |
| Makefile, README.md, ../.gitignore | 独立模拟复现入口、依赖命令、生成物忽略 |
| part4-validation.md | 实际red/green命令、退出码、SHA、物理指标和视频验证 |

上表路径以week2/为基准；docs中的设计/计划与实现分开。

## 执行与提交通则

- [ ] 实施获批后读test-driven-development及其writing-good-tests.md；不调用缺失技能。
- [ ] 在仓库根运行基线并记录真实结果：

```bash
cargo test --manifest-path week2/md/Cargo.toml --all-targets
cargo test --manifest-path week2/md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest week1/
```

Python环境若不存在，按现有README恢复；依赖/路径失败不是red。
每个任务red阶段只增加测试、依赖及最小可编译接口；新接口可使用`unimplemented!("Part 4 ...")`，不得伪造错误数值。旧入口仍工作。编译成功且实际测试在目标行为失败才可提交red；预期拒绝测试单独通过不能代替正向行为的red。
每次提交前明确暂存本任务文件及validation，运行git diff --cached --check。red提交消息用`test: Part 4 <task> (red)`，green用`feat: Part 4 <task> (green)`。validation只填写真实结果；本计划没有勾选任何执行项。

## Task 1：纯物理模型与边界

**Files:** 新增md/src/physics.rs、md/tests/physics.rs；修改md/src/lib.rs；新增part4-validation.md。

**Interfaces:** `PhysicalModel::{OpenLennardJones, PeriodicShiftedLennardJones { box_size:[f64;2], rc:f64 }}`；`validate(&self)->Result<(),String>`；`displacement(&self,a:[f64;2],b:[f64;2])->[f64;2]`；`wrap(&self,p:[f64;2])->[f64;2]`；`pair(&self,r:f64)->Result<(f64,f64),String>`返回势能和径向力；`accelerations(pos:&[[f64;2]],model:&PhysicalModel)->Result<Vec<[f64;2]>,String>`；`energies(pos:&[[f64;2]],vel:&[[f64;2]],model:&PhysicalModel)->Result<(f64,f64),String>`返回势能、动能。

- [ ] 写失败测试和可编译新接口。核心独立预期：

```rust
use md::physics::{PhysicalModel, accelerations, energies};
#[test]
fn cutoff_is_continuous_from_inside_and_force_is_not_shifted() {
    let m = PhysicalModel::PeriodicShiftedLennardJones { box_size:[8.,6.], rc:2.5 };
    let (u, f) = m.pair(2.5-1e-8).unwrap();
    assert!(u.abs() < 1e-9);
    assert!((f-md::force(2.5-1e-8)).abs() < 1e-12);
    assert!(f.abs() > 0.03);
    assert_eq!(m.pair(2.5).unwrap(), (0.,0.));
    assert_eq!(m.pair(2.6).unwrap(), (0.,0.));
}
#[test]
fn rectangular_boundaries_and_pair_counting() {
    let m = PhysicalModel::PeriodicShiftedLennardJones { box_size:[8.,6.], rc:2.5 };
    assert_eq!(m.displacement([7.5,5.5],[0.5,0.5]), [-1.,-1.]);
    assert_eq!(m.wrap([-16.5,18.5]), [7.5,0.5]);
    let p = [[0.,0.],[7.,0.]];
    let (u,k) = energies(&p,&[[0.;2];2],&m).unwrap();
    assert!((u + md::energy(2.5)).abs()<1e-12);
    assert_eq!(k,0.);
    let a = accelerations(&[[0.,0.],[1.2,0.],[0.4,1.3]],&m).unwrap();
    for axis in 0..2 { assert!(a.iter().map(|v|v[axis]).sum::<f64>().abs()<1e-12); }
}
```

- [ ] 添加rc等于半盒长拒绝、周期重合拒绝和开放模型r>rc仍有作用的断言；运行：

```bash
cargo test --manifest-path week2/md/Cargo.toml --test physics
cargo test --manifest-path week2/md/Cargo.toml --lib --test dynamics
```

- [ ] 记录新测试预期失败、旧测试通过，暂存本任务文件并提交`test: Part 4 physics (red)`。
- [ ] 实现纯函数，输入验证必须先于求倒数：模型盒长/rc有限正数，状态长度一致且N>=2，分量有限，所有成对r有限正数；计算结果也须有限。核心：

```rust
// pair内部：开放模型直接返回原函数；周期模型采用下式。
if r >= rc { return Ok((0.0, 0.0)); }
Ok((crate::energy(r)-crate::energy(rc), crate::force(r)))
// 位移每轴：d[axis] -= box_size[axis]*(d[axis]/box_size[axis]).round();
// 回绕每轴：p[axis] = p[axis].rem_euclid(box_size[axis]);
// 若浮点结果等于L，归为0；开放模型原样返回。
// 配对：a[i][axis] += radial*d[axis]/r; a[j][axis] -= radial*d[axis]/r;
```

- [ ] 重跑上述测试，记录真实结果，提交`feat: Part 4 physics (green)`。

## Task 2：System接入模型及晶格/初速度

**Files:** 修改md/src/dynamics.rs、lib.rs、Cargo.toml、Cargo.lock；新增md/src/fluid.rs、md/tests/fluid.rs。

**Interfaces:** 保留旧System::new；新增`System::with_model(pos:Vec<[f64;2]>,vel:Vec<[f64;2]>,model:PhysicalModel)->Result<Self,String>`、`rescale_temperature(&mut self,target:f64)->Result<(),String>`；内部`wrap_positions(&mut self)`。`RunConfig`字段n:usize,rho/temperature/dt:f64,eq_steps/steps/sample_every:usize,seed:u64,integrator:String；实现Default和validate。`lattice(n:usize,rho:f64)->Result<(Vec<[f64;2]>,[f64;2]),String>`；`initialize(config:&RunConfig)->Result<System,String>`。

- [ ] 写几何、速度、兼容性失败测试：

```rust
use md::fluid::{RunConfig, initialize, lattice};
#[test]
fn geometry_supports_future_sizes_without_running_them() {
    for n in [100,400,1600] {
        let (p,b) = lattice(n,0.8).unwrap();
        assert_eq!(p.len(),n);
        assert!((b[0]*b[1]-n as f64/0.8).abs()<1e-9);
        let a=(2.0/(3.0_f64.sqrt()*0.8)).sqrt();
        let m=(n as f64).sqrt() as usize;
        assert!((p[m][0]-0.5*a).abs()<1e-12);
        assert!(p.iter().all(|x|x[0]>=0. && x[0]<b[0] && x[1]>=0. && x[1]<b[1]));
    }
    for n in [0,99,121,399] { assert!(lattice(n,0.8).is_err()); }
}
#[test]
fn seeded_initial_state_has_no_com_motion_and_target_temperature() {
    let c=RunConfig::default();
    let s=initialize(&c).unwrap(); let again=initialize(&c).unwrap();
    assert_eq!(s.velocities(),again.velocities());
    for axis in 0..2 { assert!(s.velocities().iter().map(|v|v[axis]).sum::<f64>().abs()<1e-10); }
    assert!((2.*s.kinetic_energy()/(2*c.n-2) as f64-c.temperature).abs()<1e-12);
}
```

- [ ] 在dynamics内部补充周期完整步测试：单对远于rc且速度相同，跨边界后位置在盒内、速度不变；Verlet初始化1次、5步总6次刷新。现有16个测试原样保留。
- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --test fluid`及旧测试，确认目标失败，提交`test: Part 4 periodic state and initialization (red)`。
- [ ] 接入模型：旧构造保留原断言消息；新构造验证后分配缓存。势能/加速度委托Task 1，两个积分器末尾调用wrap_positions，原更新顺序不变。
- [ ] 加入兼容的rand、rand_chacha、rand_distr并锁文件。使用ChaCha8Rng::seed_from_u64，按原子顺序x/y依次采样StandardNormal；实现：

```rust
// 初速度每个分量：z * config.temperature.sqrt()，z为独立标准高斯样本。
for axis in 0..2 {
    let mean=vel.iter().map(|v|v[axis]).sum::<f64>()/vel.len() as f64;
    for v in &mut vel { v[axis]-=mean; }
}
let thermo=2.0*system.kinetic_energy()/(2*config.n-2) as f64;
let factor=(config.temperature/thermo).sqrt();
// rescale_temperature内仅将每个速度分量乘factor，不改位置/加速度。
```

晶格按j外循环i内循环生成；验证整数平方与偶数行，rho、T、dt有限正数，steps/sample_every为正且至少一帧，积分器名称合法，盒子满足rc。不初始化400/1600的动力学，仅调用lattice测试。
- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --all-targets`，通过后提交`feat: Part 4 periodic state and initialization (green)`。

## Task 3：正式驱动、轨迹文件及run CLI

**Files:** 新增md/src/trajectory.rs、md/tests/trajectory.rs、md/tests/cli.rs；修改fluid.rs、main.rs、lib.rs、Cargo.toml/Cargo.lock及.gitignore。

**Interfaces:** `RunMetadata`与run.json同名字段（Rust box_size经serde rename="box"）；`Frame { step:usize,t:f64,pos:Vec<[f64;2]>,vel:Vec<[f64;2]>,e_pot:f64,e_kin:f64 }`（后两项serde rename="E_pot"/"E_kin"）。`evolve<I:Integrator>(integrator:&I,system:&mut System,config:&RunConfig,emit:impl FnMut(Frame)->Result<(),String>)->Result<(),String>`；`run_to_directory(config:&RunConfig,dir:&Path)->Result<(),String>`；`read_trajectory(dir:&Path)->Result<(RunMetadata,Vec<Frame>),String>`，读取包括结构检查但不做三项物理验收。

- [ ] 添加serde/serde_json、clap derive和dev依赖tempfile，写真实CLI正向失败测试：

```rust
#[test]
fn run_binary_writes_complete_sampled_frames() {
    let tmp=tempfile::tempdir().unwrap();
    let output=std::process::Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run","--eq-steps","0","--steps","100","--sample-every","50","--out"])
        .arg(tmp.path()).output().unwrap();
    assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    let (meta,frames)=md::trajectory::read_trajectory(tmp.path()).unwrap();
    assert_eq!(meta.n,100); assert_eq!(frames.len(),2);
    assert_eq!(frames.iter().map(|f|f.step).collect::<Vec<_>>(),vec![50,100]);
    assert_eq!(frames[0].t,0.5); assert_eq!(frames[1].pos.len(),100);
}
```

- [ ] 在fluid.rs内部测试平衡调度：用同一真实积分器手工推进50平衡步、缩放，再推进1正式步，与evolve输出逐分量比较；另一夹具eq_steps=0、非目标温度且非零力，比较一正式步与直接Integrator::step，证明正式阶段未缩放。短测试仍N=100，非Part 5实验。
- [ ] 写serde往返测试、无初帧/无额外尾帧测试（steps=101,sample_every=50仍2帧）、非法n CLI非零测试；运行`cargo test --manifest-path week2/md/Cargo.toml --test cli --test trajectory --lib`，确认正向行为失败并提交`test: Part 4 trajectory run CLI (red)`。
- [ ] 实现驱动和流式写入：

```rust
for step in 1..=config.eq_steps {
    integrator.step(system,config.dt);
    if step%50==0 { system.rescale_temperature(config.temperature)?; }
}
for step in 1..=config.steps {
    integrator.step(system,config.dt);
    if step%config.sample_every==0 {
        emit(Frame { step,t:step as f64*config.dt,
            pos:system.positions().to_vec(),vel:system.velocities().to_vec(),
            e_pot:system.potential_energy(),e_kin:system.kinetic_energy() })?;
    }
}
```

序列化完整精度，BufWriter逐行写Frame；元数据严格匹配配置和盒子。先写同目录临时文件，成功flush后替换目标文件，失败清理自己创建的临时文件并返回错误。序列化前检查有限状态；积分数值失败转CLI非零，不能写出成功标记。
- [ ] clap采用`Run { options }`、`Check { directory:PathBuf }`、`Video { directory:PathBuf, #[arg(long)] out:PathBuf }`，后两命令本阶段明确返回未实现错误；无命令仍greeting。run默认选Verlet，euler分支调用同一evolve泛型驱动。
- [ ] read_trajectory验证必需字段、维度、有限数、配置、盒长/密度、坐标范围、连续采样步号/时间/帧数、周期重合；空白或损坏行报行号。步骤计数使用有溢出检查的整数运算。未知附加字段可忽略，不能缺必需字段。
- [ ] .gitignore加入`/week2/artifacts/`，临时视频文件位于临时目录；保持已有target规则。
- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --all-targets`，记录并提交`feat: Part 4 trajectory run CLI (green)`。

## Task 4：独立check和篡改拒绝

**Files:** 新增md/src/check.rs、md/tests/check.rs；修改main.rs、lib.rs、md/tests/cli.rs。

**Interfaces:** `CheckReport { drift:f64,t_speed:f64,mb_score:f64,passed:bool }`；`evaluate(meta:&RunMetadata,frames:&[Frame])->Result<CheckReport,String>`；`check_directory(dir:&Path)->Result<CheckReport,String>`。语法/结构/能量交叉核对错误为Err，三项教学指标失败为passed=false；CLI两者都非零。

- [ ] 写可编译接口和独立能量测试：在合法100晶格上设置已知速度，构造完整Frame；E_kin预期由手算速度个数得出，E_pot由独立i<j、minimum image及现有energy计算，evaluate必须接受能量交叉核对，不能以mock能量代替真实状态。
- [ ] 在check.rs内部给指标函数写独立精确测试（`metrics(energies:&[f64],speeds:&[f64])->Result<CheckReport,String>`是该模块私有函数）：

```rust
#[test]
fn equal_probability_speed_fixture_and_first_saved_energy() {
    let speeds:Vec<f64>=(0..24).flat_map(|j| {
        let v=(-2.0*0.5*(1.0-(j as f64+0.5)/24.0).ln()).sqrt();
        std::iter::repeat_n(v,100)
    }).collect();
    let energies=vec![-100.0;200];
    let report=metrics(&energies,&speeds).unwrap();
    assert_eq!(report.drift,0.0);
    assert!((report.t_speed-0.5).abs()<0.05);
    assert!(report.mb_score<2.0);
    let mut drifted=energies;
    drifted[180..].fill(-99.0);
    assert!((metrics(&drifted,&speeds).unwrap().drift-0.01).abs()<1e-12);
}
```

- [ ] 真二进制先生成短轨迹，保留原文件后分别：改E_pot+1、改E_kin+1、改vel但保留能量、改pos但保留能量、删必需字段、非法JSON、删除一帧、颠倒步号、错误t、数组少原子、盒外坐标、周期重合。逐个执行`md check <目录>`，断言非零及指向对应原因的stderr；不只断言可能由温度验收造成的泛泛FAIL。另构造能量自洽但三项指标不通过的文件，检查正常FAIL输出。
- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --test check --test cli --lib`，确认正向重算测试失败，提交`test: Part 4 independent trajectory check (red)`。
- [ ] 实现纯读取路径，只调用read_trajectory和physics::energies；不创建System、不调用initialize/evolve/step。对每帧分别核对两项保存能量，然后只把重算总能量和由vel得到的速率送入metrics。
- [ ] 实现指标核心：

```rust
let k=(energies.len()/10).max(1);
let first=energies[..k].iter().sum::<f64>()/k as f64;
let last=energies[energies.len()-k..].iter().sum::<f64>()/k as f64;
let drift=(last-first).abs()/energies[0].abs();
let t_speed=speeds.iter().map(|v|v*v).sum::<f64>()/(2.0*speeds.len() as f64);
let edges:Vec<f64>=(1..24).map(|k|(-2.0*t_speed*(1.0-k as f64/24.0).ln()).sqrt()).collect();
let mut counts=[0usize;24];
for &v in speeds { counts[edges.partition_point(|&b| v>=b)]+=1; }
let expected=speeds.len() as f64/24.0;
let mb_score=counts.iter().map(|&o|(o as f64-expected).powi(2)/expected).sum::<f64>()/22.0;
let passed=drift<2e-3 && (t_speed-0.5).abs()<0.05 && mb_score<2.0;
```

先拒绝空输入、零/非有限E0、非正/非有限T_speed和非有限派生指标；所有阈值严格小于。报告明确教学验收，不输出p值。
- [ ] 重跑上述测试及`cargo test --manifest-path week2/md/Cargo.toml --all-targets`，提交`feat: Part 4 independent trajectory check (green)`。

## Task 5：最近20帧RDF

**Files:** 新增md/src/rdf.rs、md/tests/rdf.rs；修改lib.rs。

**Interfaces:** `RdfWindow::new(n:usize,box_size:[f64;2],bins:usize,window:usize)->Result<Self,String>`；`push(&mut self,pos:&[[f64;2]])->Result<RdfSample,String>`；`RdfSample { centers:Vec<f64>,values:Vec<f64>,frames:usize }`。内部VecDeque保存每帧无序对计数和累计箱计数。

- [ ] 写独立面积归一化与rc外计数测试：

```rust
#[test]
fn rdf_uses_density_n_over_area_beyond_force_cutoff() {
    let mut w=md::rdf::RdfWindow::new(2,[10.,10.],5,20).unwrap();
    let s=w.push(&[[0.,0.],[3.,0.]]).unwrap();
    // r=3在[3,4)箱；rho=2/100，F=1，N=2，C=1。
    let expected=2.0/(2.0*(2.0/100.0)*std::f64::consts::PI*(16.0-9.0));
    assert!((s.values[3]-expected).abs()<1e-12);
    assert_eq!(s.frames,1);
}
#[test]
fn twenty_first_frame_evicts_first_instead_of_accumulating() {
    let mut w=md::rdf::RdfWindow::new(2,[10.,10.],5,20).unwrap();
    w.push(&[[0.,0.],[1.5,0.]]).unwrap();
    for _ in 0..19 { w.push(&[[0.,0.],[3.5,0.]]).unwrap(); }
    let last=w.push(&[[0.,0.],[3.5,0.]]).unwrap();
    assert_eq!(last.frames,20); assert_eq!(last.values[1],0.0);
}
```

- [ ] 加入跨周期边界对、r_max端点计入最后箱、r>r_max不计、F=1和F=20一致归一化测试；运行`cargo test --manifest-path week2/md/Cargo.toml --test rdf`并提交`test: Part 4 sliding RDF (red)`。
- [ ] 实现minimum image计对，bin=floor(r/dr)，r==r_max夹到最后箱，先排除r>r_max。窗口核心：

```rust
// counts为当前帧每箱无序对数；total为窗口总数，history为VecDeque。
for (sum,&c) in total.iter_mut().zip(&counts) { *sum+=c; }
history.push_back(counts);
if history.len()>window {
    let old=history.pop_front().unwrap();
    for (sum,c) in total.iter_mut().zip(old) { *sum-=c; }
}
let rho=n as f64/(box_size[0]*box_size[1]);
// 每箱：g=2.0*total[b] as f64/(history.len() as f64*n as f64*rho*area_b)。
```

- [ ] 通过测试后提交`feat: Part 4 sliding RDF (green)`；记录有限N期望(N-1)/N，不加尾部缩放。

## Task 6：真实视频管线

**Files:** 新增md/src/video.rs、md/tests/video.rs、render_fluid.py；修改main.rs、lib.rs、Cargo.toml/Cargo.lock（生产临时文件需要tempfile）。

**Interfaces:** `render_video(dir:&Path,out:&Path)->Result<(),String>`。Rust导出临时JSONL，每行`{step,t,box,pos,rdf_r,rdf_g,window_frames,window_limit}`，window_limit=20；调用脚本`render_fluid.py INPUT OUTPUT`，脚本只消费这些数值。支持环境变量MD_PYTHON指定解释器，默认python3；脚本路径基于CARGO_MANIFEST_DIR的父目录，仓库内运行。

- [ ] 先检查解释器import numpy/matplotlib、ffmpeg/ffprobe可运行、libx264可用；缺失按README安装后再做测试，不把缺依赖作为red。此检查不放进run/check/Makefile。
- [ ] 写真二进制测试：run生成100原子、100正式步、每50步2帧的短轨迹，然后执行：

```rust
let status=std::process::Command::new(env!("CARGO_BIN_EXE_md"))
    .arg("video").arg(dir).arg("--out").arg(&mp4).status().unwrap();
assert!(status.success());
assert!(std::fs::metadata(&mp4).unwrap().len()<2_000_000);
let probe=std::process::Command::new("ffprobe")
    .args(["-v","error","-count_frames","-select_streams","v:0",
           "-show_entries","stream=nb_read_frames","-of","csv=p=0"])
    .arg(&mp4).output().unwrap();
assert!(probe.status.success());
assert_eq!(String::from_utf8(probe.stdout).unwrap().trim(),"2");
```

测试用标准#[test]，不标ignored；隔离PATH的缺依赖测试启动子进程，不修改并行测试进程的全局环境。验证video可处理能量自洽但默认物理指标FAIL的短轨迹。
- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --test video`，确认未实现视频而失败，提交`test: Part 4 fluid video (red)`。
- [ ] Rust调用结构读取和RdfWindow，逐帧导出，使用Command参数数组启动Python，不拼接shell。临时文件放TempDir；依赖检查/渲染/编码任一步失败返回stderr上下文。
- [ ] Python使用Agg创建960×480双图，左图scatter且盒子等比例，右图曲线使用rdf_r/rdf_g；轴标与标题实际更新：

```python
atoms.set_offsets(frame["pos"])
line.set_data(frame["rdf_r"], frame["rdf_g"])
left.set_title(f"t={frame['t']:.2f}")
right.set_title(f"g(r): last {frame['window_frames']}/{frame['window_limit']} saved frames")
fig.canvas.draw()
encoder.stdin.write(fig.canvas.buffer_rgba())
```

固定RDF横轴0..min(box)/2；纵轴上限由整段已计算RDF最大值决定，避免帧间跳动。绘图后将RGBA管道送ffmpeg，20fps、libx264、yuv420p、无音频、faststart，不重复或丢帧。
- [ ] 按视频时长duration=F/20计算首轮码率min(900000,floor(1_700_000*8/duration)) bit/s，编码实际测大小；超限按0.7倍码率重试最多4次，每次从同一帧数据开始。失败或仍超限返回错误；成功文件先写同目录临时MP4再替换out。
- [ ] ffprobe核对帧数=F，视频文件<2,000,000字节；运行视频测试、查看抽取首/中/尾帧核对原子和窗口标签后提交`feat: Part 4 fluid video (green)`。

## Task 7：默认物理验收与独立make reproduce

**Files:** 新增week2/Makefile、md/tests/acceptance.rs；更新README.md、part4-validation.md、md/tests/cli.rs。

**Interfaces:** Makefile独立目标reproduce，只调用默认run；默认验收调用已实现二进制run、check与read_trajectory。文档不引入新接口。

- [ ] 在CLI测试加入读取Makefile并通过实际`make -C week2 reproduce`验证的集成场景，设置CARGO_TARGET_DIR到受忽略或临时构建目录时仍应使用cargo run；不要求生成视频。测试使用单独临时week2布局或输出目录隔离，避免并行测试覆盖用户artifacts。
- [ ] 新增完整默认验收测试（标准#[test]、不ignored）：

```rust
#[test]
fn default_run_passes_all_three_physical_checks() {
    let tmp=tempfile::tempdir().unwrap();
    let exe=env!("CARGO_BIN_EXE_md");
    let run=std::process::Command::new(exe).args(["run","--out"])
        .arg(tmp.path()).output().unwrap();
    assert!(run.status.success(),"{}",String::from_utf8_lossy(&run.stderr));
    let (_,frames)=md::trajectory::read_trajectory(tmp.path()).unwrap();
    assert_eq!(frames.len(),200); assert_eq!(frames[0].step,50);
    assert_eq!(frames[199].step,10000);
    let report=md::check::check_directory(tmp.path()).unwrap();
    assert!(report.drift<2e-3,"drift={}",report.drift);
    assert!((report.t_speed-0.5).abs()<0.05,"T_speed={}",report.t_speed);
    assert!(report.mb_score<2.0,"MB={}",report.mb_score);
    assert!(std::process::Command::new(exe).arg("check").arg(tmp.path()).status().unwrap().success());
}
```

- [ ] 运行`cargo test --release --manifest-path week2/md/Cargo.toml --test acceptance --test cli`。默认验收可能已经通过；本批次red必须来自尚未实现的Makefile正向测试，不能故意破坏已通过的物理算法。记录各自真实结果并提交`test: Part 4 reproduction and default acceptance (red)`。
- [ ] 实现Makefile（命令行开头是真tab）：

```make
.PHONY: reproduce
reproduce:
	cargo run --release --locked --manifest-path md/Cargo.toml -- run --out artifacts
```

Makefile测试创建TempDir，将同样的Makefile复制到临时week2，md路径链接到真实crate；在该临时目录执行make，验证两个输出文件及200帧，没有fluid.mp4，PATH无需包含Python/ffmpeg。源crate构建缓存仍为已忽略target。
- [ ] README追加准确命令与依赖分组，从week2执行：

```bash
make reproduce
./md/target/release/md check artifacts
MD_PYTHON=/tmp/amat5315-field-venv/bin/python ./md/target/release/md video artifacts --out fluid.mp4
```

解释md准确CLI形式、默认参数、N合法规则、两模型共存、三项阈值、滑动RDF及(N-1)/N期望。模拟/检查仅Rust工具链及锁定依赖；make入口额外需要make；视频另需Python/NumPy/Matplotlib和ffmpeg/ffprobe含libx264。说明CARGO_TARGET_DIR改变时二进制路径相应变化。保留原Part 1–3复现说明。
- [ ] 在真实week2执行一次make reproduce和check，保存实测指标；若三项未过，定位力/积分/温控/随机数/统计实现，不变更固定要求。确认参数和实现均符合后仍不过时，报告实际差异供用户决定，不伪造green。
- [ ] 运行release acceptance与CLI测试，成功后提交`feat: Part 4 reproduction and default acceptance (green)`。

## Task 8：最终回归与交付证据

**Files:** 仅追加week2/part4-validation.md及必要README纠正；不增加功能或新的实验。

- [ ] 检查旧16个Rust测试未删除、修改阈值或忽略；读取已有requesting-code-review技能，按用户指定的当前会话方式审查，不启动子代理。
- [ ] 视频依赖已就绪后运行最终完整检查；release模式避免100原子全量测试在debug中无谓耗时：

```bash
cargo test --release --manifest-path week2/md/Cargo.toml --all-targets
cargo test --manifest-path week2/md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest week1/
git diff --check
```

- [ ] 从week2使用Task 7已经生成的默认轨迹单独生成完整视频：

```bash
MD_PYTHON=/tmp/amat5315-field-venv/bin/python ./md/target/release/md video artifacts --out fluid.mp4
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,width,height -of json fluid.mp4
```

读取文件实际字节数，核对200帧、<2,000,000字节；抽看第1、20、21、200帧，检查窗口分别1、20、20、20与真实运动。不得把短视频测试的通过冒充默认视频通过。
- [ ] 验证git check-ignore覆盖artifacts/run.json、artifacts/traj.jsonl、md/target；git status和暂存列表不含生成轨迹/缓存。只暂存源代码、测试、Cargo.lock、Makefile、README、验证记录和最终week2/fluid.mp4；不忽略所有MP4。
- [ ] validation列出每批red/green SHA、命令/退出码/实际失败、旧测试回归、默认三指标、视频帧数/字节数及依赖版本；不预填成功。必要文档提交`docs: record Part 4 validation`。若审查发现行为缺陷，另起真实red/green修复批次，再运行受影响和最终必要测试。
- [ ] 向用户报告结果与本地提交；不push，不运行400/1600轨迹、不添加加热阶段或Part 5功能。

## 计划自审与审核边界

覆盖映射：模型/边界/rc连续性→Task1–2；初始化/阶段/文件/CLI→Task2–3；重算及三项指标→Task4/7；20帧滑动RDF→Task5；视频依赖/编码→Task6/8；Makefile独立run→Task7；旧测试和提交证据→每批及Task8。
接口统一使用PhysicalModel、RunConfig、RunMetadata、Frame、RdfSample、CheckReport；serde字段名按文件契约映射。
已核对五项用户修订、命令位置参数、RDF的N/A归一化、窗口移除、400/1600只构造及不push边界。
用户已批准执行；真实执行证据见week2/part4-validation.md。
