# Part 5 Cell List Implementation Plan

> **状态：待用户审核，本轮不执行。** 延续用户指定的当前会话执行方式，不启动子代理、不调用或安装缺失的subagent-driven-development/executing-plans/using-git-worktrees。用户批准实施后使用test-driven-development；每批先实际red并提交，再实现green并提交。

**Goal:** 用可选cell list减少周期LJ候选对，CLI默认cells，并通过force_method元数据记录实际策略和兼容旧轨迹。

**Architecture:** 新增独立邻居枚举模块，System保存ForceMethod，naive/cells共用现有成对物理计算。每次计算用当前坐标重建桶；旧纯函数和check固定naive。CLI的新默认值和旧metadata缺字段默认值分别显式定义。

**Tech Stack:** 现有Rust 2024 crate，标准库Vec/VecDeque以外不需要新容器依赖；复用serde、clap、tempfile及现有测试/视频环境。

**Spec:** `docs/superpowers/specs/2026-09-09-part5-cell-list-design.md`（22a783e）。

## Global Constraints

- 本轮只写文档，不执行计划、不push、不加热、不发布网页，不使用子代理。
- PhysicalModel与ForceMethod分开；--force naive保留全配对，--force cells为CLI默认。
- 每次计算按当前位置重建Vec<Vec<usize>>桶；nx=floor(Lx/rc)、wx=Lx/nx，y同理。
- 搜索自身及周围八格，周期格编号先去重，再仅访问j>i；无序原子对只访问一次。
- 两路径共用原displacement、hypot、pair及累加；不同时更改截断、幂运算、平方距离策略或融合缓存。
- 开放无截断模型保留naive；显式开放+cells报错；旧两原子接口和积分器行为不变。
- check和轨迹读取的纯物理校验继续用naive，不依据force_method调用cells。
- run.json只增加force_method字段，值naive/cells；缺字段读为naive，未知值/null拒绝；原10字段不变，traj.jsonl不变。
- 力分量/势能容差abs(cells-naive)<=1e-10*max(1,abs(naive))；双方须有限，不要求长轨迹逐位一致。
- 保留全部旧测试和物理阈值；仅更新明确扩展schema的字段数断言及新增必需Rust结构字段。
- profiling负载：N=400、eq_steps=200、steps=1000，两种方法显式选择。
- 后续规模测速：N=100/400/1600、eq_steps=100、steps=500，每个N/方法各3次shell time real。
- 两实验不混用，其余默认参数保持seed=2026等；本轮不运行两实验。
- 原Naive profiling 98%、约2.2秒、2076/2117样本及截图保留，不用新默认cells重写历史记录。

## 文件职责与共享接口

路径均从仓库根起。

| 文件 | 改动职责 |
| --- | --- |
| week2/md/src/neighbors.rs（新增） | ForceMethod、当前状态分桶、候选访问；内部测试候选唯一性/完整性 |
| week2/md/src/physics.rs | 新带策略的加速度/能量接口；原接口保留naive包装 |
| week2/md/src/dynamics.rs | System保存ForceMethod；显式新构造、策略访问器、刷新和势能接入 |
| week2/md/src/fluid.rs | initialize_with_force；原initialize仍naive |
| week2/md/src/trajectory.rs | metadata新字段及兼容读取；显式策略写出入口 |
| week2/md/src/main.rs | --force ValueEnum，默认Cells |
| week2/md/src/lib.rs | 导出neighbors模块，其余导出保持 |
| week2/md/tests/cells.rs（新增） | 力/势能等价、边界、截断、状态更新 |
| week2/md/tests/force_cli.rs（新增） | 真实CLI选择、默认值和新旧元数据读取 |
| week2/md/tests/cli.rs、check.rs、acceptance.rs | schema夹具适配、保留旧断言、默认/naive完整验收 |
| week2/README.md、week2/part5-cell-list-validation.md（新增） | 策略说明、真实提交/验证证据与分开的性能实验说明 |

接口约定：

```rust
// neighbors.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default,
         serde::Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ForceMethod { #[default] Naive, Cells }
pub(crate) fn visit_candidates(
    pos: &[[f64; 2]], model: &crate::physics::PhysicalModel, method: ForceMethod,
    visit: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<(), String>;
// physics.rs，原accelerations/energies签名保留，均委托Naive
pub fn accelerations_with_method(pos: &[[f64;2]], model: &PhysicalModel,
    method: ForceMethod) -> Result<Vec<[f64;2]>, String>;
pub fn energies_with_method(pos: &[[f64;2]], vel: &[[f64;2]], model: &PhysicalModel,
    method: ForceMethod) -> Result<(f64,f64), String>;
// System新增方法；原构造签名保留
pub fn with_model_and_force(p: Vec<[f64;2]>, v: Vec<[f64;2]>,
    model: PhysicalModel, method: ForceMethod) -> Result<Self, String>;
pub fn force_method(&self) -> ForceMethod;
// fluid.rs，原initialize包装Naive
pub fn initialize_with_force(c: &RunConfig, method: ForceMethod) -> Result<System, String>;
// trajectory.rs，原run_to_directory包装Naive；CLI使用新入口
pub fn run_to_directory_with_force(c: &RunConfig, dir: &std::path::Path,
    method: ForceMethod) -> Result<(), String>;
```

签名块是计划中的接口契约，不是现在写入源码的实现。

## Task 1：候选枚举及格子去重

**Files:** 新增neighbors.rs；修改lib.rs；新增part5-cell-list-validation.md。
**Consumes:** PhysicalModel::validate/wrap及周期模型的box_size/rc。
**Produces:** ForceMethod和visit_candidates；不更改现有物理/CLI默认值。

- [ ] 用户批准实施后，读取test-driven-development及writing-good-tests.md；核对工作区，不覆盖用户未跟踪viewer。
- [ ] 先运行现有基线（视频依赖缺失则按README重建，不静默跳过）：

```bash
PATH=/tmp/amat5315-ffmpeg:$PATH MD_PYTHON=/tmp/amat5315-field-venv/bin/python MPLCONFIGDIR=/tmp/amat5315-matplotlib cargo test --release --manifest-path week2/md/Cargo.toml --all-targets
cargo test --manifest-path week2/md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest week1/
```

- [ ] 在neighbors.rs写以下内部测试，新增接口允许暂用明确未实现的函数体以成功编译；不增加错误算法伪造red。

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::PhysicalModel;
    fn pairs(p: &[[f64;2]], b:[f64;2], method:ForceMethod)->Vec<(usize,usize)> {
        let m=PhysicalModel::PeriodicShiftedLennardJones{box_size:b,rc:2.5};
        let mut result=Vec::new();
        visit_candidates(p,&m,method,|i,j|{result.push((i,j));Ok(())}).unwrap();
        result
    }
    #[test]
    fn two_by_two_bins_do_not_repeat_wrapped_neighbors() {
        let p=[[0.1,0.1],[1.1,0.1],[3.1,0.1],[3.1,3.1]];
        let mut got=pairs(&p,[6.,6.],ForceMethod::Cells);
        assert_eq!(got.len(),6);
        assert!(got.iter().all(|&(i,j)|i<j));
        got.sort_unstable();got.dedup();assert_eq!(got.len(),6);
    }
    #[test]
    fn cells_reduce_candidates_without_losing_near_pairs() {
        let p=[[1.,1.],[2.,1.],[15.,15.],[16.,15.]];
        let mut got=pairs(&p,[30.,30.],ForceMethod::Cells);
        got.sort_unstable();assert_eq!(got,vec![(0,1),(2,3)]);
        assert_eq!(pairs(&p,[30.,30.],ForceMethod::Naive).len(),6);
    }
    #[test]
    fn all_interacting_pairs_are_present_without_duplicates() {
        let (mut p,b)=crate::fluid::lattice(100,0.8).unwrap();
        for (i,x) in p.iter_mut().enumerate() { x[0]+=0.03*(i as f64).sin(); }
        let got=pairs(&p,b,ForceMethod::Cells);
        let unique:std::collections::BTreeSet<_>=got.iter().copied().collect();
        assert_eq!(unique.len(),got.len());
        let m=PhysicalModel::PeriodicShiftedLennardJones{box_size:b,rc:2.5};
        for i in 0..p.len() { for j in i+1..p.len() {
            let d=m.displacement(p[i],p[j]);
            if d[0].hypot(d[1])<2.5 { assert!(unique.contains(&(i,j))); }
        }}
    }
    #[test]
    fn wrapped_boundary_pairs_are_candidates() {
        let got=pairs(&[[-0.1,1.],[0.1,1.],[12.1,5.]], [12.,9.], ForceMethod::Cells);
        assert!(got.contains(&(0,1)));
    }
}
```

- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --lib neighbors::tests`，记录可编译、实际执行、目标行为因未实现失败；旧`--test dynamics`应通过。提交：

```bash
git add week2/md/src/neighbors.rs week2/md/src/lib.rs week2/part5-cell-list-validation.md
git diff --cached --check
git commit -m "test: cell candidate enumeration (red)"
```

- [ ] 实现visit_candidates。Naive分支保留i<j；Cells先验证周期模型、有限位置和有效网格尺寸。局部桶与逐原子枚举的核心为：

```rust
// Cells分支内，nx/ny为按floor得到的有效usize，wx=Lx/nx、wy=Ly/ny。
let count=nx.checked_mul(ny).ok_or("grid size overflow")?;
let mut buckets:Vec<Vec<usize>>=Vec::new();
buckets.try_reserve_exact(count).map_err(|e|format!("grid allocation: {e}"))?;
buckets.resize_with(count,Vec::new);
let mut atom_cells=Vec::with_capacity(pos.len());
for (i,&p) in pos.iter().enumerate() {
    let wrapped=model.wrap(p);
    let cx=((wrapped[0]/wx).floor() as usize).min(nx-1);
    let cy=((wrapped[1]/wy).floor() as usize).min(ny-1);
    atom_cells.push((cx,cy));buckets[cy*nx+cx].push(i);
}
for (i,&(cx,cy)) in atom_cells.iter().enumerate() {
    let xs=[(cx+nx-1)%nx,cx,(cx+1)%nx];
    let ys=[(cy+ny-1)%ny,cy,(cy+1)%ny];
    let mut ids=Vec::with_capacity(9);
    for y in ys { for x in xs {
        let id=y*nx+x;if !ids.contains(&id) { ids.push(id); }
    }}
    for id in ids { for &j in &buckets[id] {
        if j>i { visit(i,j)?; }
    }}
}
```

维度先检查可表示、正数及安全整数加乘范围，分配失败应返回错误；实际代码用try_reserve初始化桶，避免无约束巨量网格导致进程分配中止。Naive也验证输入基本合法性。Cells+开放模型直接Err，不静默切换。桶只存在于本次调用。

- [ ] 重跑目标测试，记录结果并提交`feat: cell candidate enumeration (green)`；暂存文件同red。此批不改hypot或截断。

## Task 2：共用物理计算及System策略

**Files:** 修改physics.rs、dynamics.rs、fluid.rs；新增tests/cells.rs。
**Consumes:** Task1候选枚举；原pair/displacement/动力学接口。
**Produces:** 接口地图中的带策略物理函数、System构造/访问器、initialize_with_force；旧包装明确Naive。

- [ ] 写等价性测试以及可编译新接口。以下核心测试比较同一快照，不比较长时间轨迹：

```rust
use md::neighbors::ForceMethod;
use md::physics::{PhysicalModel,accelerations_with_method,energies_with_method};
fn compare(pos:&[[f64;2]],box_size:[f64;2]) {
    let m=PhysicalModel::PeriodicShiftedLennardJones{box_size,rc:2.5};
    let v=vec![[0.2,-0.1];pos.len()];
    let a=accelerations_with_method(pos,&m,ForceMethod::Naive).unwrap();
    let b=accelerations_with_method(pos,&m,ForceMethod::Cells).unwrap();
    for (&x,&y) in a.iter().flatten().zip(b.iter().flatten()) {
        assert!(x.is_finite() && y.is_finite());
        assert!((x-y).abs()<=1e-10*x.abs().max(1.));
    }
    let (u,k)=energies_with_method(pos,&v,&m,ForceMethod::Naive).unwrap();
    let (w,l)=energies_with_method(pos,&v,&m,ForceMethod::Cells).unwrap();
    assert!(u.is_finite() && w.is_finite());
    assert!((u-w).abs()<=1e-10*u.abs().max(1.));assert_eq!(k,l);
    for axis in 0..2 {
        let residual=b.iter().map(|x|x[axis]).sum::<f64>().abs();
        let scale=b.iter().map(|x|x[axis].abs()).sum::<f64>().max(1.);
        assert!(residual<=1e-12*scale);
    }
}
#[test]
fn perturbed_lattice_and_periodic_pairs_match() {
    let (mut p,b)=md::fluid::lattice(100,0.8).unwrap();
    for (i,x) in p.iter_mut().enumerate() {
        x[0]+=0.04*(i as f64*1.7).sin();x[1]+=0.03*(i as f64*0.9).cos();
    }
    compare(&p,b);
    compare(&[[0.1,1.],[5.9,1.],[3.,4.]], [6.,6.]);
    compare(&[[-0.1,1.],[0.1,1.],[3.,6.]], [6.,9.]);
}
#[test]
fn cutoff_and_two_cell_boxes_match() {
    for r in [2.5-1e-8,2.5,2.5+1e-8] {
        compare(&[[0.,0.],[r,0.]], [6.,6.]);
        compare(&[[0.,0.],[r,0.]], [6.,9.]);
    }
}
#[test]
fn rebuilt_membership_tracks_current_positions() {
    let mut p=[[1.,1.],[3.01,1.],[10.,8.]];
    compare(&p,[12.,12.]);
    p[1]=[2.99,1.];compare(&p,[12.,12.]);
    p[0]=[-0.1,1.];p[1]=[0.1,1.];compare(&p,[12.,12.]);
}
#[test]
fn open_cells_and_periodic_overlap_are_rejected() {
    assert!(accelerations_with_method(&[[0.,0.],[1.2,0.]],&PhysicalModel::OpenLennardJones,ForceMethod::Cells).is_err());
    let m=PhysicalModel::PeriodicShiftedLennardJones{box_size:[6.,6.],rc:2.5};
    assert!(accelerations_with_method(&[[0.,0.],[6.,0.]],&m,ForceMethod::Cells).is_err());
}
```

- [ ] 在dynamics.rs内部增加真实一步对照：同样的周期位置/速度分别用with_model_and_force构造两系统，逐步调用VelocityVerlet，覆盖内部格界和盒界；每步位置/速度使用上述容差比较，缓存计数初始化1、每步+1。可用[7.99,1]/[3.99,1]、共同速度[2,0]、盒[8,6]、dt=.01核对回绕后速度不变；用近邻[0.1,1]/[7.9,1]的小dt=1e-6核对非零力更新。
- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --test cells --lib`；记录目标失败及旧测试状态，提交`test: cell force and energy equivalence (red)`。
- [ ] 实现带策略函数，候选回调保留现有算式；不把单对物理实现复制为cells特供版本：

```rust
visit_candidates(pos,model,method,|i,j| {
    let d=model.displacement(pos[i],pos[j]);
    let r=d[0].hypot(d[1]);
    let (_,f)=model.pair(r)?;
    for axis in 0..2 {
        let component=f*d[axis]/r;
        a[i][axis]+=component;a[j][axis]-=component;
    }
    Ok(())
})?;
// 势能回调同样调用model.displacement和model.pair(d[0].hypot(d[1]))?.0。
// 动能及所有输入/有限值校验保留。
```

- [ ] System增加私有force_method；new仍Naive，with_model委托with_model_and_force(...,Naive)。新构造验证模型/策略，按实际方法初始化加速度；refresh_accelerations和potential_energy调用带策略接口。initialize_with_force复用原晶格与随机数流程，仅将method传入System；旧initialize委托Naive。借用规则和trait签名不变。
- [ ] 执行`cargo test --release --manifest-path week2/md/Cargo.toml --test cells --test dynamics --test physics --test fluid --test trajectory --lib`，通过后提交`feat: cell force and energy evaluation (green)`。记录候选完整性/去重测试通过，不预填时间或加速比。

## Task 3：CLI默认Cells与force_method兼容

**Files:** 修改trajectory.rs、main.rs；新增tests/force_cli.rs；适配tests/cli.rs、check.rs中的schema断言/夹具。
**Consumes:** Task2 initialize_with_force和System::force_method；原RunConfig、Frame不变。
**Produces:** 新metadata字段、显式策略写出入口与CLI--force。

- [ ] 先写真实二进制默认值/显式选择测试：

```rust
use std::process::Command;
use md::neighbors::ForceMethod;
#[test]
fn cli_default_and_explicit_methods_are_recorded() {
    let exe=env!("CARGO_BIN_EXE_md");
    for (option,expected) in [(None,"cells"),(Some("naive"),"naive"),(Some("cells"),"cells")] {
        let tmp=tempfile::tempdir().unwrap();
        let mut cmd=Command::new(exe);
        cmd.args(["run","--eq-steps","0","--steps","2","--sample-every","1","--out"]).arg(tmp.path());
        if let Some(method)=option {cmd.args(["--force",method]);}
        let output=cmd.output().unwrap();
        assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
        let raw:serde_json::Value=serde_json::from_slice(&std::fs::read(tmp.path().join("run.json")).unwrap()).unwrap();
        assert_eq!(raw["force_method"],expected);
        for key in ["n","rho","box","dt","temperature","eq_steps","steps","sample_every","seed","integrator"] {
            assert!(raw.get(key).is_some(),"missing original field {key}");
        }
        let (meta,frames)=md::trajectory::read_trajectory(tmp.path()).unwrap();
        assert_eq!(frames.len(),2);
        assert_eq!(meta.force_method,if expected=="cells" {ForceMethod::Cells} else {ForceMethod::Naive});
        // 短轨迹不要求三项统计PASS，但保存能量必须经naive重算一致。
        md::check::evaluate(&meta,&frames).unwrap();
    }
}
#[test]
fn legacy_metadata_defaults_to_naive_and_invalid_methods_fail() {
    let tmp=tempfile::tempdir().unwrap();
    let c=md::fluid::RunConfig{eq_steps:0,steps:2,sample_every:1,..Default::default()};
    md::trajectory::run_to_directory_with_force(&c,tmp.path(),ForceMethod::Naive).unwrap();
    let path=tmp.path().join("run.json");
    let mut json:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("force_method");
    std::fs::write(&path,serde_json::to_vec(&json).unwrap()).unwrap();
    let (meta,frames)=md::trajectory::read_trajectory(tmp.path()).unwrap();
    assert_eq!(meta.force_method,ForceMethod::Naive);
    md::check::evaluate(&meta,&frames).unwrap();
    for value in [serde_json::json!("unknown"),serde_json::Value::Null] {
        json["force_method"]=value;
        std::fs::write(&path,serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(md::trajectory::read_trajectory(tmp.path()).is_err());
    }
    assert!(!Command::new(env!("CARGO_BIN_EXE_md")).args(["run","--force","unknown"]).output().unwrap().status.success());
}
```

- [ ] 先添加新字段的编译结构及函数签名，暂不写CLI实现或正确默认逻辑；可编译骨架允许未实现。旧test/check.rs构造RunMetadata补force_method:Naive；旧cli.rs的10字段断言在schema实际切换这一批更新为11，并保留原字段验证；不得删除该旧测试来规避失败。
- [ ] 运行`cargo test --manifest-path week2/md/Cargo.toml --test force_cli --test cli --test check`，分别确认目标测试实际运行失败。若schema夹具错误导致无法编译，先修正编译后重新取得真实red。提交`test: force CLI and metadata compatibility (red)`。
- [ ] 实现明确分开的默认值：

```rust
// RunMetadata内，ForceMethod::default()为Naive，只用于旧文件缺字段。
#[serde(default)]
pub force_method: ForceMethod,
// RunArgs内，CLI显式默认Cells。
#[arg(long="force",value_enum,default_value="cells")]
force: ForceMethod,
// 写出前：
let mut s=initialize_with_force(c,method)?;
let meta=RunMetadata {config:c.clone(),box_size,force_method:s.force_method()};
```

CLI用明确的字符串default_value="cells"，枚举的Default仍保持Naive供旧文件读取。CLI调用run_to_directory_with_force；旧run_to_directory包装Naive；其他写出顺序、完整步采样和文件名保持。

- [ ] check.rs和read_trajectory仍调用旧naive energies，文件中的force_method只说明数据来源；新增值不参与选择检查算法。用cells轨迹、naive轨迹及去掉字段的旧轨迹分别执行读取/能量交叉核对，再运行全部原篡改拒绝测试。
- [ ] 运行`cargo test --release --manifest-path week2/md/Cargo.toml --test force_cli --test cli --test check --test trajectory --test cells`并记录green，提交`feat: default cell search and compatible force metadata (green)`。

## Task 4：完整回归、viewer交付与后续实验说明

**Files:** 修改tests/acceptance.rs、README.md；追加part5-cell-list-validation.md；不改课程viewer、历史Timing/Profile数值或已有图片视频。
**Consumes:** 新CLI及metadata；旧默认物理指标和视频入口。
**Produces:** 经回归的实现、可供用户加载的新版轨迹、清楚分开的后续测量命令。

- [ ] 将原默认完整验收保留为省略--force（实际Cells），并加入显式Naive完整验收；共同驱动/初始化不变。两方法均需原三项门槛，不比较逐帧同值：

```rust
// 每个method使用独立TempDir及同一原完整验收断言。
for method in ["naive","cells"] {
    let dir=tempfile::tempdir().unwrap();
    let out=std::process::Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run","--force",method,"--out"]).arg(dir.path()).output().unwrap();
    assert!(out.status.success(),"{}",String::from_utf8_lossy(&out.stderr));
    let (meta,frames)=md::trajectory::read_trajectory(dir.path()).unwrap();
    assert_eq!(frames.len(),200);
    assert_eq!(serde_json::to_value(meta.force_method).unwrap(),method);
    let report=md::check::check_directory(dir.path()).unwrap();
    assert!(report.drift<2e-3);
    assert!((report.t_speed-0.5).abs()<0.05);
    assert!(report.mb_score<2.);
}
```

此处对已正确实现行为的补充测试若直接通过如实记录，不人为破坏算法制造red。
若发现真实遗漏，先保留失败回归并提交`test: cell regression (red)`，再修复提交`fix: cell regression (green)`；没有失败则不虚构额外red。

- [ ] 执行全部回归，视频依赖若缺失按README恢复，不跳过视频测试：

```bash
PATH=/tmp/amat5315-ffmpeg:$PATH MD_PYTHON=/tmp/amat5315-field-venv/bin/python MPLCONFIGDIR=/tmp/amat5315-matplotlib cargo test --release --manifest-path week2/md/Cargo.toml --all-targets
cargo test --manifest-path week2/md/Cargo.toml --doc
/tmp/amat5315-field-venv/bin/python -m pytest week1/
git diff --check
```

- [ ] 记录所有red/green SHA、命令/退出码/失败原因/测试数，确认旧两原子断言及阈值未改变。继续在当前会话按requesting-code-review的清单自审，用户的无子代理要求优先。
- [ ] 将新版cells默认100原子数据写到独立的被忽略目录供用户viewer验证，避免覆盖Part4旧文件：

```bash
cargo run --release --locked --manifest-path week2/md/Cargo.toml -- run --force cells --out week2/artifacts/cells-viewer
week2/md/target/release/md check week2/artifacts/cells-viewer
```

本轮viewer源码检查依据是readFiles的JSON.parse/box检查及ingest按名取字段，无未知字段拒绝。
实施后在README记录“源码层面兼容，新增force_method后的用户手动加载待确认”，并提供上述两个真实文件路径。
不要修改或提交用户未跟踪的week2/week2-viewer.html；不要声称浏览器手动检查已通过。

- [ ] README增加策略、元数据默认值和重装命令；保留release opt-level=3/debug=true/strip=false。明确下列两组是**后续测量命令，本次实现验收不自动执行**：

```bash
# profiling：两方法共同负载N=400、eq=200、steps=1000。
# 从week2执行；先重装项目md，采样权限按原README临时设置并恢复。
cargo install --path md --profile release --locked --force --target-dir md/target --root "$HOME/.cargo"
samply record --save-only -o artifacts/part5-profile/naive-current.json.gz -- md run --force naive --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-naive
samply record --save-only -o artifacts/part5-profile/cells.json.gz -- md run --force cells --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-cells
```

不要覆盖历史naive.json.gz或profile-naive.png。第一条新naive采样用于同一次构建内对照，历史约2.2秒仍是历史记录。

```bash
# 规模测速：独立实验；N=100/400/1600、eq=100、steps=500，每方法各3次。
# 从week2执行，避免与profiling同时运行；这里没有采样器和构建命令。
(
    set -e
    TIMEFORMAT='%3R'
    export LC_ALL=C
    for n in 100 400 1600; do
        for method in naive cells; do
            result="artifacts/part5-scaling/n${n}-${method}"
            mkdir -p "$result"
            : > "$result/times.txt"
            for repeat in 1 2 3; do
                { time md run --force "$method" --n "$n" --eq-steps 100 --steps 500 --out "$result/run" > "$result/run-$repeat.log" 2> "$result/run-$repeat.err"; } 2>> "$result/times.txt"
            done
        done
    done
)
```

规模测试表另建，不用这组结果覆盖原100原子完整生产运行的Timing表；每组保留3次原始值及中位数/min–max。尚未实测不填数字。

- [ ] 提交README及真实验证记录；只提交代码、测试、文档，所有生成轨迹/profile/cache继续忽略。最终报告实现状态、测试、schema及viewer待手动验证状态；不push、不加热。

## 自审覆盖与当前停止点

候选完整性/两层去重→Task1；等价性/截断/边界/当前位置/开放模型/缓存→Task2；
CLI默认Cells/实际metadata/旧字段缺省Naive/非法策略/check参照→Task3；完整物理与旧测试/视频回归、viewer边界、独立测量负载→Task4。
所有跨任务接口使用相同名称；RunConfig没有增加字段，RunMetadata单独增加force_method，原10字段数断言的修改明确归入Task3。
本计划没有执行；本轮没有构建、测试、采样、规模测速或实现改动。等待用户审核。
