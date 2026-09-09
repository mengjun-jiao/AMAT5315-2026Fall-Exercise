# Part 5：cell list 邻居搜索设计

## 批准状态与范围

用户已确认主体设计及force_method元数据补充，授权保存提交设计、编写实施计划。
当前只完成文档，实施计划审核前不执行，不push，不加热，不发布网页，不使用子代理。
继续扩展week2/md；本轮唯一算法优化是邻居搜索，保持原成对物理计算和全部旧测试。
历史Naive profiling保持98%、约2.2秒、2076/2117样本及profile-naive.png，不覆盖原始证据。

## 物理模型与搜索策略

PhysicalModel继续描述开放原始LJ或周期势能平移截断LJ。
独立ForceMethod枚举包含Naive、Cells，命令行和JSON分别使用字符串naive、cells。
CLI新增--force naive/--force cells，省略--force时使用cells。
System保存实际ForceMethod；不把搜索方式塞进PhysicalModel。
旧System::new和System::with_model、旧纯物理函数保留naive语义；新增显式策略入口。
开放无截断模型继续使用naive；显式请求开放模型+cells返回清晰错误。
旧积分器trait、旧Euler/Verlet更新顺序、加速度缓存和完整步回绕不变。

## 格子列表与候选配对

每次加速度或势能计算都按传入当前位置重建局部Vec<Vec<usize>>桶，不跨调用缓存成员。
周期盒已有rc<min(Lx,Ly)/2约束；nx=floor(Lx/rc)、wx=Lx/nx，y同理，格宽至少rc。
网格维度和乘积必须有效且可表示，非法尺寸返回错误；不静默改用别的网格公式。
利用model.wrap(p)得到用于定位桶的周期等价坐标，不修改调用者的位置或速度。
桶编号按floor(x/wx)、floor(y/wy)计算，处理浮点落到上界的情况，保证索引合法。

对每个原子i，生成自身及周围八格的周期编号，先在最多9个编号内去重。
遍历各唯一桶，仅对j>i产生候选，确保每个无序原子对恰好访问一次。
两格宽的方向上-1和+1映射相同格，因此必须先去重格编号，不能只依靠j>i。
naive保持原i外层、j=i+1..N遍历。两种枚举均通过同一个候选访问接口供物理计算消费，
不把O(N²)原子对全部收集成Vec，也不引入全局HashSet去重原子对。

候选仍对原始位置调用现有displacement、hypot及pair，执行准确minimum image与r<rc判断。
沿用原径向力、U(r)-U(rc)、成对等大反向累加、势能每对一次；不替换hypot、幂运算，
不加入平方距离快速截断、力/势能融合缓存、邻居skin、并行计算或其他算术优化。

## 状态一致性与参照路径

新增System::with_model_and_force明确指定搜索方式；旧构造默认naive。
初始化与每次加速度刷新按System保存的策略计算。新位置可能在完整步回绕之前，
分桶使用周期等价位置，因此仍一致；物理位移用原minimum image。
System的势能按相同策略计算，但不增加能量缓存；动能不变。
纯accelerations/energies保留naive包装，新增带策略版本用于模拟和等价性测试。
check与结构读取的物理校验继续调用naive纯函数，不根据文件中的force_method选cells，
不信任保存能量，不推进积分器。video/RDF的算法和采样窗口不变。

## CLI、写出与旧文件兼容

RunConfig原字段不变；新增initialize_with_force和run_to_directory_with_force显式传递策略。
原initialize和run_to_directory保持naive包装，避免旧库调用者无意换算法。
CLI通过新写出入口传入默认Cells或用户指定值。
RunMetadata新增force_method字段，从System实际策略写出，而不是按CLI默认值猜测。
新增序列化值仅naive/cells；旧文件缺字段反序列化为Naive，未知值或null报错。
区分两个默认值：CLI默认Cells；旧metadata缺字段默认Naive，不用同一个隐式默认混淆。

run.json原10个字段及其数值语义不变，只增加第11个force_method；traj.jsonl字段不变。
旧Rust测试中恰好10字段的断言需按用户明确批准的schema扩展更新为11并检查force_method，
其他原字段/帧数/数值/失败行为断言保留，不删除旧测试或放宽物理阈值。
手工构造RunMetadata的旧测试夹具补充force_method:Naive，表示原夹具策略。

## 课程viewer兼容边界

本轮只读检查了本地week2/week2-viewer.html（用户已有未跟踪文件，不修改或代为提交）：
readFiles在845–848行用JSON.parse读取并检查box；ingest在233–286行按名称读取box；
温度显示按名称读取temperature/ramp_to；没有拒绝未知run字段或要求字段总数的逻辑。
据此可确认本地这版viewer的字段读取逻辑容许新增force_method，原字段不变时无需修改viewer。
这是源码层面的兼容性结论；新增字段后的浏览器自动化/用户手动加载尚未执行。
实施后应保留包含force_method的真实run.json、traj.jsonl供用户加载；用户确认前不得写成手动检查通过。

## 测试与验收

同一位置/速度快照比较每个加速度分量和总势能，建议
abs(cells-naive)<=1e-10*max(1,abs(naive))；必须先确认两边有限。
不要求不同求和顺序下长期轨迹逐位一致，不根据实际结果放宽容差。
必须覆盖固定扰动晶格、跨周期边界近邻、rc-epsilon/rc/rc+epsilon、2×2桶及单轴两桶矩形盒。
覆盖原子跨内部格界和盒界后重算、未包回但周期等价坐标、总内力为零、重合原子拒绝。
测试候选j>i且无重复，所有距离小于rc的naive原子对均在候选内；远处候选可以存在。
用计数比较候选访问量，捕捉Cells实际仍扫描全部对的退化实现，不用墙钟时间做单元测试门槛。

CLI真实二进制测试覆盖省略--force、显式naive、显式cells、非法值及文件记录的实际策略。
兼容测试覆盖缺force_method的旧文件读为naive、新值往返、未知值/null拒绝、旧轨迹仍可check。
两种策略生成的轨迹均由naive检查器核对保存能量；保留篡改拒绝和旧视频测试。
默认100原子的完整物理验收应覆盖cells及显式naive，原参数、阈值不变。
保留原两原子实验、全部Rust/Python测试、缓存测试及已提交图片视频。

每个实现批次先写可编译且实际执行失败的行为测试、提交red，再实现并验证提交green。
编译错误、缺依赖、零测试/忽略测试不算red；已通过的行为如实记录，不人为破坏。

## 分开记录的性能实验（本轮均不执行）

| 实验 | N | eq_steps | steps | 方法/重复 |
| --- | --- | --- | --- | --- |
| 优化前后profiling | 400 | 200 | 1000 | naive、cells，各自真实采样 |
| 后续规模测速 | 100、400、1600 | 100 | 500 | naive、cells，每种N/方法各3次 |

profiling比较命令中的共同负载必须为--n 400 --eq-steps 200 --steps 1000；
用--force显式区分，保持release优化和调试信息。原历史Naive截图不替换。
规模测速是独立实验，使用shell time real，不包含构建时间，记录3次原值/中位数/min–max。
两实验其余默认参数包括seed=2026不变；不把旧100原子完整运行Timing表当作新规模测速数据。
本轮只写设计和计划；后续实测前不填Cell list比例、时间或加速比。

## 设计自审

已核对搜索策略/物理模型分离、每次重建、两层去重、共用物理公式、旧开放路径、
check保持naive、CLI与缺字段默认值的区别、schema兼容、viewer源码检查与手动验证的界限。
两种性能实验分别列出，设计内没有提前执行或填入优化后的数值。
