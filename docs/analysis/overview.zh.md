# 概览

每项分析都是一条单独的命令，并且会记录结果是怎么算出来的：

```bash
mliport analyze <RUN_DIR|trajectory> <task> [task options]
```

分析结果是诊断依据，结论还要你自己下。MSD 指数、扩散系数、跳跃次数，都要结合各页面里关于采样和
模型的说明来理解。

## 分析做了什么

mliport 的分析工具把 MD 轨迹转换成物理量——径向分布函数、扩散系数、活化能、振动谱、按位点统计的
跳跃次数。轨迹是原料，分析任务是读取它的仪器。

所有分析遵循几条不变量：

- **MLIP 轨迹不是 DFT 轨迹。** 力来自势函数，势函数的好坏取决于训练数据。分析只能告诉你模型
  预测的结果，不能等同于真实材料的行为。
- **分析和运行是分开的。** 轨迹写一次；你可以用不同的窗口、物种、参数反复分析。每组参数有自己
  独立的分析目录。
- **溯源是自动的。** 每条结果都记录软件版本、模型身份、输入文件哈希和完整的分析参数，不用翻
  日志就能还原每个数字是怎么来的。

## 输入

### mliport 运行目录（推荐）

优先使用 mliport 的运行目录：里面的元数据——帧间隔、坐标约定、哪些帧属于平衡阶段、模型身份、
运行配置——都是可信的，不需要你再手动提供。

分析读取器查找 `raw/trajectory.traj`（标量时间序列用 `raw/md.csv`）和 `artifacts.json`（记录
单位、可观测量定义和文件哈希的清单）。哪些帧是采样阶段、哪些是平衡阶段，由运行自身的元数据决定。

### 外部轨迹

ASE 轨迹（`.traj`）或 VASP XDATCAR 也可以。对于这些外部文件，你需要告诉 mliport 它自己无法
知道的信息：

- `--positions-convention`：`wrapped`、`unwrapped` 或 `unknown`。如果你声明的约定和文件里记录
  的可信元数据矛盾，程序会报错而不是猜。坐标约定很重要：输运分析需要 unwrapped 坐标，而结构
  分析（如 RDF）对 wrapped 和 unwrapped 都适用。
- `--frame-interval-fs`：**保存的**相邻两帧之间的时间间隔，不是 MD 的积分步长。这个值决定所有
  和时间相关的分析（MSD、VACF、输运系数）的物理时间轴。填错了，每个扩散系数和每个谱频率都会
  偏移。

各任务还有自己需要的参数，例如可移动离子物种、RDF 的中心和近邻元素对、温度等。

### XDATCAR 和 PBC 注意事项

标准 VASP XDATCAR 无法编码 ASE 的 `pbc` 掩码。如果 MD 输入不是三维全周期的，导出时会警告
该掩码丢失；而 `artifacts.json` 记录了原始的 `trajectory.pbc` 数组。加载对应 mliport
`vasp/XDATCAR` 时，分析读取器会从清单中恢复该掩码。独立的或更早版本的 XDATCAR 没有
匹配的源 PBC 元数据，其 **PBC 未经验证**；即使 ASE 以 `pbc=[True,True,True]` 解析了它，
mliport 也不保证它适合做体相 RDF/MSD 或输运分析。有条件时请分析原始的 `raw/trajectory.traj`。

## 输出

每次分析有单独的目录 `<run>/analysis/<task>/<analysis-id>/`。id 由分析参数决定，同一条轨迹、
同样的参数再分析一次会直接用已有结果；要重新计算就加 `--force`。

- `results.json`：分析参数和结果；
- `request.json` 和 `provenance.json`：完整的分析参数，以及软件 commit、模型信息、输入文件哈希；
- CSV 表格，有图的任务还有 PNG/SVG；
- `attempts/`：每次运行的不可变记录，`latest.json` 指向最近一次。

### 分析 ID 与缓存

分析 ID 编码了任务名称、物种和关键选项（拟合窗口、bin 数等）。改变其中任何一项都会生成不同的 ID
和独立的结果目录。这意味着你可以扫描参数而不覆盖之前的结果——也意味着重复完全相同的命令不需要
重新计算。

用 `--force` 可以重算已有 ID 的结果，比如在延长轨迹或更新分析代码之后。

## 采样方面的注意事项

- 分析采样阶段的轨迹。平衡阶段的帧默认不计入，除非加 `--include-equilibration`；支持的任务可以用
  `--start-frame` / `--stop-frame` 限定范围。
- 要清楚保存帧的时间间隔，外部轨迹用 `--frame-interval-fs` 给出。
- 输运系数比 RDF、热力学量需要更长、平衡得更好的轨迹。
- 用不同的时间窗口反复分析是正常做法，每组参数都有自己的 analysis id。

### 轨迹长度参考

| 分析任务 | 最低可用 | 推荐 |
|---|---|---|
| `validate`、`thermo` | 几百帧 | 任意采样长度 |
| `rmsd`、`rdf`、`density` | 几千帧 | 10+ ps |
| `msd` | 足够到达 α ≈ 1 | 目标温度下 50+ ps |
| `transport` | 能看到扩散区间 | 每个温度 100+ ps |
| `electrolyte` | 足够的跳跃次数 | 100+ ps；慢扩散体系要更长 |
| `arrhenius` | 3 个以上温度 | 3–5 个温度，每个都收敛 |

这些只是粗略参考。扩散越慢的体系需要等比例更长的模拟。

## 没有单独页面的任务

### validate

**`thermo`** 报告温度、能量漂移和压强。`--block-size-frames` 设置分块统计的块大小。NVE 计算首先要看
能量漂移。统计量只用有限值计算（NaN 与 ±Inf 都排除，汇总统计与分块均值一致），因此结果里还有
`summary.data_integrity`：逐列给出总帧数、有限帧数，
以及哪些列含非有限值（`raw/md.csv` 里空格或无法解析的单元会变成 NaN）。由部分损坏的列算出的均值
因此能被看见，而不会显得正常。

```bash
mliport analyze runs/md validate
```

通过 `validate` 只意味着元数据自洽、文件格式正确，不保证轨迹在科学上有意义。

### thermo

`thermo` 报告温度、能量漂移和压强随时间的变化。`--block-size-frames` 设置分块统计的块大小。

```bash
mliport analyze runs/md thermo --block-size-frames 100
```

NVE 计算首先要看能量漂移：它衡量积分器守恒能量的好坏。漂移超过 ~1 meV/atom/ps 说明步长太大或
模型的力噪声太高。

NVT 计算要看温度是否在目标值附近以预期的方差波动。NPT 计算看压强和体积的轨迹，判断恒压器
是否已经平衡。

### rmsd

`rmsd` 计算考虑周期性的 RMSD 和 RMSF——快速判断晶体是否保持了结构。如果熔化或发生相变，
RMSD 会无界增长。

```bash
mliport analyze runs/md rmsd --species Li
mliport analyze runs/md rmsd --drift-reference nonmobile
```

`--species` 只看某种元素，`--drift-reference none|nonmobile|indices` 选择怎样扣除整体漂移。
对于固态电解质，`--drift-reference nonmobile` 扣除不动亚晶格的缓慢漂移，让你只看到
可移动离子的位移。

### density

`density` 生成某种可移动离子的三维周期性数密度图。输出的体积数据文件可以在 VESTA 等软件中可视化。

```bash
mliport analyze runs/md density --mobile Li --spacing 0.2 --stride 5
```

`--spacing` 设置体素大小（单位 Å，默认 0.25 Å），`--smoothing-sigma-A` 添加高斯平滑，
`--stride` 隔帧取样以加快计算。密度图显示可移动离子在哪里停留——通道、笼、优先占位点都会变得
可见。

## 验收测试用过的命令

LGPS 验收测试在真实轨迹上运行过这些命令：

```bash
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
```

`validate` 和 `thermo` 对长短轨迹都能用。每个任务都输出 JSON 和 CSV，有图的还输出 PNG 和 SVG，
不需要显示器。
