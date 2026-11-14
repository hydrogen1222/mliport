# NEP（Calorine / 原生 GPUMD）

NEP（神经进化势）是一个**势函数族**，不是一个模型。你需要自己提供针对目标元素与
工况训练好的 `nep.txt`；mliport 不训练、不下载、不附带任何 NEP 模型，也无法判断某个
`nep.txt` 对你的材料是否准确。

与另外四个后端不同，NEP 有**两条执行路径**，必须由你显式选择：

| `NEP_RUNTIME` | 设备 | 实现 | 数值精度 | 一次 GPU 调用 = |
|---|---|---|---|---|
| `ase` | `cpu` | `calorine.calculators.CPUNEP` | float64 | 一次进程内求值 |
| `ase` | `cuda[:N]` | `calorine.calculators.GPUNEP` | float32 | **启动一个 `gpumd` 进程** |
| `gpumd` | `cuda[:N]` | 原生 `gpumd` 进程 | float32 | 整个算例 |

mliport 不会推断 runtime、不会从设备反推 runtime，也不会在两者之间自动回退。二者精度、
开销模型和可执行工作流都不同，静默切换会让两个"看起来配置相同"的结果无法比较。

`NEP_RUNTIME=gpumd` 搭配 `DEVICE=cpu` 会被直接拒绝：本实现没有 CPU 原生路径，把它悄悄
降级成 `CPUNEP` 会在你不知情的情况下改变数值精度。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv-nep` | 3.10, 3.11, 3.12, 3.13 | `calorine==4.1`（NumPy/ASE 由安装器解析；gpumd 需用 CUDA Toolkit 12.9 编译） |

```bash
./scripts/install_mliport.sh --engines nep --device auto
```
<!-- END GENERATED -->

## 固定模型档案

<!-- BEGIN GENERATED: profiles -->
| Profile | 标识 | dtype | task | head | 模型 SHA-256 |
|---|---|---|---|---|---|
<!-- END GENERATED -->

Calorine 是编译扩展，因此 NEP 使用独立环境。上面的安装脚本会在该环境里装入固定版本的
`calorine` **和**可编辑安装的 mliport，所以 `.venv-nep/bin/mliport` 一定存在。手动安装时，
等价做法是在仓库根目录用同一次解析装好两者（不含 mliport 的环境没有 `mliport` 命令，也无法
运行 `doctor`）：

```bash
uv venv .venv-nep --python 3.12
uv pip install --python .venv-nep/bin/python -e './mliport[analysis-all]' calorine==4.1
```

原生路径还需要 `gpumd` 可执行文件。mliport 安装器**不**自动构建它（需要与本机 GPU 匹配
的 CUDA toolkit 与 `-arch=sm_XX`）：

```bash
git clone https://github.com/brucefan1983/GPUMD.git
cd GPUMD/src
source /path/to/cuda/env.sh          # 提供 nvcc 与 libcublas/libcusolver/libcufft
make CUDA_ARCH="-arch=sm_70" -j16    # sm_70 = V100，按你的 GPU 选择
```

mliport 在 V100（`sm_70`）上用 GPUMD 5.9.1 与 CUDA 12.9 验证过；其他 GPU 和工具包版本未经
验证（各架构的 `-arch` 取值见安装器兼容性注册表）。然后用 `NEP_GPUMD_COMMAND` /
`--gpumd-command` 指向该二进制，或把它放进 `PATH`。`mliport doctor` 在 `PATH` 或环境变量
`MLIPORT_GPUMD_COMMAND` 里查找 `gpumd`（它不读取 `NEP_GPUMD_COMMAND`）。一次性检查两条路径：

```bash
.venv-nep/bin/mliport doctor --engine nep --device cuda:0
```

没有 `gpumd` 时，`--device cpu` 仍然通过（Calorine `CPUNEP` 不需要它），`--device cuda` 会失败，
并在报错里点明缺少可执行文件。

`doctor --engine nep` 会报告 Calorine/ASE 版本、解析到的 `gpumd` 路径与 SHA-256，以及其
版本横幅首行。它**不会**加载模型：真正跑 NEP 还需要你的 `nep.txt`。

## 配置

```bash
.venv-nep/bin/mliport sp structure.vasp \
    --model-type nep \
    --model /path/to/nep.txt \
    --nep-runtime ase \
    --device cpu
```

```ini
MODEL_TYPE   = NEP
MODEL_PATH   = /path/to/nep.txt
NEP_RUNTIME  = gpumd          # ase | gpumd（必填）
NEP_GPUMD_COMMAND = /opt/gpumd/src/gpumd
DEVICE       = cuda:0
TASK         = bulk           # 仅 PBC 语义；NEP 没有 task 家族
```

同样的两个键在全部入口可用：CLI 的 `--nep-runtime` / `--gpumd-command`、Python API 的
`nep_runtime=` / `nep_gpumd_command=`、INCAR 与 `settings.ini` 的 `NEP_RUNTIME` /
`NEP_GPUMD_COMMAND`、TUI 的 runtime 选择器，以及队列任务文件中的同名参数。它们都解析成
同一个规范化配置对象，并报告"取值 + 来源"，不会各自漂移。

`MODEL_PATH` 必须是单个文件。目录会被拒绝：一个模型目录没有唯一的模型身份。

## 两条路径分别支持什么

| 工作流 | `NEP_RUNTIME=ase` | `NEP_RUNTIME=gpumd` |
|---|---|---|
| 单点 `sp` | 支持 | 支持 |
| 固定晶胞优化 `opt` | 支持 | 支持，`minimize fire\|sd` |
| 变胞优化 | 仅通过既有 ASE 约束契约 | `minimize fire ... 1` |
| MD `NVE` | 支持（ASE 积分器） | 支持（原生，推荐） |
| MD `NVT` | 支持（ASE 积分器） | 支持，需恒温器可映射（见下） |
| `NEB` / CI-NEB | 支持 | **拒绝** |
| `batch`、`confsearch` | 支持 | **拒绝** |

原生路径遇到 `neb`、`batch`、`confsearch` 会报错并指明 `NEP_RUNTIME=ase`，不会改道。
用原生进程驱动 NEB 带需要独立调度器，而静默切换 runtime 会把两个势的绝对能量混进同一次
排序。

两条路径都拒绝 DFT-D3：NEP 模型本身可能已带范德华项（`nep4-vdw`），mliport 没有证据表明
再叠加 D3 不会重复计算色散。该组合被拒绝，而不是被静默丢掉。

## 精度与"同样的数字"意味着什么

`CPUNEP` 用 float64；`GPUNEP` 与原生 `gpumd` 用 float32。GPUMD 输入里的
`precision double` 只提高**文本**输出位数，并不会把推理变成双精度。因此跨路径比较必须使用
实测的 float32 数值底噪，而不是要求逐位一致。

输出中不会出现相反的声称：`model.manifest.json` / `resolved_config.json` 记录
`numeric_precision`（`float64` 或 `float32`）、`per_call_subprocess`、`gpumd` 的
路径/SHA-256/版本，以及实际使用的 GPU UUID。

## 开销模型——不要把两者混为一谈

CUDA 上的 `NEP_RUNTIME=ase` 走 Calorine `GPUNEP`：**每一次** ASE 求值都会写输入、启动
`gpumd`、再读输出。每个 MD 步或优化步都要付进程启动和文件 I/O 的开销。mliport 在构造
计算器时会显著提示，并在身份记录里写 `per_call_subprocess: true`。

原生 `NEP_RUNTIME=gpumd` 一整段计算只启动一个进程。两者之间的 steps/s 差异是关于适配层
的陈述，不是关于 NEP 的。**不要**把 `GPUNEP` 的 ASE-MD 吞吐当作原生 GPUMD 性能。

## 原生 MD：恒温器、初速度与"不等于 ASE"的地方

GPUMD `ensemble nvt_*` 的耦合常数是 `T_coup = tau_T / dt`（无量纲），mliport 会精确换算
你的时间参数：

| mliport `THERMOSTAT` | GPUMD ensemble | 使用的 `tau_T` |
|---|---|---|
| `LANGEVIN` | `nvt_lan` | `1 / friction`（friction 单位 1/fs） |
| `BUSSI` | `nvt_bdp` | `bussi_tau`（fs） |
| `NHC` | `nvt_nhc` | `nhc_tdamp`（fs） |

`T_coup = round(tau_T / dt)` 且必须 `>= 1`；更小的值会被拒绝，因为 GPUMD 会静默地按另一种
含义解释它。实际使用的映射、`tau_T` 与 `T_coup` 都会写入运行清单。

`velocity <T> seed <N>` 得到的**不是** Maxwell–Boltzmann 分布，而且会同时把总线动量与总角
动量置零。不要把 GPUMD 初始化的 NVT 与 ASE-Maxwell 初始化的 NVT 当成等价；清单记录了
初始化方法与随机种子，差异可追溯。

写入 `model.xyz` 的速度单位是 Å/fs。ASE `momenta` 的单位是 amu·Å/(ASE 内部时间单位)，
换算是 `momentum / mass * ase.units.fs`；直接算 `p/m` 会有约 10 倍误差。

## 原生输出布局

```
<out>/native/model.xyz       扩展 XYZ：元素、坐标、晶胞、PBC、速度
<out>/native/run.in          实际执行的输入文件
<out>/native/thermo.out      原始 GPUMD 热力学输出（保留 append 语义）
<out>/native/traj.xyz        原始 dump_xyz 输出，未改动
<out>/native/stdout.log
<out>/native/stderr.log
<out>/native/restart.xyz     仅当请求 dump_restart 时
<out>/native/provenance.json
<out>/trajectory.extxyz      规范化轨迹（step、time、cell、pbc、unwrapped）
<out>/manifest.json          终态 + 身份 + 已验证的帧时间表
```

原始 GPUMD 文件在任何规范化之前写出，成功或失败都不会被删除。`gpumd` 退出码为 0 但关键
输出缺失或无法解析时，状态记为 `failed`，绝不会记为 `completed`。

### MD 何时才算 `completed`？

退出码 0 只说明进程结束了。只有当 dump 是所执行 `run.in` 的完整产物时，MD 才记为
`completed`：

- 帧数等于 `steps // dump_interval`（GPUMD 每完成 `dump_interval` 步写一帧，请求步数不是
  间隔整数倍时，末尾不足一个间隔的部分不写帧；V100 实测 20 步/间隔 10 得 2 帧，200 步/间隔
  20 得 10 帧）；
- 每一帧都有记录下来的 `Time=`（绝不推断），时间严格递增、间隔均匀等于
  `dump_interval * dt`、落在积分步网格上，且最后一帧位于所请求运行的最后一个 dump 间隔内；
- 每一帧都是输入的元素种类与顺序、完整三维 PBC、输入晶胞（原生 NVE/NVT 为定胞）以及与第
  0 帧相同的列；
- `thermo.out` 行数与帧数一致，且表头给出 `dump_thermo`、`dt_output`、`num_atoms` 时必须
  与 `run.in`/输入一致。

任一条不满足即记为 `failed`，原因写入清单与结果。规范化的 `trajectory.extxyz` 先写成
`.partial`，所有检查通过后才发布，因此失败的运行不会留下看似结果的文件，`native/` 下的原始
文件作为证据保留。清单还记录已验证的时间表（`trajectory.schedule`：帧数、首末步与时间、
间隔、坐标约定）以及轨迹的 SHA-256。

第一帧在其 dump 间隔内的偏移是 GPUMD 的实现细节，这里不做断言；帧数、间隔均匀性与对请求
窗口的覆盖已足以确定运行长度。请求步数不是间隔整数倍时，最后一次 dump 之后的步数没有任何
输出，对它们只有退出码这一项证据。

转换是单次流式处理（内存里只有一帧），长运行的后处理开销取决于单帧大小而非运行长度。

### 加载原生运行目录做分析

`TrajectoryDataset.load(<out>)`（因此所有 `mliport analyze` 任务）接受原生 GPUMD 运行目录。
清单具有权威性：必须是状态为 `completed` 的 MD 运行，`trajectory.extxyz` 必须与运行器记录
的 SHA-256 逐字节一致，并且帧必须与清单一致（帧数、原子数、PBC、坐标约定、记录的时间与
步数）。否则加载器抛出 `InvalidTrajectoryError`，不会生成数据集。数据集携带 GPUMD 记录的
时间轴与步数、dump 间隔与 `dt`、三维 PBC、GPUMD 自己使用的原子质量、`run.in` 中的目标温度，
以及实际 dump 的坐标约定（带 `unwrapped_position` 为 `unwrapped`，否则为 `wrapped`）。与清
单矛盾的显式 `positions_convention`、`md_timestep_fs`、`frame_stride_steps`、
`frame_interval_fs` 会被拒绝。原生 dump 不含速度（`dump_xyz` 不带 `vel`），因此 VACF 类分析
对它不可用。

### 单位与符号

GPUMD 写出 `virial`（eV，行主序九个分量）以及与之同号的 `stress`（virial/体积，eV/Å³）。
ASE 的应力约定符号相反：

```text
ase_stress_voigt = -gpumd_stress_voigt        # xx yy zz yz xz xy，eV/Å³
GPa              = -ase_stress * 160.2176634
```

`thermo.out` 的列为 `T KE PE sxx syy szz syz sxz sxy ax..cz`，其中应力分量单位是 **GPa**。
注意两个文件的区别：`dump_xyz` 的 `virial`/`stress` 只是**势能（virial）**部分，而
`thermo.out` 的 `sxx..sxy` 是**含动能贡献的总应力**，所以即使换算完单位，两者在有限温度下
也不相等。mliport 报告的是仅势能部分（它才是势能对形变的导数，也才与另外四个后端的 ASE
计算器可比），`thermo.out` 原样保留供检查。

两种约定都由针对 `CPUNEP` 的有限差分已知答案测试校验，而不是相信表头。

### 未折叠坐标与输运分析

`dump_xyz ... unwrapped_position` 给出展开后的轨迹，示踪 MSD、集体电荷位移、Haven 比都
依赖它。该列不可用时，清单会把 `msd_eligible` 置为 `false` 并记录原因，mliport 拒绝用稀疏
的折叠坐标推断体相输运。变胞轨迹没有统一的位移参考，本版本不用于输运。

`restart.xyz` 是覆盖式输出且不含随机数状态；仅仅存在一个文件并不等于可复现的 MD
checkpoint，清单如实记录这一点。

## GPUMD 新版模型格式（包括 NEP89）

新版 GPUMD 模型第一行列出元素，后面用 `format_version 1` 和
`model nep` 分别声明文件格式与模型类型，不再必须使用旧版的 `nep4`：

```text
nep 2 Li S
format_version 1
model nep
zbl 1 2
cutoff 6 5 274 158
```

两条 NEP 路径现在共用解析逻辑，记录元素顺序、模型类型、文件格式版本、
ZBL、元素参考能量及按元素给出的 cutoff。原生 GPUMD 的来源记录还保留
模型原文件的 SHA-256。当前**仅支持普通能量模型** `model nep`；
自旋、温度、qNEP、偶极、极化率与学习型 vdW 变种会明确拒绝。
解析器只检查元数据，不验证完整参数块；实际计算还需要支持该格式的
新版 GPUMD 可执行文件。Calorine 能否读取这种新格式尚未实测，
建议首先使用原生 GPUMD 路径验收。

## 本版本的已知边界

- 不支持 `qNEP` / `TNEP` / `nep4-vdw`：这些模型类型改变能量的含义（自由能、偶极），不能
  当作普通势使用，超出本版本已验证范围。
- 原生路径拒绝部分周期性（`T T F`），不会悄悄提升成 `T T T`。
- 变胞优化仅通过 ASE 运行时契约提供。
- 不宣称原生 checkpoint/restart 等价。
- 某个 `nep.txt` 对某种材料的准确度是模型问题，不是 mliport 的问题。
  `runtime_supported` 不等于 `physically_validated_on_target`。
