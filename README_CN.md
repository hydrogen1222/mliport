# mliport

mliport 通过同一套 VASP 风格的 CLI/TUI/Python 工作流,运行 UMA、MACE、DPA
与 GRACE 机器学习原子间势,完成单点能、结构弛豫、分子动力学、NEB 与轨迹
分析。

边界需要说清楚:

- mliport 计算的是学习得到的势能面(PES),不做 DFT 电子结构计算。
- "VASP 风格"指的是工作流与输出文件的约定(INCAR 式配置、OUTCAR/
  OSZICAR/CONTCAR/XDATCAR 等),目的是让熟悉 VASP 的用户上手成本低;
  这不是与 VASP 的物理等价性。
- 能量、力、应力全部来自所选模型。结果的精度取决于模型对你的化学体系
  的适用性,与 mliport 本身无关。

许可证:MIT。状态:**beta 验证已完成**,对应软件提交
`5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`(current-HEAD 四后端
V100-SXM2-16GB smoke:MACE/DPA/GRACE/UMA 的 T1/T2FD/T5/T6/T7,
外加 Python 3.10-3.12 CI 与干净 wheel 安装);发布物本身仍是
`2.0.0b1` beta 候选版。T3 精度、T4 静态工作流与 T8 性能保持历史证据,
明确**不**作为 current-HEAD 声明。本项目此前以 `mlipx` 发布,该名称在
PyPI/import 层面与 BASF 的同名包冲突;`mliport` 为 clean-break 改名
(旧顶层 import 不再发布)。见下方[验证](#验证)一节。

## 能运行什么

| 任务 | 命令 | 说明 |
|---|---|---|
| 单点能 | `mliport sp` | 能量、力、应力(应力取决于模型是否提供) |
| 离子弛豫 | `mliport opt` | 固定晶胞,FIRE/LBFGS/BFGS |
| 晶胞 + 离子弛豫 | `mliport opt` | FrechetCellFilter;要求模型提供应力且为三维周期体系 |
| NVE 分子动力学 | `mliport md` | velocity Verlet |
| NVT 分子动力学 | `mliport md` | Langevin、Bussi、Nosé-Hoover 链 |
| NEB / CI-NEB | `mliport neb` | 固定晶胞,IDPP 预弛豫,两阶段攀爬图像 |
| 批量计算 | `mliport batch` | 一次模型加载处理多个结构 |
| 轨迹分析 | `mliport analyze` | validate、thermo、rdf、rmsd、msd、vacf、spectrum、transport、density、arrhenius、GEMDAT 机理分析 |
| INCAR 驱动 | `mliport run -i INCAR.mliport` | VASP 风格入口 |
| 队列 | `mliport queue submit/start`、`mliport jobs` | 后台作业执行 |

ASE 能读的结构格式(POSCAR/CONTCAR、CIF、EXTXYZ 等)都可用作输入。
输出为 VASP 兼容文本文件加 JSON 记录。

## 不能替代什么

以下性质需要 DFT 代码,或在 mliport 中明确缺席:

| 性质 | mliport 中的状态 |
|---|---|
| 电子能带结构 | 不可用 |
| 态密度 | 不可用 |
| 电荷密度 / Bader / ELF | 不可用 |
| Born 有效电荷 | 不可用 |
| 介电响应 | 不可用 |
| k 点 / ENCUT / SCF 收敛 | 不适用(无 SCF) |
| NPT 分子动力学 | 未实现 |

## 安装

mliport 安装在**每个后端自己的 Python 环境内**,它不是跨环境调度器。经过
验证的安装路径是安装脚本,它为每个后端建立独立环境(四套依赖互斥,无法
共用一个环境):

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

安装脚本的行为:

- 从驱动读取 GPU 架构,按后端锁定兼容的框架版本(Volta/V100 用
  torch 2.8.0+cu126;Turing 及更新架构用 cu128 构建)。
- 按 compatibility registry 的环境名创建虚拟环境,并生成
  `./bin/mliport-mace`、`./bin/mliport-dpa`、`./bin/mliport-grace`、
  `./bin/mliport-uma` 薄 launcher(只 exec 对应环境的 CLI)。环境名、
  每个后端的 Python 范围与运行时锁定由
  [docs/installation.md](docs/installation.md) 机器生成。
- Python 按后端选择:`--python` 必须满足每个请求后端的 `requires-python`
  (UMA 需要 >=3.11),否则安装脚本在改动任何东西之前 fail closed,绝不
  静默替换版本。
- 模型权重在首次使用时下载;`--source` 控制轮子来源;`--dry-run` 只打印
  计划不执行。
- 四个后端连同 torch 与模型权重约需 10-15 GB 磁盘。
- 结束时在每个环境运行 `mliport doctor`,除非给出 `--skip-doctor`。

运行 CLI 时使用对应后端的环境(或 launcher):

```bash
.venv-mace/bin/mliport sp structure.cif --model model.model --model-type mace
.venv/bin/mliport      sp structure.cif --model uma.pt     --model-type uma
./bin/mliport-mace     sp structure.cif --model model.model --model-type mace
```

Windows 下每个环境提供 `Scripts\mliport.exe`
(`.venv-mace\Scripts\mliport.exe`);把 mliport 安装到某个"全局"环境并不会
让它去调用其他后端环境。

<details>
<summary>手动安装(按后端)</summary>

每个后端环境需要 mliport 包加引擎自身的依赖栈。权威的版本锁定在
`mliport/mliport/install/compatibility.py`;只有安装脚本会保证这些锁定与
你的 GPU 架构一致。如果手动安装,每个后端建一个环境,把 `./mliport` 安装
**进那个环境**,然后运行 `mliport doctor` 并确认全部检查通过,再开始正式
计算。

</details>


## GPU 架构兼容性

安装脚本与 `mliport setup` 会自动选择正确的 PyTorch/CUDA 轮子通道。

| GPU 系列 | 例子 | 算力 | CUDA 路线 |
|---|---|---|---|
| Maxwell | GTX 960、TITAN X | sm_50/52 | cu126 Legacy(实验性) |
| Pascal | Tesla P40、GTX 1080 Ti、P100 | sm_60/61 | cu126 Legacy |
| Volta | V100 | sm_70 | cu126 Legacy |
| Turing | RTX 20xx | sm_75 | cu128+ Modern |
| Ampere | RTX 3080 Ti、30xx | sm_80/86 | cu128+ Modern |
| Ada | RTX 4090、40xx | sm_89 | cu128+ Modern |
| Hopper | H100 | sm_90 | cu128+ Modern |
| Blackwell | RTX 50xx | sm_100/120 | cu128+ Modern |
| 无 | 仅 CPU | - | CPU 轮子 |

为什么有两条 CUDA 路线:Maxwell/Pascal/Volta 必须走 cu126 旧通道,因为
PyTorch 2.8+ 的 cu128 构建已移除 Maxwell/Pascal 支持,PyTorch 2.11+
则移除了 Volta。Turing 及更新架构走 modern 通道(torch 2.8-2.10 用
cu128,torch 2.12+ 用 cu130)。Maxwell 标为实验性,因为官方 
TensorFlow 2.20 轮子从 sm_60 起才有支持。

**架构兼容性**(来自 `mliport/install/compatibility.py`;它只描述安装
路线——上游包支持情况、锁定的后端版本、CUDA 轮子通道——*不是*负载
级别的认证)。"needs runtime smoke test"表示该 GPU 系列的安装契约
是自洽的,但 mliport 尚未在真实的该引擎 + 框架 + GPU 组合上验证;
"experimental"表示上游本身不支持或不测试该组合。安装路线冒烟测试
(引擎装好、真实模型预测)已在真实的 V100 与 RTX 4090 硬件上运行;
负载级证据目前只有下文的 V100 运行时验证。P40 使用了修正后的精确
`+cu126` 轮子锁定,但仍需要在修复后的版本上做模型冒烟复测。

| 引擎 | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | 其他 Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| MACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| DPA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| GRACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | experimental |

## 第一次计算

装好 UMA 环境后,对一个结构文件:

```bash
mliport sp POSCAR --model uma-s-1p2.pt --model-type UMA \
    --task omat --device cuda --output ./results
```

会在 `./results` 中写出 `OUTCAR`(能量、力、应力、模型身份、设备证据)
与 JSON 记录。同样的计算用 INCAR 形式:

```bash
mliport template -o INCAR.mliport sp   # 然后编辑 MODEL_PATH/TASK/DEVICE
mliport run -i INCAR.mliport
```

配置优先级:显式 CLI 参数 > INCAR 文件 > `settings.ini` > 内置默认值。
`mliport config show` 会打印完整解析后的配置,并标明每个值来自哪里。

## 选择模型

四个 beta 验证过的常用 profile,全部在同一个 OMat24 子集上完成验证:

| 后端 | 验证 profile | 训练域 | Task/head | 精度 | 应力 | 获取方式 |
|---|---|---|---|---|---|---|
| MACE | `mace-omat-0-medium.model` | OMat24 + MPtrj | bulk | float64(另有 float32) | 有 | meta 下载(CC BY 4.0) |
| DPA | `DPA-3.1-3M.pt`,分支 `Omat24` | OMat24 + MPtrj | `--head Omat24` | 上游定义 | 有 | DeepModeling 分享 |
| GRACE | `GRACE-2L-OMAT-medium-base` | OMat24 | - | 上游定义(fp32 构建) | 有 | Intel 开放下载 |
| UMA | `uma-s-1p2.pt` | OMat 等多任务头 | `--task omat` | 上游定义 | 有 | fairchem,meta 下载 |

这些是整个验证套件共用的 profile。领域专用检查点(例如针对特定化学
体系拟合的 MACE 模型、Omat24 以外的 DPA 分支)可以按同样方式加载,
但上表并不意味着这些模型获得了同等的 beta 验证覆盖,也不应把它们的
基准数字与上表 OMat24 系模型的结果直接比较。

> 不同参考计算训练出的模型,其能量不在同一热力学能量标度上。不要跨
> 模型、跨 task/head、跨参考能级混用绝对能量。同一模型同一 task 内的
> 相对能量(以该模型自身的元素参考能计算的生成能、同一模型给出的
> 势垒)才是可靠的比较对象。

## 工作流

### 单点能

`mliport sp` 把一个结构送入计算器,写出能量/力/应力。能量出现 NaN/inf
时,在任何输出文件写出之前即中止。

### 弛豫

`mliport opt` 在固定晶胞下弛豫离子位置(FIRE、LBFGS 或 BFGS)。当模型
提供应力且体系为三维周期时,通过 ASE 的 `FrechetCellFilter` 同时弛豫
晶胞与离子。输出报告最终 fmax 与步数;晶胞弛豫同时打印初始与最终
体积。

### 分子动力学

`mliport md` 支持 NVE(velocity Verlet)与三种恒温器的 NVT:Langevin、
Bussi 随机速度重标定、Nosé-Hoover 链。轨迹写为 XDATCAR 加 JSON(含
每帧能量)。积分步长与保存间隔相互独立;分析命令从轨迹元数据读取
保存间隔——保存间隔过粗会让传输分析退化,而不是悄悄给出错误结果。

### NEB / CI-NEB

`mliport neb` 在固定晶胞下计算最小能量路径:

- 端点可以直接使用或预先弛豫(`--endpoint-policy validate|relax`)。
- 初始路径:带周期映像(winding)处理的线性插值,或 IDPP 预弛豫。
- 两阶段攀爬图像:标准 NEB 收敛到力阈值后,切换 CI-NEB 收敛到
  最终判据。
- 运行中写检查点;重启时从检查点恢复几何,并保证端点身份不变。
- 势垒:从采样路径带给出正向与反向势垒(JSON 记录中的
  `barrier_forward` / `barrier_reverse`)。

> 收敛的 CI 图像只是鞍点候选。引用过渡态之前,先用鞍点 Hessian 诊断
> 验证(沿反应坐标应恰有一个虚频)。

### 分析

`mliport analyze` 处理 mliport 轨迹或外部轨迹(外部轨迹需显式给出坐标
约定 `--positions-convention`)。扩散问题的层级:

1. `msd` — 窗口化、按方向分解的 MSD 诊断。
2. `transport` — kinisi 定量示踪扩散(后验 D 与可信区间),以及由
   离子电荷得到的 Nernst-Einstein 电导率;要求固定晶胞(NVT/NVE)。
3. `electrolyte` — GEMDAT 位点映射、跳跃机理与逾渗,作为传输图像的
   机理层面交叉验证。

不确定度语义:kinisi 给出后验均值与 95% 可信区间;Nernst-Einstein
电导率只在稀疏、无关联跳跃的极限下才严格成立。示踪扩散与集体传输
之间的 Haven 比不做任何假定。

## 科学语义

- **能量**:eV,按输入晶胞的总能量。每原子值在输出中标注为 per atom。
- **力**:eV/Å。
- **应力**:ASE Voigt 顺序(`xx, yy, zz, yz, xz, xy`);以 eV/Å³ 写出,
  OUTCAR 中附 GPa 换算行。
- **task/head 身份**:模型的 task 或 head(如 UMA 的 `omat`、DPA 的
  `Omat24`)选择模型内部的参考能级。它记录在每一份输出中;打算做
  能量减法的计算之间不得更改。
- **力-能量一致性**:力是模型能量的解析导数。beta 套件用有限差分
  交叉验证;各模型结果见验证报告。
- **应力-能量一致性**:模型提供应力时,应力是能量的解析应变导数,
  beta 套件用有限差分交叉验证。
- **PBC 与 winding**:最小映像约定在 NEB 插值与位移分析中显式处理
  周期映像;外部轨迹的坐标约定与工件元数据冲突时,分析命令会拒绝
  处理。
- **约束**:ASE 约束(FixAtoms、FixSymmetry)在弛豫与 MD 中生效;
  约束精确性有回归测试(固定原子的自由度保持到机器精度)。
- **MD 步长**:由你选择。beta 套件对每个体系做步长扫描(Cu 上
  0.25-2.0 fs),以 NVE 能量漂移斜率作为判据;验证报告中记录了哪些
  步长稳定。不存在全局"安全"步长。
- **保存间隔与步长**:分析只作用于保存的帧。要分辨你测量的过程,
  保存间隔必须足够密。`mliport analyze validate` 会检查这一点,采样
  不足时报告问题而不是给出一个数字。
- **传输分析的固定晶胞要求**:kinisi 传输分析要求 NVT/NVE 轨迹。
  NPT 未实现,因此也不提供对恒压轨迹的传输分析。

## 验证

以下数字由 beta 证据记录渲染生成,不是手写。标记之间的生成块由
`validation/science/scripts/generate_beta_report.py` 从
`validation/science/reports/beta-summary.json` 产出;仓库测试会把
README 中的生成块与生成器输出逐字节比对,漂移即失败。

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

Validation status per backend, rendered from the beta evidence records (`beta-summary.json`; t1-t8 tiers, 4 backends x OMat24 common subset). `software_validated` means the mliport integration and all recorded checks passed; `model_characterized` means the workflow ran and its behavior was recorded, including honest failures (e.g. float32 arithmetic noise). Full per-test tables: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md).

| Workflow | MACE | DPA | GRACE | UMA |
|---|---|---|---|---|
| Install & doctor (CI software tests) | software_validated* | software_validated* | software_validated* | software_validated* |
| Single-point inference (4 structures) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Energy-forces consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) |
| Stress (finite-difference cross-check) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) |
| Stress-energy consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) |
| Coordinate invariance & cache | not_run | not_run | not_run | not_run |
| Fixed-cell relaxation | not_run | not_run | not_run | not_run |
| Cell relaxation | not_run | not_run | not_run | not_run |
| EOS / bulk modulus | not_run | not_run | not_run | not_run |
| Elastic constants | not_run | not_run | not_run | not_run |
| Harmonic phonons | not_run | not_run | not_run | not_run |
| Harmonic thermodynamics | not_run | not_run | not_run | not_run |
| Vacancy formation energy | not_run | not_run | not_run | not_run |
| Surface energy | not_run | not_run | not_run | not_run |
| NEB | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Saddle-point Hessian | not_run | not_run | not_run | not_run |
| Short NVE / NVT MD | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) |
| Transport analysis (demonstration) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) |
| Mechanism analysis (GEMDAT) | model_characterized (characterizedx1) | model_characterized (characterizedx1) | model_characterized (characterizedx1) | model_characterized (characterizedx1) |
| Performance (SP/MD scaling) | software_validated (passx5) | software_validated (passx5) | software_validated (passx5) | software_validated (passx5) |

Held-out OMat24 accuracy: not run on this evidence set.

`*` = CI software test only, no model involved. A cell lists the recorded statuses for that workload; per-workload rows reuse the same evidence tiers, so row counts are not additive. Full per-test tables and limitations: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md). Model identities pinned in `validation/science/model_manifest.json`.
<!-- END GENERATED -->

### 接口能力矩阵

各入口不声称等价。机器可读的
`validation/science/capability_matrix.json` 按功能记录 Python API、直接
CLI、INCAR 式键集与 TUI 是否能表达该功能。高级的显式原子映射 / 晶格
平移控制只通过 API/direct CLI/TUI 提供;INCAR 式键集刻意不包含
`neb_atom_map` 与 `neb_image_shifts`。

各 tier 含义:t1 四后端推理(公共子集)、t2 不变性与缓存、t2fd 有限
差分应力、t3 OMat24 标签对比、t4 静态工作流(弛豫、EOS、弹性、声子、
热力学、缺陷、表面)、t5 NEB、t6 MD、t7 分析、t8 性能。逐测试表格
(含每一条记录在案的失败)见
[validation/science/reports/BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md)。

套件记录在案、不作隐藏的已知局限:上游 float32 构建(DPA、UMA、
关缓存的 GRACE)在坐标不变性检查上呈现 1e-7..1e-6 eV 的算术噪声;
2x2x2 小超胞声子在小波矢处出现微小的虚声学支;内聚能与生成能记录为
unsupported,因为公共比较需要公共参考能级之外的元素参考能。


**安装契约运行时冒烟验证**(单卡 V100-SXM2-16GB,sm_70,驱动
580.173.02)——在单块 GPU 上用硬超时运行的真实模型短测试,每一条都
针对安装脚本产出的*确切*运行时。脱敏记录在
[validation/runtime/v100/](validation/runtime/v100/)(GPU 仅以 UUID 的
SHA-256 标识);每个状态对应且仅对应一条记录。MACE 记录已在当前
安装契约(`mace-torch 0.3.16` + `torch 2.8.0+cu126`)上重新验证;更早
的 torch 2.6.0+cu124 证据保留在 git 历史中。

| 后端 / 模型 | SP | 短 MD | E–F 梯度 | NEB 冒烟 | GRACE 缓存 | 记录 |
|---|---|---|---|---|---|---|
| UMA | not run | not run | not run | not run | n/a | [uma.json](validation/runtime/v100/uma.json)(该机器无离线 wheelhouse) |
| MACE float64 | passed | passed | passed | passed | n/a | [mace.json](validation/runtime/v100/mace.json) |
| MACE float32 | passed | passed | passed | passed | n/a | [mace-float32.json](validation/runtime/v100/mace-float32.json) |
| DPA `Domains_Alloy` | passed | passed | passed | passed | n/a | [dpa.json](validation/runtime/v100/dpa.json) |
| GRACE float32 | passed | passed | passed | passed | passed(ON 与 OFF) | [grace.json](validation/runtime/v100/grace.json)、[grace-nocache.json](validation/runtime/v100/grace-nocache.json) |

NEB 冒烟是 4 原子 Cu 晶胞的 5 图像固定晶胞短测试,
`saddle_validation: not_performed`——含义见 [NEB / CI-NEB](#neb--ci-neb)
一节。这份安装冒烟证据早于 beta 套件;beta 套件才是[选择模型](#选择模型)
中各模型配置的负载级证据来源。

## DFT 式验证配方

验证套件包含 EOS、弹性常数、谐波声子、热力学、缺陷与表面等配方,
构建在同一批计算器与 ASE 之上。它们是可复现脚本形式的 beta 验证配方
(`validation/science/scripts/`),不是稳定 CLI 命令;CLI 没有把它们
暴露为一级子命令。

## 输出与可复现性

一次运行的目录内容:

```
results/
├── OUTCAR        # 能量/力/应力、模型身份、设备、版本
├── OSZICAR       # 逐步日志(弛豫、MD)
├── CONTCAR       # 最终结构(弛豫)
├── XDATCAR       # 轨迹(MD)
└── *.json        # 机器可读记录,含溯源信息
```

每条记录携带:模型路径与 SHA-256、task/head、dtype、设备(请求值与
实际值、GPU UUID 哈希)、mliport 与框架版本、git commit 与科学套件
修订号、结果 schema 版本。NEB 与 MD 写检查点;分析结果按请求哈希
缓存,相同请求直接命中,请求变化即重算。

## Python API

公开接口(`mliport.api`):

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("POSCAR", model_path="mace-omat-0-medium.model",
                          model_type="MACE", device="cpu")
results = run_single_point("POSCAR", model_path="mace-omat-0-medium.model",
                           model_type="MACE", output_dir="./results")
```

另有 `run_optimization`、`run_md`、`run_neb`。以上代码片段都在仓库
测试套件中执行;片段与真实签名漂移会让 CI 失败。

## 配置优先级

CLI 参数 > INCAR 文件 > `settings.ini` > 内置默认值。`mliport config
show` 打印每个值的来源路径;`mliport config schema` 列出全部可识别的
键。TUI(`mliport tui`)以交互方式暴露同一配置空间。

## 故障排查

以下为实际观察到、可诊断的失败:

- **模型下载/鉴权失败**(UMA、MACE 的 meta 检查点):下载需要网络,
  部分检查点需要先接受许可协议。下载器的错误信息会原样透传。
- **DPA 分支用错**:DPA-3.1-3M 是多头的。`--head`/TASK 与训练分支
  不匹配时,结果会静默地来自错误的头。记录中的 `head` 字段显示实际
  使用的分支;安装 profile 锁定 `Omat24`。
- **模型不提供应力**:晶胞弛豫、弹性配方等都需要应力。mliport 以显式
  报错的方式 fail closed,不会伪造应力。
- **GPU 架构不匹配**(例如 Volta 卡装了仅含 cu128 内核的 torch
  构建):`mliport doctor` 会报告算力版本,安装脚本据此锁定对应构建;
  忽略此事的手动安装会在第一次内核启动时失败。
- **GRACE 显存**:四个后端中 GRACE 在大晶胞下最耗显存;批量计算先
  调低 `--batch-size`,再怀疑是 bug。
- **能力证据不一致**:README 表格与策划的能力注册表由测试交叉校验;
  运行时契约变更若未重新验证,CI 会失败,而不是让过时的声明随包
  发布。
- **传输采样不足**:轨迹太短或保存间隔太粗时,`mliport analyze
  transport` 会报告问题,而不是返回一个数字。

## 局限

- 学习型 PES 的质量依赖训练域。表面、缺陷与过渡态对任何模型都可能
  是分布外数据。beta 套件记录的是固定配方集上各模型的行为,不构成
  对你的化学体系的认证。
- 无电子结构(见上表)。
- 绝对能量不可跨模型、跨 task/head、跨参考能级混用。
- 物理可信的传输数值需要远长于 beta 演示运行的轨迹;套件将其标注为
  `demonstration_not_converged`。
- 无 NPT 系综。
- 未实现模型预测不确定度。
- 历史 beta 验证覆盖一种 GPU 架构(V100,sm_70)与 CPU,且早于当前验证器语义。它不证明所有
  GPU 架构都可用;在你的硬件上,请先运行 `mliport doctor` 再信任首次
  计算结果。

## 引用

mliport 目前没有自己的 DOI。请引用本仓库与你使用的版本
(元数据见 [`CITATION.cff`](CITATION.cff)),并按实际使用的 backend/分析库
分别引用对应上游软件与模型论文——见 [docs/citations.md](docs/citations.md)。
上游 DOI 绝不代表 mliport 的 DOI。

## 致谢

- [FAIR-Chem / UMA](https://github.com/facebookresearch/fairchem)
- [MACE](https://github.com/ACEsuit/mace)
- [DeePMD-kit / DPA](https://github.com/deepmodeling/deepmd-kit)
- [GRACE / tensorpotential](https://github.com/ICAMS/grace-tensorpotential)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [kinisi](https://github.com/bjmorgan/kinisi)
- [GEMDAT](https://github.com/GEMDAT-repos/GEMDAT)
- [OMat24 / Meta](https://ai.meta.com/blog/open-source-climate-modeling/)

本项目(`hydrogen1222/mliport`)与 PyPI 上另一个同名 `mliport` 项目无关。
