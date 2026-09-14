# mliport

mliport 用同一套命令行流程运行 MACE、DPA (DeepMD-kit)、GRACE
(tensorpotential) 与 UMA (FAIRChem) 机器学习势。它读取 ASE 能读的结构,
接受 INCAR 风格的选项,并写出 VASP 风格的 `OUTCAR`、`OSZICAR`、`CONTCAR`
与 `XDATCAR`,方便已经习惯 POSCAR/OUTCAR 的用户。

每次计算都会记录模型文件及其 SHA-256(可得时)、请求的设备与后端实际使用的
设备、解析后的配置和 mliport 版本。配置错误默认直接失败:拼错的键、或者把
别的 backend 的选项写进当前 backend,都不会被静默忽略。同一次计算可以从
CLI、INCAR 风格文件、TUI、后台队列或 Python API 启动。

工作流:单点、固定/可变晶胞弛豫、分子动力学(NVE,以及 Langevin、Bussi、
Nose-Hoover chain 三种 NVT 恒温器)、NEB/CI-NEB、批量扫描。分析:轨迹校验、
热力学量、RDF、RMSD/RMSF、MSD、VACF、速度谱、可动离子密度、kinisi 输运、
GEMDAT 电解质分析和 Arrhenius 拟合。

## 安装

支持 Linux 与 WSL2。安装脚本需要 `git` 和
[uv](https://docs.astral.sh/uv/);它会为每个 backend 创建一个 Python 环境,
并在 `bin/` 下生成 launcher。

```bash
# 1. 如果还没有 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 获取仓库
git clone https://github.com/hydrogen1222/mliport
cd mliport
```

NVIDIA GPU(需要可用的 CUDA 驱动;本路径已在 V100-SXM2-16GB、驱动
580.173.02 上验证):

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

仅 CPU:

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

安装脚本按 `mliport/install/compatibility.py` 固定各 backend 的上游版本,
一并安装 analysis、transport 与 electrolyte 功能,并对每个 backend 运行
`mliport doctor`。四 backend 安装默认使用 Python 3.12。各 backend 的最低
Python 版本是 MACE/GRACE `>=3.9`、DPA `>=3.10`、UMA `>=3.11`;包本身支持
Python 3.10-3.12。如果用 `--python` 覆盖解释器,所有被请求的 backend 都
必须支持该版本。

四 backend 安装完成后的环境布局:

| Backend | 环境 | Launcher |
|---|---|---|
| MACE | `.venv-mace` | `bin/mliport-mace` |
| DPA | `.venv-dpa` | `bin/mliport-dpa` |
| GRACE | `.venv-grace` | `bin/mliport-grace` |
| UMA | `.venv` | `bin/mliport-uma` |

从目标 backend 的环境或 launcher 运行:

```bash
.venv-mace/bin/mliport --help
./bin/mliport-mace --help
```

然后检查运行环境。必需项失败时命令返回非零;可选功能会单独列出:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

成功时的结尾类似:

```text
 Environment checks passed for MACE on CUDA.
```

Windows 下每个环境提供 `Scripts\mliport.exe`,安装器还会生成
`bin\mliport-*.cmd` launcher。`mliport doctor --json` 输出同样的检查结果,
便于脚本和集群配置使用。CPU 安装、自定义源、离线机器、代理、磁盘占用和
卸载步骤见 [docs/installation.md](docs/installation.md)。

### 模型

mliport 不自动下载模型。请用上游工具获取 checkpoint,再用 `--model` 指向
本地文件;本地目录/文件即可离线运行。

| Backend | Checkpoint | 获取地址 |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

验证套件使用的精确文件哈希固定记录在
[`validation/science/model_manifest.json`](validation/science/model_manifest.json)。
checkpoint 的许可证由上游项目决定;UMA checkpoint 使用非商业研究许可证。
Hugging Face 镜像与 SOCKS 代理设置见 `docs/models.md`。

## 第一次计算

仓库自带一个 50 原子的 Li10GeP2S12 (LGPS) 原胞:
`examples/structures/li10gep2s12_primitive.vasp`。它是真实的周期性晶胞,
适合做短时间验收和 CPU smoke。

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda \
  --output runs/lgps-mace-sp
```

运行会写出 `runs/lgps-mace-sp/mliport_results.json`(能量、力、应力、计时)、
`resolved_config.json`(实际使用的配置)、`OUTCAR`/`OSZICAR`/`CONTCAR` 和
`run.log`。能量与力必须是有限值;`mliport_results.json` 里记录的是
backend 实际使用的设备,而不是请求的设备。

## 选择 backend

backend 由 `--model-type`、配置中的 alias/profile 决定;无法确定时直接
失败,不会自动改用别的模型。

```bash
# MACE
.venv-mace/bin/mliport sp LGPS.vasp --model mace-omat-0-medium.model --model-type mace --task bulk --device cuda

# DPA
.venv-dpa/bin/mliport sp LGPS.vasp --model DPA-3.1-3M.pt --model-type dpa --task bulk --head Omat24 --device cuda

# GRACE
.venv-grace/bin/mliport sp LGPS.vasp --model GRACE-2L-OMAT-medium-base --model-type grace --task bulk --device cuda

# UMA
.venv/bin/mliport sp LGPS.vasp --model uma-s-1p2.pt --model-type uma --task omat --device cuda
```

| Backend | Model type | Task/head | 说明 |
|---|---|---|---|
| MACE | `mace` | `bulk` | 默认 float64;`--dtype float32` 更快、精度更低 |
| DPA | `dpa` | `bulk`,`--head Omat24` | 多任务 checkpoint,branch 从模型文件读取 |
| GRACE | `grace` | `bulk` | TensorFlow 运行时,支持 CPU 与 GPU |
| UMA | `uma`(`fairchem` alias) | `omat` | 分子用 `--task omol`;非商业研究许可证 |

DPA 与 GRACE 在框架加载后无法再切换 CUDA context。如果
`CUDA_VISIBLE_DEVICES` 尚未设置,CLI 会用一个可见目标 GPU 的新进程重启
同一条命令,并打印一行说明。Python API 不能重启调用方;在 Python 里调用
DPA/GRACE 前请自行设置 `CUDA_VISIBLE_DEVICES`。各 backend 手册见
[docs/backends/index.md](docs/backends/index.md)。

## 主要工作流

**结构弛豫** (`opt`):`--fmax`、`--max-steps` 控制收敛;`--cell-opt` 同时
弛豫晶胞与离子。

```bash
.venv-mace/bin/mliport opt LGPS.vasp --model model.model --model-type mace \
  --fmax 0.05 --max-steps 200 --cell-opt --device cuda --output runs/lgps-opt
```

**分子动力学** (`md`):`--ensemble NVE` 或 `NVT`,`--thermostat
LANGEVIN|BUSSI|NHC`,`--temp`、`--timestep`、`--steps`、`--save-interval`。
要接着已有轨迹继续跑,可从带速度的保存帧开始,并加
`--velocity-policy preserve --no-pre-relax`。NHC 需要 `--com-policy none`,
因为有约束的 NHC 不能与自动质心移除同时使用。

```bash
.venv-mace/bin/mliport md LGPS.vasp --model model.model --model-type mace \
  --ensemble NVT --thermostat LANGEVIN --temp 800 --timestep 1.0 \
  --steps 10000 --save-interval 100 --device cuda --output runs/lgps-md
```

**NEB / CI-NEB** (`neb`):端点文件定义路径,`--images` 是中间 image 数,
`--climb` 打开 climbing image。当前 checkpoint 尚未验证能量-梯度一致性,
因此需要显式的实验性 opt-in,并且结果会标注
`physical_barrier=not_claimed`:

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --images 5 --climb --allow-unvalidated-neb \
  --model model.model --model-type mace --device cuda --output runs/lgps-neb
```

用 `--resume runs/lgps-neb --output runs/lgps-neb` 续跑已经停止的 band。
已记录的步数预算属于 run identity;当前 beta 中 resume 时修改
`--max-steps` 会被拒绝。高级的显式 atom mapping/image-shift 控制只能通过
API/直接 CLI/TUI 使用。

**批量计算** (`batch`):一个目录、一个模型、一种计算类型。

```bash
.venv-mace/bin/mliport batch structures/ --pattern "*.vasp" --calc-type sp \
  --model model.model --model-type mace --device cuda --output runs/batch
```

**INCAR 风格运行** (`run`):先生成模板,编辑后再运行。

```bash
.venv-mace/bin/mliport template sp --output INCAR.mliport
# 填写 MODEL_PATH、MODEL_TYPE、TASK、DEVICE(DPA 还要填 HEAD)
.venv-mace/bin/mliport run -i INCAR.mliport -s LGPS.vasp -o runs/incar-sp
```

**队列** (`queue`、`jobs`、`kill`、`clean`):把多个计算写进 JSON 任务文件,
后台启动调度器,再查看任务。

```bash
.venv-mace/bin/mliport queue submit tasks.json
.venv-mace/bin/mliport queue start
.venv-mace/bin/mliport jobs
.venv-mace/bin/mliport queue stop
```

**TUI**:`mliport tui` 打开交互界面,需要终端;没有终端时命令会直接报错
退出,不会一直等待。

**Python API**:

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("LGPS.vasp", "model.model", model_type="mace", device="cuda")
result = run_single_point("LGPS.vasp", "model.model", model_type="mace", device="cuda")
```

API 还提供 `run_optimization`、`run_md` 与 `run_neb`。同一结构、模型和设备
下,CLI 与 API 给出的能量一致。

## 分析轨迹

`analyze` 接受 mliport run 目录或 ASE trajectory:

```bash
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md msd --mobile Li
.venv-mace/bin/mliport analyze runs/lgps-md rdf --center Li --neighbor S
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md vacf
.venv-mace/bin/mliport analyze runs/lgps-md spectrum
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
.venv-mace/bin/mliport analyze runs/lgps-md transport --mobile Li --charge 1 --fit-start-ps 100
.venv-mace/bin/mliport analyze runs/lgps-md electrolyte --mobile Li --discover-sites-from-density
```

分析结果写到 `runs/lgps-md/analysis/<task>/<id>/`,包括 JSON、CSV,以及
适合作图时的 PNG 与 SVG。绘图以 headless 方式完成,不需要 X11。

短轨迹是合法输入,但不会被过度解读:`msd` 与 `transport` 会返回明确的
insufficient-window / insufficient-sampling 状态,而不是给出扩散系数;
`transport` 的默认 lag grid 超过内存保护时会要求显式传入
`--lag-step-ps`/`--lag-stop-ps`。`arrhenius` 只拟合 `transport` 实际给出的
温度/扩散系数对,不会为失败的点编造数值。各任务的选项和语义见
[docs/analysis/index.md](docs/analysis/index.md)。

## 配置

- backend 必须显式选择:`--model-type mace|dpa|grace|uma` 加 `--model`,
  或使用配置好的 alias/profile。`fairchem` 是 `uma` 的别名。
- 严格配置是默认行为。未知键或跨 backend 键直接失败;探索性运行可以用
  `--lenient-config` 放宽。
- `mliport config` 用于查看和管理设置。项目 `settings.ini` 可以定义模型
  alias 与 profile;显式 `--settings PATH` 优先于用户级和项目级文件。
- 每次运行都会写 `resolved_config.json`,记录每个值的来源,便于事后审计。

## 验证与限制

当前版本是 beta candidate (2.0.0b3),已在干净的 V100 主机上按本文档安装,
并通过 LGPS 工作流与分析矩阵验收;下面的生成状态块给出报告链接。

INCAR 风格输入覆盖常用控制项,但不等同于全部接口。每个入口(API、直接
CLI、INCAR、TUI)能否表达某个选项记录在
`validation/science/capability_matrix.json`;标记为 `no` 或 `partial` 表示
该入口无法表达该选项,不会静默转换。

模型精度属于上游 checkpoint;短 MD 验证的是工作流而不是平衡态;当前
checkpoint 上的 NEB 属于实验性,需要 `--allow-unvalidated-neb`,并且不声明
物理势垒。输运分析采样不足时会以明确的 insufficient-sampling 状态失败,
不会给出数值。

## 文档

- [安装](docs/installation.md)
- [快速上手](docs/quickstart.md)
- [Backend](docs/backends/index.md)
- [工作流](docs/workflows/index.md)
- [分析](docs/analysis/index.md)
- [输出文件](docs/outputs.md)
- [配置](docs/configuration.md)
- [模型与许可证](docs/models.md)
- [故障排查](docs/troubleshooting.md)
- [验证](docs/validation.md)

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

Validated hardware: NVIDIA V100-SXM2-16GB (Volta, 16 GiB), driver 580.173.02, Rocky Linux 9.8. CPU installs are smoke-tested on the same host.

Current status: 2.0.0b3 beta. The four backends are installed and exercised on an LGPS structure through single point, relaxation, MD, NEB, batch, INCAR-style runs, queue, TUI, the Python API and the analysis modules.

- Full beta validation report: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md)
- LGPS fresh-install acceptance: [LGPS_ACCEPTANCE_REPORT.md](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- Model identities: [model_manifest.json](validation/science/model_manifest.json)

Per-workload tables, tier scopes and limitations live in the reports; the README keeps only this status summary.
<!-- END GENERATED -->

## 引用

见 [docs/citations.md](docs/citations.md) 与 [`CITATION.cff`](CITATION.cff)。
使用某个 backend 模型时,请同时引用上游模型并遵守其许可证。

## 许可证

mliport 包使用 [`LICENSE.md`](LICENSE.md) 中的许可证。模型 checkpoint 与上游框架
各自保留其许可证。
