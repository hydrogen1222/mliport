# mliport

[English](README.md) | **简体中文**

**mliport** 是机器学习原子间势（MLIP）的统一工作流层。它让你用同一套命令行、终端界面（TUI）、
INCAR 风格输入文件和 Python API 来运行五种后端 — **MACE**、**DPA**、**GRACE**、**UMA** 和 **NEP**。

准备好结构文件和模型检查点，就可以做单点计算、结构优化、分子动力学（NVE / NVT / NPT）、NEB 势垒搜索、
slab 上吸附构型搜索、批量扫描和轨迹分析。所有结果以 VASP 兼容格式输出（`OUTCAR`、`OSZICAR`、`CONTCAR`、
`XDATCAR`），并附上一份 JSON 记录，让你随时可以追溯每个数值是怎么算出来的。

> **mliport 只是工作流层，不含模型，也不做 DFT。**
> 输出文件格式和 VASP 一样，但里面没有 SCF 循环、没有 k 点网格、没有 POTCAR。模型对你的体系
> 准不准，取决于你选的检查点和你自己做的验证。

📖 **完整文档：** <https://hydrogen1222.github.io/mliport/>
（中英文；同样的页面在 [`docs/`](docs/) 目录下，GitHub 上也能直接看）

## 为什么用 mliport？

| 问题 | mliport 的解决方式 |
|---|---|
| 每个 MLIP 后端有自己的 API 和命令行 | 五个后端共用一套命令 |
| 换后端要改脚本 | 只改 `--model-type` 和 `--model`，其余不变 |
| 不知道实际用了哪些配置 | 每次运行都写 `resolved_config.json`，记录每个参数及来源 |
| 拼错的选项被默默忽略 | 默认严格模式 — 不认识的参数和跨后端参数都是报错 |
| 轨迹太短时给出不可靠的扩散系数 | 分析模块会报告"采样不足"，不给数值 |
| 复现别人的计算 | 自动记录模型文件 SHA-256、实际设备、mliport 版本 |

## 功能概览

- **统一工作流** — 单点、结构优化（固定 / 可变晶胞）、NVE / NVT / NPT 分子动力学、NEB / CI-NEB、
  构型搜索、批量扫描、任务队列
- **五个后端** — MACE、DPA (DeePMD-kit)、GRACE、UMA (Meta / FairChem)、NEP (GPUMD / Calorine)
- **VASP 兼容输出** — `OUTCAR`、`OSZICAR`、`CONTCAR`、`XDATCAR`，VESTA、OVITO、pymatgen 等都能读
- **轨迹分析** — MSD、示踪扩散系数、离子电导率（kinisi）、RDF、RMSD/RMSF、VACF、振动谱、
  可移动离子密度、GEMDAT 电解质分析、Arrhenius 拟合
- **NPT 动力学** — ASE 后端（MACE、DPA、GRACE、UMA）支持各向同性 MTK NPT
- **DFT-D3 色散校正** — 所有后端均可选用
- **Python API** — 六个函数：`run_single_point`、`run_optimization`、`run_md`、`run_neb`、
  `run_confsearch`、`calculate_energy`
- **完整溯源** — 每次运行记录 SHA-256、设备、配置来源和版本号

## 快速导航

| 我想…… | 去看 |
|---|---|
| 跑第一个计算 | [快速上手](docs/quickstart.md) |
| 安装一个或多个后端（GPU 或 CPU） | [安装](docs/installation.md) |
| 在答案已知的体系上检验模型 | [测试体系](docs/test-systems.md) |
| 了解五个后端并选一个 | [后端](docs/backends/index.md)、[模型](docs/models.md) |
| 设置选项（CLI、INCAR 风格、`settings.ini`） | [配置](docs/configuration.md) |
| 单点、弛豫、MD、NEB、构型搜索、批量、队列 | [工作流](docs/workflows/index.md) |
| MSD、扩散、电导率、RDF、谱、Arrhenius | [分析](docs/analysis/index.md) |
| 在 Python 里调用 mliport | [Python API](docs/api.md) |
| 了解每个输出文件的内容 | [输出文件](docs/outputs.md) |
| 解决报错 | [排错](docs/troubleshooting.md) |
| 引用 mliport 和模型 | [引用](docs/citations.md) |

## 安装

**平台：** Linux 和 WSL2。**前提：** `git` 和 [uv](https://docs.astral.sh/uv/)。

安装脚本会给每个后端单独建一个 Python 环境，并在 `bin/` 下生成对应的启动脚本。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # 还没有 uv 的话
git clone https://github.com/hydrogen1222/mliport
cd mliport

# NVIDIA GPU（需要可用的 CUDA 驱动）
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto

# 或者只用 CPU
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu

# NEP 默认不装，需要显式加上
./scripts/install_mliport.sh --engines nep --device cpu
```

| 引擎 | 环境 | 启动器 |
|---|---|---|
| MACE | `.venv-mace` | `bin/mliport-mace` |
| DPA | `.venv-dpa` | `bin/mliport-dpa` |
| GRACE | `.venv-grace` | `bin/mliport-grace` |
| UMA | `.venv` | `bin/mliport-uma` |
| NEP | `.venv-nep` | `bin/mliport-nep` |

也可以不使用启动器，直接调用对应环境里的命令：

```bash
.venv-mace/bin/mliport sp structure.cif --model model.model --model-type mace
.venv/bin/mliport sp structure.cif --model uma.pt --model-type uma
# Windows
.venv-mace\Scripts\mliport.exe sp structure.cif --model model.model --model-type mace
```

安装脚本会锁定各后端的上游版本，装好分析用的依赖，并对每个后端运行 `mliport doctor`。默认
Python 3.12；支持 3.10-3.13（UMA 需要 ≥ 3.11）。在驱动 580.173.02 的 V100 上验证过。
其他 GPU 架构的安装包和 CPU / 离线 / 代理 / 卸载等见 [安装](docs/installation.md)。

<details>
<summary>NEP 安装说明</summary>

NEP 不在默认后端列表里。安装器会在 `.venv-nep` 里装好 Calorine 和 editable 的 mliport，足够
运行 CPU 的 ASE 运行时（`CPUNEP`，float64）。原生运行时和 `GPUNEP` 还需要 `gpumd` 可执行文件，
安装器**不会**编译它（需要你的 CUDA toolkit 和 `-arch=sm_XX`）。编译好之前请用 `--device cpu`；
编译方法见 [NEP 后端](docs/backends/nep.md)。

</details>

### 检查安装

```bash
./bin/mliport-mace doctor --engine mace --device auto
```

检查通过时最后一行是 `Environment checks passed for MACE on CUDA.`。
写脚本时可用 `mliport doctor --json` 拿到 JSON 格式的结果。

### 获取模型

mliport 不会自动下载模型。请从上游下载，再用 `--model` 传入路径：

| 引擎 | 检查点 | 来源 |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |
| NEP | 你自己的 `nep.txt` | 针对你的元素和条件训练；mliport 不附带也不下载 |

验证用的模型哈希：[`validation/science/model_manifest.json`](validation/science/model_manifest.json)。
模型许可证以上游为准；UMA 仅限非商业研究。Hugging Face 镜像和 SOCKS 代理见 [模型](docs/models.md)。

## 第一次计算

仓库里带了一个 50 原子的 Li₁₀GeP₂S₁₂（LGPS）晶胞，CPU 上也能很快跑完：

```bash
./bin/mliport-mace sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda \
  --output runs/lgps-mace-sp
```

`runs/lgps-mace-sp/` 里会生成：

| 文件 | 内容 |
|---|---|
| `mliport_results.json` | 能量、力、应力、耗时、溯源信息 |
| `resolved_config.json` | 每个参数的取值和来源 |
| `OUTCAR` | VASP 兼容输出 |
| `CONTCAR` | 最终结构 |
| `run.log` | 完整日志 |

输出目录里已有结果时程序会报错（用 `--overwrite` 覆盖）。结果里记录的设备是后端实际用的设备。

## 选择后端

后端由 `--model-type`（或配置好的别名）指定，没有时直接报错 — 程序不会替你选模型。

```bash
./bin/mliport-mace  sp LGPS.vasp --model mace-omat-0-medium.model --model-type mace  --task bulk --device cuda
./bin/mliport-dpa   sp LGPS.vasp --model DPA-3.1-3M.pt --model-type dpa --task bulk --head Omat24 --device cuda
./bin/mliport-grace sp LGPS.vasp --model GRACE-2L-OMAT-medium-base --model-type grace --task bulk --device cuda
./bin/mliport-uma   sp LGPS.vasp --model uma-s-1p2.pt --model-type uma --task omat --device cuda
./bin/mliport-nep   sp LGPS.vasp --model nep.txt --model-type nep --nep-runtime ase --device cpu
```

| 引擎 | `--model-type` | Task / head | 说明 |
|---|---|---|---|
| MACE | `mace` | `bulk` | 默认 float64；`--dtype float32` 更快 |
| DPA | `dpa` | `bulk`、`--head Omat24` | 多任务检查点 — 必须指定 head |
| GRACE | `grace` | `bulk` | TensorFlow 运行时；CPU 和 GPU |
| UMA | `uma`（别名 `fairchem`） | `omat` | 分子用 `--task omol`；仅限非商业研究 |
| NEP | `nep` + `--nep-runtime ase\|gpumd` | `bulk`（仅 PBC） | 运行时不会自动推断 |

DPA 和 GRACE 的框架加载后不能再换 GPU，需要事先设好 `CUDA_VISIBLE_DEVICES`。
详见 [后端](docs/backends/index.md)。

## 主要工作流

**结构优化** — `--fmax` 控制收敛，`--cell-opt` 同时优化晶胞：

```bash
./bin/mliport-mace opt LGPS.vasp --model model.model --model-type mace \
  --fmax 0.05 --max-steps 200 --cell-opt --device cuda --output runs/lgps-opt
```

**分子动力学** — NVE、NVT（Langevin / Bussi / NHC）或 NPT（仅 ASE 后端）：

```bash
./bin/mliport-mace md LGPS.vasp --model model.model --model-type mace \
  --ensemble NVT --thermostat LANGEVIN --temp 800 --timestep 1.0 \
  --steps 10000 --save-interval 100 --device cuda --output runs/lgps-md
```

**NEB / CI-NEB** — 最小能量路径，含梯度预检：

```bash
./bin/mliport-mace neb --initial initial.vasp --final final.vasp \
  --images 5 --climb \
  --model model.model --model-type mace --device cuda --output runs/lgps-neb
```

**构型搜索** — slab 上的低能吸附构型：

```bash
./bin/mliport-mace confsearch run --slab slab.vasp --molecule conformers.xyz \
  --n-starts 20 --cycles 5 \
  --model model.model --model-type mace --task bulk --device cuda --output runs/ads
```

**批量、队列、TUI 和 Python：**

```bash
./bin/mliport-mace batch structures/ --pattern "*.vasp" --calc-type sp \
  --model model.model --model-type mace --device cuda --output runs/batch

./bin/mliport-mace queue submit tasks.json       # 之后：queue start、jobs、queue stop
./bin/mliport-mace tui                           # 交互式终端界面
```

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("LGPS.vasp", "model.model", model_type="mace", device="cuda")
result = run_single_point("LGPS.vasp", "model.model", model_type="mace", device="cuda")
```

详见 [工作流](docs/workflows/index.md)。

## 分析轨迹

`analyze` 的输入可以是 mliport 的运行目录，也可以是 ASE 轨迹：

```bash
./bin/mliport-mace analyze runs/lgps-md validate
./bin/mliport-mace analyze runs/lgps-md thermo
./bin/mliport-mace analyze runs/lgps-md msd --mobile Li
./bin/mliport-mace analyze runs/lgps-md rdf --center Li --neighbor S
./bin/mliport-mace analyze runs/lgps-md transport --mobile Li --charge 1 --fit-start-ps 100
./bin/mliport-mace analyze runs/lgps-md electrolyte --mobile Li --discover-sites-from-density
./bin/mliport-mace analyze runs/lgps-md vacf
./bin/mliport-mace analyze runs/lgps-md spectrum
./bin/mliport-mace analyze runs/lgps-md density --mobile Li
```

结果写到 `runs/lgps-md/analysis/<task>/<id>/`，有 JSON、CSV、PNG 和 SVG。轨迹太短时，
`msd` 和 `transport` 会报告采样不足，不给扩散系数；`arrhenius` 只用你提供的数据点。
详见 [分析](docs/analysis/index.md)。

## 配置

后端必须明确指定（`--model-type` 加 `--model`，或别名；`nep` 还需要 `--nep-runtime`）。
配置检查默认严格：不认识的参数和跨后端参数都是错误。`--lenient-config` 可降为警告。
`mliport config` 查看设置；`settings.ini` 可定义别名；`--settings PATH` 优先于用户和项目配置。
每次运行都写 `resolved_config.json`，记录每个参数的取值和来源。详见 [配置](docs/configuration.md)。

## 状态与限制

当前版本是 beta candidate（2.0.0b5）。我们在一台全新的 V100 机器上按上面的步骤安装，跑通了 LGPS 的各个
工作流和分析。下面这段由验证报告自动生成。

INCAR 风格输入只支持常用参数。显式原子映射和 image-shift 控制只能通过 API、直接 CLI 或 TUI 设置。API、命令行、INCAR 和 TUI 各自支持哪些参数，记录在
`validation/science/capability_matrix.json` 里；标为 `no` 或 `partial` 的项在该入口无法设置，程序也不会
自动换成别的设置。

NEP 的限制：`nep.txt` 要你自己提供，mliport 无法判断它是否适合你的材料。原生 `gpumd` 运行时没有 CPU 路径，
遇到 NEB、批量和 confsearch 会直接拒绝，不会改用 ASE 运行时；DFT-D3 会被拒绝；不支持 `qNEP`、`TNEP` 和
`nep4-vdw` 模型；部分周期性边界会被拒绝。NEP 不在下面这份验证汇总的范围内：它在真实硬件上的证据是一台 V100
配 GPUMD 5.9.1 和 CUDA 12.9，其他 GPU 和 toolkit 没有验证过。详见 [docs/backends/nep.zh.md](docs/backends/nep.zh.md)。

模型精度由上游模型决定。短时间 MD 只用来检查流程能否跑通，不代表体系已经平衡。NEB 预检通过只说明力和
能量自洽，不代表势垒准确。采样不足时，输运分析会报告 insufficient-sampling，不给出数值。

构型搜索（`confsearch`）是新功能。CI 里用合成模型做了测试，但还没有用真实模型在 GPU 上验证，去重容差也
只在合成结构上标定过。目前请把它的排序结果当作初筛，交给 DFT 复核。

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

Scientific campaign target: commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).
LGPS fresh-install acceptance target: commit `51d2a78` ([acceptance report](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)).
Current release: 2.0.0b5 (package metadata; the release tag records the exact commit).

Validated hardware: NVIDIA V100-SXM2-16GB (Volta, 16 GiB), driver 580.173.02, Rocky Linux 9.8. CPU installs are smoke-tested on the same host.

Current status: 2.0.0b5 beta. The four backends are installed and exercised on an LGPS structure through single point, relaxation, MD, NEB, batch, INCAR-style runs, queue, TUI, the Python API and the analysis modules.

- Full beta validation report: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md)
- LGPS fresh-install acceptance: [LGPS_ACCEPTANCE_REPORT.md](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- Model identities: [model_manifest.json](validation/science/model_manifest.json)

Per-workload tables, tier scopes and limitations live in the reports; the README keeps only this status summary.

<!-- END GENERATED -->

**已知限制：**

- INCAR 风格输入只支持常用参数。
- NEP：`nep.txt` 要你自己提供，mliport 无法判断它适不适合你的材料。原生 `gpumd` 运行时
  没有 CPU 路径，NEB、批量和 confsearch 会被拒绝。不在验证范围内。见 [NEP](docs/backends/nep.md)。
- 模型精度由上游模型决定。
- 构型搜索（`confsearch`）是新功能：CI 里用合成模型测试过，还没有在 GPU 上用真实模型验证。
  目前请把排序结果当作初筛，交给 DFT 复核。

## 引用

见 [引用](docs/citations.md) 和 [`CITATION.cff`](CITATION.cff)。使用了哪个模型，请同时引用
该模型的论文并遵守其许可证。

## 许可证

mliport 的许可证见 [`LICENSE.md`](LICENSE.md)。模型文件和上游框架沿用各自的许可证。
