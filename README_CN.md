# mliport

[English](README.md) | **简体中文**

mliport 用一套命令行、终端界面（TUI）和 Python API 运行四种机器学习势：MACE、DPA、GRACE 和 UMA。
准备好结构和模型文件，就可以做结构优化、分子动力学、NEB 势垒、slab 上吸附分子的构型搜索和轨迹分析。
输出沿用 VASP 的文件格式：`OUTCAR`、`OSZICAR`、`CONTCAR`、`XDATCAR`。

mliport 只负责组织计算流程，本身不含模型，也不做 DFT。输出文件只是格式和 VASP 一样，里面没有 SCF 和
POTCAR。模型对你的体系准不准，取决于你选的模型文件。

主要功能：

- **四个后端，用法一致。** 单点、结构优化（固定或可变晶胞）、NVE/NVT 分子动力学（Langevin、Bussi、
  Nosé–Hoover 链）、NEB 和 CI-NEB、slab 吸附构型搜索、批量计算和任务队列。可以用命令行、INCAR 风格
  的输入文件、TUI 或 Python 调用。
- **不做默认猜测。** 没有默认后端。选项拼错，或者用了别的后端的选项，程序会报错停下。
- **完整的计算记录。** 每次运行都记录模型文件和它的 SHA-256、请求的设备和实际使用的设备、最终生效
  的配置以及 mliport 版本。
- **轨迹分析。** 轨迹检查、热力学量、RDF、RMSD/RMSF、MSD、VACF、振动谱、可移动离子密度、kinisi
  输运、GEMDAT 电解质分析和 Arrhenius 拟合。轨迹太短时会报告采样不足。


## 文档

完整文档见 <https://hydrogen1222.github.io/mliport/>（中英文），同样的页面也在 [`docs/`](docs/)
目录里，可以直接在 GitHub 上看。

| 我想…… | 去看 |
|---|---|
| 跑第一个计算 | [快速上手](docs/quickstart.zh.md) |
| 在答案已知的体系上检验一个模型 | [测试体系](docs/test-systems.zh.md) |
| 安装一个或多个后端（GPU 或 CPU） | [安装](docs/installation.zh.md) |
| 了解 MACE、DPA、GRACE、UMA 是什么并选一个 | [后端](docs/backends/index.zh.md)、[模型](docs/models.zh.md) |
| 通过 CLI、INCAR 风格文件或 `settings.ini` 设置选项 | [配置](docs/configuration.zh.md) |
| 单点、弛豫、MD、NEB、构型搜索、批量、队列 | [工作流](docs/workflows/index.zh.md) |
| MSD、扩散、电导率、RDF、谱、Arrhenius | [分析](docs/analysis/index.zh.md) |
| 了解每个输出文件的内容 | [输出文件](docs/outputs.zh.md) |
| 解决报错 | [排错](docs/troubleshooting.zh.md) |
| 引用 mliport 和模型 | [引用](docs/citations.zh.md) |

## 安装

支持 Linux 和 WSL2，需要先装好 `git` 和 [uv](https://docs.astral.sh/uv/)。安装脚本会给每个后端单独建一个
Python 环境，并在 `bin/` 下生成对应的启动脚本。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # 还没有 uv 的话
git clone https://github.com/hydrogen1222/mliport
cd mliport

# NVIDIA GPU（需要可用的 CUDA 驱动）
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto

# 或者只用 CPU
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

安装脚本会锁定各后端的上游版本，装好分析用的依赖，最后对每个后端运行一次 `mliport doctor`。四个后端
一起装时默认用 Python 3.12。mliport 本身支持 Python 3.10-3.13；用 `--python` 指定版本时，这个版本要被
所有选中的后端支持（UMA 要 3.11 及以上）。这套流程在驱动版本 580.173.02 的 V100 上验证过。其他架构的
GPU 要用不同的安装包，见 [docs/installation.md](docs/installation.md)，CPU、离线、代理和卸载也在那里。

| 引擎 | 环境 | 启动器 |
|---|---|---|
| MACE | `.venv-mace` | `bin/mliport-mace` |
| DPA | `.venv-dpa` | `bin/mliport-dpa` |
| GRACE | `.venv-grace` | `bin/mliport-grace` |
| UMA | `.venv` | `bin/mliport-uma` |

用哪个后端，就在哪个后端的环境里运行 mliport，或者用它的启动脚本：

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
./bin/mliport-mace --help
```

检查通过时，最后一行是 `Environment checks passed for MACE on CUDA.`。Windows 下每个环境里是
`Scripts\mliport.exe`，启动脚本是 `bin\mliport-*.cmd`。写脚本时可以用 `mliport doctor --json` 拿到
JSON 格式的检查结果。

### 模型

mliport 不会自动下载模型。请从上游下载，再用 `--model` 传入路径；模型在本地时可以离线使用。

| 引擎 | 检查点 | 来源 |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

验证时用的模型哈希记录在 [`validation/science/model_manifest.json`](validation/science/model_manifest.json)。
模型的许可证以上游为准，UMA 只能用于非商业研究。Hugging Face 镜像和 SOCKS 代理的设置见
[docs/models.md](docs/models.md)。

## 第一次计算

仓库里带了一个 50 个原子的 Li10GeP2S12（LGPS）晶胞，CPU 上也能很快跑完：

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda \
  --output runs/lgps-mace-sp
```

`runs/lgps-mace-sp/` 里会生成 `mliport_results.json`（能量、力、应力、耗时）、
`resolved_config.json`（实际生效的配置）、`OUTCAR`、`CONTCAR` 和 `run.log`，结构优化和 MD 还有
`OSZICAR`。输出目录里已有结果时程序会报错，运行期间目录会被锁住；确定要覆盖时加 `--overwrite`。
结果里记录的设备是后端实际用的设备。

## 选择后端

后端由 `--model-type` 或配置好的模型别名指定，两者都没有时直接报错，程序不会替你选模型。

```bash
.venv-mace/bin/mliport  sp LGPS.vasp --model mace-omat-0-medium.model --model-type mace  --task bulk --device cuda
.venv-dpa/bin/mliport   sp LGPS.vasp --model DPA-3.1-3M.pt --model-type dpa --task bulk --head Omat24 --device cuda
.venv-grace/bin/mliport sp LGPS.vasp --model GRACE-2L-OMAT-medium-base --model-type grace --task bulk --device cuda
.venv/bin/mliport       sp LGPS.vasp --model uma-s-1p2.pt --model-type uma --task omat --device cuda
```

| 引擎 | `--model-type` | Task / head | 说明 |
|---|---|---|---|
| MACE | `mace` | `bulk` | 默认 float64；`--dtype float32` 更快但精度更低 |
| DPA | `dpa` | `bulk`、`--head Omat24` | 多任务模型，必须指定 head |
| GRACE | `grace` | `bulk` | 基于 TensorFlow，支持 CPU 和 GPU |
| UMA | `uma`（也可写 `fairchem`） | `omat` | 分子用 `--task omol`；仅限非商业研究 |

DPA 和 GRACE 的框架加载后就不能再换 GPU。没有设置 `CUDA_VISIBLE_DEVICES` 时，命令行会新开一个设好 GPU
的进程来运行；在 Python 里调用时，请在导入这些框架之前设置 `CUDA_VISIBLE_DEVICES`。详见
[docs/backends/index.md](docs/backends/index.md)。

## 主要工作流

**结构优化。** `--fmax` 和 `--max-steps` 控制收敛，`--cell-opt` 同时优化晶胞。

```bash
.venv-mace/bin/mliport opt LGPS.vasp --model model.model --model-type mace \
  --fmax 0.05 --max-steps 200 --cell-opt --device cuda --output runs/lgps-opt
```

**分子动力学。** `--ensemble` 可选 `NVE` 或 `NVT`，`--thermostat` 可选 `LANGEVIN`、`BUSSI` 或 `NHC`。
要从保存了速度的某一帧接着跑，用 `--velocity-policy preserve --no-pre-relax`。用 NHC 时，默认的
`--com-policy auto` 会在开始时把质心动量清零，避免整体漂移影响 MSD 和输运分析。

```bash
.venv-mace/bin/mliport md LGPS.vasp --model model.model --model-type mace \
  --ensemble NVT --thermostat LANGEVIN --temp 800 --timestep 1.0 \
  --steps 10000 --save-interval 100 --device cuda --output runs/lgps-md
```

**NEB 和 CI-NEB。** 两个端点文件定义路径，`--images` 设置中间 image 数，`--climb` 开启爬坡 image。
正式计算前，mliport 会在你的路径上用有限差分（考虑了数值噪声）检查力是否等于能量的负梯度，不一致就
停止；模型声明力是直接预测的，也会被拒绝。检查结果保存为 `gradient_preflight`。这一步只检查力和能量
是否自洽，不代表势垒准确。

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --images 5 --climb \
  --model model.model --model-type mace --device cuda --output runs/lgps-neb
```

续算用 `--resume runs/lgps-neb --output runs/lgps-neb`。步数上限属于这次运行的固定参数，续算时修改
`--max-steps` 会报错。指定原子对应关系、image 平移这类高级设置，只能通过 API、命令行和 TUI 使用。

**构型搜索。** `confsearch` 用来找周期性 slab 一侧单个吸附分子的低能构型。起点可以是你准备的结构，也可以
用 `--slab`/`--molecule` 随机摆放。每个起点在固定晶胞下跑几轮"升温、保温、降温"的 Langevin MD，途中取
出的结构都做局部优化、分类（骨架断键的分子单独存放）和去重。最后按能量给出这次搜索找到的最低几个结构，
附上判断搜索是否充分的统计，并导出 POSCAR 和 CP2K 文件供 DFT 复核。有限次搜索无法保证找到全局最小。
`--dry-run` 只检查输入和估算计算量，不加载模型。

```bash
.venv-mace/bin/mliport confsearch run --slab slab.vasp --molecule conformers.xyz \
  --n-starts 20 --cycles 5 \
  --model model.model --model-type mace --task bulk --device cuda --output runs/ads
```

中断的搜索用 `confsearch resume` 续算，跑完的可以用 `confsearch extend --cycles N` 追加轮数，设置相同的几次
搜索可以用 `confsearch merge` 合并。详细参数和注意事项见
[docs/workflows/confsearch.md](docs/workflows/confsearch.md)。

**批量计算、INCAR 风格输入、任务队列、TUI 和 Python。**

```bash
.venv-mace/bin/mliport batch structures/ --pattern "*.vasp" --calc-type sp \
  --model model.model --model-type mace --device cuda --output runs/batch

.venv-mace/bin/mliport template sp --output INCAR.mliport     # 编辑 MODEL_PATH、MODEL_TYPE 等
.venv-mace/bin/mliport run -i INCAR.mliport -s LGPS.vasp -o runs/incar-sp

.venv-mace/bin/mliport queue submit tasks.json                # 之后：queue start、jobs、queue stop
```

`mliport tui` 打开交互界面（需要在终端里运行）。在 Python 里调用：

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("LGPS.vasp", "model.model", model_type="mace", device="cuda")
result = run_single_point("LGPS.vasp", "model.model", model_type="mace", device="cuda")
```

## 分析轨迹

`analyze` 的输入可以是 mliport 的运行目录，也可以是 ASE 轨迹：

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

结果写到 `runs/lgps-md/analysis/<task>/<id>/`，有 JSON、CSV，以及 PNG 和 SVG 图，不需要显示器。轨迹太短
时，`msd` 和 `transport` 会报告采样不足，不给扩散系数；`arrhenius` 只用你提供的数据点。各项分析的参数见
[docs/analysis/index.md](docs/analysis/index.md)。

## 配置

后端必须明确指定（`--model-type mace|dpa|grace|uma` 加 `--model`，或者用模型别名）。配置检查是严格的：
不认识的参数和别的后端的参数都会报错，试算时可以用 `--lenient-config` 降为警告。`mliport config` 用来查看
设置；项目里的 `settings.ini` 可以定义别名，`--settings PATH` 指定的文件优先于用户和项目配置。每次运行都
会写出 `resolved_config.json`，记录每个参数的取值和来源。

## 状态与限制

当前版本是 beta candidate（2.0.0b4）。我们在一台全新的 V100 机器上按上面的步骤安装，跑通了 LGPS 的各个
工作流和分析。下面这段由验证报告自动生成。

INCAR 风格输入只支持常用参数。API、命令行、INCAR 和 TUI 各自支持哪些参数，记录在
`validation/science/capability_matrix.json` 里；标为 `no` 或 `partial` 的项在该入口无法设置，程序也不会
自动换成别的设置。

模型精度由上游模型决定。短时间 MD 只用来检查流程能否跑通，不代表体系已经平衡。NEB 预检通过只说明力和
能量自洽，不代表势垒准确。采样不足时，输运分析会报告 insufficient-sampling，不给出数值。

构型搜索（`confsearch`）是新功能。CI 里用合成模型做了测试，但还没有用真实模型在 GPU 上验证，去重容差也
只在合成结构上标定过。目前请把它的排序结果当作初筛，交给 DFT 复核。

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

Scientific campaign target: commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).
LGPS fresh-install acceptance target: commit `51d2a78` ([acceptance report](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)).
Current release: 2.0.0b4 (package metadata; the release tag records the exact commit).

Validated hardware: NVIDIA V100-SXM2-16GB (Volta, 16 GiB), driver 580.173.02, Rocky Linux 9.8. CPU installs are smoke-tested on the same host.

Current status: 2.0.0b4 beta. The four backends are installed and exercised on an LGPS structure through single point, relaxation, MD, NEB, batch, INCAR-style runs, queue, TUI, the Python API and the analysis modules.

- Full beta validation report: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md)
- LGPS fresh-install acceptance: [LGPS_ACCEPTANCE_REPORT.md](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- Model identities: [model_manifest.json](validation/science/model_manifest.json)

Per-workload tables, tier scopes and limitations live in the reports; the README keeps only this status summary.
<!-- END GENERATED -->

## 引用

见 [docs/citations.md](docs/citations.md) 和 [`CITATION.cff`](CITATION.cff)。使用了哪个模型，请同时引用该
模型的论文，并遵守它的许可证。

## 许可证

mliport 的许可证见 [`LICENSE.md`](LICENSE.md)。模型文件和上游框架沿用各自的许可证。
