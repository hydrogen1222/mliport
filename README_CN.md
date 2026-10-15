# mliport

用同一套命令行、同一个终端界面和同一个 Python API 来运行机器学习势：MACE、DPA、GRACE 和 UMA。你给它
一个结构和一个检查点，它就能做弛豫、跑分子动力学、找 NEB 势垒或分析轨迹，并写出 VASP 用户熟悉的文
件：`OUTCAR`、`OSZICAR`、`CONTCAR`、`XDATCAR`。

**文档：<https://hydrogen1222.github.io/mliport/>**（默认英文，用顶栏的语言切换按钮可切到中文）。同样
的页面也在 [`docs/`](docs/) 里。

mliport 是工作流层，不是模型，也不是 DFT 程序。文件是 VASP 风格的，但没有 SCF，也没有 POTCAR；一个势
对你的化学体系对不对，取决于你选的检查点。

它能给你：

- **四个后端，同一套工作流。** 单点、固定与可变晶胞弛豫、NVE/NVT 分子动力学（Langevin、Bussi、
  Nosé–Hoover 链）、NEB 与 CI-NEB、批量扫描和作业队列；入口可以是 CLI、INCAR 风格文件、TUI 或
  Python。
- **不会悄悄猜测。** 没有默认后端；拼错的选项，或者属于另一个后端的选项，都会让运行停下。配置错误
  默认直接失败。
- **一份可追溯的记录。** 每次运行都记录模型文件及其 SHA-256、请求的与实际的设备、解析后的配置和
  mliport 版本。
- **数据太短时会明说的分析。** 轨迹校验、热力学量、RDF、RMSD/RMSF、MSD、VACF、振动谱、可移动离子
  密度、kinisi 输运、GEMDAT 电解质分析和 Arrhenius 拟合。

## 安装

支持 Linux 与 WSL2。需要 `git` 和 [uv](https://docs.astral.sh/uv/)；安装脚本会为每个后端建一个 Python
环境，并在 `bin/` 下为每个后端生成启动器。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # 还没有 uv 的话
git clone https://github.com/hydrogen1222/mliport
cd mliport

# NVIDIA GPU（需要可用的 CUDA 驱动）
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto

# 或者只用 CPU
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

安装脚本会固定上游后端的版本、安装分析所需的额外依赖，并对每个引擎运行 `mliport doctor`。四个引擎一
起装时默认用 Python 3.12；包本身支持 Python 3.10-3.13，如果你传了 `--python`，所请求的每个后端都必须
接受它（UMA 需要 3.11 或更高）。这套安装在驱动为 580.173.02 的 V100 上验证过；更老和更新的 GPU 家族走
别的 wheel 路线，列在 [docs/installation.md](docs/installation.md) 里，那里还有 CPU、离线、代理的配置
和卸载方法。

| 引擎 | 环境 | 启动器 |
|---|---|---|
| MACE | `.venv-mace` | `bin/mliport-mace` |
| DPA | `.venv-dpa` | `bin/mliport-dpa` |
| GRACE | `.venv-grace` | `bin/mliport-grace` |
| UMA | `.venv` | `bin/mliport-uma` |

请在你要用的那个后端的环境里运行 mliport，或者通过它的启动器：

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
./bin/mliport-mace --help
```

结果良好时，输出以 `Environment checks passed for MACE on CUDA.` 结尾。在 Windows 上，每个环境里有
`Scripts\mliport.exe`，安装脚本还会写出 `bin\mliport-*.cmd` 启动器。`mliport doctor --json` 会以 JSON
给出同样的检查，方便脚本使用。

### 模型

mliport 不下载检查点。请用上游工具取得检查点，再用 `--model` 传入路径；本地目录可以离线使用。

| 引擎 | 检查点 | 来源 |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

验证套件用到的哈希固定在 [`validation/science/model_manifest.json`](validation/science/model_manifest.json)。
许可证归上游项目所有；UMA 的检查点采用非商业研究许可。Hugging Face 镜像和 SOCKS 代理见
[docs/models.md](docs/models.md)。

## 第一次计算

仓库里带了一个 50 原子的 Li10GeP2S12（LGPS）晶胞，小到可以在 CPU 上做冒烟测试：

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda \
  --output runs/lgps-mace-sp
```

在 `runs/lgps-mace-sp/` 里你会得到 `mliport_results.json`（能量、力、应力、耗时）、
`resolved_config.json`（实际用到的配置）、`OUTCAR`、`CONTCAR`、`run.log`，弛豫或 MD 运行还会有
`OSZICAR`。如果输出目录里已经有结果，运行会拒绝它，并在运行期间锁住该目录；`--overwrite` 表示你有意
覆盖。结果里的设备是后端报告的设备，不是你请求的设备。

## 选择后端

后端来自 `--model-type` 或配置好的别名。两者都没有时运行会失败；mliport 绝不会自己换成另一个模型。

```bash
.venv-mace/bin/mliport  sp LGPS.vasp --model mace-omat-0-medium.model --model-type mace  --task bulk --device cuda
.venv-dpa/bin/mliport   sp LGPS.vasp --model DPA-3.1-3M.pt --model-type dpa --task bulk --head Omat24 --device cuda
.venv-grace/bin/mliport sp LGPS.vasp --model GRACE-2L-OMAT-medium-base --model-type grace --task bulk --device cuda
.venv/bin/mliport       sp LGPS.vasp --model uma-s-1p2.pt --model-type uma --task omat --device cuda
```

| 引擎 | `--model-type` | Task / head | 需要知道的 |
|---|---|---|---|
| MACE | `mace` | `bulk` | 默认 float64；`--dtype float32` 更快但精度更低 |
| DPA | `dpa` | `bulk`、`--head Omat24` | 多任务检查点；缺少 head 是错误，不是猜测 |
| GRACE | `grace` | `bulk` | TensorFlow 运行时，支持 CPU 和 GPU |
| UMA | `uma`（`fairchem` 是别名） | `omat` | 分子用 `--task omol`；非商业研究许可 |

DPA 和 GRACE 在框架加载之后无法切换 GPU，所以当 `CUDA_VISIBLE_DEVICES` 未设置时，CLI 会在设好正确
GPU 的新进程里重启命令。从 Python 调用时，请在导入它们之前设置 `CUDA_VISIBLE_DEVICES`。更多见
[docs/backends/index.md](docs/backends/index.md)。

## 主要工作流

**结构弛豫。** `--fmax` 和 `--max-steps` 控制收敛；`--cell-opt` 让晶胞与离子一起弛豫。

```bash
.venv-mace/bin/mliport opt LGPS.vasp --model model.model --model-type mace \
  --fmax 0.05 --max-steps 200 --cell-opt --device cuda --output runs/lgps-opt
```

**分子动力学。** `--ensemble` 取 `NVE` 或 `NVT`，`--thermostat` 取 `LANGEVIN`、`BUSSI` 或 `NHC`。运行可
以从带速度的已保存帧继续，用 `--velocity-policy preserve --no-pre-relax`。使用 NHC 时，默认的
`--com-policy auto` 会把质心动量一次性清零，这样整个晶胞不会漂移，也不会让基于 MSD 的输运产生偏差。

```bash
.venv-mace/bin/mliport md LGPS.vasp --model model.model --model-type mace \
  --ensemble NVT --thermostat LANGEVIN --temp 800 --timestep 1.0 \
  --steps 10000 --save-interval 100 --device cuda --output runs/lgps-md
```

**NEB 与 CI-NEB。** 两个端点文件确定能带，`--images` 设定中间镜像数，`--climb` 打开爬升镜像。第一步之
前，mliport 会在你自己的能带上检查模型的力是否为其能量的梯度（考虑噪声的有限差分），不是就拒绝继续；
像 UMA 这样直接预测力的模型一律被拒绝。测量结果保存为 `gradient_preflight`。它只是一致性的抽查，不是
对精度的陈述。

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --images 5 --climb \
  --model model.model --model-type mace --device cuda --output runs/lgps-neb
```

用 `--resume runs/lgps-neb --output runs/lgps-neb` 续跑。步数预算是运行身份的一部分，所以续跑时改
`--max-steps` 会被拒绝。显式的原子映射与镜像平移这类高级控制，仅在 API、直接 CLI 和 TUI 中可用
。

**批量、INCAR 风格运行、队列、TUI、Python。**

```bash
.venv-mace/bin/mliport batch structures/ --pattern "*.vasp" --calc-type sp \
  --model model.model --model-type mace --device cuda --output runs/batch

.venv-mace/bin/mliport template sp --output INCAR.mliport     # 编辑 MODEL_PATH、MODEL_TYPE 等
.venv-mace/bin/mliport run -i INCAR.mliport -s LGPS.vasp -o runs/incar-sp

.venv-mace/bin/mliport queue submit tasks.json                # 之后：queue start、jobs、queue stop
```

`mliport tui` 打开交互式应用（需要终端）。从 Python 调用：

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("LGPS.vasp", "model.model", model_type="mace", device="cuda")
result = run_single_point("LGPS.vasp", "model.model", model_type="mace", device="cuda")
```

## 分析轨迹

`analyze` 接受 mliport 运行目录或 ASE 轨迹：

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

结果写到 `runs/lgps-md/analysis/<task>/<id>/`，格式为 JSON、CSV，合适时还有 PNG 和 SVG；绘图无需显
示器。短轨迹会被接受，但不会被过度解读：`msd` 和 `transport` 返回明确的采样不足状态，而不是扩散系
数；`arrhenius` 只拟合你给它的点。各任务的选项见 [docs/analysis/index.md](docs/analysis/index.md)。

## 配置

后端始终是显式的（`--model-type mace|dpa|grace|uma` 加 `--model`，或使用别名）。配置是严格的：未知
的键和跨后端的键都是错误，探索性运行可以用 `--lenient-config` 把它们降级。`mliport config` 用来查看设
置；项目里的 `settings.ini` 可以定义别名，`--settings PATH` 的优先级高于用户和项目文件。每次运行都会
写出 `resolved_config.json`，记录每个值的来源。

## 状态与限制

这是一个 beta candidate（2.0.0b4）。它已在干净的 V100 机器上按这些说明安装，并跑通了 LGPS 工作流和
分析矩阵；下面的区块由验证报告生成。

INCAR 风格输入覆盖常用控制项，不是全部接口。每个入口（API、直接 CLI、INCAR、TUI）能表达哪些选项，
记录在 `validation/science/capability_matrix.json` 里；那里标为 `no` 或 `partial` 就表示该入口无法表
达，也不会被悄悄转换。

模型精度属于上游检查点。短 MD 检查的是工作流，不是平衡。NEB 的预检说明力与能量一致，这并不意味着势
垒本身准确。输运分析在采样不足时以明确的 insufficient-sampling 状态失败，而不是返回一个数。

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

见 [docs/citations.md](docs/citations.md) 和 [`CITATION.cff`](CITATION.cff)。如果你使用了某个后端模型，请
引用上游模型并遵守其许可证。

## 许可证

mliport 包采用 [`LICENSE.md`](LICENSE.md) 中的许可证。模型检查点和上游框架保留各自的许可证。
