# mliport

mliport 是一个**工作流层**，用来运行机器学习原子间势（MLIP）。它支持五种后端 — MACE、
DPA、GRACE、UMA 和 NEP — 统一用同一套命令行、INCAR 风格输入文件、终端界面（TUI）和
Python API 调用。准备好结构文件和模型检查点，就可以做单点计算、结构优化、分子动力学
（NVE、NVT、NPT）、NEB 势垒搜索、slab 吸附构型搜索、批量扫描和轨迹分析。所有结果
以 VASP 兼容格式输出（`OUTCAR`、`OSZICAR`、`CONTCAR`、`XDATCAR`），并附上一份 JSON
记录，完整记载计算过程中的每个设置。

mliport **不是**模型、不是训练器、也不是 DFT 代码。输出文件格式和 VASP 相同，但里面
没有 SCF 循环、没有 k 点网格、没有 POTCAR。模型对你的体系准不准，完全取决于你选择的
检查点和你自己做的验证。

## 设计理念

**一套流程，五个后端。**
同一组命令 `mliport sp`、`mliport opt`、`mliport md`、`mliport neb` 对所有后端通用。
切换模型只需改 `--model-type` 和 `--model`，其余不变。这让你很容易在同一个结构上比较
不同后端的结果，或者用新模型重跑已有的计算设置。

**不做默认猜测。**
没有默认后端。拼错参数、用了别的后端的参数、或忘记指定 DPA 多任务检查点的 head，程序
都会报错停下。配置检查默认严格 — mliport 不会默默忽略不认识的参数，也不会自动切换模型。

**可复现性内建。**
每次运行都写一份 `resolved_config.json`，记录模型文件路径及其 SHA-256 哈希、请求的设备
和实际使用的设备、每个参数的取值及来源（默认值、settings.ini、INCAR 文件还是命令行），
以及 mliport 版本号。你永远可以还原一个结果是怎么算出来的。

**分析知道自己的极限。**
分析管线 — MSD、示踪扩散系数、离子电导率、RDF、VACF、振动谱、GEMDAT 电解质分析、
Arrhenius 拟合 — 在轨迹太短无法得到可靠结果时，会报告明确的"采样不足"状态，而不是
给你一个不应该相信的数值。

## 功能概览

| 功能 | 说明 |
|---|---|
| 统一工作流 | 单点、结构优化（固定/可变晶胞）、NVE/NVT/NPT 分子动力学、NEB/CI-NEB、构型搜索、批量、任务队列 |
| 五个后端 | MACE、DPA (DeepPotential)、GRACE、UMA (Meta)、NEP (GPUMD) |
| VASP 兼容输出 | `OUTCAR`、`OSZICAR`、`CONTCAR`、`XDATCAR` — 任何能读 VASP 文件的工具都能用 |
| INCAR 风格输入 | 写一个 `INCAR.mliport` 文件，用熟悉的 VASP 式语法；`mliport run` 一键运行 |
| 严格配置 | 不认识的键、跨后端的键、类型不匹配都是错误，不是警告 |
| NPT 分子动力学 | ASE 后端（MACE、DPA、GRACE、UMA）支持各向同性 MTK NPT 系综 |
| DFT-D3 色散校正 | 所有后端均可选用 DFT-D3 校正 |
| 轨迹分析 | MSD、扩散系数、电导率、RDF、VACF、振动谱、密度、GEMDAT、Arrhenius |
| Python API | 六个函数（`run_single_point`、`run_optimization`、`run_md`、`run_neb`、`run_confsearch`、`calculate_energy`）用于脚本集成 |
| 模型溯源 | 每次运行记录 SHA-256、设备、配置来源和版本号 |

## 快速体验

仓库自带一个 50 原子的 Li₁₀GeP₂S₁₂（LGPS）结构用于测试。装好 MACE 后：

```bash
./bin/mliport-mace sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk
```

[快速上手](quickstart.md)会带你逐步走完整个流程，包括安装和获取模型。

## 下一步

| 我想…… | 去看 |
|---|---|
| 安装并跑第一个计算 | [快速上手](quickstart.md) |
| 了解环境、Python 版本和 GPU 设置 | [安装](installation.md) |
| 在答案已知的体系上检验模型 | [测试体系](test-systems.md) |
| 选择模型，了解哪些经过了验证 | [模型](models.md)、[后端](backends/index.md) |
| 通过 CLI、INCAR 文件或 `settings.ini` 配置参数 | [配置](configuration.md) |
| 跑结构优化、MD、NEB、构型搜索或批量计算 | [工作流](workflows/index.md) |
| 分析轨迹（MSD、RDF、电导率……） | [分析](analysis/index.md) |
| 在 Python 脚本中调用 mliport | [Python API](api.md) |
| 了解每个输出文件的内容 | [输出文件](outputs.md) |
| 看看到底验证了什么 | [验证](validation.md) |
| 解决报错 | [排错](troubleshooting.md) |
| 从旧名 `mlipx` 迁移过来 | [迁移](migration-from-mlipx.md) |
| 引用 mliport 和它运行的模型 | [引用](citations.md) |
| 参与 mliport 开发 | [开发](development.md) |

## 平台与要求

mliport 在 **Linux** 和 **WSL2** 上运行，支持 Python **3.10–3.13**。GPU 加速需要
NVIDIA 显卡和可用的 CUDA 驱动。安装器使用 [uv](https://docs.astral.sh/uv/) 管理
Python 环境。不支持 macOS。

源码：[github.com/hydrogen1222/mliport](https://github.com/hydrogen1222/mliport)
