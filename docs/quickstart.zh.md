# 快速上手

本指南带你从克隆仓库到完成单点计算、结构优化和短 MD 模拟，大约十分钟。我们以
**MACE** 作为第一个后端，因为它的检查点文件开源且体积小；其他四个后端的用法在
末尾。

## 前提条件

你需要一台 Linux 机器（或 Windows 上的 WSL2），并具备：

- **git** — 用于克隆仓库。
- **[uv](https://docs.astral.sh/uv/)** — 安装器使用的 Python 环境管理工具。
  如尚未安装，运行
  `curl -LsSf https://astral.sh/uv/install.sh | sh`。
- **NVIDIA GPU + CUDA 驱动**（可选）— 用于 GPU 加速计算。纯 CPU 模式对小体系
  同样可用。

## 1. 克隆与安装

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto
```

安装器会创建专用 Python 环境（`.venv-mace/`），将 mliport 和 MACE 后端安装
到其中，在 `bin/mliport-mace` 生成启动脚本，并自动运行 `mliport doctor` 做
健全性检查。整个过程需要几分钟。

若机器没有 NVIDIA GPU，请使用 `--device cpu`。要一次安装多个后端，传入
`--engines mace dpa grace uma`。完整说明——离线安装、代理配置、磁盘空间、
手动安装——请参阅[安装](installation.md)。

## 2. 验证安装

```bash
./bin/mliport-mace doctor --engine mace --device auto
```

安装正常时会输出：

```
Environment checks passed for MACE on CUDA.
```

（如果安装时使用了 `--device cpu`，则显示 `on CPU`）。若检查失败，请参阅
[故障排查](troubleshooting.md)。

!!! tip "脚本友好的 JSON 输出"
    `mliport doctor --json` 输出机器可读的结果，适用于 CI 流水线和自动化部署
    脚本。

## 3. 获取模型检查点

mliport 不会自动下载或内置模型权重——你需要从上游项目获取。以 MACE-OMAT-0
（基于 OMat24 数据训练的通用基础模型）为例：

```bash
# 方式 A：从 GitHub Release 页面下载
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model

# 方式 B：从 Hugging Face 下载（需要 huggingface-cli）
huggingface-cli download ACEsuit/MACE-OMAT-0 mace-omat-0-medium.model
```

将文件放在方便的位置。以下示例中，请将
`/path/to/mace-omat-0-medium.model` 替换为实际路径。

!!! note "选择哪个检查点"
    [模型](models.md)页面列出了所有支持的检查点、下载链接、哈希值和许可条款。
    MACE-OMAT-0-medium 是周期性体相材料的良好起点。

## 4. 运行单点计算

仓库自带一个 50 原子的 Li₁₀GeP₂S₁₂（LGPS）测试结构。计算其能量和力：

```bash
./bin/mliport-mace sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk \
  --output runs/lgps-sp
```

mliport 会打印摘要并将结果写入 `runs/lgps-sp/`：

| 文件 | 内容 |
|---|---|
| `mliport_results.json` | 能量、力、应力、计时、溯源信息 |
| `resolved_config.json` | 每个配置值及其来源 |
| `OUTCAR` | VASP 兼容输出（能量、力、应力） |
| `CONTCAR` | POSCAR 格式的最终结构 |
| `run.log` | 完整运行日志 |

使用 MACE-OMAT-0 检查点计算 LGPS 晶胞，能量约为 −216 eV，化学式为
`Ge2Li20P4S24`（50 个原子）。

!!! warning "输出目录保护"
    如果 `runs/lgps-sp/` 中已有结果，mliport 会拒绝覆盖。使用 `--overwrite`
    可以强制替换，或选择不同的 `--output` 路径。

## 5. 结构优化

```bash
./bin/mliport-mace opt examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk \
  --fmax 0.05 --max-steps 200 \
  --output runs/lgps-opt
```

`--fmax` 是力收敛阈值（单位 eV/Å）；`--max-steps` 限制最大优化步数。优化是否
真正收敛记录在 `mliport_results.json` 中——未收敛的运行会如实报告。弛豫后的
结构保存为 `CONTCAR`。

如需同时优化晶胞形状和体积，添加 `--cell-opt`。

## 6. 运行短 MD 模拟

```bash
./bin/mliport-mace md examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk \
  --ensemble NVT --thermostat LANGEVIN --temp 600 \
  --timestep 1.0 --steps 1000 --save-interval 10 \
  --output runs/lgps-md
```

这将运行 1000 步 NVT Langevin 动力学（600 K），每 10 步保存一帧。轨迹写入
`XDATCAR`（可用 VESTA、OVITO 等打开），热力学量记录在 `OSZICAR` 中。

!!! tip "选择系综"
    mliport 支持三种 MD 系综：`NVE`（微正则）、`NVT`（正则，可选 Langevin/
    Bussi/NHC 恒温器）和 `NPT`（等温等压，仅 ASE 后端）。完整选项见
    [MD 工作流](workflows/md.md)。

## 7. 从 Python 读取结果

所有结果存储在 `mliport_results.json` 中。读取方法：

```python
import json
from pathlib import Path

# 读取单点计算结果
with open("runs/lgps-sp/mliport_results.json") as f:
    record = json.load(f)

calc = record["calculation"]
print(f"能量:  {calc['results']['energy']:.4f} eV")
print(f"设备:  {record['environment']['actual_device_type']}")
print(f"模型:  {record['model']['path']}")
print(f"SHA-256: {record['model']['sha256'][:16]}...")

# 读取优化结果
with open("runs/lgps-opt/mliport_results.json") as f:
    opt = json.load(f)

print(f"已收敛: {opt['calculation']['optimization']['converged']}")
print(f"步数:   {opt['calculation']['optimization']['steps']}")
```

在更大的工作流中，Python API 更为便捷：

## 7. 任务文件夹的写法

上面的命令可以用，但每天重复敲没必要。推荐的日常用法是一个任务一个文件夹：`INCAR.mliport`
和结构文件放在一起，然后 `mliport run` 不带任何参数。

```text
LiDS-on-LPSC/
  INCAR.mliport
  POSCAR
```

```bash
cd LiDS-on-LPSC
mliport run
```

`run` 会把结果写进 INCAR 旁边的 `results/`；`--output` 或 `OUTPUT_DIR` 仍然可以指定别的
目录。`MODEL_PATH` 只写文件名时，程序会去仓库的 `models/` 目录里找，所以把
`mace-omat-0-medium.model` 放进 `models/` 之后，`MODEL_PATH = mace-omat-0-medium.model`
就能直接用。每个键的含义见 [INCAR 键参考](incar.md)。

## 8. Python API

```python
# 只获取能量（eV）
energy = calculate_energy(
    "examples/structures/li10gep2s12_primitive.vasp",
    "/path/to/mace-omat-0-medium.model",
    model_type="mace", task="bulk"
)

# 获取完整结果字典
result = run_single_point(
    "examples/structures/li10gep2s12_primitive.vasp",
    "/path/to/mace-omat-0-medium.model",
    model_type="mace", task="bulk",
    output_dir="runs/api-sp"
)
```

完整参考见 [Python API](api.md)。

## 尝试其他后端

每个后端有自己的 Python 环境和启动脚本。使用 `--engines <名称>` 安装，通过
对应的启动脚本运行：

=== "DPA"

    ```bash
    ./scripts/install_mliport.sh --engines dpa --device auto

    ./bin/mliport-dpa sp examples/structures/li10gep2s12_primitive.vasp \
      --model DPA-3.1-3M.pt --model-type dpa \
      --task bulk --head Omat24 \
      --output runs/lgps-dpa-sp
    ```

    DPA 使用多任务检查点——必须通过 `--head` 选择分支。详见
    [DPA 后端](backends/dpa.md)。

=== "GRACE"

    ```bash
    ./scripts/install_mliport.sh --engines grace --device auto

    ./bin/mliport-grace sp examples/structures/li10gep2s12_primitive.vasp \
      --model GRACE-2L-OMAT-medium-base --model-type grace \
      --task bulk \
      --output runs/lgps-grace-sp
    ```

    GRACE 模型是 TensorFlow SavedModel 目录，而非单个文件。详见
    [GRACE 后端](backends/grace.md)。

=== "UMA"

    ```bash
    ./scripts/install_mliport.sh --engines uma --device auto

    ./bin/mliport-uma sp examples/structures/li10gep2s12_primitive.vasp \
      --model uma-s-1p2.pt --model-type uma \
      --task omat \
      --output runs/lgps-uma-sp
    ```

    UMA 体相材料使用 `--task omat`，分子使用 `--task omol`。其检查点为非商业
    研究许可。详见 [UMA 后端](backends/uma.md)。

=== "NEP"

    ```bash
    ./scripts/install_mliport.sh --engines nep --device cpu

    ./bin/mliport-nep sp examples/structures/li10gep2s12_primitive.vasp \
      --model /path/to/nep.txt --model-type nep \
      --nep-runtime ase --task bulk \
      --output runs/lgps-nep-sp
    ```

    NEP 需要你提供针对目标体系训练好的 `nep.txt`，并且必须显式选择
    `--nep-runtime ase` 或 `--nep-runtime gpumd`。详见
    [NEP 后端](backends/nep.md)。

!!! info "没有默认后端"
    省略 `--model-type`（且没有模型别名提供）始终会报错。mliport 从不猜测使用
    哪个模型——你始终清楚地知道是哪个模型产生了某个数值。

## 下一步

| 主题 | 页面 |
|---|---|
| 了解哪个模型适合你的体系 | [模型](models.md) |
| 配置运行、INCAR 文件和 `settings.ini` | [配置](configuration.md) |
| 运行 MD、NEB、构型搜索、批量扫描 | [工作流](workflows/index.md) |
| 分析轨迹（MSD、扩散、RDF……） | [分析](analysis/index.md) |
| 从 Python 使用 mliport | [Python API](api.md) |
| 查看已验证的内容 | [验证](validation.md) |
| 修复错误 | [故障排查](troubleshooting.md) |
