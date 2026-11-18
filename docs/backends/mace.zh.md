# MACE

MACE 是一种等变消息传递神经网络，用于构建原子间势函数。在 mliport 中，MACE 检查点通过 `mace-torch` 包加载。本页介绍 MACE 的架构原理、安装运行方法和注意事项。

## 模型介绍

### 架构

MACE（Multi-Atomic Cluster Expansion，多原子簇展开）在每个消息传递层内部直接构造高阶多体相互作用特征。在传统的消息传递网络中，每层聚合来自邻居原子的信息（二体相互作用），多体效应只有在堆叠多层之后才能涌现。MACE 则不同——它使用原子簇展开形式，在每层内显式构造三体、四体乃至更高阶的特征。这意味着通常只需要两层消息传递就能得到精确的势函数，模型紧凑且高效。

关键创新在于 MACE 使用了等变特征：内部表示在旋转和反射变换下正确变换，因此预测的能量自动满足刚体变换的不变性。力是预测能量的精确梯度，具有保守性——这是能量守恒 MD 和基于梯度的方法（如 NEB）所必需的性质。

参考文献：[Batatia 等，NeurIPS 2022](https://arxiv.org/abs/2206.07697)

### 示例检查点的训练数据

`mace-omat-0-medium` 检查点在 [OMat24](https://arxiv.org/abs/2410.12771) 上训练。OMat24 是 Meta 发布的约 1.1 亿个 VASP 计算数据集，对象是体相无机结构。其中许多结构被刻意偏离平衡态——随机扰动原子位置和高温构型——这提高了模型在 MD 模拟中的鲁棒性，因为原子可能临时访问高能构型。

参考 DFT 水平是 **PBE / PBE+U**（VASP 5.4，Materials Project 风格）：+U 校正仅应用于氧化物和氟化物中的部分过渡金属。这意味着模型再现的是 PBE(+U) 的计算结果，而不是实验测量值。

较早的 `MACE-MP-0` 系列在 MPtrj（Materials Project 轨迹数据集）上训练，同样是 PBE / PBE+U 水平（[Batatia 等，2024](https://arxiv.org/abs/2401.00096)）。完整的检查点列表及其许可证维护在 [mace-foundations](https://github.com/ACEsuit/mace-foundations)。

### 对你意味着什么

- 能量和力逼近的是体相无机材料上的 PBE(+U) 计算结果。
- 模型**不含色散项** — 可以用 `--d3` 添加 DFT-D3 校正，但这是在 ML 势函数之上的经验校正。
- 表面、分子和界面**不在** OMat24 训练检查点的训练分布内。在这些体系上使用前，先拿几个结构和你自己的 DFT 对比。
- 商用前请确认检查点的许可证——不同检查点有不同的许可条款。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines mace --device auto
```
<!-- END GENERATED -->

安装后验证：

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

## 固定的 profile

<!-- BEGIN GENERATED: profiles -->
| Profile | 标识 | dtype | task | head | 模型 SHA-256 |
|---|---|---|---|---|---|
| `mace_omat` | MACE-OMAT-0 medium, float64 inference | float64 | bulk | None | `d4b14be9afa2...` |
| `mace_omat_float32` | MACE-OMAT-0 medium, float32 inference | float32 | bulk | None | `d4b14be9afa2...` |
<!-- END GENERATED -->

`mace_omat` 两个 profile 用的都是上游发布的 `mace-omat-0-medium.model`，以 `validation/science/model_manifest.json` 里的 SHA-256 为准。下载链接见[模型](../models.md)。

## 运行

### 命令行

```bash
./bin/mliport-mace sp structure.cif \
  --model /path/to/mace-omat-0-medium.model --model-type mace
```

### INCAR 文件

```ini
MODEL_TYPE   = MACE
MODEL_PATH   = /path/to/mace-omat-0-medium.model
TASK         = bulk
DTYPE        = float64
```

```bash
./bin/mliport-mace run INCAR.mliport
```

### Python API

```python
from mliport.api import run_single_point

results = run_single_point(
    structure="structure.cif",
    model_path="/path/to/mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
)
print(f"Energy: {results['energy']:.6f} eV")
```

### 精度

`--dtype float64`（`mace_omat` profile 的默认值）是精度优先模式。`--dtype float32` 更快但精度更低。mliport 将两种精度视为不同的模型标识——它们的结果不会被混合，切换精度会记录在运行溯源中。请有意识地使用 float32，并在比较数据时注明。

## 注意事项

- **Head。** 如果检查点提供多个 head（例如 MACE-MH-1），必须用 `--head` 指定一个；mliport 不会替你选择。单 head 检查点不需要额外设置。

- **只接受能量模型。** 没有势能 head 的检查点会在运行前被拒绝。这包括纯偶极/极化率模型（MACE-MDP 系列）、MACE-POLAR 可极化静电模型、以及带电荷/自旋联合嵌入的模型（MACE-OMOL 系列）。错误信息会说明原因和应使用的上游加载方式。

- **精度属于模型标识的一部分。** 记录的 dtype 始终来自后端的实际报告值。不要在不注明的情况下比较 float32 和 float64 的能量。

- **应力**只有在加载的检查点提供时才会输出（检查 `implemented_properties`）。不同检查点情况不同，请通过 `mliport doctor` 或验证记录中的 `stress_supported` 字段确认。

- **NEB** 可以与 MACE 配合使用，因为力是能量的梯度（保守力）。运行时的一致性检查见 [NEB](../workflows/neb.md)。

- **DFT-D3 色散校正**可以通过 `--d3` 添加到任何 MACE 计算中。这会在 ML 预测的能量和力之上添加经验 Grimme D3 校正。

## 引用 MACE

mliport 采用 MIT 许可证。上游 MACE 软件和模型检查点需要另外引用；完整参考文献列表见[引用](../citations.md)。
