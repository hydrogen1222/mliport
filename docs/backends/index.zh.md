# 后端

mliport 支持五种机器学习原子间势后端。每个后端封装了不同的神经网络架构及其上游框架，但它们向 mliport 呈现相同的接口：传入结构和检查点文件，mliport 返回统一格式的能量、力和（如果可用的）应力。

## 什么是后端？

mliport 中的"后端"由三部分组成：

1. **神经网络架构** — 根据原子核位置预测原子能量和力的数学模型（MACE、DPA、GRACE、UMA 或 NEP）。
2. **上游框架** — 加载和评估训练好的检查点的 Python 包（`mace-torch`、`deepmd-kit`、`tensorpotential`、`fairchem-core` 或 `calorine`）。
3. **训练好的检查点** — 包含学习到的权重的文件。检查点决定了模型了解哪些化学体系、在什么 DFT 参考水平下、以及适用于哪些类型的系统（体相、表面、分子）。

mliport 每次计算只运行一个后端。后端通过 `--model-type`（或 INCAR 文件中的 `MODEL_TYPE`）选择，也可以从 `settings.ini` 中的模型别名或配置文件选择，否则运行会以 `No MLIP backend was selected` 错误停止。没有默认值，也没有回退机制。

## 后端对比

| | MACE | DPA | GRACE | UMA | NEP |
|---|---|---|---|---|---|
| 架构 | 等变消息传递 + 多体特征 | 线图序列上的消息传递（对 → 角度 → …） | 图原子簇展开（1L 局部，2L 半局部） | eSEN + 线性专家混合 | 神经进化势（前馈，GPU 优化） |
| 示例检查点 | `mace-omat-0-medium` | `DPA-3.1-3M` | `GRACE-2L-OMAT-medium-base` | `uma-s-1p2` | 用户自行训练的 `nep.txt` |
| 训练数据 | OMat24（约 1.1 亿结构） | OpenLAM-v1（31 个数据集） | OMat24 | OMat24 + OMol25 + OC20 + ODAC25 + OMC25 | 用户提供 |
| 参考 DFT 水平 | PBE / PBE+U（体相） | 按 head 选择（Omat24 = PBE/PBE+U） | PBE / PBE+U（体相） | 按 task 选择（omat = PBE/PBE+U） | 取决于训练数据 |
| 关键选择器 | `--dtype`（float32/float64） | `--head`（数据集分支） | 模型构建版本（default/fp64） | `--task`（omat/omol/oc20/…） | `--nep-runtime`（ase/gpumd） |
| 框架 | PyTorch | PyTorch | TensorFlow | PyTorch | Calorine (ASE) 或原生 GPUMD |
| 环境 | `.venv-mace` | `.venv-dpa` | `.venv-grace` | `.venv` | `.venv-nep` |

以上示例检查点均不包含色散校正。只有 UMA 的训练数据包含表面、分子和多孔材料。基础模型再现的是其参考 DFT 泛函的结果，而非实验值。模型在*你的*体系上的表现需要通过几个 DFT 单点计算来验证。参见[模型](../models.md)了解如何诚实地比较后端。

## 如何选择后端

正确的后端取决于你的体系和计算需求：

**我需要研究表面、分子或金属有机框架**
→ **UMA** 是唯一一个在非体相系统上训练过的示例模型（OC20 用于催化表面，OMol25 用于分子，ODAC25 用于 MOF）。其他四个示例模型仅在体相无机晶体上训练，将它们用于表面或分子属于外推。

**我需要双精度（float64）计算**
→ **MACE** 配合 `--dtype float64`，或 **GRACE** 的 fp64 构建版本。

**我需要最大吞吐量来运行长时间 MD**
→ **NEP** 原生 GPUMD 运行时（`--nep-runtime gpumd`）在整个运行过程中只启动一个 GPU 进程，开销最小。**GRACE 1L** 模型以牺牲部分精度换取速度。UMA 的 `turbo` 推理模式启用 `torch.compile` 和 TF32。

**我有自己训练的势函数**
→ **NEP** 专为用户训练的 `nep.txt` 文件设计。

**我需要单个模型覆盖最广泛的元素**
→ **DPA**（OpenLAM-v1 覆盖 31 个数据集）或 **UMA**（五个数据集系列覆盖体相、表面、分子、MOF 和分子晶体化学）。

**我需要运行 NEB 或构型搜索**
→ 力是能量梯度（保守力）的任何后端均可。MACE、DPA、GRACE 和 UMA 都满足要求。NEP 仅在 `--nep-runtime ase` 下可用；原生 GPUMD 运行时不支持 NEB 或构型搜索。

## 独立环境

每个后端有自己的 Python 虚拟环境，因为上游框架之间存在不兼容的依赖关系：

| 后端 | 环境 | 分离原因 |
|---|---|---|
| MACE | `.venv-mace` | `mace-torch` 固定特定 PyTorch 版本 |
| DPA | `.venv-dpa` | `deepmd-kit` 需要自己的 torch + C++ 扩展 |
| GRACE | `.venv-grace` | `tensorpotential` 使用 TensorFlow 而非 PyTorch |
| UMA | `.venv` | `fairchem-core` 固定自己的 torch 版本 |
| NEP | `.venv-nep` | `calorine` 是编译的 C 扩展 |

安装脚本会创建你需要的环境：

```bash
./scripts/install_mliport.sh --engines mace --device auto
```

验证后端是否可以正常运行：

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

完整安装步骤请参见[安装](../installation.md)。

## GPU 隔离

DPA 和 GRACE 需要在框架加载 CUDA 上下文之前选择 GPU。命令行会自动处理：在设置 `CUDA_VISIBLE_DEVICES` 后重启进程。从 Python API 调用时，你需要在导入框架之前自行设置环境变量：

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from mliport.api import run_single_point
```

## 页面结构

以下每个后端页面都遵循相同的结构：

1. **模型介绍** — 架构、训练数据、力的含义
2. **安装** — 环境和安装命令
3. **固定配置** — mliport 注册的检查点
4. **运行** — 命令行和 INCAR 示例
5. **注意事项** — 陷阱、限制和后端特有选项
