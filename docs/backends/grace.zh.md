# GRACE

GRACE（Graph Atomic Cluster Expansion）模型是导出的 TensorFlow SavedModel，通过 `tensorpotential` 加载。与其他基于 PyTorch 的后端不同，GRACE 运行在 TensorFlow 上，因此有自己的独立环境和 GPU 显存管理方式。

## 模型是什么

### 架构

GRACE 把原子团簇展开（ACE）从局部原子环境推广到树形图。ACE 的核心思想是用原子密度关联的系统展开来表示每个原子的局部环境——类似于傅里叶级数表示函数的方式，但适配了原子邻域的几何结构。

GRACE 有两个变体：

- **1L（单层，局部）：** 每个原子的能量只取决于截断半径内的近邻。速度最快，适用于长程结构关联不太重要的体系。
- **2L（两层，半局域）：** 在局部展开之上添加一层等变消息传递，使每个原子能"看到"其近邻的环境。这捕捉了半局域相互作用——不仅取决于附近有哪些原子，还取决于那些近邻本身的排列方式。

选择时需要注意：1L 更快，2L 对半局域效应显著的体系（大多数块体晶体）更准确。

参考文献：[Bochkarev, Lysogorskiy & Drautz, PRX 2024](https://arxiv.org/abs/2311.16326)

### 示例检查点的训练数据

`-OMAT-base` 模型用 OMat24 训练（PBE / PBE+U，体相无机，见 [MACE 页面](mace.md)），能量、力和应力的权重相同。`-OMAT` 版本在此基础上用更大的能量权重微调，`-OAM` 版本再用 MPtrj 和 Alexandria 的子集微调（[Lysogorskiy 等，2025](https://arxiv.org/abs/2508.17936)）。

力是能量的梯度（保守力），模型按导出版本的精度进行计算。

### 对你意味着什么

- 参考水平（PBE / PBE+U）和盲区（不含色散，只用体相数据训练）与其他 OMat24 模型相同。
- 最看重速度时选 1L，最看重精度时选 2L。
- 许可证请查模型发布页面。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |

```bash
./scripts/install_mliport.sh --engines grace --device auto
```
<!-- END GENERATED -->

安装完成后验证：

```bash
.venv-grace/bin/mliport doctor --engine grace --device auto
```

## 固定的 profile

<!-- BEGIN GENERATED: profiles -->
| Profile | 标识 | dtype | task | head | 模型 SHA-256 |
|---|---|---|---|---|---|
| `grace_omat` | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | bulk | None | `d497af4c7d9a...` |
| `grace_omat_fp64` | GRACE-2L-OMAT-medium-base fp64 build | float64 | bulk | None | `53b945efce37...` |
<!-- END GENERATED -->

## 运行

### 命令行

```bash
./bin/mliport-grace sp structure.cif \
  --model /path/to/saved_model --model-type grace
```

`--model` 路径必须指向 SavedModel **目录**（即包含 `saved_model.pb` 的那个目录），而不是单个文件。官方导出的 SavedModel 从上游 AMS-ICAMS-RUB 发布渠道获取。

### INCAR 文件

```ini
MODEL_TYPE   = GRACE
MODEL_PATH   = /path/to/saved_model
TASK         = bulk
```

```bash
./bin/mliport-grace run INCAR.mliport
```

### Python API

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 必须在 import 之前设置

from mliport.api import run_single_point

results = run_single_point(
    structure="structure.cif",
    model_path="/path/to/saved_model",
    model_type="grace",
    task="bulk",
)
```

## 需要知道的事

- **没有 head。** GRACE 模型不暴露 mliport 层面的 head。`--task` 设置周期性边界语义（`bulk` 或 `molecule`）。

- **GPU 隔离。** TensorFlow 的 GPU 可见性必须在进程级别固定。CLI 会自动通过设置 `CUDA_VISIBLE_DEVICES` 重启进程来处理。从 Python API 和队列 worker 调用时，你需要同样的隔离——在导入之前设置环境变量。

- **GPU 显存管理。** TensorFlow 默认在启动时预留几乎所有可见显存。mliport 默认启用按需分配（`--gpu-memory-growth`），你还可以用 `--gpu-memory-limit-mb` 设置显式上限。这能防止 GRACE 挤占同一 GPU 上其他进程的显存。

- **近邻列表缓存。** 默认启用 Verlet 风格的近邻缓存（`--neighbor-cache`、`--neighbor-skin`，默认 skin 为 1.5 Å）。只有当某个原子移动超过 skin 距离的一半时才重建近邻表。这通过避免每步都做完整的近邻搜索来加速 MD 和优化。

- **精度**由模型构建版本决定。固定的 profile 包含上游默认版本和 fp64 版本，你可以根据精度需求选择。

- **应力**取决于检查点——并非所有导出版本都包含应力预测。

## 引用 GRACE

mliport 采用 MIT 许可证，只把 GRACE 当作另外安装的软件调用，不打包、不再分发 GRACE 的代码或权重。
上游软件和模型需要另外引用，见[引用](../citations.md)。

## 许可

**GRACE 软件本身的许可证。** `tensorpotential` 采用
[Academic Software Licence（ASL）](https://github.com/ICAMS/grace-tensorpotential/blob/master/LICENSE.md)，
不是开源许可证：它只允许学术、非商业用途，且对衍生作品是互惠（类似 GPLv2）的。它的 PyPI 元数据
声明为 `License: Academic Software License (ASL)`。安装 `grace` 环境就是在运行该软件，因此商业或
商业研究用途需要向权利方（ACEworks GmbH）另行取得商业许可，这与检查点自身的许可证无关。mliport
不内嵌 GRACE 代码，所以这条限制不改变 mliport 的 MIT 许可证，只改变你可以用这个后端做什么。
