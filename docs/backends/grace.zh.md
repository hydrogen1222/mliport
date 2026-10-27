# GRACE

GRACE 模型是导出的 TensorFlow SavedModel，通过 `tensorpotential` 加载。模型路径是
SavedModel 的**目录**，而不是单个文件。

## 模型是什么

- **结构。** GRACE（图原子团簇展开）把原子团簇展开从局部"星形图"推广到树形图：`1L`
  模型是局部的、速度快；`2L` 模型多一层等变消息传递，用来描述半局域相互作用
  （[Bochkarev、Lysogorskiy 与 Drautz，PRX 2024](https://arxiv.org/abs/2311.16326)）。
- **示例检查点的训练数据。** `-OMAT-base` 模型用 OMat24 训练（PBE / PBE+U，体相无机，
  见 [MACE 页面](mace.md)），能量、力和应力的权重相同；`-OMAT` 版本在此基础上用更大的
  能量权重微调，`-OAM` 版本再用 MPtrj 和 Alexandria 的子集微调
  （[Lysogorskiy 等，2025](https://arxiv.org/abs/2508.17936)）。
- **力**是能量的梯度（保守力）；模型按下面列出的导出版本的精度计算。
- **对你意味着什么。** 参考水平和盲区与其他 OMat24 模型相同（不含色散，只用体相数据训
  练）。最看重速度时选 `1L`，最看重精度时选 `2L`。许可证请查模型发布页面。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |

```bash
./scripts/install_mliport.sh --engines grace --device auto
```
<!-- END GENERATED -->

## 固定的 profile

<!-- BEGIN GENERATED: profiles -->
| Profile | 标识 | dtype | task | head | 模型 SHA-256 |
|---|---|---|---|---|---|
| `grace_omat` | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | bulk | None | `d497af4c7d9a...` |
| `grace_omat_fp64` | GRACE-2L-OMAT-medium-base fp64 build | float64 | bulk | None | `53b945efce37...` |
<!-- END GENERATED -->

## 运行

```bash
.venv-grace/bin/mliport sp structure.cif \
  --model /path/to/saved_model --model-type grace
```

官方导出的 SavedModel 来自上游 AMS-ICAMS-RUB 的发布渠道；把 `--model`
（`MODEL_PATH`）指向你解压出来的那个目录。

## 需要知道的事

- **没有 head。** GRACE 模型不暴露 mliport 层面的 head；`--task` 是明确的周期性边
  界语义（`bulk` 或 `molecule`）。
- **GPU 隔离。** TensorFlow 的 GPU 可见性必须按进程固定。`CUDA_VISIBLE_DEVICES`
  未设置时，命令行会重启命令；Python API 和队列 worker 也需要同样的隔离。
- **GPU 显存与近邻列表**可以调节。TensorFlow 通常会占满几乎所有可见显存，所以显存
  按需增长默认是开启的（`--gpu-memory-growth`、`--gpu-memory-limit-mb`）。Verlet
  风格的近邻缓存（`--neighbor-cache`、`--neighbor-skin`，默认 skin 为 1.5 Å）默认也
  是开启的，当某个原子移动超过 skin 的一半时才重建近邻表。
- **精度**跟随模型：固定的 profile 里包含上游默认构建和 fp64 构建，见上表。
- **应力**取决于模型。

mliport 采用 MIT 许可。上游软件和模型需要另行引用，见[引用](../citations.md)。
