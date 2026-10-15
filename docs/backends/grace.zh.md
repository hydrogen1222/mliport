# GRACE

GRACE 模型是导出的 TensorFlow SavedModel，通过 `tensorpotential` 加载。模型路径是
SavedModel 的**目录**，而不是单个文件。

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
