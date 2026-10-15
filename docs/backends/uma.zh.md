# UMA（FAIRChem）

Meta FAIR 的 UMA 模型通过上游的 `fairchem-core` 加载。`uma` 和 `fairchem`（上游包
名）是同一个运行时，作为 `--model-type` 用哪个都行。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv` | 3.11, 3.12, 3.13 | `fairchem-core==2.21.0` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines uma --device auto
```
<!-- END GENERATED -->

## 固定的 profile

<!-- BEGIN GENERATED: profiles -->
| Profile | 标识 | dtype | task | head | 模型 SHA-256 |
|---|---|---|---|---|---|
| `uma_omat` | UMA-s 1.2 (OMat task head) | upstream/model-defined | omat | None | `ba5c0d912efa...` |
<!-- END GENERATED -->

## 运行

```bash
.venv/bin/mliport sp structure.cif \
  --model /path/to/uma.pt --model-type uma --task omat
```

模型请从上游 fairchem/UMA 的发布渠道获取，使用前核对 SHA-256。mliport 从不再分发
模型权重，UMA 的许可证是仅限非商业研究的许可，见[引用](../citations.md)。

## 需要知道的事

- **`--task` 选择任务族**（周期性块体用 `omat`，分子用 `omol`，催化用 `oc20`）。UMA
  没有单独的 head 字段。这些任务族不能互换：分子要用 `omol` 任务，周期性晶体要用
  `omat` 任务。
- **推理模式。** `--inference-mode` 接受 `default` 或 `turbo`。Turbo（TF32 加
  `torch.compile`）是可选加入的，MD 默认不使用它，对精度敏感的运行因此保持在默认
  模式。`--activation-checkpointing` 用速度换 GPU 显存。
- **精度**由模型决定；mliport 只报告框架版本，不假装能控制权重。
- **不支持 NEB。** UMA 的预测器会报告 `direct_forces`：它的力是直接预测出来的，而
  不是能量的梯度。NEB 需要的是后者，所以一致性预检总是拒绝这类模型（见
  [NEB](../workflows/neb.md)）。
- **应力**只有在模型提供时才会报告。

mliport 采用 MIT 许可。上游软件和模型需要另行引用。
