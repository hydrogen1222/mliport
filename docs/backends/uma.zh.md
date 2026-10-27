# UMA（FAIRChem）

Meta FAIR 的 UMA 模型通过上游的 `fairchem-core` 加载。`uma` 和 `fairchem`（上游包
名）是同一个运行时，作为 `--model-type` 用哪个都行。

## 模型是什么

- **结构。** UMA 是 eSEN 类等变网络，截断半径 6 Å，其线性层是*线性专家混合*：专家权重
  按体系的组成、电荷、自旋和任务一次性选定，所以总参数量虽大，推理时实际参与计算的参
  数很少（[Wood 等，2025](https://arxiv.org/abs/2506.23971)）。
- **训练数据。** 约 5 亿个结构，来自 OMat24（体相无机，PBE / PBE+U）、OMol25（分子，
  ωB97M-V）、OC20（催化表面）、ODAC25（金属有机框架）和 OMC25（分子晶体）。这些数据集
  的 DFT 设置各不相同，所以你要选 **task**：每个 task 复现对应数据集的参考水平，不同
  task 给出的能量不能互相比较。
- **力。** UMA 论文报告 UMA-S 和 UMA-M 是保守力模型（在直接预测力的预训练之后，再微调成
  用求导得到力和应力）。mliport 不依赖这个说法：NEB 和构型搜索在运行时读取预测器的
  `direct_forces` 标志并实测能量–力一致性（见下文）。
- **对你意味着什么。** 在这几个示例后端里，只有 UMA 的训练数据同时包含表面、分子和多孔
  材料，而不只是体相；所以做 slab 上的吸附时，先拿你自己的 DFT 对比 `omat` 和 `oc20`
  两个 task 再决定。它的许可证仅限非商业研究。

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
- **NEB 和构型搜索**要求力是能量的梯度。mliport 在运行时判断，而不是依据本页：预测器
  报告 `direct_forces = true` 就拒绝，否则用有限差分预检在你的结构上实测一致性（见
  [NEB](../workflows/neb.md)）。
- **应力**只有在模型提供时才会报告。

mliport 采用 MIT 许可。上游软件和模型需要另行引用。
