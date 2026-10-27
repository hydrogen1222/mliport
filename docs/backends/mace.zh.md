# MACE

MACE 模型通过 `mace-torch` 加载。mliport 里一个 MACE profile 由"模型文件 + 精度"确定，两种精度
算作两个不同的模型：默认用 `float64`，精度优先；`float32` 单独记录，结果不和 float64 混用。

## 模型是什么

- **结构。** MACE 是等变消息传递网络，每一层都构造高阶多体特征（把原子团簇展开放进
  消息传递里），所以两层就足以得到精确的势
  （[Batatia 等，NeurIPS 2022](https://arxiv.org/abs/2206.07697)）。
- **示例检查点的训练数据。** `mace-omat-0` 用 [OMat24](https://arxiv.org/abs/2410.12771)
  训练：Meta 发布的约 1.1 亿个 VASP 计算，对象是体相无机结构，其中很多结构刻意偏离
  平衡（随机扰动和高温帧）。参考水平是 PBE / PBE+U（VASP 5.4），与 Materials Project
  一致：只对氧化物和氟化物中的部分过渡金属加 +U。较早的 `MACE-MP-0` 系列用 MPtrj
  训练，也是 PBE / PBE+U（[Batatia 等，2024](https://arxiv.org/abs/2401.00096)）；全部检查点
  及其许可证列在 [mace-foundations](https://github.com/ACEsuit/mace-foundations)。
- **力**是能量的梯度（保守力），所以 MACE 能通过 NEB 和构型搜索的梯度预检。
- **对你意味着什么。** 能量和力逼近的是体相无机材料上的 PBE(+U)。模型不含色散（见
  `--d3`）；表面、分子和界面不在训练分布里，在这些体系上使用前，先拿几个结构和你自己
  的 DFT 对比。商用前请先确认检查点的许可证。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines mace --device auto
```
<!-- END GENERATED -->

## 固定的 profile

<!-- BEGIN GENERATED: profiles -->
| Profile | 标识 | dtype | task | head | 模型 SHA-256 |
|---|---|---|---|---|---|
| `mace_omat` | MACE-OMAT-0 medium, float64 inference | float64 | bulk | None | `d4b14be9afa2...` |
| `mace_omat_float32` | MACE-OMAT-0 medium, float32 inference | float32 | bulk | None | `d4b14be9afa2...` |
<!-- END GENERATED -->

`mace_omat` 两个 profile 用的都是上游发布的 `mace-omat-0-medium.model`，以
`validation/science/model_manifest.json` 里的 SHA-256 为准。下载链接见[模型](../models.md)。

## 运行

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model /path/to/mace-omat-0-medium.model --model-type mace
```

`--task` 指定体系是周期性的（`bulk`）还是分子（`molecule`）。`--dtype float32` 更快但精度更低，
比较数据时要注明用的是哪种精度。

## 需要知道的事

- **Head。** 模型有多个 head 时，必须用 `--head` 指定一个。只有一个 head 的模型不用写。
- **精度属于模型信息的一部分。** 记录的精度来自后端实际报告的值。`float32` 和 `float64` 的能量
  不要在不注明的情况下直接比较。
- **应力**只有模型支持时才会输出（见 `implemented_properties`），不同模型情况不同。可以查验证
  记录里的 `stress_supported` 字段，或运行 `mliport doctor` 确认。
- **NEB** 可以用 MACE，因为它的力是能量求导得到的；运行时的一致性检查见
  [NEB](../workflows/neb.md)。

mliport 采用 MIT 许可证。上游软件和模型需要另外引用，见[引用](../citations.md)。
