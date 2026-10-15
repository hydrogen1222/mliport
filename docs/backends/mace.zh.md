# MACE

MACE 模型通过 `mace-torch` 加载。一个 MACE profile 是"模型文件加精度"，mliport 把
两种精度当作不同的模型：`float64` 是首选的、以精度为先的 profile，`float32` 是单
独的身份，它的结果绝不会和 float64 的混在一起。

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

`mace_omat` 系列 profile 使用上游 MACE 发布的 `mace-omat-0-medium.model`，通过
`validation/science/model_manifest.json` 里的 SHA-256 引用。下载链接见
[模型](../models.md)。

## 运行

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model /path/to/mace-omat-0-medium.model --model-type mace
```

`--task` 是明确的周期性边界语义（`bulk` 或 `molecule`）。`--dtype float32` 更快但
精度更低；请有意识地使用，并在比较数字时说明。

## 需要知道的事

- **Head。** 如果模型提供了多个 head，必须用 `--head` 指明一个，mliport 不会替你
  挑。只有单个 head 的模型什么都不用写。
- **精度是身份的一部分。** 记录下来的 dtype 永远来自后端，而不是某种假设。不要在
  没有说明的情况下拿 `float32` 的能量去和 `float64` 的比。
- **应力**只有在加载的模型提供它时才会报告（`implemented_properties`），而不同模
  型之间情况不同，所以请查看验证记录里的 `stress_supported` 字段，或者运行
  `mliport doctor`，不要想当然。
- **NEB** 能和 MACE 配合使用，因为它的力来自能量的导数；运行时的一致性检查见
  [NEB](../workflows/neb.md)。

mliport 采用 MIT 许可。上游软件和模型需要另行引用，见[引用](../citations.md)。
