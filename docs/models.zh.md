# 模型

mliport 本身不包含模型。它通过适配器加载别人训练好的势函数，用来跑你的计算，并记录每个
结果是哪个模型算出来的。看结果时要分清哪些归 mliport 管、哪些归模型管：

| mliport 负责 | 模型文件和你自己负责 |
|---|---|
| 加载并选定模型 | 权重，以及训练它们用的数据 |
| 生效配置与溯源记录 | 能量所属的参考态 |
| 拒绝后端表达不了的选项 | 势函数本身有多准 |
| 确认后端运行环境真的能跑 | 绝对能量标度与各种约定 |

## 五个后端地位相同

`mace`、`dpa`、`grace`、`uma`、`nep` 五个后端地位相同，`fairchem` 是 `uma` 的别名。没有默认
后端：如果命令行、模型别名和 profile 都没有指定后端，程序会报错停下。

| 后端 | 适配器 | 框架 | 常见模型文件 |
|---|---|---|---|
| MACE | `mace-torch` | PyTorch | `.model` |
| DPA | `deepmd-kit` | PyTorch | `.pt` / `.pth` |
| GRACE | `tensorpotential` | TensorFlow | SavedModel 目录 |
| UMA | `fairchem-core` | PyTorch | `.pt` |
| NEP | `calorine`（ASE）/ 原生 `gpumd` | Python / GPUMD | `nep.txt` |

各后端的模型介绍、固定 profile、精度、head 的含义和已知限制见各自页面：
[MACE](backends/mace.md)、[DPA](backends/dpa.md)、[GRACE](backends/grace.md)、
[UMA](backends/uma.md)、[NEP](backends/nep.md)。NEP 需要单独安装，并且必须显式指定
`NEP_RUNTIME`；表里其余几行只要后端环境装好就能用。

## 模型支持状态

设备列里的 `verified` 只表示**mliport 能正确调用这个模型**：模型能加载，能量、力以及（模型
文件里定义了的）应力都是有限值，几步 MD 和一个小 NEB 能跑完，记录下来的模型哈希、head、精度
和设备与请求的一致。它不代表模型对你的体系算得准；势函数的精度是另一个问题，这张表回答不了。

| 状态 | 含义 |
|---|---|
| `verified` | 在该设备上、在对应后端自己的环境里，全部可调用性检查都通过，并且留有证据。 |
| `untested` | 适配器应该能加载，但还没有证据。 |
| `needs_runtime_upgrade` | 必须先改动固定的运行环境版本，才能试着调用。 |
| `unsupported` | 已知在该设备上加载不了或跑不起来，或者适配器无法表达模型实际计算的东西；mliport 会直接报错，而不是硬跑。 |

| 标签 | 引擎 | 模型文件 | Head / task | 精度 | CPU | GPU | 证据 |
|---|---|---|---|---|---|---|---|
| `uma_omat` | uma | `uma-s-1p2.pt` | `omat` | 由模型决定 | `untested` | `verified` | [beta 验证](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md) |
| `mace_omat` | mace | `mace-omat-0-medium.model` | 默认 | `float64` | `untested` | `verified` | [beta 验证](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md) |
| `mace_omat_float32` | mace | `mace-omat-0-medium.model` | 默认 | `float32` | `untested` | `verified` | [beta 验证](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md) |
| `dpa_omat` | dpa | `DPA-3.1-3M.pt` | `Omat24` | 由模型决定 | `untested` | `verified` | [beta 验证](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md) |
| `grace_omat` | grace | `GRACE-2L-OMAT-medium-base` | 默认 | 由模型决定 | `untested` | `verified` | [beta 验证](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md) |
| `grace_omat_fp64` | grace | `GRACE-2L-OMAT-medium-base-fp64` | 默认 | `float64` | `untested` | `verified` | [beta 验证](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md) |
| `grace_3l_omat_large_ft_am` | grace | `GRACE-3L-OMAT-large-ft-AM` | 默认 | 由模型决定 | `untested` | `untested` | — |
| `grace_2l_smax_omat_large` | grace | `GRACE-2L-SMAX-OMAT-large` | 默认 | 由模型决定 | `untested` | `untested` | — |
| `dpa31_sse_pbe` | dpa | `DPA-3.1-3M.pt` | `Domains_SSE_PBE` | 由模型决定 | `untested` | `untested` | — |
| `dpa31_sse_pbesol` | dpa | `DPA-3.1-3M.pt` | `Domains_SSE_PBESol` | 由模型决定 | `untested` | `untested` | — |
| `dpa4_neo_omat` | dpa | `DPA4-Neo-OMat24-v20260805.pt` | 单任务（待核实） | 由模型决定 | `untested` | `untested` | — |
| `dpa4_air_omat` | dpa | `DPA4-Air-OMat24-v20260805.pt` | 单任务（待核实） | 由模型决定 | `untested` | `untested` | — |
| `dpa4_plus_omat` | dpa | `DPA4-Plus-OMat24-v20260805.pt` | 单任务（待核实） | 由模型决定 | `untested` | `untested` | — |
| `mace_matpes_r2scan` | mace | `MACE-MATPES-r2SCAN-0` medium | 单 head（待核实） | `float64` | `untested` | `untested` | — |
| `mace_mh1_omat_pbe` | mace | `MACE-MH-1` | `omat_pbe` | `float64` | `untested` | `untested` | — |
| `mace_mh1_matpes_r2scan` | mace | `MACE-MH-1` | `matpes_r2scan` | `float64` | `untested` | `untested` | — |
| `uma_s_1p2p1_omat` | uma | `uma-s-1p2p1.pt` | `omat` | 由模型决定 | `needs_runtime_upgrade` | `needs_runtime_upgrade` | — |
| `uma_s_1p2p1_ft_r2scan_omat` | uma | `uma-s-1p2p1-ft-r2scan.pt` | `omat` | 由模型决定 | `needs_runtime_upgrade` | `needs_runtime_upgrade` | — |
| `nep89_gpumd` | nep | NEP89 `nep.txt` | — | 由模型决定 | `unsupported` | `verified` | [运行时证据](https://github.com/hydrogen1222/mliport/pull/75) |
| `uma_s_1pt` | uma | `uma-s-1.pt` | `omat` | 由模型决定 | `unsupported` | `unsupported` | 上游已知的广延性缺陷 |

有一行需要说明：NEP89 只在原生 GPU 运行时上验证过，因为 Calorine 的 ASE 运行时解析不了这种
模型格式，所以 NEP89 在 CPU 上是 `unsupported`，而不只是没测。

归档的 `uma-s-1.pt` 故意**不**放进这张表。它是上游归档版本，有已知的广延性缺陷；mliport 没有
对它做过检查，也不推荐使用。它的状态不该写成 `unsupported`，因为 `unsupported` 的含义是
mliport 会拒绝这个模型，而实际上没有任何东西拒绝它。

固定 profile 的哈希、来源 URL 和许可证都在
[模型清单](https://github.com/hydrogen1222/mliport/blob/main/validation/science/model_manifest.json)
里；每个状态背后的可调用性检查见[验证](validation.md)。

## 如何公平地比较结果

下面几条是比较计算结果时的基本要求，mliport 会把检查这些要求所需的信息都记录下来：

- **不同模型的能量不能直接比。** 形成能、势垒这类能量差，必须用同一个模型、同一个 head、
  同一套参考态算出来。
- **算完不代表算对。** NEB 顺利跑完，只说明流程没问题；势函数能不能描述你的体系，要另外
  检验。
- **基准测试的成绩不能随便推广。** 在 OMat24 等基准上的表现，只说明这个模型在那套测试集上
  的水平，换一类材料可能完全不同。
- **task 和 head 属于模型的一部分。** UMA 的 task（`omat`、`omol` 等）以及 DPA、MACE 的
  head 会改变模型算出来的结果。mliport 会记录实际用的 task 和 head，多任务模型必须由你指定
  用哪个分支。
- **精度也属于模型的一部分。** MACE 的 `float32` 和 `float64` 是两个不同的 profile，比较它
  们的能量时要注明。

## 指定后端

最直接的办法是加命令行参数：

```bash
mliport sp structure.cif --model model.model --model-type mace
```

常用的模型可以在 `settings.ini` 里定义别名（别名里必须写明对应的后端）：

```ini
[model:mace_omat]
engine = mace
path = /models/mace-omat-0-medium.model
task = bulk
```

```bash
mliport sp structure.cif --model-alias mace_omat
```

`--model` 也可以直接写别名。别名、profile 和命令行参数同时存在时谁优先，见[配置](configuration.md)。

## 记录了哪些模型信息

每次运行都会保存实际使用的 `model_type`、`model_path`、task、head 和精度，后端和框架的版本，
请求的设备和实际使用的设备，以及 mliport 的 commit。这些都一致时，两个结果才能比较。各字段
的含义见[输出文件](outputs.md)。

## 模型下载

| 引擎 | 模型文件 | 来源 |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations 发布页 `mace_omat_0`](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

验证用的就是这些文件。下面的 SHA-256 是对上游发布文件计算的（清单里不记录本地路径）：

| Profile | 文件 | SHA-256（前缀） |
|---|---|---|
| `mace_omat` | `mace-omat-0-medium.model` | `d4b14be9afa2...` |
| `dpa_omat` | `DPA-3.1-3M.pt` | `86dd3a804d78...` |
| `grace_omat` | `GRACE-2L-OMAT-medium-base` | SavedModel 目录 |
| `uma_omat` | `uma-s-1p2.pt` | `ba5c0d912efa...` |

完整哈希在
[`validation/science/model_manifest.json`](https://github.com/hydrogen1222/mliport/blob/main/validation/science/model_manifest.json)。

!!! note "验收时的下载情况"
    LGPS 验收时，MACE 的模型文件是在单独的缓存目录里重新下载的，哈希和记录一致。从 Hugging
    Face 下载这一步**没有**在验收机器上测通：那台机器的 SOCKS/httpx 代理设置和 Hugging Face
    客户端的默认设置不兼容，所以验收时用的是四个模型文件的本地副本，并核对了哈希。如果你的
    网络也要走代理，可以参考下面的做法。

模型文件的许可证由上游项目决定，其中 UMA 只允许非商业的研究用途。

## 代理与离线使用

从 Hugging Face 下载用的是 `huggingface_hub` 里的 `httpx`。如果走 SOCKS 代理
（`ALL_PROXY=socks5...`），请在用来下载的环境里装上 socks 扩展：

```bash
uv pip install 'httpx[socks]' --python .venv/bin/python
```

有些代理配置会在 `NO_PROXY` 里写入 `[::1]` 这种带方括号的地址，httpx 会报 `Invalid port`。
把它改成只含普通主机名即可：

```bash
NO_PROXY=localhost,127.0.0.1 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('deepmodelingcommunity/DPA-3.1-3M', 'DPA-3.1-3M.pt'))"
```

本地的模型文件不需要联网，把路径传给 `--model` 就行；离线运行安装脚本时加 `--source offline`。
