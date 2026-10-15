# 模型

mliport 不是模型。它通过适配器加载别人训练好的势函数，用它们跑你的工作流，并把
"到底是哪个模型干的活"如实记下来。读结果时，请记住这样的分工。

| mliport 负责 | 模型文件和你自己负责 |
|---|---|
| 加载并选定模型 | 权重，以及训练它们用的数据 |
| 生效配置与溯源记录 | 能量所属的参考态 |
| 拒绝后端表达不了的选项 | 势函数本身有多准 |
| 确认后端运行环境真的能跑 | 绝对能量标度与各种约定 |

## 四个后端，没有偏爱

`mace`、`dpa`、`grace`、`uma` 是地位平等的四个运行适配器；`fairchem` 被接受为
`uma` 的另一个名字。没有隐含的默认值。一次运行如果没有指明后端（既没有直接指
定，也没有通过模型别名或 profile 指定），就会报错停下。

| 后端 | 适配器 | 框架 | 常见模型文件 |
|---|---|---|---|
| MACE | `mace-torch` | PyTorch | `.model` |
| DPA | `deepmd-kit` | PyTorch | `.pt` / `.pth` |
| GRACE | `tensorpotential` | TensorFlow | SavedModel 目录 |
| UMA | `fairchem-core` | PyTorch | `.pt` |

固定的 profile、精度、head 语义和已知限制见各后端页面：
[MACE](backends/mace.md)、[DPA](backends/dpa.md)、[GRACE](backends/grace.md)、
[UMA](backends/uma.md)。

## 老老实实地比较结果

有些习惯能让计算在科学上站得住脚，mliport 会记录你遵守它们所需的信息：

- **不同模型的能量不可比。** 形成能或势垒必须来自同一个模型、同一个 head、同一
  套参考态。悄悄混用模型是科学上的错误，不是一个功能。
- **工作流跑通不说明准确。** 适配器把你的 NEB 跑完，只说明管线没问题，不说明势
  函数能描述你的化学体系。
- **基准数字不会自动外推。** OMat24 的结果刻画的是固定下来的那个模型在那个基准
  上的表现，并不适用于所有材料。
- **任务和 head 也是模型的一部分。** UMA 的任务族（`omat`、`omol` 等）以及 DPA、
  MACE 的 head 会改变模型计算的东西。mliport 记录实际用的 task 和 head，并且拒
  绝去猜多任务模型的分支。
- **精度同样是模型的一部分。** MACE 的 `float32` 与 `float64` 是不同的 profile；
  比较它们的能量时务必说明。

## 为一次运行选后端

最快的方式是加参数：

```bash
mliport sp structure.cif --model model.model --model-type mace
```

常用的模型可以在 `settings.ini` 里定义别名（别名必须声明它对应哪个引擎）：

```ini
[model:mace_omat]
engine = mace
path = /models/mace-omat-0-medium.model
task = bulk
```

```bash
mliport sp structure.cif --model-alias mace_omat
```

`--model` 也接受别名。别名、profile 和命令行参数如何叠加，见[配置](configuration.md)。

## 关于模型，记录了什么

每次运行都会保存生效的 `model_type`、`model_path`、task、head 和 dtype，后端和
框架版本，请求的设备和实际的设备，以及 mliport 的 commit。只有这些都一致，两个
结果才可比。各字段的含义见[输出文件](outputs.md)。

## 模型文件去哪里拿

| 引擎 | 模型文件 | 来源 |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations 发布页 `mace_omat_0`](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

验证套件固定在这些文件上。它们的 SHA-256 对应的是发布出来的文件（本地路径从不写
进清单）：

| Profile | 文件 | SHA-256（前缀） |
|---|---|---|
| `mace_omat` | `mace-omat-0-medium.model` | `d4b14be9afa2...` |
| `dpa_omat` | `DPA-3.1-3M.pt` | `86dd3a804d78...` |
| `grace_omat` | `GRACE-2L-OMAT-medium-base` | SavedModel 目录 |
| `uma_omat` | `uma-s-1p2.pt` | `ba5c0d912efa...` |

完整哈希在
[`validation/science/model_manifest.json`](https://github.com/hydrogen1222/mliport/blob/main/validation/science/model_manifest.json)。

!!! note "验收时哪些下载了、哪些没有"
    LGPS 验收时，MACE 的发布文件是在隔离的缓存里下载的，哈希与固定值一致。
    Hugging Face 的冷下载路径在验收机器上**没有**通过网络验证：那台机器的
    SOCKS/httpx 配置不接受 Hugging Face 客户端的默认代理设置，所以验收改用了四个
    模型文件的本地副本，并校验了哈希。如果你的网络需要代理，请参考下面的做法。

模型文件的许可证归上游项目所有。特别是 UMA 的许可证是非商业的研究许可。

## 代理与离线使用

Hugging Face 的下载经由 `huggingface_hub` 里的 `httpx`。在 SOCKS 代理
（`ALL_PROXY=socks5...`）之后，请在你用来下载的环境里安装 socks 扩展：

```bash
uv pip install 'httpx[socks]' --python .venv/bin/python
```

有些代理配置还会往 `NO_PROXY` 里写 `[::1]` 这样带方括号的条目，httpx 会以
`Invalid port` 拒绝。请把列表限制为普通主机名：

```bash
NO_PROXY=localhost,127.0.0.1 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('deepmodelingcommunity/DPA-3.1-3M', 'DPA-3.1-3M.pt'))"
```

本地磁盘上的模型完全不需要网络：把路径传给 `--model`，涉及安装脚本时再加
`--source offline`。
