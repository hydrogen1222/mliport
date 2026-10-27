# 后端

mliport 每次计算只使用一个后端。后端来自 `--model-type`，或来自设置里的模型别名
/ profile，否则运行会以 `No MLIP backend was selected` 停下。不存在"找不到就退
回到别的"这种事。

| 后端 | 页面 | `--model-type` | 示例中使用的模型 |
|---|---|---|---|
| MACE | [MACE](mace.md) | `mace` | `mace-omat-0-medium.model` |
| DPA（DeepMD-kit） | [DPA](dpa.md) | `dpa` | `DPA-3.1-3M.pt`，配合 `--head Omat24` |
| GRACE | [GRACE](grace.md) | `grace` | `GRACE-2L-OMAT-medium-base` |
| UMA（FAIRChem） | [UMA](uma.md) | `uma`（或 `fairchem`） | `uma-s-1p2.pt`，配合 `--task omat` |

每个后端住在自己的环境里（`.venv-mace`、`.venv-dpa`、`.venv-grace`、`.venv`），
因为它们的框架和 Python 要求各不相同；原理见[安装](../installation.md)。想确认某
个后端真的能跑，用 `doctor`：

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

DPA 和 GRACE 需要在框架加载之前就选好 GPU。命令行会通过重启自己、在新进程里设置
`CUDA_VISIBLE_DEVICES` 来处理；从 Python API 调用时，则需要你自己在调用进程里设置。

下面每个页面都按同一条线索组织：这个后端怎么装、mliport 固定了哪些模型
profile、怎么运行、要留意什么。环境表和 profile 表是从兼容性注册表和验证清单生成
的，所以不会和代码脱节。

## 一览

| | MACE | DPA | GRACE | UMA |
|---|---|---|---|---|
| 结构 | 带多体特征的等变消息传递 | 线图序列上的消息传递（原子对、键角……） | 图原子团簇展开（`1L` 局部，`2L` 半局域） | 带线性专家混合的 eSEN 网络 |
| 示例检查点的训练数据 | OMat24 | OpenLAM-v1（31 个数据集） | OMat24 | OMat24、OMol25、OC20、ODAC25、OMC25 |
| 示例的参考水平 | PBE / PBE+U，体相 | 取决于 head；`Omat24` = PBE / PBE+U | PBE / PBE+U，体相 | 取决于 task；`omat` = PBE / PBE+U |
| 你要选的 | 精度（`float64` / `float32`） | `--head` | 导出的版本 | `--task` |
| 框架 | PyTorch | PyTorch（`.pb` 为 TensorFlow） | TensorFlow | PyTorch |
| 示例的许可证 | 请查检查点 | 请查检查点 | 请查检查点 | 仅限非商业研究 |

这些检查点都不含色散，也只有 UMA 的训练数据在体相之外还包括表面和分子。基础模型复现
的是它的参考泛函，而不是实验；它在*你的*体系上复现得好不好，需要用几个 DFT 单点来检
查。如何公平地比较后端，见[模型](../models.md)页面。
