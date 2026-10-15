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
