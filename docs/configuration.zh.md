# 配置

大多数时候你会把所有设置都写在命令行上。运行变复杂以后，可以把设置挪到
`settings.ini`、模型别名或 INCAR 风格的作业文件里。它们都汇入同一个解析器，解析器
按下面的顺序叠加各层，后面的覆盖前面的：

```text
built-in defaults
  < settings.ini
  < model alias ([model:NAME])
  < profile ([profile:NAME])
  < INCAR / job file
  < CLI flags / API arguments
```

解析器从不把两层对同一个选项的想法"融合"起来。每个选项最终只有一个值、只来自一
个来源，这个来源会写进 `resolved_config.json`（见[输出文件](outputs.md)）。想知道
某次运行为什么是那样表现的，看这个文件就有答案。

## 一次运行需要什么

| 需要什么 | 从哪里来 |
|---|---|
| 计算类型 | 子命令（`sp`、`opt`、`md`、`neb`、`batch`），或 `CALC_TYPE` / `CALCULATION` |
| 后端 | `--model-type` / `MODEL_TYPE`，或声明了 `engine` 的模型别名 |
| 模型路径 | `--model` / `MODEL_PATH`，或别名里的 `path` |
| 结构 | 位置参数；`run` 命令下用 `--structure` 或 INCAR 的结构键 |
| DPA 分支 | 多任务模型用 `--head` / `HEAD` |
| NEB 端点 | `--initial` / `--final`，或 `NEB_INITIAL` / `NEB_FINAL` |

不存在隐含的 UMA 默认值，也不存在别的默认值。如果任何一层都没有指明后端，解析器
会停下来：

```text
Error: No MLIP backend was selected.

Choose one of:
  --model-type mace
  --model-type dpa
  --model-type grace
  --model-type uma

or use a configured model alias/profile.
```

## 默认就是严格模式

拼错的键，或属于另一个后端的键，都是致命错误。另一种做法——给个警告然后带着错误
的设置继续跑——恰恰就是错误结果被产出、又被人相信的途径。

```bash
mliport sp structure.cif --model m.model --model-type mace --inference-mode turbo
# Error: Option 'inference_mode' is not applicable to engine 'mace' ...
```

做探索性工作时可以切回旧的"警告后继续"行为，但只能是你有意为之：

```bash
mliport sp ... --lenient-config
```

```ini
[general]
strict_config = false
```

查看配置可以用：`mliport config validate settings.ini` 检查文件，
`mliport config explain <KEY>` 看某个值从哪来。`mliport config paths` 列出
`settings.ini` 的搜索位置，`mliport config init` 生成一个起始文件，
`mliport config show` 打印生效配置，`mliport config schema` 列出所有可识别的键。

## settings.ini

搜索路径（优先级从高到低）由 `mliport config paths` 打印。设置 `MLIPORT_SETTINGS`
可以指向一个明确的文件；全局的 `--settings PATH` 参数优先级比它还高。

```ini
[general]
strict_config = true

[model:mace_omat]
engine = mace
path = /models/mace-omat-0-medium.model
task = bulk

[md]
ensemble = NVT
temperature = 300

[profile:quick]
calc_type = sp
device = cuda
```

- `[model:NAME]` 段必须声明 `engine`（`uma`、`mace`、`dpa` 或 `grace`；`fairchem`
  是 `uma` 的另一个名字）。
- `[profile:NAME]` 段把计算器和运行选项打包在一起，但不指定模型。
- 以计算类型命名的段（`[sp]`、`[md]`、`[neb]` 等）只为该类型设置默认值；
  `[general]`、`[resources]`、`[output]`、`[safety]` 对所有类型生效。

## INCAR 风格的作业

`mliport run --incar INCAR.mliport` 读取一个 VASP INCAR 风格的键值文件。这只是写
法上的风格，它并不是 DFT 输入。

```text
CALC_TYPE = SP
MODEL_TYPE = MACE
MODEL_PATH = /models/mace-omat-0-medium.model
DEVICE = cpu                        # explicit cuda / cuda:0 for GPU runs
```

让 mliport 生成模板，后端由你自己填：

```bash
mliport template sp --engine mace --output INCAR.mace
mliport template sp --output INCAR.neutral   # backend-neutral, MODEL_TYPE=REQUIRED
```

## 只属于某个后端的选项

| 选项 | 后端 | 作用 |
|---|---|---|
| `TASK` / `--task` | UMA | 任务族（`omat`、`omol`、`oc20` 等） |
| `TASK` / `--task` | MACE、DPA、GRACE | 明确的 PBC 语义（`bulk` / `molecule`） |
| `HEAD` / `--head` | DPA、MACE | 模型的分支或 head（许多 MACE 模型用 `default`） |
| `INFERENCE_MODE` | UMA | `default`、`turbo` 或 `trained` 推理预设 |
| `ACTIVATION_CHECKPOINTING` | UMA | 节省 GPU 显存 |
| `DEFAULT_DTYPE` | MACE | 模型精度 |
| `GPU_MEMORY_GROWTH`、`GPU_MEMORY_LIMIT_MB` | GRACE | TensorFlow 的 GPU 显存策略 |
| `NEIGHBOR_CACHE`、`NEIGHBOR_SKIN` | GRACE | 近邻列表缓存策略 |
| `TORCH_NUM_THREADS` | UMA、MACE、DPA | CPU 算子内线程数 |

在严格模式下，把这些选项交给另一个后端会报错，选项绝不会被悄悄丢掉。

## 色散校正（DFT-D3）

MLIP 模型文件里不包含色散校正。在纯 PBE 数据上训练的模型（MACE 的 `omat` head、UMA
的 `omat`、大多数材料模型）学到的是 PBE 势能面，而 PBE 本身没有长程色散。`--d3` 在
模型之上叠加 Grimme DFT-D3 的能量、力和应力，对所有后端都适用，实现来自
`torch-dftd`：

| 选项 | 默认值 | 作用 |
|---|---|---|
| `D3` / `--d3` | 关闭 | 在模型之上叠加 DFT-D3 |
| `D3_XC` / `--d3-xc` | `pbe` | 模型训练数据的参考泛函，决定 D3 参数 |
| `D3_DAMPING` / `--d3-damping` | `bj` | 阻尼函数：`zero`、`bj`、`zerom` 或 `bjm` |

```bash
uv pip install --python .venv-mace/bin/python torch-dftd
mliport opt slab.vasp --model model.model --model-type mace --task bulk --d3
```

只能给训练参考里不含色散的模型加 D3，并用 `--d3-xc` 写明这个参考泛函；mliport 无法从
MACE、DPA 或 GRACE 的模型文件里读出它。UMA 每个任务的参考是已知的：只有 `omat`
（PBE）和 `oc20`（RPBE）接受 D3；`omol`、`omc`、`odac` 和 `oc25` 的参考已经包含色散
或未确认不含色散，会被拒绝。只写 `D3_XC` 或 `D3_DAMPING` 而不开 `D3` 会报错，而不是
记录一个实际上不生效的设置。NEB 暂不支持 D3，TUI 也还没有这个选项。

校正会随运行一起记录：JSON 输出的模型元数据里有 `dispersion` 字段，OUTCAR 风格的文件
头里有一行 `Dispersion:`。`torch-dftd` 需要 PyTorch，MACE、DPA 和 UMA 环境里都已经
有了。

## 科学参数的默认值

温度、时间步长、步数、恒温器、优化器、`fmax`、NEB 设置等的默认值都在 mliport 内
置默认里，按计算类型划定作用范围，并为每次运行记录在 `resolved_config.json` 中。
权威列表一条命令就能看到：

```bash
mliport config schema
mliport config show --settings settings.ini
```

## 输出控制

| 键 | 含义 |
|---|---|
| `DEVICE` / `--device` | `cpu`、`cuda` 或 `cuda:N`；请求的设备和实际使用的设备分开记录 |
| `OUTPUT_DIR` / `--output` | 运行目录 |
| `WRITE_OUTCAR`、`WRITE_XDATCAR`、`WRITE_TRAJECTORY` | 写出哪些文件 |
| `WRITE_FORCES`、`WRITE_STRESS` | 写出哪些数组和列 |

NEB 拒绝关闭它那些必需的输出，而不会递给你一份悄悄缺东西的结果。

## 环境变量

| 变量 | 作用 |
|---|---|
| `MLIPORT_SETTINGS` | 明确指定 `settings.ini` 路径 |
| `MLIPORT_JOBS_DIR` | 队列与作业注册表目录 |
| `MLIPORT_CANCEL_GRACE_SECONDS` | 取消作业时的宽限时间 |
| `MLIPORT_LEASED_GPU_UUID` | 租给队列 worker 的那块 GPU |
| `CUDA_VISIBLE_DEVICES` | 设备可见性；队列租用 GPU 时会设置它，所以不要在运行中的作业里覆盖它 |

## TUI

`mliport tui` 提供同样的配置项（包括明确的后端选择），在选定后端和模型之前不会开始
运行。它和命令行通过同一个队列辅助函数提交作业，所以两者不会各走各的。
