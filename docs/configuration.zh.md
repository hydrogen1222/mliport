# 配置

简单的计算把参数都写在命令行上就行。参数多了以后，可以写进 `settings.ini`、模型别名或
INCAR 风格的输入文件。这些来源由同一个解析器处理，按下面的顺序叠加，后面的覆盖前面的：

```text
built-in defaults
  < settings.ini
  < model alias ([model:NAME])
  < profile ([profile:NAME])
  < INCAR / job file
  < CLI flags / API arguments
```

同一个选项在多层里都出现时，只取优先级最高的那个值，不做合并。每个选项的最终值和它的来源
都写在 `resolved_config.json` 里（见[输出文件](outputs.md)）。想知道某次计算为什么用了某个
参数，查这个文件即可。

## 必需的输入

| 需要什么 | 从哪里来 |
|---|---|
| 计算类型 | 子命令（`sp`、`opt`、`md`、`neb`、`batch`、`confsearch`），或 `CALC_TYPE` / `CALCULATION` |
| 后端 | `--model-type` / `MODEL_TYPE`，或声明了 `engine` 的模型别名 |
| 模型路径 | `--model` / `MODEL_PATH`，或别名里的 `path` |
| 结构 | 位置参数；`run` 命令下用 `--structure` 或 INCAR 的结构键 |
| DPA 分支 | 多任务模型用 `--head` / `HEAD` |
| NEB 端点 | `--initial` / `--final`，或 `NEB_INITIAL` / `NEB_FINAL` |
| confsearch 起点 | `CS_STARTS` + `CS_ADSORBATE_INDICES`，或 `--slab` / `--molecule` / `--n-starts` |

后端没有默认值。所有层里都没有指定后端时，程序会报错：

```text
Error: No MLIP backend was selected.

Choose one of:
  --model-type mace
  --model-type dpa
  --model-type grace
  --model-type uma

or use a configured model alias/profile.
```

## 默认使用严格模式

拼错的参数名，或者属于别的后端的参数，都会直接报错。如果只给个警告然后继续跑，很容易算出
错误结果却没人发现。

```bash
mliport sp structure.cif --model m.model --model-type mace --inference-mode turbo
# Error: Option 'inference_mode' is not applicable to engine 'mace' ...
```

做探索性计算时，可以手动切换成"只警告、继续运行"：

```bash
mliport sp ... --lenient-config
```

```ini
[general]
strict_config = false
```

几个查看配置的命令：`mliport config validate settings.ini` 检查文件；
`mliport config explain <KEY>` 显示某个值的来源；`mliport config paths` 列出 `settings.ini`
的查找位置；`mliport config init` 生成一个模板文件；`mliport config show` 打印最终生效的配置；
`mliport config schema` 列出所有参数名。

## settings.ini

`mliport config paths` 会按优先级从高到低列出查找位置。环境变量 `MLIPORT_SETTINGS` 可以
指定一个文件，命令行的 `--settings PATH` 优先级更高。

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

- `[model:NAME]` 段必须写 `engine`（`uma`、`mace`、`dpa` 或 `grace`；`fairchem` 等同于
  `uma`）。
- `[profile:NAME]` 段把一组计算参数打包在一起，不包含模型。
- 以计算类型命名的段（`[sp]`、`[md]`、`[neb]` 等）只对该类型生效；`[general]`、
  `[resources]`、`[output]`、`[safety]` 对所有类型生效。

## INCAR 风格的输入文件

`mliport run --incar INCAR.mliport` 读取一个仿照 VASP INCAR 写法的参数文件。只是格式像，
内容和 DFT 无关。

```text
CALC_TYPE = SP
MODEL_TYPE = MACE
MODEL_PATH = /models/mace-omat-0-medium.model
DEVICE = cpu                        # explicit cuda / cuda:0 for GPU runs
```

可以让 mliport 生成模板，后端自己填：

```bash
mliport template sp --engine mace --output INCAR.mace
mliport template sp --output INCAR.neutral   # backend-neutral, MODEL_TYPE=REQUIRED
```

## 后端专用的选项

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

严格模式下，把这些选项用在别的后端上会报错。

## 色散校正（DFT-D3）

MLIP 模型本身不含色散校正。用 PBE 数据训练的模型（MACE 的 `omat`、UMA 的 `omat` 和大多数
材料模型）学到的是 PBE 势能面，PBE 不含长程色散。`--d3` 会在模型结果上加上 Grimme DFT-D3 的
能量、力和应力，所有后端都能用，实现来自 `torch-dftd`：

| 选项 | 默认值 | 作用 |
|---|---|---|
| `D3` / `--d3` | 关闭 | 在模型之上叠加 DFT-D3 |
| `D3_XC` / `--d3-xc` | `pbe` | 模型训练数据的参考泛函，决定 D3 参数 |
| `D3_DAMPING` / `--d3-damping` | `bj` | 阻尼函数：`zero`、`bj`、`zerom` 或 `bjm` |

```bash
uv pip install --python .venv-mace/bin/python torch-dftd
mliport opt slab.vasp --model model.model --model-type mace --task bulk --d3
```

只有训练数据不含色散的模型才能加 D3，并且要用 `--d3-xc` 写明训练数据的泛函，因为 mliport
没法从 MACE、DPA、GRACE 的模型文件里读出这个信息。UMA 各 task 的参考泛函是已知的：只有
`omat`（PBE）和 `oc20`（RPBE）可以加 D3；`omol`、`omc`、`odac`、`oc25` 的参考已经含色散或
无法确认不含，加 D3 会报错。只写了 `D3_XC` 或 `D3_DAMPING` 却没开 `D3`，也会报错，避免记录
一个实际没生效的设置。NEB 暂不支持 D3。TUI 配置页里的 DFT-D3 开关对应同样的三个选项（NEB
的配置页不显示这个开关）。

是否加了 D3 会记录下来：JSON 结果的模型信息里有 `dispersion` 字段，OUTCAR 文件头里有一行
`Dispersion:`。`torch-dftd` 依赖 PyTorch，MACE、DPA 和 UMA 的环境里都已经有了。

## 科学参数的默认值

温度、步长、步数、恒温器、优化器、`fmax`、NEB 参数等都有内置默认值，按计算类型区分，每次
运行实际用的值都记录在 `resolved_config.json` 里。完整列表可以这样查看：

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

NEB 的必需输出不能关闭，关闭会报错。

## 环境变量

| 变量 | 作用 |
|---|---|
| `MLIPORT_SETTINGS` | 明确指定 `settings.ini` 路径 |
| `MLIPORT_JOBS_DIR` | 队列与作业注册表目录 |
| `MLIPORT_CANCEL_GRACE_SECONDS` | 取消作业时的宽限时间 |
| `MLIPORT_LEASED_GPU_UUID` | 分配给队列 worker 的 GPU |
| `CUDA_VISIBLE_DEVICES` | 可见的 GPU；队列分配 GPU 时会设置它，不要在运行中的作业里修改 |

## TUI

`mliport tui` 提供同样的配置项，同样要求先选定后端和模型才能运行。TUI 和命令行通过同一套
代码提交作业，两边的行为一致。分析页包含所有 `analyze` 任务和常用参数；不常用的参数（例如
GEMDAT 自动识别位点）只能在命令行里设置。
