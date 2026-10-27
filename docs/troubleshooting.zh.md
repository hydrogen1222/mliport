# 故障排查

mliport 的大多数报错都是有意为之：在猜测会给出一个看上去正确的错误答案的地方，它选择停下。所以
解决办法几乎总是补上缺失的信息，而不是去找一个能把检查关掉的开关。下面每一条都给出现象、常见
原因、怎么确认，以及该怎么做。

## 安装

**安装器还没装任何东西就退出了。** 原因可能是引擎名未知、`--python` 的版本被所请求的某个后端
拒绝，或是在没有受支持 GPU 的机器上用了 `--device cuda`。加上 `--dry-run` 再运行一次，它打印
的计划和错误会点明是哪条约束。请选一个所有被请求后端都接受的 Python 版本（UMA 需要 3.11 或
更高），以及机器上真有的设备。见[安装](installation.md)。

**启动器缺失，或指向了错误的环境。** 安装器没有跑完，或者 mliport 是手动装的。`ls bin/` 和
`readlink -f bin/mliport-mace` 能看出它指向哪里。请为该引擎重新运行安装器，不要去改启动器。

## 设备与 GPU

**明明要了 `--device cuda`，后端却跑在 CPU 上，或者第一个核函数就崩溃。** 要么框架的 wheel 里
没有适合你 GPU 架构的核函数，要么 GPU 不可见。`mliport doctor --engine <engine> --device
cuda` 和 `nvidia-smi` 会告诉你是哪一种。请使用安装器为你的架构固定的 wheel 路线，并且不要在排
队作业里覆盖 `CUDA_VISIBLE_DEVICES`。

**某次运行记录了 `actual_device: unknown`。** 后端没有暴露它用的是哪个设备，或者运行回退了。请
把这条记录当作未归属到 GPU；先修好可见性并重新运行，再写下方法学上的结论。

## 模型与配置

**找不到模型，或校验和对不上。** 路径错误、下载不完整，或者是另一个检查点。把 SHA-256 与
`validation/science/model_manifest.json` 对照（`mliport doctor` 会报告它能报告的部分），再从上游
重新下载。千万不要把另一个检查点改名来凑数。

**`No MLIP backend was selected.`** 没有默认后端。请传 `--model-type mace|dpa|grace|uma`，或在
`[model:...]` 别名里声明 `engine`。

**`Strict config validation failed` 或 `not applicable to engine`。** 键拼错了，或者给了属于另一个
后端的选项。报错会点出键名和它的来源，`mliport config explain <KEY>` 会给出规范写法。请修正这
个键。只有当你确实想把错误降为警告时，才用 `--lenient-config`（或 `strict_config = false`）。

**DPA 检查点要求一个 `HEAD`。** 检查点有多个分支而你没说是哪一个。`dp --pt show <model>
model-branch` 会列出它们，用 `--head` 传入你要的那个。mliport 在这里会失败，绝不会悄悄选一个
"默认"head，所以不会因此产出一个来自错误 head 的结果。

**应力，或者 `--cell-opt`，被拒绝。** 加载的检查点没有实现应力。`mliport doctor --model <model>
--engine <engine>` 会报告 `has_stress`。请换一个支持应力的检查点，或者固定晶胞运行。应力张量绝
不会由力凭空造出来。

**改了模型或选项，好像毫无效果。** 复用了已有的运行目录或分析 ID。把 `resolved_config.json` 和结
果里的模型标识与你的请求对照一下；重算分析用 `--force`，或者写到新的 `--output`。mliport 不会在
模型标识改变后复用同一个计算器。

## 运行

**MD 运行中止：力超过 `--fmax-abort`、能量为 NaN、温度失控。** 时间步太大、结构没有弛豫、电荷
或自旋态不一致，或者模型被用在了它的适用范围之外。看看中止前后的 `raw/md.csv` 和
`vasp/OUTCAR`；中止记录里写着阈值。请先预弛豫、缩短时间步、检查 task，然后带着记录下来的随机
种子重跑。调高 `--fmax-abort` 只是把问题藏起来。

**NEB 不收敛。** 常见原因是端点没有弛豫或彼此不匹配、原子映射错误、镜像太少或步数太少。看看能
带能量和 `run_context.json`。修正端点或映射，增加镜像或步数，或者调整弹簧常数。鞍点候选不等于经
过验证的过渡态；用 Hessian 确认是单独的一步。

**NEB 被拒绝："not verifiably the gradient of its energy"。** 第一步之前，NEB 会在你的能带上把模型
的力与其能量的有限差分相比较。被拒绝说明力与能量不一致（直接预测力的模型）、某个值
不是有限数，或者能量分辨率太粗而无法判断（大的 float32 晶胞：请用 float64 或更小的晶胞）。报错信
息给出镜像、方向和误差，完整记录在 `NEBGradientConsistencyError.summary` 里。

**NEB 续跑提示 "Resume fingerprint is incompatible"。** 续跑会在原来的运行目录里沿用记录下来的选
项，步数预算是这个身份的一部分。续跑时改 `--max-steps` 会被拒绝，去掉它就会使用记录下来的预算。

## 分析

**`analyze` 要求补充选项。** `transport` 需要 `--mobile`、`--charge` 和 `--fit-start-ps`，当默认滞后时
间网格会越过内存保护线时，还需要显式的 `--lag-step-ps` 和 `--lag-stop-ps`。`electrolyte` 需要
`--sites` 或 `--discover-sites-from-density` 二选一。`rdf` 需要 `--center` 和 `--neighbor`；`msd`
需要 `--mobile`。

**输运拒绝拟合，或者 `D` 的不确定度大得离谱。** 扩散区间里保存的帧太少、保存帧间隔不对，或者漂
移策略掩盖了信号。结果 JSON 里有滞后时间网格、拟合窗口和采样诊断。请运行得更长、写明帧间隔，
并在显式的扩散窗口上重新拟合。不要为了挤出一个数字而降低采样要求。

**GEMDAT 报告 `backend_error`。** GEMDAT 后端失败了（位点映射、密度参数、缺少依赖）。这不是
"零次跳跃"。请读 `results.json` 里的错误，试试显式的 `--sites`，并修正定义或环境。保留
`backend_error` 记录，而不是去报告一个机理。

**GEMDAT 在长轨迹上耗时极长。** 在 1 ns、400 原子的 LGPS 运行上发现位点很贵。请分析其中的一段并
显式给出 `--frame-interval-fs`，或者给它更多时间。

## 全新安装的注意事项（2.0.0b3）

这些是在一台干净的 V100 机器上遇到、并在 2.0.0b3 之前修好的问题。写在这里，是为了让你碰到时能认
出这条信息。

**"no local Python 3.10-3.13 runtime with packaging"。** 安装器需要一个 3.10–3.13 的解释器来规划安
装。现在它会选 uv 管理的那个；如果它没有 `packaging`，就在一次性的 `uv run --with packaging` 环
境里运行规划器。如果你自己设置了 `MLIPORT_INSTALL_PYTHON`，它必须是 3.10–3.13 且能导入
`packaging`；`uv python install 3.12` 可以得到一个。

**`--engines mace dpa grace uma` 被当作多余参数拒绝。** 已修复：`--engines` 同时接受空格分隔和逗号分
隔两种写法。

**doctor 通过了，但 `analyze transport` 或 `electrolyte` 不可用。** 已修复：安装器现在会安装
`mliport[analysis-all]`，其中包含 scipy、matplotlib、kinisi 和 gemdat。请重新运行安装器，不要手动补
包。

**DPA 因 `No package metadata was found for mpich` 失败。** `deepmd-kit` 通过它的 `torch` extra 读取
MPICH 元数据。DPA 环境现在会装上匹配的 `mpich`，所以重装 DPA 就够了：

```bash
./scripts/install_mliport.sh --engines dpa --device auto --clean
```

**DPA 或 GRACE 提示 "requires process-level device isolation"。** 当 `CUDA_VISIBLE_DEVICES` 未设置
时，CLI 会在设好它的新进程里重启 DPA 和 GRACE 命令，所以文档里那条各环境对应的命令可以直接用。
Python API 无法重启调用者：请在导入这些框架之前 export `CUDA_VISIBLE_DEVICES=<uuid>`（CPU 用
`""`）。GPU 的 UUID 可以用 `nvidia-smi --query-gpu=uuid --format=csv,noheader` 得到。

**`mliport tui` 在脚本里启动时卡住。** TUI 需要交互式终端，没有时现在会报错退出。无界面的自动化请
使用 Python API。

**在 SOCKS 代理后面下载 Hugging Face 失败。** 在下载环境里安装 `httpx[socks]`，并且不要在
`NO_PROXY` 里放带方括号的条目；见[模型](models.md)。
