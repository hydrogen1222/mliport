# 故障排查

mliport 的大多数报错是故意设计的：在信息不足、只能靠猜的地方，程序宁可停下来，也不给出一个
看起来正常但其实错误的结果。所以遇到报错，通常是补上缺的信息，很少需要去关掉某个检查。下面
每一条都按"现象、常见原因、怎么确认、怎么解决"来写。

## 安装

**安装脚本还没开始装就退出了。** 可能是后端名写错了，`--python` 指定的版本不被某个后端支持，或
者在没有可用 GPU 的机器上用了 `--device cuda`。加上 `--dry-run` 再跑一次，它打印的计划和报错
会指出是哪一条不满足。Python 版本要所有后端都支持（UMA 要 3.11 及以上），设备要是机器上真有
的。见[安装](installation.md)。

**启动脚本不存在，或者指向了错的环境。** 安装没有跑完，或者 mliport 是手动装的。用 `ls bin/` 和
`readlink -f bin/mliport-mace` 看它指向哪里。重新运行该后端的安装即可，不要手动改启动脚本。

## 设备与 GPU

**指定了 `--device cuda`，结果跑在 CPU 上，或者一开始计算就崩溃。** 要么框架没有编译适合你这块
GPU 的代码，要么 GPU 不可见。运行 `mliport doctor --engine <engine> --device cuda` 和
`nvidia-smi` 可以区分是哪种情况。请用安装脚本为你的 GPU 选的版本，并且不要在排队的作业里修改
`CUDA_VISIBLE_DEVICES`。

**运行记录里写着 `actual_device: unknown`。** 后端没有报告它用的是哪个设备，或者运行退回到了
别的设备上。这条记录不能算作 GPU 上的结果。先解决可见性问题再重跑，之后再写方法部分。

## 模型与配置

**找不到模型，或者校验和对不上。** 路径错了、下载不完整，或者是另一个模型文件。把 SHA-256 和
`validation/science/model_manifest.json` 对照一下（`mliport doctor` 能检查的部分会报告），然后从
上游重新下载。不要把别的模型文件改名顶替。

**`No MLIP backend was selected.`** 没有默认后端。加 `--model-type mace|dpa|grace|uma`，或者在
`[model:...]` 别名里写明 `engine`。

**`Strict config validation failed` 或 `not applicable to engine`。** 参数名拼错了，或者用了别的
后端的参数。报错会指出参数名和它的来源，`mliport config explain <KEY>` 会给出正确写法。改掉这个
参数即可。只有确实想把这类错误降为警告时，才用 `--lenient-config`（或 `strict_config = false`）。

**DPA 模型要求指定 `HEAD`。** 这个模型有多个分支，而你没有指定用哪个。`dp --pt show <model>
model-branch` 会列出所有分支，用 `--head` 传入你要的那个。mliport 不会替你选一个"默认"分支，
所以也不会出现用错分支算出结果的情况。

**应力或 `--cell-opt` 被拒绝。** 加载的模型不支持应力。`mliport doctor --model <model> --engine
<engine>` 会报告 `has_stress`。换一个支持应力的模型，或者改为固定晶胞计算。程序不会用力去估算
应力。

**改了模型或参数，结果却没变。** 可能复用了已有的运行目录或分析 ID。把 `resolved_config.json`
和结果里的模型信息与你的设置对照一下；重算分析时加 `--force`，或者换一个新的 `--output`。模型
变了之后，mliport 不会继续用旧的计算器。

## 运行

**MD 中途停止：力超过 `--fmax-abort`、能量出现 NaN、温度失控。** 常见原因有步长太大、结构没有
预先优化、电荷或自旋不对，或者体系超出了模型的适用范围。查看停止前后的 `raw/md.csv` 和
`vasp/OUTCAR`，停止时的阈值写在记录里。先预优化结构、减小步长、检查 task，然后用记录下来的随机
种子重跑。调高 `--fmax-abort` 只会掩盖问题。

**NEB 不收敛。** 常见原因是端点没有优化好或者两端不匹配、原子对应关系错了、image 太少或步数太
少。查看能带能量和 `run_context.json`，修正端点或原子对应关系，增加 image 或步数，或者调整弹簧
常数。找到的鞍点只是候选，要确认是真正的过渡态，还需要另外做 Hessian 分析。

**NEB 被拒绝："not verifiably the gradient of its energy"。** NEB 在第一步之前会在你的路径上用
有限差分检查：力是否等于能量的负梯度。被拒绝的原因可能是力和能量不一致（模型直接预测力）、出现
了非有限值，或者能量精度不够、无法判断（例如很大的 float32 晶胞，可以改用 float64 或更小的晶胞）。
报错信息会给出是哪个 image、哪个方向、误差多大，完整记录在
`NEBGradientConsistencyError.summary` 里。

**NEB 续算提示 "Resume fingerprint is incompatible"。** 续算会在原目录里沿用原来的参数，步数上限
也是其中之一。续算时改 `--max-steps` 会被拒绝；去掉这个参数，就会用原来记录的值。

## 分析

**`analyze` 提示缺少参数。** `transport` 需要 `--mobile`、`--charge` 和 `--fit-start-ps`；如果默认
的时间间隔网格会占用过多内存，还需要明确给出 `--lag-step-ps` 和 `--lag-stop-ps`。`electrolyte`
需要 `--sites` 或 `--discover-sites-from-density` 二选一。`rdf` 需要 `--center` 和 `--neighbor`，
`msd` 需要 `--mobile`。

**输运分析拒绝拟合，或者 `D` 的误差大得离谱。** 扩散区间里保存的帧太少、帧间隔设置不对，或者
漂移校正掩盖了信号。结果 JSON 里有时间间隔网格、拟合窗口和采样诊断信息。解决办法是跑得更长、
写明帧间隔，并在明确的扩散区间上重新拟合。不要为了得到一个数而降低采样要求。

**GEMDAT 报 `backend_error`。** 这表示 GEMDAT 本身运行失败了（位点识别、密度参数或缺少依赖），
并不代表"没有发生跳跃"。看 `results.json` 里的报错，试试明确指定 `--sites`，修正位点定义或环境。
保留 `backend_error` 这条记录，不要据此讨论扩散机理。

**GEMDAT 在长轨迹上非常慢。** 在 1 ns、400 个原子的 LGPS 轨迹上识别位点很耗时。可以只分析其中
一段，并明确给出 `--frame-interval-fs`，或者多等一会儿。

## 全新安装时遇到过的问题（2.0.0b3）

下面这些问题是在一台全新的 V100 机器上遇到的，在 2.0.0b3 之前已经修好。列在这里，方便你看到
类似报错时对号入座。

**"no local Python 3.10-3.13 runtime with packaging"。** 安装脚本需要一个 3.10–3.13 的 Python
来制定安装计划。现在它会用 uv 管理的 Python；如果其中没有 `packaging`，就临时用
`uv run --with packaging` 运行。如果你自己设置了 `MLIPORT_INSTALL_PYTHON`，它必须是 3.10–3.13
并且能导入 `packaging`，可以用 `uv python install 3.12` 装一个。

**`--engines mace dpa grace uma` 被当成多余参数。** 已修复：`--engines` 现在既接受空格分隔，也接受
逗号分隔。

**doctor 通过了，但 `analyze transport` 或 `electrolyte` 用不了。** 已修复：安装脚本现在会安装
`mliport[analysis-all]`，包括 scipy、matplotlib、kinisi 和 gemdat。重新运行安装脚本即可，不要手动
补装。

**DPA 报 `No package metadata was found for mpich`。** `deepmd-kit` 通过它的 `torch` 依赖读取
MPICH 的信息。DPA 环境现在会装上对应版本的 `mpich`，重装 DPA 即可：

```bash
./scripts/install_mliport.sh --engines dpa --device auto --clean
```

**DPA 或 GRACE 提示 "requires process-level device isolation"。** 没有设置 `CUDA_VISIBLE_DEVICES`
时，命令行会新开一个设好该变量的进程来运行 DPA 和 GRACE，所以文档里的命令可以直接用。Python
API 没法重启调用它的程序，请在导入这些框架之前设置 `CUDA_VISIBLE_DEVICES=<uuid>`（用 CPU 时设为
`""`）。GPU 的 UUID 可以用 `nvidia-smi --query-gpu=uuid --format=csv,noheader` 查到。

**在脚本里启动 `mliport tui` 时卡住。** TUI 需要交互式终端，现在没有终端时会直接报错退出。无界面
的自动化请用 Python API。

**通过 SOCKS 代理从 Hugging Face 下载失败。** 在下载用的环境里安装 `httpx[socks]`，并且
`NO_PROXY` 里不要有带方括号的地址；见[模型](models.md)。
