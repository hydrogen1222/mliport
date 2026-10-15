# 安装

mliport 不是装在一个大环境里的。每个后端（MACE、DPA、GRACE、UMA）都带着自己的
深度学习依赖，这些依赖彼此并不总能和平共处，所以安装脚本会为**每个后端单独建一
个环境**，再把 mliport 装进每个环境里。在 `.venv-mace` 里运行的 `mliport` 只能
跑 MACE，`.venv` 里的只能跑 UMA，以此类推。需要哪些后端就装哪些，然后从对应的
环境里调用 `mliport`。

## 安装脚本

先装好 [uv](https://docs.astral.sh/uv/)，再克隆仓库：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hydrogen1222/mliport
cd mliport
```

有 NVIDIA GPU 且驱动正常的机器（这条路线在 V100-SXM2-16GB、驱动 580.173.02 上
测试过）：

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

没有 GPU 的机器：

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

只需要部分后端时少写几个 engine 即可。`--dry-run` 只打印每一步而不执行。除非
加了 `--skip-doctor`，安装脚本结束时会在每个环境里跑一遍 `mliport doctor`。分
析、输运和电解质相关功能随同一次安装一起装好，不需要第二步。

安装脚本还会在 `./bin/` 里写几个小启动器：

```bash
./bin/mliport-mace sp structure.cif --model model.model --model-type mace
./bin/mliport-uma  sp structure.cif --model uma.pt --model-type uma
```

启动器只负责选 Python 环境，不会替你设置模型、dtype 或任何科学选项。

## 哪个环境跑什么

<!-- BEGIN GENERATED: backend-matrix -->
| 后端 | 环境 | Python | 运行时 |
|---|---|---|---|
| UMA (FAIRChem) | `.venv` | 3.11, 3.12, 3.13 | `fairchem-core==2.21.0` (torch 2.8.0) |
| MACE | `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |
| DPA (DeepMD-kit) | `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit==3.1.3` (torch 2.10.0) |
| GRACE | `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |
<!-- END GENERATED -->

`Python` 一列是安装脚本对该后端接受的版本：mliport 支持的 Python 版本、后端自己
的 `requires-python`、以及它所需框架 wheel 的 Python 范围，三者的交集。这张表由
兼容性注册表（`mliport/mliport/install/compatibility.py`）生成，精确的版本钉
(pin) 也在那里。

你可以用 `--python` 强制指定解释器，但它必须满足你所请求的*每一个*后端。不可能
成功的组合，例如 `--python 3.10 --engines uma`，会在安装任何东西之前就被拒绝；
安装脚本绝不会悄悄换成别的版本。

## 检查安装

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
.venv-dpa/bin/mliport  doctor --engine dpa  --device auto
.venv-grace/bin/mliport doctor --engine grace --device auto
.venv/bin/mliport       doctor --engine uma  --device auto
```

必需检查全部通过时命令以 0 退出，并以类似
`Environment checks passed for MACE on CUDA.` 的一行结尾。报告里还会列出可选功
能（TUI、分析、输运、电解质、绘图）。想用真实模型试一下，加上
`--model PATH --task TASK --structure FILE`。`doctor --json` 给出同样内容的机器可
读版本，方便集群自动化部署。

## 运行 mliport

从拥有该后端的环境里调用命令行：

```bash
.venv-mace/bin/mliport  sp structure.cif --model model.model --model-type mace
.venv-dpa/bin/mliport   sp structure.cif --model model.pt   --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model saved_model --model-type grace
.venv/bin/mliport       sp structure.cif --model uma.pt     --model-type uma
```

`./bin/` 里的启动器执行的正是这些命令。在 Windows 上，每个环境里有一个
`Scripts\mliport.exe`，安装脚本也会在 POSIX 启动器旁边写 `bin\mliport-*.cmd`：

```bat
.venv-mace\Scripts\mliport.exe sp structure.cif --model model.model --model-type mace
```

### DPA、GRACE 与 GPU

DPA 和 GRACE 的框架一旦加载，就无法再移动 CUDA 上下文。当 `CUDA_VISIBLE_DEVICES`
没有设置时，命令行会用你指定的那块 GPU 可见的状态在新进程里重启同一条命令，并
打印一行简短说明。Python API 没办法重启调用它的程序，所以在 Python 里使用 DPA
或 GRACE 时，请在导入框架之前设置好 `CUDA_VISIBLE_DEVICES`（用 `""` 表示 CPU）。

## 模型

mliport 不下载模型。请用上游工具取得模型文件，再用 `--model` 传入路径；本地文件
离线也能用。模型去哪里拿、验证套件用到的文件名和 SHA-256 在[模型](models.md)页面。

如果通过 SOCKS 代理从 Hugging Face 下载，请在你用来下载的环境里装
`httpx[socks]`，并且让 `NO_PROXY` 里不要出现 `[::1]` 这类带方括号的条目，httpx
不接受它们。`HF_ENDPOINT=https://hf-mirror.com` 可以选用一些站点使用的 Hugging
Face 镜像。安装脚本本身走 `uv`，读取的是同样的代理变量。

## GPU 架构

PyTorch 和 TensorFlow 为不同代的 GPU 提供不同的 CUDA 构建，安装脚本会替你选对：
Maxwell、Pascal、Volta 走 cu126 旧通道，Turing 及更新的走 cu128 或更高。
`mliport setup` 会打印它检测到的 GPU 和将采用的路线。

<!-- BEGIN GENERATED: gpu-routes -->
| 引擎 | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| MACE | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| DPA | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| GRACE | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 实验性 |
<!-- END GENERATED -->

请仔细看这些格子：`实验性` 表示上游框架不测试该 GPU 家族，`需运行时冒烟测试` 表
示路线存在，但 mliport 还没有在这类硬件上留下真实运行记录。两者都不等于"已验
证"。只有 V100 有硬件证据，其他家族是通过仔细检查 wheel 和原生扩展得出的分类：
编译进去的架构列表、`+cu126`/`+cu128` wheel 的哈希、`cuobjdump` 输出，都在
[架构报告](https://github.com/hydrogen1222/mliport/blob/main/validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md)
里。`mliport doctor` 会对它找到的 GPU 打印这个分类，而不是声称"已验证"。

Turing 之前的 GPU 依赖 CUDA 12.6 通道，而 PyTorch 从 2.15 起不再发布它。在这些环
境里请保持安装脚本给出的版本，不要升级框架。

## 离线机器

用 `--source offline` 指向准备好的 wheel 缓存，或者先在联网机器上装好再把环境拷
过去。需要搬运的只有两样：uv 的缓存（`UV_CACHE_DIR`）和模型目录。设置
`UV_PYTHON_DOWNLOADS=never` 可以让安装脚本不去下载托管的解释器。

## 磁盘占用与卸载

四个后端的 GPU 安装大约占 25–35 GB，主要是 PyTorch 和 TensorFlow；CPU 安装小一
些。要移除某个后端，直接删除它的环境目录（`.venv-mace`、`.venv-dpa`、
`.venv-grace` 或 `.venv`）即可。仓库和你的模型文件不受影响，`./bin/mliport-*` 启
动器随 `bin/` 目录一起删除。想恢复时重新运行安装脚本。

## 手动安装

每个后端环境都需要 mliport 本身加上该引擎的依赖。权威的版本钉 (pin) 在
`mliport/mliport/install/compatibility.py` 里，只有安装脚本才能让它们和你的 GPU
架构保持一致。如果你仍然要手动装：每个后端建一个环境，把 `./mliport` 装进每个
环境，然后在里面运行 `mliport doctor`，全部检查通过之前不要相信结果。只装一次
mliport 就指望它能进到其他环境里，是行不通的。
