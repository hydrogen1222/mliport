# 安装

四个后端（MACE、DPA、GRACE、UMA）各自依赖不同版本的深度学习框架，放在同一个环境里经常
冲突。所以安装脚本给**每个后端单独建一个环境**，每个环境里各装一份 mliport。`.venv-mace`
里的 `mliport` 只能跑 MACE，`.venv` 里的只能跑 UMA，其余类推。需要哪个后端就装哪个，用的
时候从对应环境里调用 `mliport`。

## 安装脚本

先装好 [uv](https://docs.astral.sh/uv/)，再克隆仓库：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hydrogen1222/mliport
cd mliport
```

有 NVIDIA GPU 且驱动正常的机器（在 V100-SXM2-16GB、驱动 580.173.02 上测试过）：

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

没有 GPU 的机器：

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

只要部分后端时，`--engines` 后面少写几个就行。`--dry-run` 只打印要执行的步骤，不真正安装。
安装结束时脚本会在每个环境里运行一次 `mliport doctor`（加 `--skip-doctor` 可以跳过）。分析、
输运和电解质相关的依赖会一起装好，不用另外安装。

安装脚本还会在 `./bin/` 里生成几个启动脚本：

```bash
./bin/mliport-mace sp structure.cif --model model.model --model-type mace
./bin/mliport-uma  sp structure.cif --model uma.pt --model-type uma
```

启动脚本只负责选对 Python 环境，模型、精度等计算参数仍要你自己指定。

## 哪个环境跑什么

<!-- BEGIN GENERATED: backend-matrix -->
| 后端 | 环境 | Python | 运行时 |
|---|---|---|---|
| UMA (FAIRChem) | `.venv` | 3.11, 3.12, 3.13 | `fairchem-core==2.21.0` (torch 2.8.0) |
| MACE | `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |
| DPA (DeepMD-kit) | `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit==3.1.3` (torch 2.10.0) |
| GRACE | `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |
<!-- END GENERATED -->

`Python` 一列是该后端可用的 Python 版本，取 mliport、后端本身和所需框架三者支持范围的交集。
这张表由兼容性注册表（`mliport/mliport/install/compatibility.py`）自动生成，具体的版本锁定
也写在那里。

可以用 `--python` 指定解释器版本，但它必须同时满足所选的*每一个*后端。像
`--python 3.10 --engines uma` 这种装不上的组合，会在开始安装前就报错，脚本不会自作主张换
版本。

## 检查安装

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
.venv-dpa/bin/mliport  doctor --engine dpa  --device auto
.venv-grace/bin/mliport doctor --engine grace --device auto
.venv/bin/mliport       doctor --engine uma  --device auto
```

必需的检查都通过时，命令以 0 退出，最后一行类似 `Environment checks passed for MACE on CUDA.`。
报告里还会列出可选功能（TUI、分析、输运、电解质、绘图）是否可用。想顺便用真实模型试算一次，
加上 `--model PATH --task TASK --structure FILE`。`doctor --json` 输出同样内容的 JSON，方便在
集群上做自动化部署。

## 运行 mliport

从对应后端的环境里调用命令：

```bash
.venv-mace/bin/mliport  sp structure.cif --model model.model --model-type mace
.venv-dpa/bin/mliport   sp structure.cif --model model.pt   --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model saved_model --model-type grace
.venv/bin/mliport       sp structure.cif --model uma.pt     --model-type uma
```

`./bin/` 里的启动脚本执行的就是这些命令。Windows 上每个环境里有一个 `Scripts\mliport.exe`，
安装脚本也会在 `bin\` 里生成 `mliport-*.cmd`：

```bat
.venv-mace\Scripts\mliport.exe sp structure.cif --model model.model --model-type mace
```

### DPA、GRACE 与 GPU

DPA 和 GRACE 的框架一旦加载，就不能再换 GPU。如果没有设置 `CUDA_VISIBLE_DEVICES`，命令行
会新开一个进程，只让你指定的那块 GPU 可见，再重新执行同一条命令，并打印一行说明。Python
API 没法重启调用它的程序，所以在 Python 里用 DPA 或 GRACE 时，请在导入框架之前自己设置
`CUDA_VISIBLE_DEVICES`（设为 `""` 表示用 CPU）。

## 模型

mliport 不负责下载模型。请用上游工具下载模型文件，再用 `--model` 传入路径；本地文件离线也
能用。下载地址、验证时用到的文件名和 SHA-256 见[模型](models.md)。

如果要通过 SOCKS 代理从 Hugging Face 下载，请在用来下载的环境里装 `httpx[socks]`，并且
`NO_PROXY` 里不要有 `[::1]` 这种带方括号的写法，httpx 不认。设置
`HF_ENDPOINT=https://hf-mirror.com` 可以改用镜像站。安装脚本通过 `uv` 联网，读取的是同样的
代理变量。

## GPU 架构

PyTorch 和 TensorFlow 针对不同代的 GPU 提供不同的 CUDA 版本，安装脚本会自动选择：
Maxwell、Pascal、Volta 用 cu126，Turing 及更新的用 cu128 或更高。`mliport setup` 会打印检测
到的 GPU 和准备采用的安装方案。

<!-- BEGIN GENERATED: gpu-routes -->
| 引擎 | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| MACE | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| DPA | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| GRACE | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 实验性 |
<!-- END GENERATED -->

表中的标记含义：`实验性` 表示上游框架不测试这一代 GPU；`需运行时冒烟测试` 表示有可用的安装
方案，但 mliport 还没在这类硬件上实际跑过。这两种都还没验证。目前只有 V100 有实测记录，其
他型号的分类是检查 wheel 和原生扩展得出的（编译进去的架构列表、`+cu126`/`+cu128` wheel 的
哈希、`cuobjdump` 输出），详见
[架构报告](https://github.com/hydrogen1222/mliport/blob/main/validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md)。
`mliport doctor` 会对检测到的 GPU 打印这个分类。

Turing 之前的 GPU 依赖 CUDA 12.6 版本，而 PyTorch 从 2.15 起不再提供。这类机器上请保持安装
脚本装的版本，不要升级框架。

## 离线机器

用 `--source offline` 指向事先准备好的 wheel 缓存，或者在联网机器上装好再把环境拷过去。需要
拷贝的只有 uv 的缓存（`UV_CACHE_DIR`）和模型目录。设置 `UV_PYTHON_DOWNLOADS=never` 可以禁止
安装脚本下载 Python 解释器。

## 磁盘占用与卸载

四个后端都装 GPU 版大约占 25–35 GB，大部分是 PyTorch 和 TensorFlow；CPU 版小一些。要删掉
某个后端，直接删除它的环境目录（`.venv-mace`、`.venv-dpa`、`.venv-grace` 或 `.venv`）即可，
仓库和模型文件不受影响。`./bin/mliport-*` 启动脚本可以连同 `bin/` 目录一起删。需要时重新
运行安装脚本即可恢复。

## 手动安装

每个后端环境里都要装 mliport 和该后端的依赖。准确的版本锁定在
`mliport/mliport/install/compatibility.py`，安装脚本会按你的 GPU 型号选对版本，所以推荐用脚本
安装。如果一定要手动装：每个后端建一个环境，在每个环境里都装一次 `./mliport`，然后运行
`mliport doctor`，全部检查通过后再开始计算。只在一个环境里装 mliport，其他环境是用不了的。
