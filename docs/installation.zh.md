# 安装

## 为什么每个后端一个环境？

机器学习原子间势依赖深度学习框架（PyTorch、TensorFlow、JAX），而这些框架会锁定特定版本的 CUDA 库、NumPy 等底层包。当两个后端需要*不同*版本的同一个库时（MACE 需要 PyTorch 2.8，DPA 需要 2.11，GRACE 运行在 TensorFlow 上），把它们放在同一个环境中会产生无法解决的版本冲突。

mliport 的解决方案是**为每个后端构建独立的虚拟环境**。每个环境包含 mliport 本身以及该后端的依赖栈，其中的 `mliport` 命令只能调用对应的后端。这确保了每条依赖链的清洁和每次运行的可重现性。

| 后端 | 环境 | 可运行的内容 |
|---|---|---|
| UMA (FAIRChem) | `.venv` | 仅 UMA 检查点 |
| MACE | `.venv-mace` | 仅 MACE 检查点 |
| DPA (DeePMD-kit) | `.venv-dpa` | 仅 DPA 检查点 |
| GRACE | `.venv-grace` | 仅 GRACE 检查点 |
| NEP (Calorine / GPUMD) | `.venv-nep` | 仅 NEP 势函数 |

你只需要安装自己使用的后端。如果只用 MACE，就只安装 MACE。

## 前置条件

运行安装器之前需要两样东西：

1. **uv** — 一个快速的 Python 包管理器，集虚拟环境管理和依赖解析于一体，比 `pip` + `venv` 快得多：

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. **mliport 仓库**：

    ```bash
    git clone https://github.com/hydrogen1222/mliport
    cd mliport
    ```

在 GPU 机器上还需要可用的 NVIDIA 驱动。安装器在 V100-SXM2-16GB（驱动 580.173.02）上经过测试。

## 运行安装器

### GPU 机器（推荐）

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

`--device auto` 会自动检测 GPU 并选择对应的 CUDA wheel 通道。你可以只安装部分后端——`--engines mace dpa` 会跳过 GRACE 和 UMA。

### 仅 CPU 机器

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

CPU 安装体积更小（无 CUDA 库），适用于所有机器，但推理速度较慢。适合后处理、分析和笔记本电脑上的测试。

### Windows（PowerShell）

`scripts/install_mliport.sh` 是 Bash 脚本，不能直接在原生 PowerShell 中运行。
在仓库根目录下，使用 Python 原生安装器：

```powershell
$env:PYTHONPATH = Join-Path $PWD "mliport"
uv run --no-project --with packaging --python 3.12 -- python -m mliport.install --engines mace --device auto
```

安装器会在 `.venv-*/Scripts/` 下创建 Windows 虚拟环境，并生成原生
`bin/mliport-*.cmd` 启动器。运行 MACE 可以使用
`.venv-mace\Scripts\mliport.exe` 或 `bin\mliport-mace.cmd`。
具体后端能否在 Windows 上构建、使用 GPU，仍取决于各上游项目的支持情况；
存在启动器不代表该后端的 GPU 运行时一定可用。

### 安装器的工作流程

对于你指定的每个引擎，安装器会：

1. 创建虚拟环境（`.venv-mace`、`.venv-dpa` 等）。
2. 以可编辑模式安装 mliport 及分析扩展。
3. 安装该后端的框架（锁定版本）。
4. 在 `./bin/` 下写入启动器脚本（如 `./bin/mliport-mace`）。
5. 运行 `mliport doctor` 验证安装（除非指定了 `--skip-doctor`）。

整个过程是确定性的：同一命令在同一机器上产生相同的环境，因为框架版本锁定在兼容性注册表（`mliport/mliport/install/compatibility.py`）中。

### 常用安装器选项

| 选项 | 作用 |
|---|---|
| `--dry-run` | 打印每个步骤但不执行 |
| `--clean` | 删除并重建指定引擎的环境 |
| `--skip-doctor` | 跳过最后的验证步骤 |
| `--python 3.12` | 强制使用特定 Python 版本（必须满足所有指定的后端） |
| `--source offline` | 使用预先准备的 wheel 缓存 |

## NEP：可选后端

NEP 不在默认引擎列表中，因为其原生 GPU 运行时需要单独编译的 `gpumd` 二进制文件，安装器无法自动构建。

**仅 CPU 的 NEP** 可以直接使用——Calorine ASE 运行时无需编译：

```bash
./scripts/install_mliport.sh --engines nep --device cpu
```

**GPU NEP** 需要你自行用 CUDA Toolkit 12.9（或兼容版本）编译 `gpumd`，然后将其放到 `PATH` 上或设置 `MLIPORT_GPUMD_COMMAND`：

```bash
./scripts/install_mliport.sh --engines nep --device auto
# 结尾的 doctor 检查会失败，直到 gpumd 可用。
# 编译 gpumd 后重新运行安装器，或使用 --skip-doctor
# 并直接调用 .venv-nep/bin/mliport。
```

详见 [NEP](backends/nep.md) 的构建说明和运行时细节。

## 各环境运行什么

<!-- BEGIN GENERATED: backend-matrix -->
| 后端 | 环境 | Python | 运行时 |
|---|---|---|---|
| UMA (FAIRChem) | `.venv` | 3.11, 3.12, 3.13 | `fairchem-core==2.25.0` (torch 2.13.0) |
| MACE | `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |
| DPA (DeepMD-kit) | `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit @ git+https://github.com/deepmodeling/deepmd-kit.git@0edeaef5617e9878edeb2346bf37bb6a035a4f51` (torch 2.11.0) |
| NEP (Calorine / GPUMD) | `.venv-nep` | 3.10, 3.11, 3.12, 3.13 | `calorine==4.1`（NumPy/ASE 由安装器解析；gpumd 需用 CUDA Toolkit 12.9 编译） |
| GRACE | `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |
<!-- END GENERATED -->

`Python` 列显示安装器对该后端接受的版本范围，是 mliport 自身 `requires-python`、后端约束和可用框架 wheel 的交集。此表由兼容性注册表自动生成。

可以用 `--python` 强制指定版本，但它必须满足*所有*指定的后端。不可能的组合如 `--python 3.10 --engines uma`（UMA 需要 3.11+）会在安装前被拒绝。

## 验证安装

安装完成后，验证每个后端：

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
.venv-dpa/bin/mliport  doctor --engine dpa  --device auto
.venv-grace/bin/mliport doctor --engine grace --device auto
.venv/bin/mliport       doctor --engine uma  --device auto
.venv-nep/bin/mliport   doctor --engine nep  --device cpu     # 如果安装了 NEP
```

每个命令在检查通过时返回 0，并报告可选功能（TUI、分析、输运、电解质、绘图）。要用真实模型测试，添加 `--model PATH --task TASK --structure FILE`。机器可读输出用 `doctor --json`，便于集群部署脚本使用。

## 运行 mliport

### 直接使用环境路径

从拥有该后端的环境调用 `mliport`：

```bash
.venv-mace/bin/mliport  sp structure.cif --model model.model --model-type mace
.venv-dpa/bin/mliport   sp structure.cif --model model.pt   --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model saved_model --model-type grace
.venv/bin/mliport       sp structure.cif --model uma.pt     --model-type uma
.venv-nep/bin/mliport   sp structure.cif --model nep.txt    --model-type nep --nep-runtime ase --device cpu
```

### 使用启动器

安装器在 `./bin/` 下生成的便捷脚本会为你选择正确的环境：

```bash
./bin/mliport-mace sp structure.cif --model model.model --model-type mace
./bin/mliport-uma  sp structure.cif --model uma.pt --model-type uma
```

启动器只选择 Python 环境，不设置模型、精度或任何科学选项。

在 Windows 上，每个环境有 `Scripts\mliport.exe`，安装器会生成原生 `bin\mliport-*.cmd` 启动器；POSIX shell 启动器只会在 POSIX 系统上生成。

### DPA 和 GRACE：GPU 设备隔离

DPA（DeePMD-kit）和 GRACE（TensorFlow）的框架一旦加载就无法迁移 CUDA 上下文。当 `CUDA_VISIBLE_DEVICES` 未设置时，CLI 会自动在新进程中重启相同命令并使请求的 GPU 可见，同时打印说明。

Python API 无法重启调用者。如果从 Python 使用 DPA 或 GRACE，需要在导入框架前设置 `CUDA_VISIBLE_DEVICES`：

```bash
export CUDA_VISIBLE_DEVICES=0          # 使用 GPU
export CUDA_VISIBLE_DEVICES=""         # 使用 CPU
```

GPU 的 UUID 可通过 `nvidia-smi --query-gpu=uuid --format=csv,noheader` 获取。

## 模型

mliport 不下载检查点。从上游项目获取所需模型，通过 `--model` 传入路径即可；本地文件无需网络。检查点来源和验证套件使用的锁定文件名见[模型](models.md)页面。

如果通过 SOCKS 代理从 Hugging Face 下载，需在下载环境中安装 `httpx[socks]`，并确保 `NO_PROXY` 中没有方括号条目如 `[::1]`。`HF_ENDPOINT=https://hf-mirror.com` 可选择部分站点使用的 Hugging Face 镜像。

## GPU 架构

PyTorch 和 TensorFlow 为不同 GPU 代际提供不同的 CUDA 构建版本。安装器会检测你的 GPU 并自动选择正确的 wheel 通道：

- **Turing 及更新**（RTX 20xx/30xx/40xx、A100、H100）：使用 `cu128` 或 `cu130`。
- **Volta**（V100）：使用 `cu126`，兼容较旧的驱动。
- **Maxwell 和 Pascal**：`cu126` 遗留通道；上游标记为 `experimental`。

当同时可见多张 NVIDIA GPU 时，安装器会检查所选框架 wheel 是否覆盖**全部**可见 GPU。
如果不存在共同的兼容路线，`--device cuda` 会在创建环境前拒绝执行，
`--device auto` 则回退到 CPU。可以通过 `CUDA_VISIBLE_DEVICES` 只暴露兼容的 GPU 子集。
GRACE 使用的官方 TensorFlow 2.20 wheel 不包含 Maxwell GPU 内核，因此 Maxwell 上的
GRACE 也遵循相同的拒绝/CPU 回退规则；其他仅标记为实验性的架构不会被一概视为不可用。

CUDA 13 通道（`cu130`，UMA 的 torch 2.13 使用）需要 580+ 驱动，且不包含 Turing 以下的核函数，因此 Volta 保持在 `cu126`。

<!-- BEGIN GENERATED: gpu-routes -->
| 引擎 | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| MACE | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| DPA | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 |
| NEP | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 实验性 |
| GRACE | 实验性 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 需运行时冒烟测试 | 实验性 |
<!-- END GENERATED -->

仔细阅读表格中的状态：`experimental` 意味着上游框架未在该 GPU 系列上测试，`needs runtime smoke test` 意味着路径存在但 mliport 尚无该硬件上的实际运行记录。只有 V100 有硬件证据。详细的架构分析见[架构报告](https://github.com/hydrogen1222/mliport/blob/main/validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md)。

Turing 之前的 GPU 依赖 CUDA 12.6 通道。PyTorch 从 2.15 起停止发布该通道，因此在这些环境中请保持安装器的锁定版本，不要升级框架。

`mliport setup` 会打印检测到的 GPU 和将要使用的路线。

## 重新运行安装器

安装器对已存在的环境有明确的处理约定：

- **兼容环境**（请求的 Python 版本的真实 venv，不含其他后端的包）：复用；锁定包就地重新解析。
- **不兼容环境**（普通目录、错误的 Python、或已包含其他后端的框架）：**拒绝**。拒绝信息会说明原因并建议添加 `--clean` 来删除重建。

`--clean` 只删除指定引擎的环境。如果只需重建 MACE，`--clean --engines mace` 不会影响其他环境。

## 离线机器

对于隔离网络的集群，在联网机器上准备 wheel 缓存，然后指向它：

```bash
./scripts/install_mliport.sh --engines mace --device cpu --source offline
```

只需要搬运两样东西：uv 的 wheel 缓存（`UV_CACHE_DIR`）和模型文件。设置 `UV_PYTHON_DOWNLOADS=never` 阻止安装器获取托管的解释器。

## 磁盘空间与卸载

四引擎 GPU 安装大约占 25–35 GB，主要是 PyTorch 和 TensorFlow。CPU 安装更小。要移除某个后端，删除其环境目录（`.venv-mace`、`.venv-dpa`、`.venv-grace`、`.venv-nep` 或 `.venv`）即可。仓库和模型文件不受影响。启动器随 `bin/` 目录一起删除。

## 手动安装

每个后端环境需要 mliport 本身加上该引擎的依赖栈。权威的版本锁定在 `mliport/mliport/install/compatibility.py` 中。如果手动安装：

1. 为每个后端创建独立环境。
2. 在每个环境中安装 `./mliport`。
3. 在每个环境中运行 `mliport doctor`。
4. 所有检查通过前不要信任结果。

只安装一次 mliport 然后期望它能访问其他环境是行不通的——每个环境都是独立设计的。

## 常见问题

**"安装器未安装任何东西就退出了。"** 未知的引擎名、某后端不支持的 `--python` 版本、或在没有 GPU 的机器上使用 `--device cuda`。使用 `--dry-run` 查看计划和失败的约束。

**"启动器缺失或指向了错误的环境。"** 安装器未完成，或 mliport 是手动安装的。用 `ls bin/` 和 `readlink -f bin/mliport-mace` 检查。为该引擎重新运行安装器。

**"无法安装所有后端——版本冲突。"** 这正是独立环境存在的原因。每个后端独立安装即可；它们不需要共存于同一环境。

更多安装问题见[故障排除](troubleshooting.md)。
