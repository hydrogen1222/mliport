# mlipx — MLIP eXtended

**一个 VASP 风格的 CLI / TUI / Python API，用于机器学习原子间势（MLIP）计算。**

mlipx 将四个 MLIP 引擎统一到同一套接口后面——**UMA (FAIRChem)**（默认）、**MACE**、**DPA (DeepMD-kit)**、**GRACE**——支持单点能（SP）、几何优化（OPT）、分子动力学（MD）、批量计算和带验证的轨迹分析，并输出 VASP 兼容文件（OUTCAR、CONTCAR、XDATCAR、OSZICAR）。

```
structure.cif ──▶  MLIP 引擎 (UMA/MACE/DPA/GRACE)  ──▶  能量、力、应力
```

---

## 支持的引擎

| `--model-type` | 引擎 | 后端包 | Task 取值 |
|---|---|---|---|
| `uma`（默认，别名 `fairchem`） | UMA — FAIRChem | `fairchem-core` | `omat` / `omol` / `oc20` / `oc25` / `odac` / `omc` |
| `mace` | MACE | `mace-torch` | `bulk` / `molecule` |
| `dpa` | DPA — DeepMD-kit | `deepmd-kit` | `bulk` / `molecule` |
| `grace` | GRACE | `tensorpotential` | `bulk` / `molecule` |

每个引擎必须运行在**独立的 Python 环境**中，因为依赖互相冲突（UMA 需要 `e3nn>=0.5`；MACE 固定 `e3nn==0.4.4`；DPA 固定 `torch==2.10`；GRACE 使用 TensorFlow）。**不要**把四个引擎装进同一个环境。

---

## 安装

### 一键安装（推荐）

```bash
# 克隆并安装 uv（如果已有 uv 可跳过）
git clone https://github.com/hydrogen1222/mlipx.git
cd mlipx
curl -LsSf https://astral.sh/uv/install.sh | sh

# 自动检测 GPU 并安装全部四个引擎
./scripts/install_mlipx.sh
```

常用变体：

```bash
./scripts/install_mlipx.sh --device cpu        # 纯 CPU 机器
./scripts/install_mlipx.sh --engines uma,mace  # 只装 UMA + MACE
./scripts/install_mlipx.sh --source china      # 交互选择国内镜像（编号 1–5）
./scripts/install_mlipx.sh --source china-ustc --non-interactive  # 固定源/脚本用法
./scripts/install_mlipx.sh --clean             # 全部重建
./scripts/install_mlipx.sh --dry-run           # 预览不安装
```

如果上一次安装中途停止，请使用 `--clean` 重新运行，让各引擎的残缺环境在验证前完整重建。
GPU 环境体积很大：安装全部四个隔离环境需要预留数十 GiB。GRACE 安装完成后通常约占
6–7 GiB，下载和解压 CUDA wheel 时还需要额外的临时空间。

运行 `./scripts/install_mlipx.sh --help` 查看全部参数。

### 手动安装（四个环境）

如果喜欢手动安装，请为每个引擎创建一个 venv：

```bash
# UMA（默认）——和其他引擎一样显式安装
uv venv --python 3.12 .venv
uv pip install --no-config --python .venv/bin/python \
  "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv/bin/python \
  -e ./mlipx "fairchem-core==2.21.0"
.venv/bin/mlipx doctor --engine uma --device auto

# MACE
uv venv --python 3.12 .venv-mace
uv pip install --no-config --python .venv-mace/bin/python \
  "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv-mace/bin/python \
  -e ./mlipx "e3nn==0.4.4" "mace-torch==0.3.16"
.venv-mace/bin/mlipx doctor --engine mace --device auto

# DPA / DeepMD
uv venv --python 3.12 .venv-dpa
uv pip install --no-config --python .venv-dpa/bin/python \
  "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv-dpa/bin/python \
  -e ./mlipx "deepmd-kit==3.1.3"
.venv-dpa/bin/mlipx doctor --engine dpa --device auto

# GRACE
uv venv --python 3.12 .venv-grace
uv pip install --no-config --python .venv-grace/bin/python \
  -e ./mlipx "tensorflow[and-cuda]==2.20.0" "tensorpotential==0.6.0"
uv pip install --no-config --python .venv-grace/bin/python \
  "nvidia-cudnn-cu12==9.3.0.75"
.venv-grace/bin/mlipx doctor --engine grace --device auto
```

使用对应的命令前缀：

| 引擎 | 环境 | 前缀 |
|---|---|---|
| UMA | `.venv` | `.venv/bin/mlipx ...` |
| MACE | `.venv-mace` | `.venv-mace/bin/mlipx ...` |
| DPA | `.venv-dpa` | `.venv-dpa/bin/mlipx ...` |
| GRACE | `.venv-grace` | `.venv-grace/bin/mlipx ...` |

### GPU 兼容性

安装器和 `mlipx setup` 会自动选择正确的 PyTorch/CUDA wheel。

| GPU 系列 | 代表显卡 | 计算能力 | CUDA 路线 |
|---|---|---|---|
| Maxwell | GTX 960、TITAN X | sm_50/52 | cu126 Legacy（⚠️ 实验性） |
| Pascal | **Tesla P40**、GTX 1080 Ti、P100 | sm_60/61 | cu126 Legacy |
| Volta | **V100** | sm_70 | cu126 Legacy |
| Turing | RTX 20xx | sm_75 | cu128+ Modern |
| Ampere | **RTX 3080 Ti**、30xx | sm_80/86 | cu128+ Modern |
| Ada | **RTX 4090**、40xx | sm_89 | cu128+ Modern |
| Hopper | H100 | sm_90 | cu128+ Modern |
| Blackwell | RTX 50xx | sm_100/120 | cu128+ Modern |
| 无 | 仅 CPU | — | CPU wheels |

> **为什么有两条 CUDA 路线？** Maxwell/Pascal/Volta 必须使用 **cu126 Legacy** 通道：PyTorch 2.8+ 从 cu128 构建中移除了 Maxwell/Pascal，PyTorch 2.11+ 从 cu128+ 中移除了 Volta。Turing+ 使用**现代**通道（torch 2.8–2.10 用 cu128，torch 2.12+ 用 cu130）。Maxwell 标记为实验性，因为 TensorFlow 2.20 官方 wheel 从 sm_60 开始构建。

**各引擎验证状态**（来自 `mlipx/install/compatibility.py`；V100 和 RTX 4090 已经过真机测试。P40 使用修正后的精确 `+cu126` wheel pin，但仍需在修复后重新做模型 smoke test）：

| 引擎 | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | 其他 Turing+ |
|---|---|---|---|---|---|
| UMA | experimental | needs smoke test | **verified** | **verified** | needs smoke test |
| MACE | experimental | needs smoke test | **verified** | **verified** | needs smoke test |
| DPA | experimental | needs smoke test | **verified** | **verified** | needs smoke test |
| GRACE | experimental | needs smoke test | **verified** | **verified** | needs smoke test |

### 下载源

PyPI 包和 PyTorch wheel 分开处理。安装器**不会修改**你的全局
`~/.config/uv/uv.toml`，而是使用 `UV_NO_CONFIG=1` 和进程级环境变量。
在真实终端中使用 `--source china` 会显示编号菜单：清华 TUNA、阿里云、
中科大 USTC、腾讯云或官方源。国内选项的 PyTorch wheel 均使用阿里云镜像；
直接回车默认选择清华 TUNA。在 CI、管道等非终端环境中不会弹出提示，默认仍为
清华 TUNA；也可显式加 `--non-interactive`。
`offline` 模式还要求本机已有 `uv` 可找到的目标 Python；安装器不会为了创建
环境而偷偷下载解释器。

| `--source` | PyPI | PyTorch wheel | 适用场景 |
|---|---|---|---|
| `auto` → `official` | pypi.org | download.pytorch.org | 默认 |
| `official` | pypi.org | download.pytorch.org | 显式使用官方源 |
| `china` | 交互选择；默认清华 TUNA | mirrors.aliyun.com（`--find-links`） | 中国大陆交互安装 |
| `china-aliyun` | mirrors.aliyun.com | mirrors.aliyun.com | 固定使用阿里云 |
| `china-ustc` | mirrors.ustc.edu.cn | mirrors.aliyun.com | 固定使用中科大 |
| `china-tencent` | mirrors.cloud.tencent.com | mirrors.aliyun.com | 固定使用腾讯云 |
| `offline` | 仅本地缓存 | 仅本地缓存 | 已有目标 Python 的离线机器 |
| `custom` | 你的环境变量 | 你的环境变量 | 高级用户 |

---

## 快速开始

### 单点能（UMA）

```bash
.venv/bin/mlipx sp structure.cif --model uma-s-1.pt --task omat --device cpu
```

### 几何优化

```bash
.venv/bin/mlipx opt structure.cif --model uma-s-1.pt --task omat \
  --cell-opt --fmax 0.02
```

### 分子动力学

```bash
.venv/bin/mlipx md structure.cif --model uma-s-1.pt --task omat \
  --device cuda --steps 10000
```

### 其他引擎

```bash
# MACE
.venv-mace/bin/mlipx sp bulk.cif --model mace.model \
  --model-type mace --task bulk --head default --device cuda:0

# DPA / DeepMD
.venv-dpa/bin/mlipx opt bulk.cif --model dpa.pt \
  --model-type dpa --task bulk --head Domains_SSE_PBE \
  --device cuda:0 --fmax 0.05

# GRACE（--model 指向整个 SavedModel 目录）
.venv-grace/bin/mlipx sp bulk.cif --model grace_model/ \
  --model-type grace --task bulk --device cuda:0 \
  --gpu-memory-limit-mb 6144
```

### INCAR 文件（VASP 风格）

```bash
.venv/bin/mlipx template sp          # 生成 INCAR.sp
.venv/bin/mlipx run -i INCAR.sp -s structure.cif
```

示例 `INCAR.sp`：

```ini
CALC_TYPE   = SP
MODEL_TYPE  = UMA        # 或 MACE / DPA / GRACE
MODEL_PATH  = uma-s-1.pt
TASK        = omat       # UMA: omat/omol/...；其他: bulk/molecule
DEVICE      = cpu
```

### 批量计算

```bash
.venv/bin/mlipx batch structures/ --model uma-s-1.pt \
  --model-type uma --task omat --device cuda \
  --calc-type sp --pattern "*.cif" --output batch_results
```

每个输入结构会得到独立的输出子目录；根目录生成 `batch_summary.json`。

---

## 轨迹分析

`mlipx analyze` 与计算后端无关：可以在 UMA `.venv` 中分析任意引擎产生的轨迹，而无需加载模型后端。

```bash
# 先验证（检查时间轴、PBC、坐标约定、任务资格）
.venv/bin/mlipx analyze results/LGPS-800K validate

# 热力学
.venv/bin/mlipx analyze results/LGPS-800K thermo

# RDF / 配位数
.venv/bin/mlipx analyze results/LGPS-800K rdf \
  --center Li --neighbor S --rmax 6 --cn-cutoff 3

# MSD
.venv/bin/mlipx analyze results/LGPS-800K msd \
  --mobile Li --axes x,y,z,xyz --drift-reference nonmobile

# 密度 / VACF / 速度谱 / Arrhenius
.venv/bin/mlipx analyze results/LGPS-800K density --mobile Li --spacing 0.25
.venv/bin/mlipx analyze results/LGPS-800K vacf --species Li
.venv/bin/mlipx analyze results/LGPS-800K spectrum --species Li --taper one-sided-cosine

# 由多个温度下的独立输运结果拟合 Arrhenius
# （每个值用重复的 flag 分别传入）
.venv/bin/mlipx analyze RUN arrhenius \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 1e-10 --diffusivity 2e-10 --diffusivity 5e-10 \
  --diffusivity-std 0.1e-10 --diffusivity-std 0.2e-10 --diffusivity-std 0.5e-10
```

### 分析层级：MSD、transport 与 electrolyte

三个命令承担不同职责：

- `msd` 是诊断视图：多时间起点 MSD、局部 `alpha` 和 native OLS 拟合窗口；其 OLS 结果明确标记 `publication_grade=false`，不是定量输运权威。
- `transport` 是定量输运权威。kinisi 给出示踪 `D_tracer`，并按
  `sigma_NE_tracer=n(z e)^2D_tracer/(k_B T)` 传播后验。启用
  `--collective-conductivity` 时，MSCD 包含轨迹中表示的离子相关，得到
  `sigma_collective`，并推导 `D_sigma=sigma_collective k_B T/(n(z e)^2)`。
  Haven 比明确为 `H_R=D_tracer/D_sigma=sigma_NE_tracer/sigma_collective`，
  correlation factor 为 `sigma_collective/sigma_NE_tracer`。可选
  `--jump-diffusion` 的 MSTD/`D_J` 是总/跳跃位移诊断，不是示踪扩散。
- `electrolyte` 只做 GEMDAT 机制分析：密度、位点、占据、transition、驻留、跳跃、集体事件和渗流路径。GEMDAT 端点/COM 扩散率和 Haven 只放在 `diagnostic_crosscheck`，并标记 `publication_transport_authority=false`。

协方差感知的输运分析使用 [kinisi 2.x](https://joss.theoj.org/papers/10.21105/joss.05984)。必须显式给出拟合起点：

```bash
.venv/bin/mlipx analyze RUN transport --mobile Li --charge 1 \
  --drift-reference nonmobile --fit-start-ps 40 \
  --lag-step-ps 2 --lag-stop-ps 200 --random-seed 0

# LGPS：示踪 + collective Einstein 电导率
mlipx analyze RUN transport \
  --mobile Li --charge 1 --drift-reference nonmobile \
  --fit-start-ps 40 --lag-step-ps 2 --lag-stop-ps 200 \
  --collective-conductivity --random-seed 0
```

关键科学规则：

- **绝不猜测。** `--charge` 必须显式给出；温度来自运行目录（外部轨迹用 `--temperature-K`）；`wrapped`/`unwrapped` 必须描述文件真实情况。
- **仅支持固定晶胞。** 变晶胞输运不受支持。
- **漂移校正必须显式选择：** `none`、`nonmobile` 或 `indices`。
- **`--lag-step-ps` / `--lag-stop-ps`** 只稀疏化 kinisi 的 lag 时间网格，不对轨迹帧降采样；两者必须同时使用。
- **native MSD OLS 诊断** 只有在同时给出 `--fit-start-ps` 和 `--fit-stop-ps` 时才产生；mlipx 不会自动判断 publication 拟合窗口。transport 使用显式 kinisi `--fit-start-ps` 与选定 lag 网格。
- **Nernst–Einstein 示踪电导率**（`sigma_NE_tracer`）会报告后验均值 / 标准差 / 95% 后验可信区间。它不是总物理不确定性，也不自动等于实验或集体电导率。
- `sigma_NE_tracer` 忽略 distinct 离子相关；`sigma_collective` 是所分析经典 MD 轨迹与所选电荷模型内的 collective Einstein 离子电导率，不自动等于实验块体/多晶电导率。
- kinisi CrI 是条件后验区间，不包含模型、有限尺寸或 replica 不确定性；`--collective-system-particles` 是按索引分组的统计参数，不是独立 MD replica。Haven 区间若可给出，会明确标注为独立边际后验近似，因为没有建模示踪/集体协方差。
- 安全完成 PBC 重构的 0.1 ps 保存间隔可用于长时 Einstein/MSCD 分析；它不能替代短时 VACF 或 Green–Kubo 所需的密集采样。

### 电解质机制分析（可选 GEMDAT）

GEMDAT 是可选的后端，用于位点映射、跳跃和渗流分析：

```bash
python -m pip install -e './mlipx[analysis,electrolyte]'
.venv/bin/mlipx analyze RUN electrolyte --mobile Li --sites Li_sites.cif \
  --jump-dimensions 3 --percolation-axes xyz
```

位点来源是必须的（`--sites` 或 `--discover-sites-from-density`）。GEMDAT 的扩散系数不会被提升为优于 kinisi 的估计。
显式 CIF 位点是可复现路径；密度分割是探索性的，并记录 resolution/background/峰数等元数据。GEMDAT 自由能路径 barrier 是有限温度占据推导量，不是 NEB 势能迁移 barrier。

### 输出与复现

每个分析任务写入 `RUN/analysis/TASK/REQUEST_HASH/`：

```
request.json
provenance.json
results.json
task-specific CSV/NPZ
PNG and SVG when applicable
diagnostics.json when applicable
```

Transport 另外生成 `transport_summary.csv`、压缩的 `kinisi_arrays.npz` 和
`transport_msd`；启用 collective/jump 时分别增加 `transport_mscd` 与
`transport_mstd`。Electrolyte 生成压缩 `electrolyte_arrays.npz`、汇总/transition/
jump/percolation CSV、CIF 位点产物以及无 GUI 的密度/自由能/机制图。transport
与 electrolyte 的 scientific revision 会进入缓存键，语义变化后不会静默复用旧结果。

请求哈希包含源指纹、选择、范围、坐标轴、漂移定义、科学参数和后端版本。相同请求会被复用，除非使用 `--force`。

---

## 接口

| 接口 | 命令 | 适用场景 |
|---|---|---|
| CLI | `mlipx sp/opt/md/batch/...` | 脚本、HPC、自动化 |
| TUI | `mlipx tui` | 交互式探索 |
| Python API | `from mlipx.api import run_single_point, ...` | 自定义工作流 |
| INCAR | `mlipx run -i INCAR` | VASP 风格配置 |

### Python API

```python
from mlipx.api import run_single_point, run_md, calculate_energy

result = run_single_point("structure.cif", "uma-s-1.pt", task="omat")
energy = calculate_energy("structure.cif", "uma-s-1.pt", task="omat")
```

---

## INCAR 配置

| 类别 | 键 | 默认值 |
|---|---|---|
| 计算 | `CALC_TYPE` | —（`SP` / `OPT` / `MD`） |
| 模型 | `MODEL_TYPE` | `UMA` |
| 模型 | `MODEL_PATH` | — |
| 模型 | `TASK` | `omat`（UMA）/ `bulk`（其他） |
| 模型 | `DEVICE` | `cpu` |
| 模型 | `HEAD` | —（MACE/DPA 多任务） |
| 模型 | `DTYPE` | `float64`（MACE） |
| 输出 | `WRITE_OUTCAR` | `.TRUE.` |
| 输出 | `WRITE_XDATCAR` | `.TRUE.` |
| 输出 | `WRITE_TRAJECTORY` | `.TRUE.` |
| 输出 | `WRITE_JSON` | `.TRUE.` |
| 优化 | `FMAX` | `0.05` |
| 优化 | `MAX_STEPS` | `500` |
| 优化 | `OPT_ALGO` | `FIRE` |
| 优化 | `CELL_OPT` | `.FALSE.` |
| MD | `MD_ENSEMBLE` | `NVT` |
| MD | `TEMPERATURE` | `300` |
| MD | `TIMESTEP` | `1.0` |
| MD | `STEPS` | `1000` |
| MD | `THERMOSTAT` | `LANGEVIN` |
| MD | `SAVE_INTERVAL` | `10` |

完整的带注释关键字列表见 `mlipx template sp/opt/md` 生成的模板。

---

## 输出文件

每次计算会生成自包含的输出目录：

```
OUTPUT/
├── OUTCAR                 VASP 风格文本输出
├── CONTCAR                最终结构
├── XDATCAR                VASP 轨迹（若开启）
├── mlipx_results.json     机器可读结果
├── raw/
│   ├── trajectory.traj    标准 ASE 轨迹
│   └── md.csv             MD 时间序列（若为 MD）
└── artifacts.json         provenance / 版本 / 语义
```

高通量场景下可使用 `--no-write-outcar --no-write-xdatcar` 跳过 VASP 互操作文本输出；标准轨迹仍然保留。

---

## 后台任务与队列

后台任务通过队列 JSON 接口提交：

```bash
# 1. 用 JSON 描述任务
cat > tasks.json <<'JSON'
{
  "max_concurrent": 1,
  "tasks": [
    {
      "name": "opt-uma-1",
      "calc_type": "opt",
      "structure": "/path/a.cif",
      "model": "/path/uma-s-1.pt",
      "model_type": "uma",
      "device": "cuda:0",
      "options": {"fmax": 0.05}
    }
  ]
}
JSON

# 2. 提交并启动
.venv/bin/mlipx queue submit tasks.json
.venv/bin/mlipx queue start            # 后台调度器
.venv/bin/mlipx queue status

# 3. 管理
.venv/bin/mlipx jobs                   # 查看运行中/完成/失败任务
.venv/bin/mlipx kill <job-id>          # 终止运行中任务
.venv/bin/mlipx clean                  # 清理已完成/失败记录
```

每个任务可以使用自己的 Python 环境 / 引擎 / 模型。TUI 也内置队列控制。

---

## 资源控制

| 选项 | 作用 |
|---|---|
| `--cpu-threads N` | CPU 线程数（UMA/MACE/DPA 用 PyTorch；GRACE 用 TF） |
| `--gpu-memory-growth` | GRACE：按需增长 TF GPU 显存（默认开启） |
| `--gpu-memory-limit-mb MIB` | GRACE：TF GPU 显存硬上限 |
| `--inference-mode turbo` | 仅 UMA：快速推理预设 |
| `--activation-checkpointing` | 仅 UMA：节省显存 |
| `--dtype float32` | MACE：选择 float32 提速（默认 float64） |

---

## 故障排除

### "no kernel image is available for execution on the device"

你的 PyTorch 构建没有对应 GPU 的内核。Maxwell/Pascal/Volta 使用 cu126 Legacy 通道；Turing+ 使用现代通道。运行：

```bash
.venv/bin/mlipx setup     # 本机报告
./scripts/install_mlipx.sh --dry-run
```

### "No edges found in structure"（未找到边）

原子间距离超过截断、晶胞无效或 PBC 设置错误。检查输入结构并使用正确的 `--task`（周期体系 `omat`/`bulk`；分子体系 `omol`/`molecule`）。

### CUDA 显存不足

使用 `--device cpu`、更小的模型，或 UMA `--activation-checkpointing`。GRACE 设置 `--gpu-memory-limit-mb`。

### 安装器报告 `No space left on device`

重试前先检查仓库所在文件系统和 uv 缓存：

```bash
df -h . "$(uv cache dir)"
du -sh "$(uv cache dir)" .venv* 2>/dev/null
uv cache clean
```

如果中断在 GRACE 安装阶段，应保留已经验证通过的 UMA/MACE/DPA，只重建 GRACE。
安装期间建议至少留出 10–12 GiB；实际占用会随 wheel 版本和 uv 链接模式变化。

```bash
./scripts/install_mlipx.sh --source china --engines grace --clean
```

如果 uv 缓存位于空间较小的根分区，而另一文件系统空间充足，可在重试时设置
`UV_CACHE_DIR=/larger/path/uv-cache`。仓库所在文件系统仍需容纳最终的
`.venv-grace` 环境。

### MACE 环境不兼容

MACE 不能与 UMA 共用环境。使用 `.venv-mace/bin/mlipx ...`（安装器会自动创建）。

### MD 原子爆炸

NVT 默认预弛豫（最多 50 步 FIRE）；NVE 默认关闭，需要时手动开启。

---

## 开发

使用独立的开发环境，避免与 UMA 运行时 `.venv` 冲突：

```bash
# 1. 创建开发环境
uv venv --python 3.12 .venv-dev

# 2. 安装 mlipx（含 dev + analysis extras）
uv pip install --python .venv-dev/bin/python -e './mlipx[dev,analysis]'

# 3. 运行测试（无需安装重型 ML 后端——backend 测试会 mock/跳过）
.venv-dev/bin/python -m pytest tests -q
```

UMA 通过外部 `fairchem-core` 依赖提供。核心代码位于 `mlipx/mlipx/`；
安装/兼容性逻辑位于 `mlipx/mlipx/install/`。

分析 extras（可选）：`./mlipx[analysis]`（scipy/matplotlib）、
`./mlipx[transport]`（kinisi）、`./mlipx[electrolyte]`（gemdat），或
`./mlipx[analysis-all]` 一次性安装三者。

---

## 许可证

MIT License。mlipx 基于 [FAIRChem](https://github.com/FAIR-Chem/fairchem)（Copyright © Meta Platforms, Inc. and affiliates），MIT 许可。详见 [`LICENSE.md`](LICENSE.md) 和 [`mlipx/LICENSE`](mlipx/LICENSE)。
