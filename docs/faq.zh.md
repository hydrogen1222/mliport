# 常见问题与实用配方

常见问题的回答，以及日常任务的完整示例。

## 安装与配置

### 支持哪些 Python 版本？

**Python 3.10 到 3.13**，运行于 **Linux** 或 **WSL2**。不支持 macOS。GPU 加速需要
NVIDIA 显卡和可用的 CUDA 驱动。

### 需要安装全部五个后端吗？

不需要。只安装你要用的后端即可。每个后端有自己独立的 Python 虚拟环境，互不干扰。
安装脚本根据 `--engines` 参数为每个后端创建一个环境：

```bash
# 只装 MACE
./scripts/install_mliport.sh --engines mace --device auto

# 同时装 MACE 和 UMA（两个独立环境）
./scripts/install_mliport.sh --engines mace uma --device auto
```

### 怎么检查安装是否正常？

运行 `mliport doctor`：

```bash
./bin/mliport-mace doctor --engine mace --device auto
```

它会检查后端的 Python 包是否可导入、请求的设备是否可用，还可以（可选地）测试模型文件能否
加载。加上 `--model` 和 `--task` 可以探测具体的模型文件。

### 能在 conda 环境里用 mliport 吗？

安装脚本使用 [uv](https://docs.astral.sh/uv/) 并创建自己的虚拟环境。如果你需要
把 mliport 装进已有的 conda 环境，可以用 pip 手动安装 — 详见[安装](installation.md)
中的手动安装步骤。

### 怎么更新 mliport？

拉取最新代码，对你用的每个后端重新运行安装脚本：

```bash
git pull
./scripts/install_mliport.sh --engines mace --device auto
```

安装脚本会检测到已有环境并进行更新。

---

## 模型与后端

### 怎么选后端？

取决于你的使用场景：

| 场景 | 推荐后端 |
|---|---|
| 通用体相材料 | MACE (OMAT-0) 或 UMA |
| 分子和分子体系 | UMA (`--task omol`) 或 MACE (`--task omol`) |
| 针对特定体系自训练的势函数 | NEP（训练自己的 `nep.txt`） |
| 需要 TensorFlow 生态 | GRACE |
| DeePMD 生态 / 多任务模型 | DPA |

五个后端都支持所有工作流类型（单点、优化、MD、NEB、构型搜索）。主要差异在于可用的模型、
许可条款和生态兼容性。

详见[后端](backends/index.md)页面。

### 模型文件在哪里下载？

mliport 从不自动下载或内置模型权重。你需要从各项目的发布页面自行获取。[模型](models.md)
页面列出了所有已知的模型文件，附带下载链接、SHA-256 哈希和许可信息。

### `--task` 是做什么的？

多头模型（MACE、UMA）在不同物理语义的数据集上训练。`--task` 选择使用哪个头：

- `omat` / `bulk` — 周期性体相材料（OMat24 数据集）
- `omol` — 分子和分子体系
- `oc20` / `oc25` — Open Catalyst 数据集（表面上的吸附物）

具体有哪些可用的 task 取决于模型文件。用错 task 会产生物理上无意义的结果（参考能量不
对），因此 mliport 要求你显式指定。

### `--head` 和 `--task` 有什么区别？

`--task` 是 MACE/UMA 选择模型头的概念，按训练数据集区分。`--head` 是 DPA/DeepMD
选择多任务模型分支的概念。两者的作用相同 — 选择多任务模型的哪个输出通道 — 只是不同后端
的命名不同。

### 可以混合不同模型的绝对能量吗？

**不行。** 不同模型（甚至同一模型的不同 task）使用不同的参考能量零点。跨模型或跨 task
比较或相减绝对能量在物理上没有意义。请在同一个模型 + task 组合内比较能量*差*（反应能、
能垒、相对稳定性）。

---

## 运行计算

### 为什么 mliport 不指定 `--model-type` 就不让跑？

因为没有默认后端。mliport 要求你说清每个数字是哪个模型算出来的，这样结果的归属才不会
含糊。如果不想每次都打 `--model-type`，可以在 `settings.ini` 里设置模型别名 — 详见
[配置](configuration.md)。

### 怎么在多次运行间复用设置？

创建 `settings.ini` 文件：

```ini
[defaults]
device = cuda
strict_config = true

[model:mace-omat]
model_path = /models/mace-omat-0-medium.model
model_type = mace
task = bulk

[profile:production-md]
ensemble = NVT
thermostat = BUSSI
bussi_tau = 100
temperature = 600
timestep = 2.0
steps = 100000
save_interval = 100
```

然后使用别名和配置文件：

```bash
mliport md structure.vasp --model-alias mace-omat --profile production-md \
  --output runs/md-001
```

### `mliport run` 是干什么的？

`mliport run` 读取 INCAR 风格的输入文件（`INCAR.mliport`），里面用类似 VASP 的语法
写所有计算参数：

```
CALC_TYPE = md
MODEL_PATH = /models/mace-omat-0-medium.model
MODEL_TYPE = mace
TASK = bulk
MD_ENSEMBLE = NVT
TEMPERATURE = 600
TIMESTEP = 2.0
STEPS = 10000
```

```bash
mliport run --incar INCAR.mliport --output runs/md-from-incar
```

对于熟悉 VASP 输入文件的用户很方便，也可以把一次计算记录在一个自包含的文件里。

### 怎么批量算多个结构？

用 `mliport batch`：

```bash
mliport batch sp structures/*.cif \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --output runs/batch-sp
```

每个结构会在输出路径下得到自己的子目录。

### MD 跑失败了能恢复吗？

MD 可以通过指定相同的输出目录来恢复 — mliport 会从最后保存的检查点继续。NEB 支持
显式的 `--resume`。构型搜索也用 `--resume` 加原始输出目录来恢复。

### MD 的预弛豫是什么？

`--pre-relax` 标志（或 API 中的 `pre_relax=True`）会在开始 MD 之前做一次快速的
原子位置优化。这可以消除过大的初始力，避免 MD 前几步的不稳定。它只优化原子位置 — 不弛豫
晶胞，也不消除应力。

---

## NPT 分子动力学

### 哪些后端支持 NPT？

NPT 使用 ASE 的 `IsotropicMTKNPT` 积分器，仅支持 **UMA、MACE、GRACE 和 DPA**
后端。NEP **不支持** NPT — 无论是 GPUMD 运行时还是 Calorine/ASE 运行时 —
因为引擎会拒绝 `{uma, mace, grace, dpa}` 之外的所有模型类型执行 NPT。

### NPT 需要哪些参数？

至少需要：

```bash
mliport md structure.vasp --model model.pt --model-type mace --task bulk \
  --ensemble NPT --temp 300 --pressure-gpa 0.0 \
  --npt-tdamp 100 --npt-pdamp 1000
```

- `--temp`：目标温度，单位 K
- `--pressure-gpa`：目标压力，单位 GPa（0 = 常压，正值 = 压缩）
- `--npt-tdamp`：MTK 热浴阻尼时间，单位 fs（默认 100）
- `--npt-pdamp`：MTK 压浴阻尼时间，单位 fs（默认 1000）

### NPT 轨迹的晶胞会变化吗？

是的。晶胞的**体积**在每一步都会变化（`IsotropicMTKNPT` 积分器对晶胞做各向同性缩放 —
晶胞形状的六个自由度不能独立变化）。NPT 的主要轨迹输出是 ASE `.traj` 文件，记录每帧的
晶胞；NPT 运行**不会**生成标准的固定头部 `XDATCAR`。对 NPT 轨迹不应直接使用假设
固定晶胞的分析工具。

---

## 分析

### "采样不足"是什么意思？

mliport 的分析流水线会检查轨迹是否足够长，以产生可靠的统计量。如果检测到轨迹太短
（例如 MSD 还没到扩散区间，或 RDF 还没收敛），它会报告明确的"采样不足"状态，而不是
返回一个不可靠的数字。解决办法是跑更长的轨迹或增加采样。

### 怎么分析 MD 轨迹？

语法为 `mliport analyze <运行目录或轨迹文件> <任务> [选项]`。
数据源（运行目录或轨迹文件）写在任务名**之前**：

```bash
# 验证轨迹元数据和任务可分析性
mliport analyze runs/md validate

# 热力学诊断
mliport analyze runs/md thermo

# 均方位移 — 必须指定 --mobile
mliport analyze runs/md msd --mobile Li

# 径向分布函数 — 必须指定 --center 和 --neighbor
mliport analyze runs/md rdf --center Li --neighbor S --bins 200

# 速度自相关（需要轨迹中包含真实速度数据）
mliport analyze runs/md vacf

# 示踪扩散系数和 NE 电导率 — 三个标志都必须提供
mliport analyze runs/md transport --mobile Li --charge 1 --fit-start-ps 5.0

# 从温度/扩散系数数组拟合 Arrhenius
mliport analyze runs/md arrhenius \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 1.2e-10 --diffusivity 3.4e-10 --diffusivity 7.8e-10
```

全部 11 个分析任务见[分析](analysis/index.md)。

### 可以分析其他程序的轨迹吗？

mliport 可以分析自己的运行目录（包含溯源元数据）以及外部的 XDATCAR 和 ASE `.traj`
文件。对于外部轨迹，可能需要指定 `--positions-convention`（wrapped/unwrapped）和
`--frame-interval-fs`（当轨迹缺少时间元数据时）。

---

## 输出与可重复性

### 每种计算会产生哪些文件？

| 文件 | 所有类型 | 说明 |
|---|---|---|
| `mliport_results.json` | ✓ | 能量、力、应力、耗时、溯源信息 |
| `resolved_config.json` | ✓ | 每个配置项的值及其来源层 |
| `run.log` | ✓ | 完整的运行日志 |
| `OUTCAR` | sp, opt | VASP 兼容的力和能量 |
| `OSZICAR` | opt, md | 逐步收敛 / 热力学信息 |
| `CONTCAR` | sp, opt | POSCAR 格式的最终结构 |
| `XDATCAR` | md | VASP 格式的轨迹 |

完整说明见[输出文件](outputs.md)。

### 怎么知道结果是哪个模型算的？

每个 `mliport_results.json` 都记录了模型文件路径和 SHA-256 哈希、请求的设备和实际
使用的设备、每个配置项的来源（默认值、settings.ini、INCAR 还是 CLI），以及 mliport
版本。`resolved_config.json` 还提供带来源标注的完整配置。

### 为什么 mliport 拒绝覆盖已有结果？

为了防止你意外销毁已完成的计算。如果确实要覆盖，在 CLI 上加 `--overwrite`，或在 API
中传 `overwrite=True`。

---

## NEP 特有问题

### `--nep-runtime` 是什么？

NEP 有两种运行时：

- **`ase`** — 通过 Calorine Python 包用 ASE 接口运行 NEP。CPU 上用 `CPUNEP`，
  CUDA 上用 `GPUNEP`。支持 mliport 的大部分功能，但**不支持 NPT**。
- **`gpumd`** — 直接启动原生 GPUMD 可执行文件。仅限 GPU，大规模 MD 性能最高，
  不支持 NPT，功能集也有所不同。

使用 NEP 时必须指定 `--nep-runtime`。mliport 从不猜测该用哪个运行时。

### 需要 `--nep-gpumd-command` 吗？

仅在使用 `--nep-runtime gpumd` 或 Calorine 的 `GPUNEP` 计算器需要它时才需要。
指向系统上的 `gpumd` 可执行文件：

```bash
--nep-runtime gpumd --nep-gpumd-command /usr/local/bin/gpumd
```

---

## 快速排障

| 现象 | 常见原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: mace` | 当前环境没有安装 MACE | 用 `--engines mace` 重跑安装脚本 |
| `Unknown config key: xxx` | 拼写错误或严格模式下用了不属于该后端的选项 | 检查拼写；参见[配置](configuration.md) |
| `CUDA out of memory` | 体系太大，GPU 显存不够 | 换小模型、减小批次或用 CPU |
| `Output directory exists` | 上次运行的结果还在 | 用 `--overwrite` 或换个 `--output` 路径 |
| `No model_type specified` | 没给 `--model-type` 也没有别名 | 加 `--model-type` 或设置模型别名 |

更多内容见[故障排查](troubleshooting.md)。
