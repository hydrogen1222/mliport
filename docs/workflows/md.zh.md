# 分子动力学

`mliport md` 通过积分经典运动方程生成轨迹——原子位置和速度的时间序列，用于采样热力学系综。轨迹是 mliport 所有[分析](../analysis/index.md)工具的原材料：扩散系数、径向分布函数、振动光谱等。

## 物理原理

分子动力学求解牛顿运动方程：

$$m_i \ddot{\mathbf{r}}_i = \mathbf{F}_i = -\nabla_i E$$

其中力 $\mathbf{F}_i$ 由 MLIP 模型在每个时间步计算。轨迹通过 velocity Verlet 算法积分，可以耦合恒温器来控制温度。

### 系综

- **NVE**（微正则系综）：原子数、体积和总能量恒定。无恒温器。用于检验能量守恒，或需要不受扰动的动力学时（如通过 VACF 计算振动光谱）。
- **NVT**（正则系综）：通过恒温器维持恒定温度。最常用的选择，适用于平衡态采样和输运系数提取。
- **NPT**（等温等压系综）：通过耦合的恒温器和恒压器维持恒定温度和压力。用于在目标压力下平衡晶胞体积，然后切换到 NVT 进行正式模拟。

## 命令行

```bash
mliport md structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --ensemble NVT --temp 300 --timestep 1.0 --steps 1000 \
  --equilibration-steps 500 --save-interval 10 \
  --output runs/md
```

### 800 K NHC 恒温器的正式模拟

```bash
mliport md structure.cif --model uma.pt --model-type uma --task omat \
  --ensemble NVT --temp 800 --timestep 1.0 \
  --equilibration-steps 5000 --steps 50000 --save-interval 10 \
  --thermostat NHC \
  --output runs/md-800K
```

### NPT 晶胞平衡

```bash
mliport md structure.cif --model model.pt --model-type mace \
  --ensemble NPT --temp 800 --pressure-gpa 0 \
  --npt-tdamp 100 --npt-pdamp 1000 \
  --timestep 0.5 --steps 50000 --save-interval 10 \
  --no-pre-relax --output runs/npt-800K
```

NPT 使用 ASE 的**各向同性 Martyna–Tobias–Klein (MTK)** 恒温恒压器，适用于 UMA、MACE、GRACE 和 DPA 后端。NEP/GPUMD 不支持 NPT。详见下方的 [NPT 详细说明](#npt-详细说明)。

## Python API

```python
from mliport.api import run_md

# NVT 正式模拟
results = run_md(
    structure="structure.cif",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    ensemble="NVT",
    temperature=800,
    timestep=1.0,
    steps=50000,
    thermostat="NHC",
    save_interval=10,
    output_dir="runs/md-800K",
)
print(f"Final temperature: {results['temperature']:.1f} K")
```

该函数接受 ASE `Atoms` 对象或文件路径。返回一个字典，包含温度、能量和轨迹元信息。

!!! note "NPT 参数为关键字专用"
    调用 `run_md` 进行 NPT 模拟时，NPT 参数必须以关键字形式传入：
    `pressure_gpa=0.0`、`npt_tdamp=100.0` 等，不能按位置传入。
    这避免了 `MDRunner` 历史上的位置参数歧义（Issue #102）。

## 选项

### 动力学参数

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--ensemble` | `NVT` | `NVE`、`NVT` 或 `NPT` |
| `--temp` | 300 K | 目标温度 |
| `--timestep` | 1.0 fs | 积分步长 |
| `--steps` | 1000 | 正式模拟的步数 |
| `--equilibration-steps` | 0 | 平衡步数（同一系综，在正式模拟之前） |
| `--save-interval` | 10 | 每 N 步写入一帧轨迹 |
| `--seed` | 随机 | 随机种子；省略时自动生成并记录 |

### 恒温器（NVT）

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--thermostat` | `LANGEVIN` | `LANGEVIN`、`BUSSI` 或 `NHC` |
| `--friction` | — | Langevin 摩擦系数 (1/fs) |
| `--bussi-tau` | — | Bussi/CSVR 耦合时间 (fs) |
| `--nhc-tdamp` | — | Nosé–Hoover chain 阻尼时间 (fs) |
| `--nhc-tchain` | — | NHC 链长度 |
| `--nhc-tloop` | — | NHC 积分子步数 |

### NPT 参数

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--pressure-gpa` | 0 | 外加静水压力，单位 GPa（正值为压缩） |
| `--npt-tdamp` | — | MTK 恒温器阻尼时间 (fs) |
| `--npt-pdamp` | — | MTK 恒压器阻尼时间 (fs) |
| `--npt-tchain` / `--npt-pchain` | 3 | MTK 链长度 |
| `--npt-tloop` / `--npt-ploop` | 1 | MTK 积分子步数 |

### 预优化

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--pre-relax` | NVT 默认开启，NVE/NPT 默认关闭 | 在 MD 之前做一次仅原子位置的优化 |
| `--pre-relax-steps` | 50 | 预优化最大步数 |
| `--pre-relax-fmax` | 0.1 eV/Å | 预优化力的阈值 |

### 安全与输出

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--fmax-abort` | 20 eV/Å | 任何原子受力超过此值时中止运行 |
| `--velocity-policy` | — | 初始速度的处理方式 |
| `--com-policy` | `auto` | 质心运动的处理方式 |
| `--write-outcar` | true | 写入 VASP 风格的 OUTCAR |
| `--write-xdatcar` | true | 写入 VASP 风格的 XDATCAR（仅 NVT/NVE） |
| `--write-trajectory` | true | 写入 ASE `.traj` 轨迹文件 |

## 预优化：门槛而非建议

如果预优化在 `--pre-relax-steps` 步内没有达到 `--pre-relax-fmax`，运行会**在正式 MD 开始前停止**。从受力很大的结构开始跑动力学很少是你想要的，而悄悄地"差不多就行"会掩盖这个问题。报错信息会告诉你该怎么办：增加步数预算、在有科学依据的前提下放宽阈值、或用 `--no-pre-relax` 关闭预优化。预优化成功的结果记录在 `md_provenance.pre_relax` 中。

## 恒温器：选哪个以及为什么重要

恒温器控制 NVT 模拟中的温度。选择不同的恒温器会影响动力学，从而影响你提取的输运系数。

- **Langevin**（默认）：随机恒温器，添加摩擦力和随机力。简单且稳定。摩擦参数决定恒温器的耦合强度。*注意*：Langevin 动力学不是 Hamilton 的，因此从 Langevin 轨迹计算的速度自相关函数和振动光谱会带有恒温器伪影。计算光谱时请用 NVE 或 NHC。
- **Bussi/CSVR**：确定性恒温器，通过缩放速度来匹配正则分布（Bussi, Donadio and Parrinello, J. Chem. Phys. **126**, 014101, 2007）。折中方案：对动力学的扰动最小，同时保证正确的温度分布。
- **NHC**（Nosé–Hoover chain）：基于扩展系统动力学的确定性恒温器（Martyna et al., J. Chem. Phys. **97**, 2635, 1992）。产生真正的正则采样且不扰动动力学。计算振动光谱和扩散系数的最佳选择。**不能与 FixCom 或其他 ASE 约束共存。** mliport 在加载模型*之前*就会检查约束——如果输入结构带有任何约束，运行会立即报错并给出说明，而不是在加载模型后才崩溃。

**报告扩散系数时务必注明所用的恒温器。**同一温度下使用不同恒温器的两次模拟是不同的动力学体系。

### NHC 与质心运动

NHC 不能与 `FixCom` 约束一起运行。在默认的 `--com-policy auto`（等效于 `initialize_only`）下，mliport 在初始化速度后将质心动量清零一次，之后 NHC 会保持它为零。因此体系有 $3N-3$ 个动能自由度，恒温器的设定温度按 $(3N-3)/(3N)$ 缩放（记录为 `nhc_set_temperature_K`）。使用 `--com-policy none` 时，随机的质心动量会保留，晶胞整体做匀速平移漂移；如果分析时不去除漂移，MSD 会被它主导。

## 步长、保存间隔与分析

`--timestep` 是积分步长；`--save-interval` 决定每隔多少步保存一帧。步长 1 fs、`--save-interval 10` 意味着每 10 fs 存一帧。输运分析需要知道的是这个**保存帧间距**——不是积分步长。

把平衡阶段排除在统计之外：用 `--equilibration-steps` 设置平衡步数，只分析正式模拟段。NVE 和 NVT 是不同的动力学，它们的轨迹不应混在一起平均。

## NPT 详细说明

NPT 运行各向同性的 Martyna–Tobias–Klein (MTK) 恒温恒压器。可用于 UMA、MACE、GRACE 和 DPA 后端（不支持 NEP/GPUMD）。

**前提条件：**

- 三维周期性边界条件
- 有限正体积的晶胞
- 模型能提供**应力**
- 不能有原子/质心约束，不能是部分周期性

**压力约定：** 正压力为压缩方向（GPa）。转换为 ASE 内部使用的 eV/Å³ 的过程会被记录。

**XDATCAR 与分析：** 标准 XDATCAR 仅适用于定胞 MD，因此 NPT 会禁用 XDATCAR（并记录原因）。轨迹保存在 `raw/trajectory.traj` 中，每帧包含晶胞和动量。`md.csv` 文件记录每个保存帧的瞬时总压（包含动能项）和体积。

**输运分析注意事项：** NPT 轨迹被标记为变胞/不适合输运分析。**不要**从 NPT 数据计算传统的定胞 MSD、Einstein–Helfand 电导率或扩散系数。对于固态电解质，推荐的流程是：

1. 在目标温度和压力下用 NPT 平衡体积/晶胞
2. 通过独立的统计检验提取平衡时段的平均晶胞（NPT 的最后一帧**不**自动等于平均晶胞）
3. 以平衡后的晶胞启动一次**单独的**定胞 NVT 或 NVE 正式模拟

**帧重启：** NPT 的帧重启不保留隐藏的 MTK 链/恒压器状态，因此不是严格的轨迹续算。

## 继续运行

MD 没有 `--resume` 标志。要继续一条轨迹，读取一个携带动量的保存帧，然后启动一次**新的**运行：

```bash
mliport md last_frame.traj --model mace-omat-0-medium.model \
  --model-type mace --velocity-policy preserve --no-pre-relax \
  --output runs/md-part2
```

新运行保留输入的位置和速度，不会重新抽取 Maxwell–Boltzmann 分布。它有自己独立的运行目录和记录，第一帧等于上一段的最后一帧。

## 什么情况会停止运行

- 任何原子受力超过 `--fmax-abort` 或出现 NaN 时立即中止，不会写入损坏的帧。
- 写输出失败时会中止，而不是悄悄丢帧。
- 模型、结构缺失或不支持的选项组合会在动力学开始前就报错。

## 输出文件

| 文件 | 位置 | 内容 |
|---|---|---|
| `trajectory.traj` | `raw/` | 完整的 ASE 轨迹，包含位置、速度、晶胞 |
| `md.csv` | `raw/` | 每帧的能量、温度、压力、体积 |
| `OUTCAR`、`OSZICAR` | `vasp/` | VASP 兼容的能量和力记录 |
| `XDATCAR` | `vasp/` | VASP 兼容的轨迹（仅 NVT/NVE，定胞） |
| `mliport_results.json` | `.` | 汇总：温度、能量、计时 |
| `resolved_config.json` | `.` | 完整的已解析配置 |
| `run_context.json` | `.` | 模型身份和来源信息 |
| `artifacts.json` | `.` | 文件列表，包含单位和哈希 |
| `run.log` | `.` | 人类可读的日志 |

各文件的通用约定见[输出文件](../outputs.md)。
