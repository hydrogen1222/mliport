# CLI 参考

mliport 是一个命令行工具。每次计算都以 `mliport <命令>` 开始，命令决定工作流类型。
本页列出所有命令及其主要选项。详细的物理背景和实例请参阅各工作流页面。

## 全局选项

写在命令名之前：

```bash
mliport [--version] [--verbose] <命令> ...
```

| 选项 | 含义 |
|---|---|
| `--version` | 显示版本并退出 |
| `--verbose` | 增加日志详细程度 |

## 计算命令

### `mliport run`

从 INCAR 风格的输入文件运行计算。模型、结构、计算类型和所有选项都写在一个文件里。

```bash
mliport run --incar INCAR.mliport --output runs/my-calc
```

| 选项 | 必需 | 说明 |
|---|---|---|
| `--incar` | 是 | INCAR 参数文件路径 |
| `--output` | 是 | 输出目录 |
| `--overwrite` | — | 允许覆盖已有输出 |

### `mliport sp`

单点能量、力和应力。

```bash
mliport sp structure.vasp --model model.pt --model-type mace --task bulk \
  --output runs/sp-001
```

主要选项：

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--model` | — | 模型文件路径（必需） |
| `--model-type` | — | 后端：`uma`、`mace`、`dpa`、`grace`、`nep`（必需） |
| `--task` | — | 模型头：`omat`、`omol`、`oc20` 等 |
| `--head` | — | DPA 多任务头 |
| `--device` | `auto` | `auto`、`cuda`、`cpu`、`cuda:N` |
| `--output` | — | 输出目录（必需） |

### `mliport opt`

几何优化（原子位置；可选晶胞）。

```bash
mliport opt structure.vasp --model model.pt --model-type mace --task bulk \
  --fmax 0.01 --output runs/opt-001
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--optimizer` | `FIRE` | 算法：`FIRE`、`BFGS`、`LBFGS` |
| `--fmax` | 0.05 | 力收敛标准，eV/Å |
| `--max-steps` | 500 | 最大优化步数 |
| `--cell-opt` | 关 | 同时优化晶胞形状/体积 |
| `--fix-symmetry` | 关 | 在弛豫过程中保持空间群对称性 |

### `mliport md`

分子动力学。

```bash
mliport md structure.vasp --model model.pt --model-type mace --task bulk \
  --ensemble NVT --temp 600 --steps 50000 --output runs/md-001
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--ensemble` | `NVT` | `NVT`、`NVE` 或 `NPT` |
| `--temp` | 300 | 温度，K |
| `--timestep` | 1.0 | 步长，fs |
| `--steps` | 1000 | MD 步数 |
| `--thermostat` | `LANGEVIN` | `LANGEVIN`、`BUSSI`、`NHC` |
| `--friction` | 0.001 | Langevin 摩擦系数，1/fs |
| `--bussi-tau` | 1000 | Bussi/CSVR 耦合时间，fs |
| `--nhc-tdamp` | 100 | NHC 恒温器阻尼时间，fs |
| `--save-interval` | 10 | 每 *n* 步保存一次 |
| `--equilibration-steps` | 0 | 生产阶段前的平衡步数 |
| `--pre-relax` | 自动 | MD 前快速弛豫位置 |

**NPT 专用选项**（需要 UMA、MACE、GRACE 或 DPA）：

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--pressure-gpa` | 0.0 | 目标压强，GPa |
| `--npt-tdamp` | 100 | MTK 恒温器阻尼，fs |
| `--npt-pdamp` | 1000 | MTK 恒压器阻尼，fs |

### `mliport neb`

微动弹性带法（NEB）过渡态搜索。

```bash
mliport neb initial.vasp final.vasp --model model.pt --model-type mace \
  --task bulk --output runs/neb-001
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--n-images` | 7 | 中间镜像数 |
| `--climb` | 关 | 启用爬坡镜像 NEB |
| `--neb-method` | `improvedtangent` | NEB 方法 |
| `--fmax` | 0.03 | 力收敛标准，eV/Å |
| `--max-steps` | 1000 | 最大 NEB 步数 |
| `--endpoint-policy` | `validate` | `validate`、`relax` 或 `none` |

### `mliport confsearch`

构型搜索（吸附物放置 + 多周期退火 + 淬火）。

```bash
mliport confsearch run slab.vasp adsorbate.xyz \
  --model model.pt --model-type mace --task bulk --output runs/cs-001
```

子命令：`run`、`place`、`resume`、`extend`、`merge`、`export`。

### `mliport batch`

对多个结构进行批量单点或优化计算。

```bash
mliport batch sp structures/*.cif \
  --model model.pt --model-type mace --task bulk --output runs/batch-sp
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--resume` | 关 | 跳过已计算的结构 |
| `--max-workers` | 自动 | 并行工作线程数 |

## 分析

```bash
mliport analyze <运行目录或轨迹> <任务> [选项]
```

数据源（运行目录或轨迹文件）写在任务名**之前**。

| 任务 | 说明 | 主要必需选项 |
|---|---|---|
| `validate` | 轨迹元数据和任务兼容性 | — |
| `thermo` | 温度、能量漂移、压强 | — |
| `rdf` | 径向分布函数 | `--center`、`--neighbor` |
| `rmsd` | 周期性位移 RMSD/RMSF | — |
| `msd` | 均方位移 | `--mobile` |
| `transport` | 示踪扩散系数和 NE 电导率 | `--mobile`、`--charge`、`--fit-start-ps` |
| `density` | 三维数密度图 | `--mobile` |
| `arrhenius` | 从 D(T) 数据做 Arrhenius 拟合 | `--temperature` (×N)、`--diffusivity` (×N) |
| `electrolyte` | GEMDAT 位点分析 | `--mobile` |
| `vacf` | 速度自相关函数 | — |
| `spectrum` | 由 VACF 得到振动光谱 | — |

完整说明请参阅[分析](analysis/index.md)。

## 工具命令

### `mliport config`

管理配置。子命令：`show`、`validate`、`resolve`。

### `mliport template`

为指定计算类型生成 INCAR 模板。

```bash
mliport template md > INCAR.mliport
```

### `mliport doctor`

检查安装和硬件。

```bash
mliport doctor --engine mace --device auto
```

| 选项 | 说明 |
|---|---|
| `--engine` | 探测特定后端 |
| `--device` | 探测特定设备 |
| `--json` | 机器可读 JSON 输出 |

### `mliport setup`

将后端安装到隔离的虚拟环境中。

```bash
mliport setup --engines mace uma --device auto
```

### `mliport queue`

后台作业管理。子命令：`submit`、`status`、`cancel`、`log`。

### `mliport jobs`

列出后台作业。

### `mliport kill`

按 ID 终止运行中的后台作业。

### `mliport clean`

清除已完成运行的中间文件。

### `mliport xdatcar`

在轨迹格式之间转换（XDATCAR ↔ ASE .traj）。

### `mliport state`

查看或修改运行状态。子命令：`show`、`reset`。
