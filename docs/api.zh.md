# Python API

mliport 提供六个高层函数，用于从 Python 脚本运行 MLIP 计算。它们是 CLI 子命令
（`sp`、`opt`、`md`、`neb`、`confsearch`）的程序化等价物，共享同一个配置解析器，
因此内置默认值、模型别名和 `settings.ini` 配置文件的行为与命令行完全一致。

```python
from mliport.api import (
    calculate_energy,
    run_single_point,
    run_optimization,
    run_md,
    run_neb,
    run_confsearch,
)
```

## 何时使用 Python API

CLI 适合一次性计算和批处理脚本。Python API 适合：

- **筛选循环** — 遍历一系列结构，将能量或力收集到 DataFrame 中。
- **多阶段工作流** — 先优化结构，再跑 MD，再分析，全在一个脚本中完成，
  可以根据结果进行程序化分支。
- **Jupyter notebook** — 交互式探索结果，无需调用 shell。
- **与其他代码集成** — 从 ASE 优化器、遗传算法或工作流管理器中调用 mliport。

## 公共参数

六个函数共享以下参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `structure` | `Atoms \| str \| Path` | ASE `Atoms` 对象，或结构文件路径（CIF、POSCAR、XYZ 等）。 |
| `model_path` | `str` | 模型文件路径。 |
| `model_type` | `str \| None` | 后端引擎：`"uma"`、`"mace"`、`"dpa"`、`"grace"` 或 `"nep"`。`None` 由解析器决定（从别名或设置）。 |
| `task` | `str \| None` | 模型 head / 任务名（如 `"omat"`、`"omol"`、`"bulk"`）。 |
| `device` | `str \| None` | `"cpu"` 或 `"cuda"`。`None` = 自动检测。 |
| `output_dir` | `str` | 输出文件目录。默认为 `"./results"`。 |
| `job_name` | `str \| None` | 可选的作业名称，用于在输出目录内组织结果。 |
| `verbose` | `bool` | 打印进度到 stdout。默认 `True`（`calculate_energy` 默认 `False`）。 |
| `settings_path` | `str \| None` | 显式指定 `settings.ini` 文件路径。 |
| `model_alias` | `str \| None` | `settings.ini` 中 `[model:NAME]` 节定义的模型别名。 |
| `profile` | `str \| None` | `settings.ini` 中 `[profile:NAME]` 节定义的可复用配置。 |
| `strict_config` | `bool \| None` | 覆盖严格配置行为。 |
| `overwrite` | `bool` | 允许覆盖已有输出（默认 `False`）。 |
| `default_dtype` | `str \| None` | MACE 数据类型覆盖（`"float32"` 或 `"float64"`）。 |
| `head` | `str \| None` | MACE head 或 DeepMD/DPA 多任务分支名。 |
| `inference_mode` | `str \| None` | 推理模式覆盖（例如 UMA 的 `"turbo"` 模式需要显式启用）。 |
| `nep_runtime` | `str \| None` | NEP 运行时：`"ase"` 或 `"gpumd"`。`model_type="nep"` 时必填。 |
| `nep_gpumd_command` | `str \| None` | `gpumd` 可执行文件路径（`nep_runtime="gpumd"` 时需要）。 |

所有函数还接受 `**kwargs`，额外选项会转发给配置解析器。

---

## `run_single_point`

计算结构的能量、力和应力，不移动任何原子。

```python
result = run_single_point(
    structure="POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    output_dir="runs/sp",
)
```

**返回值：** `dict`，包含 `energy`（eV）、`forces`（eV/Å 数组）、
`stress`（Voigt 6 分量，eV/Å³）和 `time`（秒）等键。

### 示例：在同一结构上比较不同后端

```python
from mliport.api import run_single_point

backends = {
    "mace": {"model_path": "mace-omat-0-medium.model", "task": "bulk"},
    "uma":  {"model_path": "uma-s-1p2.pt", "task": "omat"},
}

for name, opts in backends.items():
    result = run_single_point(
        "structure.cif",
        model_type=name,
        output_dir=f"runs/compare-{name}",
        **opts,
    )
    print(f"{name}: {result['energy']:.4f} eV")
```

---

## `run_optimization`

优化原子位置（可选同时优化晶胞参数），直到力收敛到阈值以下。

```python
result = run_optimization(
    structure="POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    fmax=0.02,           # eV/Å
    max_steps=500,
    optimizer="FIRE",    # FIRE、BFGS 或 LBFGS
    cell_opt=True,       # 同时优化晶胞
    output_dir="runs/opt",
)
```

**额外参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `fmax` | `float \| None` | 解析器默认值 | 力收敛阈值（eV/Å）。 |
| `max_steps` | `int \| None` | 解析器默认值 | 最大优化步数。 |
| `optimizer` | `str \| None` | 解析器默认值 | 算法：`"FIRE"`、`"BFGS"` 或 `"LBFGS"`。 |
| `cell_opt` | `bool \| None` | `False` | 是否优化晶胞参数。 |
| `fix_symmetry` | `bool \| None` | `False` | 优化过程中保持对称性。 |

**返回值：** `dict`，包含 `energy`、`converged`（bool）、`nsteps`（int）、
`forces`、`stress` 等。

### 示例：批量优化一组结构

```python
from pathlib import Path
from mliport.api import run_optimization

structures = Path("candidates").glob("*.cif")
for cif in sorted(structures):
    result = run_optimization(
        cif,
        model_path="mace-omat-0-medium.model",
        model_type="mace", task="bulk",
        fmax=0.02, max_steps=300,
        output_dir=f"runs/opt-{cif.stem}",
    )
    status = "✓" if result["converged"] else "✗"
    print(f"{cif.name}: {result['energy']:.4f} eV  {status}")
```

---

## `run_md`

运行分子动力学模拟。

```python
result = run_md(
    structure="POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    ensemble="NVT",
    temperature=600,        # K
    timestep=1.0,           # fs
    steps=10000,
    thermostat="LANGEVIN",
    save_interval=10,
    output_dir="runs/md",
)
```

**额外参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `ensemble` | `str \| None` | `"NVT"`、`"NVE"` 或 `"NPT"`。 |
| `temperature` | `float \| None` | 温度（K）。 |
| `timestep` | `float \| None` | 时间步长（fs）。 |
| `steps` | `int \| None` | MD 步数。 |
| `thermostat` | `str \| None` | NVT 恒温器：`"LANGEVIN"`、`"BUSSI"` 或 `"NHC"`。 |
| `friction` | `float \| None` | Langevin 摩擦系数（1/fs）。 |
| `bussi_tau` | `float \| None` | Bussi/CSVR 耦合时间（fs）。 |
| `nhc_tdamp` | `float \| None` | Nosé-Hoover 链阻尼时间（fs）。 |
| `nhc_tchain` | `int \| None` | Nosé-Hoover 链长度。 |
| `nhc_tloop` | `int \| None` | Nosé-Hoover 恒温器积分子步数。 |
| `save_interval` | `int \| None` | 保存轨迹帧的间隔。 |
| `pre_relax` | `bool \| None` | MD 前先预优化位置（减少初始力尖峰）。 |
| `com_policy` | `str \| None` | 质心策略：`"auto"`、`"none"`、`"initialize_only"` 或 `"constraint"`。 |

**返回值：** `dict`，包含 `temperature`、`energy`、`time` 等。

!!! note "NPT 系综"
    NPT 分子动力学使用 ASE `IsotropicMTKNPT` 积分器，仅 ASE 后端
    （MACE、DPA、GRACE、UMA）支持。NEP 的 `gpumd` 运行时不支持 NPT。NPT 额外
    参数：`pressure_gpa`、`npt_tdamp`、`npt_pdamp`、`npt_tchain`、`npt_pchain`、
    `npt_tloop`、`npt_ploop`。

### 示例：使用 Bussi 恒温器的 NVT MD

```python
from mliport.api import run_md

result = run_md(
    "relaxed_CONTCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace", task="bulk",
    ensemble="NVT",
    thermostat="BUSSI",
    bussi_tau=100.0,      # 100 fs 耦合时间
    temperature=800,
    timestep=2.0,
    steps=50000,
    save_interval=50,
    pre_relax=True,
    output_dir="runs/bussi-md",
)
```

---

## `run_neb`

运行 NEB（弹性带方法）或 CI-NEB（攀爬镜像 NEB）计算，寻找两个端点结构之间的最小
能量路径。

```python
result = run_neb(
    initial="initial_POSCAR",
    final="final_POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    n_intermediate_images=7,
    climb=True,
    output_dir="runs/neb",
)
```

**额外参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `initial` | `Atoms \| str \| Path \| None` | 初始端点结构。新运行必填。 |
| `final` | `Atoms \| str \| Path \| None` | 终态端点结构。新运行必填。 |
| `resume` | `str \| Path \| None` | 检查点目录路径，用于恢复运行。 |
| `n_intermediate_images` | `int \| None` | 沿带的中间镜像数。 |
| `climb` | `bool \| None` | 启用攀爬镜像 NEB。 |
| `method` | `str \| None` | NEB 方法。 |
| `interpolation` | `str \| None` | 初始路径的插值方案。 |
| `spring` | `float \| None` | NEB 弹簧常数（eV/Å²）。 |
| `fmax` | `float \| None` | 力收敛阈值（eV/Å）。 |
| `max_steps` | `int \| None` | 最大优化步数。 |
| `pre_fmax` | `float \| None` | 预优化的力阈值。 |
| `pre_max_steps` | `int \| None` | 预优化的步数上限。 |
| `endpoint_policy` | `str \| None` | 端点优化的处理策略。 |
| `checkpoint_interval` | `int \| None` | 全带检查点间隔步数。 |

**返回值：** `dict`，包含带能量、势垒高度、收敛状态和逐镜像数据。

### 示例：恢复 NEB 运行

```python
from mliport.api import run_neb

# 从原始输出目录中的检查点恢复
result = run_neb(
    resume="runs/neb",
    output_dir="runs/neb",
)
print(f"Barrier: {result['barrier']:.3f} eV")
```

!!! warning "NEB 为固定晶胞"
    mliport 的 NEB 工作流在固定晶胞下运行。不支持变晶胞 NEB。

---

## `run_confsearch`

运行基于 MD 的构型搜索，用于表面上的吸附物放置。工作流通过高温和低温 MD 循环
探索构型空间，然后对最稳定的构型进行排序和导出。

```python
result = run_confsearch(
    slab="slab.vasp",
    molecule="molecule.xyz",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    n_starts=5,
    cycles=3,
    output_dir="runs/confsearch",
)
```

**关键参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `starts` | `Any` | 预先构建的起始构型（路径或路径列表）。 |
| `slab` | `str \| Path \| None` | slab 结构的路径。 |
| `molecule` | `str \| Path \| None` | 吸附物分子的路径。 |
| `n_starts` | `int \| None` | 随机起始放置的数量。 |
| `adsorbate_indices` | `str \| None` | 作为吸附物的原子索引。 |
| `cycles` | `int \| None` | 加热/冷却循环的次数。 |
| `mode` | `str \| None` | 搜索模式。 |
| `temperature_high` | `float \| None` | 高温 MD 目标温度（K）。 |
| `temperature_low` | `float \| None` | 低温淬火目标温度（K）。 |
| `top_k` | `int \| None` | 导出的排名最高构型数。 |
| `export_window` | `float \| None` | 导出的能量窗口（eV）。 |
| `refine` | `bool \| None` | 是否细化最优候选。 |
| `dry_run` | `bool` | 仅验证输入不运行（默认 `False`）。 |
| `resume` | `bool` | 恢复之前的搜索（默认 `False`）。 |

**返回值：** `dict`，内容为 `search_results.json` — 排序后的候选构型及其能量和结构。

!!! note "筛选而非全局优化"
    排序结果是一种筛选产物：它报告找到的最低能量极小值并附带饱和度证据，
    而非保证的全局最小值。导出的候选构型用于后续 DFT 重新排序。

---

## `calculate_energy`

便利函数，仅返回能量（eV）作为浮点数。默认使用临时目录，因此重复调用不会冲突。

```python
from mliport.api import calculate_energy

energy = calculate_energy(
    "structure.cif",
    "mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
)
print(f"{energy:.4f} eV")
```

**额外参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `relax` | `bool` | `False` | 计算能量前先优化结构。 |
| `fmax` | `float \| None` | 解析器默认值 | 预优化的力阈值。 |
| `max_steps` | `int \| None` | 解析器默认值 | 预优化的步数上限。 |

**返回值：** `float` — 能量，单位 eV。

### 示例：候选结构的能量筛选

```python
from pathlib import Path
from mliport.api import calculate_energy

candidates = sorted(Path("structures").glob("*.cif"))
energies = []
for cif in candidates:
    e = calculate_energy(
        cif,
        "mace-omat-0-medium.model",
        model_type="mace", task="bulk",
    )
    energies.append((cif.name, e))
    print(f"{cif.name}: {e:.4f} eV")

# 按能量排序
for name, e in sorted(energies, key=lambda x: x[1]):
    print(f"{name}: {e:.4f} eV")
```

---

## 配置解析

所有 API 函数经过与 CLI 相同的 `resolve_config()` 管线。优先级从高到低：

1. 函数的显式关键字参数
2. CLI 层覆盖（来自 `**kwargs`）
3. INCAR 文件值（适用时）
4. 配置文件值（`settings.ini` 中的 `[profile:NAME]`）
5. 模型别名值（`settings.ini` 中的 `[model:NAME]`）
6. 设置默认值（`settings.ini` 中的 `[defaults]`）
7. 内置默认值

这意味着你可以在 `settings.ini` 中设定常用选项，每次调用时只覆盖不同的部分：

```ini
; settings.ini
[defaults]
device = cuda
strict_config = true

[model:mace-omat]
model_path = /models/mace-omat-0-medium.model
model_type = mace
task = bulk
```

```python
from mliport.api import run_single_point

# model_path、model_type 和 task 来自别名
result = run_single_point(
    "structure.cif",
    model_path="",   # 别名提供时不使用
    model_alias="mace-omat",
    output_dir="runs/sp",
)
```

详见[配置](configuration.md)。

## 错误处理

所有函数抛出标准 Python 异常：

- `ValueError` — 无效输入（结构文件不存在、选项不兼容、恢复时未指定 output_dir）。
- `FileExistsError` — 输出目录已有结果且 `overwrite=False`。
- 引擎特定错误 — 模型加载失败、CUDA 错误等。

函数**不会**捕获和静默错误。如果计算失败，异常会带着描述性信息传播。

## 线程安全

每次函数调用创建自己的 `CalculationEngine` 实例并同步运行。使用不同输出目录的
多线程调用是安全的。GPU 计算通常不应在单进程内并行 — 请使用多进程。
