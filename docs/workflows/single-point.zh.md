# 单点计算

单点计算在**不移动任何原子**的情况下，计算一个结构的**能量**、**力**和（模型支持时）**应力张量**。它是 mliport 中最基本的工作流：所有其他工作流——优化、MD、NEB——的核心都是单点计算。

## 计算内容

给定一个结构和一个模型，单点计算输出：

- **总能量** $E$，单位 eV——模型对此原子构型的预测值。这不是 DFT 总能量；它的参考基准取决于模型的训练数据。
- **力** $\mathbf{F}_i = -\nabla_i E$，单位 eV/Å——能量对每个原子位置的负梯度。结构优化和 MD 积分的就是这些力。
- **应力张量** $\sigma_{\alpha\beta}$，单位 eV/Å³（模型提供时）——能量对应变的导数。晶胞优化和 NPT 动力学需要它。

单点计算是在正式长时间计算之前，验证模型、结构和设备能否正常协同工作的最低成本方式。

## 命令行

```bash
mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --output runs/sp
```

结构可以是 ASE 能读取的任何格式：CIF、XYZ、POSCAR、extxyz 等。结构必须与模型预期匹配——周期性晶体需要 bulk task（UMA 用 `omat`，其他后端用 `bulk`），孤立分子需要 molecular task（`omol` 或 `molecule`）。

### 不同后端的示例

```bash
# DPA 指定 head
mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda:0 \
  --output runs/dpa-sp

# GRACE
mliport sp structure.cif --model saved_model \
  --model-type grace --output runs/grace-sp

# UMA
mliport sp structure.cif --model uma-s-1p2.pt \
  --model-type uma --task omat --output runs/uma-sp

# NEP（通过 Calorine ASE 计算器）
mliport sp structure.cif --model nep.txt \
  --model-type nep --nep-runtime ase --device cpu \
  --output runs/nep-sp
```

## Python API

```python
from mliport.api import run_single_point, calculate_energy

# 完整的单点计算结果
results = run_single_point(
    structure="structure.cif",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    output_dir="runs/sp",
)
print(f"Energy: {results['energy']:.4f} eV")
print(f"Max force: {max(abs(f).max() for f in results['forces']):.4f} eV/Å")

# 只需要能量（使用临时目录，自动清理）
energy = calculate_energy(
    "structure.cif",
    "mace-omat-0-medium.model",
    model_type="mace",
)
```

`run_single_point` 接受 ASE `Atoms` 对象或文件路径。返回一个字典，包含 `energy`、`forces`、`stress`（如果可用）以及计时信息。`calculate_energy` 是一个便捷封装，只返回能量浮点数。

## 选项

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--model` | *（必填）* | 模型检查点的路径 |
| `--model-type` | *（必填）* | 后端引擎：`mace`、`dpa`、`grace`、`uma` 或 `nep` |
| `--task` | 取决于后端 | task 类别：`bulk`、`molecule`、`omat`、`omol` 等 |
| `--head` | *（自动或必填）* | 当检查点含有多个分支时，指定 DPA 分支或 MACE head |
| `--device` | `cpu` | `cpu`、`cuda`、`gpu` 或 `cuda:N`；请求的和实际使用的设备都会记录 |
| `--charge`、`--spin` | 未设置 | 分子的电子态元数据（UMA `omol`） |
| `--inference-mode` | 后端默认值 | 后端特定的精度和推理设置 |
| `--dtype` | 后端默认值 | 浮点精度（`float32` 或 `float64`） |
| `--write-outcar` | `true` | 写入 VASP 风格的 OUTCAR |
| `--write-stress` | `true` | 在输出中包含应力（如果模型提供） |
| `--overwrite` | `false` | 允许覆盖输出目录中已有的结果 |

## 输出文件

输出目录中包含：

| 文件 | 内容 |
|---|---|
| `OUTCAR` | VASP 兼容的输出，包含能量、力和应力 |
| `CONTCAR` | 输入结构的副本（原子没有移动） |
| `mliport_results.json` | 机器可读的结果：能量、力、应力、计时 |
| `resolved_config.json` | 每个活跃选项及其来源 |
| `run_context.json` | 模型身份、设备、mliport 版本 |
| `run.log` | 人类可读的计算日志 |

各文件的通用约定见[输出文件](../outputs.md)。

## 错误处理

mliport 遇到问题会立即报错：

- 模型无法加载时会给出明确的错误信息——mliport 绝不会自行替换为其他模型或默认后端。
- 多 task 的 DPA 检查点如果没指定 `--head`，会停止并列出可用的 head。
- 与后端不匹配的 task（如对仅支持 bulk 的后端使用 `molecule`）会被拒绝。
- 能量或力出现非有限值（NaN 或无穷大）时会中止运行，不会写入错误的结果。

退出码 `1` 表示计算失败；退出码 `0` 表示计算完成且结果已写入输出目录。

## 如何解读结果

单点能量只能与来自**相同模型身份**（相同模型、head、dtype、task）的另一个单点结果比较。不同模型的绝对能量因参考基准的任意性而不具可比性。详见[模型](../models.md)中关于模型选择和"模型身份"的说明。

如果你需要快速计算能量差——例如两种多型体之间的差异——对两者用同一模型做单点计算然后相减：

```python
from mliport.api import calculate_energy

e_alpha = calculate_energy("alpha.cif", "mace-omat-0-medium.model", model_type="mace")
e_beta  = calculate_energy("beta.cif",  "mace-omat-0-medium.model", model_type="mace")
print(f"ΔE = {(e_beta - e_alpha) * 1000:.1f} meV")
```
