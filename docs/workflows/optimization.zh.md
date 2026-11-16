# 结构优化

`mliport opt` 通过在势能面上沿下坡方向迭代，将原子受力（以及可选的晶胞应力）降到收敛阈值以下，从而最小化结构的势能。

## 物理原理

结构优化求解的是：

$$\min_{\{\mathbf{r}_i\}} E(\{\mathbf{r}_i\})$$

约束条件为每个原子 $i$ 的受力 $|\mathbf{F}_i| = |\nabla_i E| < f_\text{max}$。优化器在每一步计算力，然后给出一组新的原子位置，使能量降低。这是一个**局域**最小化过程：它找到的是从初始结构出发最近的极小值，不是全局最小值。

开启 `--cell-opt` 后，晶胞矢量也会利用应力张量进行优化，允许体积和形状调整。这需要模型提供应力、结构为完全周期性、且 task 为 bulk。

## 命令行

```bash
mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/opt
```

### 晶胞优化

```bash
mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.02 --cell-opt --output runs/cell-opt
```

`--cell-opt` 有严格的前提条件：需要应力支持、完全周期性和 bulk task。对于孤立分子、部分周期性结构或不支持应力的模型，它会停下来并给出解释，而不是悄悄地只优化原子位置。

### 不同后端的示例

```bash
# GRACE
mliport opt structure.cif --model saved_model \
  --model-type grace --fmax 0.03 --max-steps 200 \
  --output runs/grace-opt

# DPA 带晶胞优化
mliport opt structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda \
  --fmax 0.02 --cell-opt --output runs/dpa-opt
```

## Python API

```python
from mliport.api import run_optimization

results = run_optimization(
    structure="structure.cif",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    fmax=0.02,
    max_steps=200,
    cell_opt=True,
    output_dir="runs/opt",
)
print(f"Converged: {results['converged']}")
print(f"Steps: {results['nsteps']}")
print(f"Final energy: {results['energy']:.4f} eV")
```

该函数接受 ASE `Atoms` 对象或文件路径。返回一个字典，包含 `energy`、`forces`、`converged`、`nsteps` 以及优化轨迹。

## 选项

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--fmax` | 0.05 eV/Å | 力的收敛阈值 |
| `--max-steps` | 500 | 最大步数上限 |
| `--optimizer` | `FIRE` | 优化算法：`FIRE`、`BFGS` 或 `LBFGS` |
| `--cell-opt` | 关闭 | 同时优化晶胞；需要应力支持、完全 PBC 和 bulk task |
| `--fix-symmetry` | 关闭 | 优化过程中保持晶体对称性 |
| `--model`、`--model-type`、`--task`、`--head`、`--device` | — | 模型和后端选择，同[单点计算](single-point.md) |

### 如何选择优化器

- **FIRE**（默认）：稳健且简单；适用于离平衡位置较远的结构。使用自适应时间步的 velocity Verlet 算法。
- **BFGS**：拟牛顿法，逐步构建近似 Hessian 矩阵。接近极小值时收敛更快，但在力场噪声较大时可能遇困难。
- **LBFGS**：有限内存版 BFGS，对大体系比 BFGS 更省内存。

## 输出文件

| 文件 | 内容 |
|---|---|
| `CONTCAR` | 最终（优化后）的结构 |
| `OUTCAR` | VASP 兼容的每步能量和力的记录 |
| `OSZICAR` | 每步一行的能量记录（类似 VASP 的离子步） |
| `mliport_results.json` | 收敛状态、步数、最终能量和力 |
| `resolved_config.json` | 完整的已解析配置 |
| `run.log` | 人类可读的日志 |

## 收敛与退出码

| 结果 | 退出码 | `converged` 字段 |
|---|---|---|
| 在 `--max-steps` 步内力降到 `--fmax` 以下 | `0` | `true` |
| 达到步数上限仍未收敛 | `2` | `false` |
| 运行时错误或配置错误 | `1` | — |

达到步数上限时，部分轨迹会被保留并记录原因。使用前应检查这个结构——它是一个**未收敛**的优化结果，不是成功的结果。

## 结果的含义

优化后的结构是**当前模型**势能面上的极小值，不一定是 DFT 的极小值。它只在同一模型身份和参考基准下有意义。如果使用了 `--cell-opt`，参考体积已发生变化，在与其他计算比较能量之前请先检查最终晶胞。

常见的工作流程是：先用 MLIP 做优化，然后对 MLIP 优化后的结构做 DFT 单点计算，获取 MLIP 几何构型下的 DFT 能量。这比完整的 DFT 优化便宜得多，但几何结构的精度取决于势函数的质量。
