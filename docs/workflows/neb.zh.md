# NEB 与 CI-NEB

`mliport neb` 使用 nudged elastic band (NEB) 方法寻找两个端点结构之间的**最小能量路径**，并报告能垒。可选的 climbing image (CI-NEB) 变体能精修鞍点估计。

## 物理原理

Nudged elastic band 方法（Henkelman et al., J. Chem. Phys. **113**, 9901, 2000）在两个已优化的端点之间串联一组 image，然后同时优化所有 image。弹簧力使 image 沿路径均匀分布，同时垂直于路径方向的真实力驱动每个 image 趋向最小能量路径。

**CI-NEB**（Henkelman et al., J. Chem. Phys. **113**, 9978, 2000）中，能量最高的 image 不再受弹簧力约束，而是沿路径方向向上攀爬以逼近鞍点。这样无需增加 image 数量就能获得更好的能垒估计。

能垒定义为：

$$E_\text{barrier} = E_\text{saddle} - E_\text{reactant}$$

其中 $E_\text{saddle}$ 是收敛能带上最高 image 的能量。mliport 报告的是**采样**能垒（最高 image 减去端点），而非拟合的鞍点。

## 命令行

```bash
mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace \
  --images 7 --climb --output runs/neb
```

你需要提供两个原子顺序一致的结构（顺序不同时使用 `--atom-map` 指定对应关系）、一个明确指定后端的模型，以及足够容纳 `images + 2` 个计算器副本的内存。

### 使用 IDPP 插值

```bash
mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace \
  --images 7 --climb --interpolation idpp \
  --output runs/neb-idpp
```

IDPP（image-dependent pair potential）插值（Smidstrup et al., J. Chem. Phys. **140**, 214106, 2014）通过最小化原子间距的变化来构建比线性插值更好的初始路径。对于原子大幅重排、线性插值会导致原子重叠的情况，推荐使用。

## Python API

```python
from mliport.api import run_neb

results = run_neb(
    initial="initial.vasp",
    final="final.vasp",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    n_intermediate_images=7,
    climb=True,
    output_dir="runs/neb",
)
print(f"Forward barrier: {results['barrier_forward_sampled_eV']:.3f} eV")
print(f"Reverse barrier: {results['barrier_reverse_sampled_eV']:.3f} eV")
print(f"Reaction energy: {results['reaction_energy_eV']:.3f} eV")
```

## 梯度预检：为何要先检查

NEB 只有在模型的力是能量的梯度时才有意义。否则能带是按照与能量面不一致的力来优化的，得到的能垒毫无意义。因此 mliport 会先做检查。

在端点和插值 image 上，它沿多个方向比较力的投影 $-\mathbf{F} \cdot \hat{\mathbf{u}}$ 与能量的中心有限差分。两个步长通过 Richardson 外推组合，容差考虑了模型 dtype 的舍入噪声。以下情况会拒绝运行：

- 力不是能量的梯度
- 出现非有限值
- 能量分辨率不足以下定论（例如大的 float32 晶胞）
- 模型直接预测力（`direct_forces = true`；一律拒绝）

检查结果保存在 `run_context.json` 和 `neb_results.json` 的 `gradient_preflight` 中。这是在几个点上的抽查，并非对整个势能面的证明。

## 选项

### 能带设置

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--initial`、`--final` | *（必填）* | 端点结构 |
| `--images` | 7 | 中间 image 的数量（不含端点） |
| `--climb` | 关闭 | 启用 climbing image 以精修鞍点 |
| `--interpolation` | `linear` | 初始路径插值方式：`linear` 或 `idpp` |
| `--path-convention` | `mic` | `mic`（最小映像约定）或 `unwrapped` |
| `--spring` | 0.1 eV/Å² | image 间的弹簧常数 |
| `--atom-map` | — | 按初态原子顺序列出对应的末态原子编号（0-based），逗号分隔 |
| `--image-shifts` | — | 每个原子的整数晶格平移，如 `0,0,0;1,0,0` |

### 收敛

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--fmax` | 0.03 eV/Å | 力的收敛阈值 |
| `--max-steps` | 1000 | 步数上限 |
| `--pre-fmax` | 0.1 eV/Å | 第一阶段阈值（CI 启用前） |
| `--pre-max-steps` | 300 | 第一阶段步数上限 |
| `--maxstep` | 0.1 Å | 优化器最大步长 |
| `--min-distance` | 0.5 Å | image 中原子间距小于此值时拒绝 |

### 端点处理

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--endpoint-policy` | `validate` | `validate`（检查端点已优化）或 `relax`（先优化端点） |
| `--endpoint-fmax` | 0.02 eV/Å | 端点力的阈值 |
| `--endpoint-steps` | 500 | 端点优化的步数预算（使用 `relax` 时） |

### IDPP 调参

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--idpp-fmax` | — | IDPP 收敛阈值 |
| `--idpp-steps` | — | IDPP 步数上限 |
| `--idpp-mic` | — | IDPP 是否使用最小映像约定 |

## 端点策略：检查还是优化

`--endpoint-policy validate`（默认）要求端点已经优化好：如果任一端点的力超过 `--endpoint-fmax`，mliport 会停下来并指出是哪一端。使用 `relax` 时会先优化两个端点，然后再构建能带。无论哪种方式，端点的处理都会改变参考能量，因此务必记录使用了哪种策略。

## 输出文件

| 文件 | 内容 |
|---|---|
| `neb_results.json` | 能量、力、收敛状态、能垒数值 |
| `run_context.json` | 模型身份和梯度预检结果 |
| `resolved_config.json` | 已解析配置 |
| `artifacts.json` | 文件列表及哈希 |
| `checkpoints/` | 定期保存的完整能带检查点 |
| `vasp_path/` | 每个 image 一个 `POSCAR`（VASP NEB 布局） |

`neb_results.json` 中的能垒字段：

- `barrier_forward_sampled_eV`：最高 image 减去初态能量
- `barrier_reverse_sampled_eV`：最高 image 减去终态能量
- `reaction_energy_eV`：终态减去初态能量
- `barrier_status`：是否找到内部峰值

## 续算

```bash
mliport neb --resume runs/neb --output runs/neb
```

续算在原来的目录里继续，并核对记录下来的信息：模型、task、head、精度，晶胞和 PBC，约束，能带
设置和各项参数。当前 beta 版本里步数上限也在核对范围内，续算时改 `--max-steps` 会报
`Resume fingerprint is incompatible` 并拒绝运行，错误信息会列出不一致的字段（例如 `neb_options.max_steps (checkpoint 200 != requested 123)`）。省略某个选项表示沿用检查点记录的值，显式传入相同的值则允许；被拒绝时检查点保持不变。

续算恢复能带几何结构但重启优化器；它不重放之前的优化器状态。收敛状态和最终能垒可能与一次不中断的运行不同，因此务必验证收敛。

## 如何解读能垒

- 收敛的 NEB 鞍点候选并非经过验证的过渡态。需要做 Hessian（频率）计算来确认它沿反应坐标恰好有一个虚频模式。
- 能垒只在同一模型身份和同一 NEB 设置（image 数、插值方式、端点策略、弹簧常数）下有意义。
- 原子对应关系在化学上必须正确——错误的映射会产生虚假能垒。

## 功能支持矩阵

并非每个 NEB 选项都能在每个入口使用：

| 功能 | API | CLI | INCAR | TUI |
|---|---|---|---|---|
| 端点 | 支持 | 支持 | 支持 | 支持 |
| 原子映射 | 支持 | 支持 | 不支持 | 支持 |
| image 晶格平移 | 支持 | 支持 | 不支持 | 支持 |
| 续算 | 支持 | 支持 | 不支持 | 支持 |
| 路径约定 | 支持 | 支持 | 支持 | 支持 |
| 插值方式 | 支持 | 支持 | 支持 | 支持 |
| Climbing image | 支持 | 支持 | 支持 | 支持 |
| 预优化 | 支持 | 支持 | 支持 | 支持 |
| 最小距离保护 | 支持 | 支持 | 支持 | 支持 |

"不支持"意味着该功能需要通过 API、CLI 或 TUI 使用——不会被静默地翻译。
