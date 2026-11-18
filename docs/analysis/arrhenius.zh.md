# Arrhenius

`arrhenius` 对几个温度下分别算出的扩散系数做 Arrhenius 拟合，给出活化能，也可以外推到其他温度。
它只拟合你提供的数据点，不会凭空生成。

```bash
mliport analyze runs/md-800K arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11
```

每个数值前都要写一次选项名。温度和扩散系数按顺序一一对应，个数必须相同。命令里的运行目录只决定
结果写到哪里，拟合用的数字全部来自这些选项。

## 物理背景

### Arrhenius 方程

Arrhenius 方程描述热激活过程如何随温度加速：

$$
D(T) = D_0 \, \exp\!\left(- \frac{E_a}{k_B T}\right)
$$

其中 *D*₀ 是指前因子（单位 m²/s），*E*_a 是**活化能**（可移动离子跳跃需要克服的能垒），
*k*_B 是玻尔兹曼常数，*T* 是绝对温度。

对两边取自然对数：

$$
\ln D = \ln D_0 - \frac{E_a}{k_B} \cdot \frac{1}{T}
$$

这是 1/*T* 对 ln *D* 空间中的一条直线。斜率给出 *E*_a，截距给出 *D*₀。

### 活化能的含义

*E*_a 是离子跳跃的有效势垒高度。在固态电解质中：

- **低 *E*_a**（0.1–0.3 eV）→ 快离子传导，如最好的超离子导体（LGPS、Li₇La₃Zr₂O₁₂）。
- **高 *E*_a**（> 0.5 eV）→ 慢扩散，典型的"普通"离子晶体在中等温度下的表现。
- 模拟得到的表观 *E*_a 取决于温度范围、模型、晶胞大小和系综。在与实验对比验证之前，它是模拟
  的性质，不是材料常数。

### Arrhenius 关系何时失效

Arrhenius 方程假设在整个温度范围内只有一种主导机制。以下情况可能导致失效：

- **发生了相变** — 斜率出现突变。
- **多种机制竞争** — 有效活化能是随温度变化的加权平均。
- **体系未达到平衡** — 某个温度下的轨迹不够长，扩散系数没有收敛。

非 Arrhenius 行为在 1/*T* 对 ln *D* 图上表现为弯曲。始终检查线性度和拟合的 r²。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--temperature T` | 温度，单位 K（可重复；必需） |
| `--diffusivity D` | 对应的扩散系数，单位 m²/s（可重复；必需） |
| `--diffusivity-std S` | 每个点的误差（可选），用作拟合权重（可重复） |
| `--extrapolate-temperature T` | 外推到的温度，结果单独记录（可重复） |
| `--source-run-id ID` | 记录每个点来自哪次计算（可重复） |

### 误差加权

提供 `--diffusivity-std` 时，拟合在 log 空间使用**加权最小二乘**。每个点的权重为
1 / σ_ln(D)，其中 σ_ln(D) = σ_D / D（对数的误差传播）。这让确定性更高的点（通常是高温下
MSD 收敛更快的点）获得更大的影响。

不提供 `--diffusivity-std` 时，拟合是无权重的普通最小二乘，对所有点一视同仁。只有两个点时，
直线完全确定，无法估计不确定性。

## 输出

`results.json` 里有活化能 `Ea`、指前因子、拟合质量、输入数据和外推结果，图保存在
`<run>/analysis/arrhenius/<analysis-id>/`。

| 字段 | 内容 |
|---|---|
| `activation_energy_eV` | *E*_a，单位 eV |
| `activation_energy_std_eV` | *E*_a 的不确定性（两个点时为 null） |
| `preexponential_factor_m2_s` | *D*₀，单位 m²/s |
| `r_squared` | 决定系数 |
| `extrapolations` | 每个外推温度的预测 *D* 值，以及是否超出拟合范围 |
| `warnings` | 两点拟合的注意事项、外推警告 |

两组数个数不一致、有非正值、温度少于两个、或者所有温度都相同，都会报错。

## 结果解读

### 拟合质量

- **r² 接近 1** — 数据被单条 Arrhenius 直线很好地描述。只有三个点时，即使有真实散布，
  r² > 0.95 也是常见的。
- **r² < 0.9** — 温度范围跨越了机制变化、某个点是异常值（轨迹未收敛）、或者数据本身噪声大。
- **两个点** — 拟合在数学上精确（r² = 1），不提供统计检验。用三个或更多温度。

### 报告时要写明

- 所有扩散系数必须用同一个模型、同一套 MD 设置（步长、晶胞、系综、漂移处理）算出来，否则拟合
  混入了不同的条件。
- 外推到测量范围以外的结果是预测值，结果里也按预测值记录。外推越远越不可靠。
- 写清温度范围、数据点数、拟合质量、活化能及其不确定性。

### 用 `transport` 的结果作为输入

通常的做法是把各温度下 `transport` 算出的扩散系数填进来。验收测试就是这样做的，用了三个温度下
的 LGPS 结果：

```bash
.venv-grace/bin/mliport analyze results/uma-800K.traj arrhenius \
  --frame-interval-fs 100 --positions-convention unwrapped \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 8.475718e-10 --diffusivity 3.479545e-09 --diffusivity 4.103223e-09 \
  --diffusivity-std 7.356329e-11 --diffusivity-std 2.407381e-10 --diffusivity-std 2.640917e-10
```

命令里的轨迹只决定输出位置，`--positions-convention` 这类轨迹相关的选项在拟合中不起作用，
会记为已忽略。这次拟合得到 `Ea = 0.316 eV`（r² = 0.88）。

### 带外推的例子

```bash
mliport analyze runs/md arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11 \
  --diffusivity-std 0.1e-12 --diffusivity-std 0.2e-12 \
  --diffusivity-std 0.4e-12 --diffusivity-std 0.8e-12 \
  --extrapolate-temperature 300 --extrapolate-temperature 1200
```

如果成功完成的输运计算少于两个，流程会报告 `insufficient_transport_data`，不给出活化能。

## 下一步

- 要计算输入 Arrhenius 拟合的扩散系数：[输运系数](transport.md)
- 了解每个温度下的扩散机制：[GEMDAT](electrolyte-gemdat.md)
- 检查轨迹是否处于扩散区间：[MSD](msd.md)
