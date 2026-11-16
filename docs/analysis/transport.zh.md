# 输运（kinisi）

`transport` 用 [kinisi](https://github.com/kinisi-dev/kinisi) 库从 MD 轨迹估计可移动离子的
示踪扩散系数及其统计不确定性，并换算为 Nernst–Einstein 离子电导率。kinisi 通过分块平均和贝叶斯
拟合正确处理 MSD 在不同滞后时间之间的相关性。

```bash
./bin/mliport-mace analyze runs/lgps-md transport \
  --mobile Li --charge 1 --fit-start-ps 50 --temperature-K 800
```

!!! tip "先看 MSD，再算输运"
    先运行 [MSD](msd.md) 任务，确认轨迹处于扩散区间（α ≈ 1）。`transport` 提取的是
    扩散系数——如果轨迹还没有进入扩散区间，无论统计处理多严格，数字都不可靠。

## 物理背景

### 从 MSD 到扩散系数

Einstein 关系把 MSD 的长时间极限和自扩散系数联系起来：

$$
D = \lim_{t \to \infty} \frac{\text{MSD}(t)}{2 \, d \, t}
$$

其中 *d* 是空间维数。实际操作中，在扩散区间对 MSD 拟合直线，从斜率提取 *D*。难点在于估计
这个斜率的**不确定性**。

### 为什么用 kinisi？

对 MSD 做简单线性拟合会低估不确定性，因为相邻滞后时间的 MSD 值是**相关的**——它们共享时间原点。
kinisi 通过以下方式解决这个问题：

1. 将轨迹分块，计算不同滞后时间之间的 MSD 协方差矩阵。
2. 做贝叶斯拟合，考虑这些相关性。
3. 给出 *D* 的后验分布，从中得到可信区间（不确定性）。

这比简单最小二乘更严格，避免报告不真实的小误差棒。

### Nernst–Einstein 电导率

**Nernst–Einstein 关系**把示踪扩散系数和离子电导率联系起来：

$$
\sigma_\text{NE} = \frac{n \, (z \, e)^2 \, D}{k_B \, T}
$$

其中 *n* 是可移动离子的数密度，*z* 是电荷数，*e* 是元电荷，*k*_B 是玻尔兹曼常数，
*T* 是温度。

!!! warning "Nernst–Einstein 假设离子运动不相关"
    Nernst–Einstein 电导率假设每个离子独立扩散。在真实材料中，离子-离子关联（**Haven 比**）
    可能使真实电导率偏离 NE 估计。用 `--collective-conductivity` 计算考虑关联的集体
    （Onsager）电导率。

### Haven 比

Haven 比量化示踪输运和电荷输运之间的偏差：

$$
H_R = \frac{D_\text{tracer}}{D_\text{charge}}
$$

*H*_R = 1 表示不相关的运动（NE 精确）。*H*_R < 1 表示协同运动使电导率高于个体扩散的预测。
*H*_R > 1 表示反关联运动降低了电导率。对很多固态电解质，*H*_R 在 0.3–0.8 之间。

## 用法

### 必需选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动离子（如 `Li`、`Na`） |
| `--charge Z` | 离子电荷，以 *e* 为单位（如 Li⁺ 写 `1`） |
| `--fit-start-ps` | 扩散拟合窗口的起点（ps） |

### 常用选项

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--temperature-K` | — | Nernst–Einstein 换算所用的温度 |
| `--dimensions` | `xyz` | 参与计算的笛卡尔方向（`xyz` 的子集） |
| `--lag-step-ps` | — | 滞后时间网格间距（稀疏化协方差矩阵） |
| `--lag-stop-ps` | — | 最大滞后时间（和 `--lag-step-ps` 要么都写，要么都不写） |
| `--drift-reference` | `nonmobile` | 骨架漂移校正 |
| `--drift-mode` | `arithmetic_mean` | 漂移计算方法 |

### 集体电导率

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--collective-conductivity` | false | 同时计算 Onsager（电荷集体）电导率 |
| `--collective-system-particles` | 1 | kinisi 统计组数（不是独立 MD 副本） |

### 跳跃扩散

| 选项 | 含义 |
|---|---|
| `--jump-diffusion` | 同时运行 kinisi 的跳跃扩散分析，作为诊断参考 |

### 采样控制

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--n-samples` | 1000 | 后验样本数 |
| `--n-walkers` | 32 | MCMC 行走者数 |
| `--n-burn` | 500 | 烧入步数 |
| `--n-thin` | 10 | 稀疏化间隔 |
| `--random-seed` | 0 | 随机种子，保证可复现 |
| `--parser-memory-limit-gib` | 4 | 内存守卫：估计峰值超过此值时拒绝分析 |

### 回退选项

| 选项 | 含义 |
|---|---|
| `--allow-reconstructed-fallback` | 使用 kinisi 未经验证的位移重建方法（结果标记 `publication_grade=false`） |

## 示例

在 800 K 下对 LGPS 中的锂做完整的输运分析：

## `publication_grade` 的含义边界

`publication_grade` 是来源（provenance）标记，不是对物理结论的判定。输运结果里的它为 `true`，只说明
注入 kinisi 的是精确的连续位移数组，且在拟合前逐帧与保存的轨迹核对过（`results.json` 里的
`publication_grade_scope` 写明了这一点）。`--allow-reconstructed-fallback` 诊断路径、以及包含实验性
组件的结果，该标记为 `false`。

`collective_conductivity` 和 `jump_diffusion` 里嵌套的 `publication_grade` 回答的是另一个问题：电荷分组
没有被拆开（`--collective-system-particles 1`），所以跨组关联被保留。分组 `> 1` 属于实验性诊断，会被
显式标记。

两个标记都**不**表示：轨迹足够长、拟合区间处于扩散区间、结果可以外推到别的晶胞尺寸或温度、或者 MLIP
本身准确。这些不是该适配器能判定的，因此每个输运结果都记录
`quality.statistical_adequacy = "not_assessed"` 并给出原因。能否发表要由你自己的收敛检查（lag 窗口的
`alpha`、后验宽度、多副本/多窗口、晶胞尺寸检查）决定，而不是这个标记。

```bash
./bin/mliport-mace analyze runs/lgps-md-800K transport \
  --mobile Li --charge 1 \
  --fit-start-ps 50 --lag-step-ps 5 --lag-stop-ps 500 \
  --temperature-K 800 \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --collective-conductivity --jump-diffusion
```

## 输出文件

结果写入 `<run>/analysis/transport/<analysis-id>/`：

| 文件 | 内容 |
|---|---|
| `results.json` | *D* 及其不确定性、NE 电导率、集体电导率（如果请求了）、采样诊断、溯源信息 |
| `transport_summary.csv` | 主要数字汇总在一行 |

## 结果解读

### 应报告的信息

报告输运结果时要写明：

1. **扩散系数及其不确定性** — 不能只报告 *D* 的均值。
2. **哪种电导率** — 基于示踪扩散的 Nernst–Einstein，还是集体电导率。
3. **拟合窗口** — 从哪里开始和结束，以及为什么这样选。
4. **漂移处理** — 骨架漂移校正方式会改变 *D*。
5. **晶胞** — 定容 NVT 限制扩散机制；NPT 允许晶胞呼吸。太小的晶胞会抑制扩散。

### 几种估计方法的比较

同一条轨迹上的不同估计方法回答的问题不同：

| 方法 | 说明 | 适用场景 |
|---|---|---|
| 简单最小二乘 | MSD 斜率 / (2d)，不处理协方差 | 快速检查 |
| kinisi 后验 | 示踪 *D* 及其可信区间 | 正式发表 |
| Nernst–Einstein | σ = n(ze)²D/(k_BT) | 从示踪 *D* 得到电导率 |
| 集体电导率 | 电荷加权 MSD 的 Onsager 电导率 | 包含关联的完整电导率 |

kinisi 与简单最小二乘的比值偏离 1 很多时，拟合窗口可能不在扩散区间——检查 [MSD 的 α 曲线](msd.md)。

### 失败情况

以下情况会报错并给出明确原因：

- 轨迹对于请求的拟合窗口太短
- 帧间隔未知（外部轨迹没有提供 `--frame-interval-fs`）
- 坐标约定与存储的元数据矛盾
- 估计内存超过 `--parser-memory-limit-gib`

轨迹太短时返回采样不足状态——mliport 不会凭空制造扩散系数。

## 实测例子

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns transport \
  --mobile Li --charge 1 --fit-start-ps 100 \
  --lag-step-ps 1 --lag-stop-ps 200
```

对仓库里 1 ns 的 GRACE LGPS 轨迹，这条命令得到 `D = 1.885e-09 m²/s`，95% 可信区间为
`[1.492e-09, 2.275e-09] m²/s`（拟合区间 100–200 ps）。

## 下一步

- 了解扩散机制（哪些位点、怎么跳跃）：[GEMDAT](electrolyte-gemdat.md)
- 从多个温度提取活化能：[Arrhenius](arrhenius.md)
- 回头检查扩散区间：[MSD](msd.md)
