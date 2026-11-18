# 均方位移（MSD）

`msd` 任务计算可移动原子的均方位移和局部指数 α(t)，告诉你轨迹处于什么扩散状态。

```bash
./bin/mliport-mace analyze runs/lgps-md msd --mobile Li --axes xyz
```

## 物理背景

### 什么是 MSD？

**均方位移**衡量原子从初始位置走了多远，对所选物种的所有原子和所有可能的时间起点
取平均：

$$
\text{MSD}(t) = \left\langle \frac{1}{N} \sum_{i=1}^{N} \left| \mathbf{r}_i(t_0 + t) - \mathbf{r}_i(t_0) \right|^2 \right\rangle_{t_0}
$$

其中 *N* 是可移动原子数，**r**ᵢ 是原子 *i* 的展开（unwrapped）坐标，尖括号表示对
时间起点 *t*₀ 求平均。

在自由扩散（Fick 扩散）的体系中，MSD 随时间线性增长：

$$
\text{MSD}(t) = 2 d \, D \, t
$$

其中 *d* 是空间维数（`--axes xyz` 时 d=3，`--axes xy` 时 d=2 等），*D* 是**自扩散
系数**。这就是 **Einstein 关系** — 从 MD 模拟中提取扩散系数的理论基础。

!!! important "必须使用展开坐标"
    MSD 需要**展开**（unwrapped）坐标 — 跟踪每个原子真实位移、不折回模拟盒子的
    坐标。mliport 运行目录始终存储展开坐标。对于外部轨迹，要指定
    `--positions-convention unwrapped`；如果约定错误，MSD 会被人为限制在晶胞尺寸以内。

### 局部指数 α(t)

局部指数刻画 MSD 随时间的瞬时标度行为：

$$
\alpha(t) = \frac{d \ln \text{MSD}}{d \ln t}
$$

其值对应不同的扩散状态：

| α 值 | 状态 | 物理含义 |
|---|---|---|
| α ≈ 2 | 弹道运动 | 极短时间 — 原子还没碰撞 |
| α > 1 | 超扩散 | 关联运动、定向流动 |
| **α ≈ 1** | **扩散（Fick）** | **正常扩散 — Einstein 关系成立** |
| α < 1 | 亚扩散 | 笼效应、陷阱、受限运动 |
| α ≈ 0 | 局域化 | 原子在原地振动，无净输运 |

典型的固态电解质轨迹在最短时间尺度上 α ≈ 2（弹道振动），在中间时间尺度 α 下降
到 1 以下（笼内振荡 — 原子困在配位壳层里），最终在长时间尺度趋近 1（原子逃逸并
扩散）。α 达到 1 的时刻是 Fick 扩散的起始 — 只有在此之后提取的扩散系数才有意义。

!!! warning "MSD 本身不是扩散系数"
    `msd` 任务是**诊断工具**，它告诉你轨迹是否足够长、已进入扩散状态。要提取带
    不确定度的扩散系数，请用 [transport](transport.md) 任务，它使用 kinisi 的分块
    平均方法。

## 用法

### 必需选项

| 选项 | 说明 |
|---|---|
| `--mobile SPECIES` | 扩散物种（如 `Li`、`Na`、`O`）。**必需。** |

### 常用选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--axes` | `xyz` | 空间维度：`x`、`y`、`z` 或组合如 `xy`、`xyz` |
| `--drift-reference` | `nonmobile` | 漂移扣除的参考系：`none`、`nonmobile`、`all`、`indices` |
| `--drift-mode` | `mass_weighted_com` | 漂移计算方式：`none`、`arithmetic_mean`、`mass_weighted_com` |
| `--method` | `fft` | MSD 算法：`fft`（快速傅里叶，O(N log N)）或 `direct`（O(N²)） |

### 拟合窗口

| 选项 | 说明 |
|---|---|
| `--fit-start-ps` | 线性拟合窗口起始（皮秒） |
| `--fit-stop-ps` | 线性拟合窗口终止 |

拟合窗口记录在结果中，但不影响原始 MSD — 它标记你认为属于扩散区域的时段。

### Alpha 控制

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--alpha-window-decades` | 0.25 | 局部对数回归窗口的全宽度 |
| `--alpha-min-points` | 5 | 窗口内的最少数据点数 |
| `--alpha-min-origins` | 8 | 所需的最少时间起点数 |
| `--alpha-consistency-band` | 0.2 | α = 1 附近的半带宽，视为"扩散" |
| `--alpha-max-lag-ps` | — | 只忽略 α 中的长时延迟（原始 MSD 不变） |
| `--alpha-focus-band LOW HIGH` | — | α 图的纵轴范围 |
| `--alpha-log-x` | false | 对 α 图使用对数时间轴 |

### 分析窗口

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--start-frame` | — | 起始帧 |
| `--stop-frame` | — | 结束帧 |
| `--include-equilibration` | false | 包含平衡阶段的帧（默认排除） |

## 解读结果

### 怎么看 α 曲线

看 **α(t) 曲线的整体形状**，不要只看单个数字：

1. **短时间**（< 1 ps）：α ≈ 2（弹道）。正常。
2. **中间时间**（1–50 ps）：α 下降到 1 以下。这是笼内振荡 — 原子在局域环境中振动。
3. **长时间**（> 50 ps，因体系而异）：α 趋近 1。这是扩散的起始。此区域的 MSD
   才有意义。

如果 α 始终未进入一致性带（默认 1 ± 0.2），说明轨迹太短或该温度下体系不扩散。
mliport 会标记为 `not_diffusive`。

### 漂移扣除

框架漂移 — 晶格的缓慢整体位移 — 必须在计算可移动物种的 MSD 前扣除。默认设置
（`--drift-reference nonmobile --drift-mode mass_weighted_com`）扣除所有不可移动
原子的质心运动。务必报告所用的漂移策略。

### 示例

800 K 下 LGPS 中锂扩散的完整 MSD 分析：

```bash
./bin/mliport-mace analyze runs/lgps-md-800K msd \
  --mobile Li --axes xyz \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --fit-start-ps 50 --fit-stop-ps 400 \
  --alpha-focus-band 0.5 1.5 --alpha-log-x
```

## 输出文件

结果写到 `<run>/analysis/msd/<analysis-id>/`：

| 文件 | 内容 |
|---|---|
| `results.json` | MSD 值、α 值、拟合参数、采样状态 |
| `msd.csv` | 时间（ps）对 MSD（Å²）的表格 |
| `alpha.csv` | 时间（ps）对 α(t) 的表格 |
| `msd.png`、`msd.svg` | MSD 图 |
| `alpha.png`、`alpha.svg` | 局部指数图（带一致性带） |

## 短轨迹

5 步轨迹也能接受 — 分析会运行并报告明确的"扩散窗口不足"状态。这对自动化测试和
流水线验证有用。实际的 MSD 测量至少需要几百个保存帧，并让拟合窗口覆盖 α ≈ 1 的区域。

## 下一步

- 提取带不确定度的扩散系数：[输运](transport.md)
- 映射产生扩散的位点和跳跃：[GEMDAT](electrolyte-gemdat.md)
- 将多温度扩散系数组合为活化能：[Arrhenius](arrhenius.md)
