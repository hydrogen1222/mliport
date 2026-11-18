# GEMDAT 电解质分析

`electrolyte` 用 [GEMDAT](https://gemdat.readthedocs.io/) 把可移动离子的位置归到各个晶体学位点上，
统计位点之间的跳跃，并寻找渗流通道。它回答的是**机理**问题：离子停在哪些位点、怎样跳跃。扩散快慢
请用 `transport` 分析。

```bash
mliport analyze runs/md electrolyte --mobile Li --sites sites.yaml --temperature-K 800
mliport analyze runs/md electrolyte --mobile Li --discover-sites-from-density
```

!!! tip "先看输运，再看机理"
    先运行 [transport](transport.md) 确认体系存在扩散并测量其速率。GEMDAT 回答的是扩散*怎么*
    发生——哪些位点被占据、哪些跳跃频繁、是否存在渗流通道。

## 物理背景

### 基于位点的机理分析

在晶态固体电解质中，可移动离子不是在均匀介质中扩散。它们在离散的**晶体学位点**——势能面上的
局部极小值——之间跳跃。输运机制取决于哪些位点存在、它们如何连接，以及离子在它们之间跳跃的频率。

GEMDAT 把连续的 MD 轨迹转化为离散的事件序列：

1. **定义位点** — 从已知晶体结构获取，或从可移动离子的时间平均密度中识别。
2. **逐帧映射** — 把每个可移动离子分配到最近的位点（在捕获半径内）。
3. **检测跃迁** — 当一个离子改变了分配的位点，就是一个跃迁事件。
4. **筛选为跳跃** — 满足最短停留时间要求的跃迁才被计入跳跃（极短暂的到访后立刻返回不算跳跃）。
5. **分析连通性** — 构建跃迁矩阵，计算跳跃速率，沿选定的晶体学方向寻找渗流通道。

### 从密度识别位点

当没有参考位点结构时，GEMDAT 可以从轨迹本身识别位点：

1. 用间距 `--resolution-A`（默认 0.5 Å）的网格构建可移动离子的三维数密度直方图。
2. 找到高于 `--background-level`（默认：最大值的 0.1）的密度峰。
3. 围绕峰值分割密度，定义位点体积。
4. 把位点体积转为 pymatgen `Structure`，供后续分析使用。

这是探索性的——位点取决于网格分辨率、背景阈值和轨迹长度。必须写明识别参数。

### 自由能景观与渗流

从密度直方图出发，GEMDAT 计算**自由能图**：

$$
F(\mathbf{r}) = -k_B T \ln \frac{\rho(\mathbf{r})}{\rho_\text{max}}
$$

其中 ρ(**r**) 是局部密度。位点之间的自由能垒是从占据率推导的迁移垒类似物，但**不等于** NEB
势能垒——它包含有限温度的熵贡献，并且依赖轨迹长度和网格分辨率。

**渗流通道**是自由能景观中沿给定晶体学方向贯穿模拟晶胞的连通路径。沿 *x*、*y* 和 *z* 方向
都存在渗流通道，暗示（但不证明）三维长程离子输运。

### 跳跃扩散系数

GEMDAT 可以从跳跃统计估计扩散系数：

$$
D_\text{jump} = \frac{\langle l^2 \rangle}{2 \, d \, \tau}
$$

其中 ⟨*l*²⟩ 是跳跃距离的均方值，*d* 是维数（`--jump-dimensions`），τ 是跳跃之间的平均时间。
这是**诊断性的交叉检查**，不能替代 kinisi 的输运分析。跳跃扩散系数和示踪扩散系数可能不同，
原因包括关联运动、不完全的位点分配，以及离散跳跃与连续位移之间的差异。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动离子（必需） |
| `--sites PATH` 或 `--discover-sites-from-density` | 位点文件，或从密度自动识别（必须二选一） |
| `--resolution-A` | 密度网格间距，单位 Å（默认 0.5） |
| `--background-level` | 密度峰识别阈值（默认 0.1） |
| `--site-radius-A` | 位点的捕获半径 |
| `--minimal-residence` | 至少停留多少帧才算一次占据（默认 0） |
| `--jump-dimensions` | 跳跃分析的维度：1、2 或 3（默认 3） |
| `--percolation-axes` | 检查渗流的方向（默认 `xyz`） |
| `--temperature-K` | 计算自由能和衍生量用的温度 |
| `--drift-reference`、`--drift-mode`、`--drift-indices` | 骨架漂移校正（mode 默认 `arithmetic_mean`） |

### 位点来源

必须二选一：

- **`--sites PATH`** — 显式的位点文件（CIF、POSCAR 或 pymatgen 支持的格式）。晶格必须和轨迹
  匹配。这是可复现的选择：位点已知且固定。
- **`--discover-sites-from-density`** — 从轨迹自身的密度图中识别位点。这是探索性的：位点取决于
  轨迹、分辨率和背景阈值。必须写明这些参数。

## 输出

`results.json` 中包含位点列表、占据和跳跃次数、跳跃方向和停留时间统计、渗流诊断，以及溯源信息。

| 部分 | 内容 |
|---|---|
| `summary` | 位点数量、跳跃数量、跳跃扩散系数、温度、漂移处理、GEMDAT 版本 |
| `arrays` | 密度直方图、自由能、跃迁和跳跃矩阵、峰体素 |
| `tables` | 跃迁事件、停留时间、含距离和跳跃速率的跳跃表 |
| `structures` | 位点结构、占据率注释的位点、参考构型 |
| `paths` | 各检测方向上的渗流路径坐标和自由能 |

### 状态码

| 状态 | 含义 |
|---|---|
| `success` | 计算了跃迁且跳跃具有物理意义 |
| `no_events` | 分析正常运行，确实没有发现跃迁（物理上的零） |
| `backend_error` | GEMDAT 自身出错——跳跃次数**未知**，而非零 |
| `invalid_input` | 位点结构和轨迹不兼容 |

GEMDAT 出错时，结果标为**后端失败**，绝不是"零次跳跃"。只有分析正常完成且确实没有跃迁时，才报告
零次跳跃。mliport 内置了一个独立的最近位点交叉检查，用以区分真正的"无跃迁"和 GEMDAT 在空事件
集合上的堆栈错误。

## 结果解读

### 报告时要写明

1. **位点来源** — 显式文件还是从密度识别，以及识别参数（分辨率、背景阈值）。
2. **跳跃次数和停留准则** — `--minimal-residence` 改变有多少事件被计入跳跃。
3. **漂移处理** — 骨架漂移校正改变帧到位点的映射。
4. **渗流方向** — 检查了哪些维度和方向。
5. **温度** — 影响自由能景观的计算。

### 与输运的对比

GEMDAT 的跳跃扩散系数和 kinisi 的示踪扩散系数回答的问题不同。它们应该在同一数量级，但不会精确
一致。如果二者相差超过 5 倍，需要排查：位点定义可能没有捕获实际机理，或者轨迹太短导致跳跃统计
未收敛。

### 计算量

在 1 ns、400 个原子的完整轨迹上识别位点很耗时。验收测试只取其中 30 ps：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns electrolyte \
  --mobile Li --discover-sites-from-density
```

### 完整示例

基于密度识别位点的完整机理分析：

```bash
mliport analyze runs/md-800K electrolyte --mobile Li \
  --discover-sites-from-density --resolution-A 0.2 --minimal-residence 2 \
  --jump-dimensions 3 --percolation-axes xyz --temperature-K 800
```

GEMDAT 关于部分占据的警告来自上游库，不影响你指定的位点。

## 下一步

- 用适当的不确定性估计测量扩散速率：[输运系数](transport.md)
- 从多个温度提取活化能：[Arrhenius](arrhenius.md)
- 检查轨迹是否处于扩散区间：[MSD](msd.md)
