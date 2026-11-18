# RDF

`rdf` 计算中心元素和近邻元素之间的径向分布函数，并给出配位数。

```bash
mliport analyze runs/md rdf --center Li --neighbor S --rmax 6.0 --bins 200
```

两种元素都必须写。对 LGPS，常看的元素对是 Li–Li、Li–S 和 P–S。

## 物理背景

### 什么是 RDF？

**径向分布函数**（RDF），通常写作 *g(r)*，度量一种原子在另一种原子周围的局部密度随距离的变化，
相对于均匀分布：

$$
g(r) = \frac{1}{4 \pi r^2 \, \Delta r} \cdot \frac{V}{N_\text{center} \, N_\text{neighbor}} \sum_{i \in \text{center}} \sum_{j \in \text{neighbor}} \delta(r - r_{ij})
$$

实际计算中，连续的 *δ* 被宽度为 Δ*r* 的直方图 bin 代替。结果的含义是：

- **g(r) → 1**（大 *r* 处）：局部结构与均匀流体无法区分。这是液体或充分混合体系的预期长程极限；
  如果你的 RDF 在大 *r* 处不趋近 1，说明截断太短或晶胞太小。
- **g(r) > 1**（某个 *r* 处）：在这个距离上找到原子的概率高于随机分布——对应一个**配位壳层**。
- **g(r) ≈ 0**（短 *r* 处）：排斥核，原子不会重叠。

### 配位数

配位数是 *g(r)* 到截断半径的积分：

$$
n(r_c) = 4 \pi \, \rho_\text{neighbor} \int_0^{r_c} g(r) \, r^2 \, dr
$$

其中 ρ 是近邻元素的数密度。截断半径应放在 *g(r)* 的第一个极小值处——那是第一配位壳层的边界。
报告配位数时不写截断半径是没有意义的。

### RDF 能告诉你什么

RDF 是**结构的统计平均**，不是成键定义。它回答的问题是："平均而言，在 A 元素周围距离 *r* 内有
多少个 B 元素的近邻？"单个快照的 RDF 很嘈杂；在足够多的帧上取平均后，才得到平衡态的对关联函数。

对于固态电解质，RDF 能揭示：

- **骨架完整性** — 晶体骨架在晶格决定的距离上呈现尖锐峰。峰展宽或移位意味着结构变化；峰消失
  意味着熔化或非晶化。
- **可移动离子的配位环境** — Li–S 或 Li–O RDF 的第一峰展示平均配位壳层的几何构型。
- **模型质量** — 将 MLIP RDF 与 DFT-MD 参考对比，可以判断势函数是否重现了局部结构。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--center SPECIES`、`--neighbor SPECIES` | 中心元素和近邻元素（都必需） |
| `--rmax` | 最大半径，单位 Å |
| `--bins` | 直方图格数（默认 200） |
| `--cn-cutoff` | 计算配位数所用的截断半径，单位 Å |
| `--stride` | 每隔 N 帧取一帧（默认 1） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--positions-convention`、`--frame-interval-fs` | 外部轨迹需要补充的信息 |

长轨迹可以用 `--start-frame`、`--stop-frame` 和 `--stride` 减少计算量：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns rdf \
  --center Li --neighbor S --start-frame 0 --stop-frame 2000 --stride 5
```

## 输出

分元素对的 RDF、配位数，以及 `<run>/analysis/rdf/<analysis-id>/` 下的 CSV、PNG 和 SVG。
元素不存在或有效帧太少会报错；外部轨迹需要说明坐标约定。

| 文件 | 内容 |
|---|---|
| `results.json` | 完整的 RDF、配位数、元数据和溯源信息 |
| `rdf.csv` | *r*（Å）和 *g(r)* 表格 |
| `rdf.png`、`rdf.svg` | RDF 图 |

## 结果解读

### 怎么读 g(r)

1. **第一峰位置** — 最可能的最近邻距离。对 LGPS 中的 Li–S，通常在 2.4–2.5 Å 左右。
2. **第一峰高度** — 第一配位壳层的有序程度。峰越高越窄，说明配位环境越有序。
3. **第一极小值位置** — 这里就是 `--cn-cutoff` 应该放的地方。
4. **长程行为** — *g(r)* 应该在远低于 `--rmax` 的距离上收敛到 1。如果在大 *r* 处仍有振荡或
   偏离 1，可能是晶胞太小或轨迹太短。

### 常见陷阱

- **晶胞尺寸效应**：如果 `--rmax` 超过了最短晶格矢量的一半，大 *r* 处的 RDF 会包含周期镜像，
  变得不可靠。把 `--rmax` 控制在 *L*/2 以下。
- **统计不足**：太少的帧会给出很嘈杂的 RDF。用采样阶段的几千帧。
- **坐标约定**：wrapped 和 unwrapped 坐标计算的 RDF 结果相同（因为周期距离不受影响），但
  `--positions-convention` 仍然需要正确设置，以通过一致性检查。

### 与参考对比

验证模型时，用相同的 bin 宽度、截断半径和帧数把 MLIP RDF 和 DFT-MD 参考进行对比。峰位差异
超过 0.1 Å 或配位数差异超过 0.5，说明模型与 DFT 在局部结构上有分歧。

```bash
mliport analyze runs/md-800K rdf --center Li --neighbor O \
  --rmax 8 --bins 240 --cn-cutoff 3.0 --stride 5
```
