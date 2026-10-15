# 输运（kinisi）

`transport` 用 kinisi 分析器，从 MD 轨迹估计可移动离子的示踪扩散系数并给出不确定度；对带电
物种，再换算成 Nernst–Einstein 离子电导率。

```bash
mliport analyze runs/md transport --mobile Li --charge 1 \
  --fit-start-ps 50 --temperature-K 800
```

你需要：一条生产轨迹，已知的**保存帧**间隔（外来文件用 `--frame-interval-fs` 给出），可移动
物种，它的电荷（以 e 为单位），以及扩散拟合的起点。平衡阶段的帧不参与分析。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动物种（必需） |
| `--charge` | 离子电荷，以 e 为单位（必需；用于电导率） |
| `--fit-start-ps` | 扩散拟合的起点（必需） |
| `--lag-step-ps`、`--lag-stop-ps` | 显式指定更稀疏的 kinisi 滞后时间网格；要么都给，要么都不给 |
| `--dimensions` | 参与估计的笛卡尔轴，取 `xyz` 的子集（默认 `xyz`） |
| `--temperature-K` | Nernst–Einstein 换算所用的温度 |
| `--drift-reference`、`--drift-mode`、`--drift-indices` | 骨架漂移校正（mode 默认 `arithmetic_mean`） |
| `--collective-conductivity`、`--collective-system-particles` | 同时计算电荷集体（Onsager）电导率；后者是 kinisi 按索引顺序划分的统计分组（默认 1），它们不是独立的 MD 副本 |
| `--jump-diffusion` | 同时运行 kinisi 的跳跃扩散分析器，作为诊断 |
| `--n-samples`、`--n-walkers`、`--n-burn`、`--n-thin`、`--random-seed` | kinisi 采样控制（默认 1000、32、500、10、0） |
| `--parser-memory-limit-gib` | 估计的内存峰值超过该值时拒绝解析（默认 4） |
| `--allow-reconstructed-fallback` | 仅作诊断：退回到 kinisi 未经验证的位移重构；结果会标记 `publication_grade=false` |

滞后时间网格稀疏化的是滞后时间，不是轨迹帧。轨迹很长时，正是稀疏网格让 kinisi 的协方差矩阵
留在内存保护线以内。

## 输出什么

`results.json` 里有带不确定度的 `D`，电导率各项（基于示踪扩散的 Nernst–Einstein 项，以及
你要求时的集体项），滞后时间网格与拟合窗口，采样诊断，以及来源信息（模型标识、软件提交）。
`transport_summary.csv` 把关键数字放在一行里。

轨迹太短、拟合窗口低于所需采样量、帧间隔未知、坐标约定冲突、内存估计超限，这些情形都会
带着原因失败。只有你明确允许时才会使用重构回退，而且会被记录下来。

## 怎么读结果

- 有限轨迹给出的扩散系数，只有在拟合窗口位于扩散区间、且报告了采样误差时才值得信任。
  请看 kinisi 的不确定度，不要只看 `D` 的均值。
- Nernst–Einstein 电导率假设离子运动互不关联，所以除非同时算出集体电导率，它并不是完整
  的电导率。请写明你报告的是哪一个。
- 恒体积的晶胞会限制机理，过于局促的晶胞可能压低或扭曲扩散。请写明晶胞，以及它是否被
  弛豫过。
- 骨架漂移必须明确处理，漂移策略会改变 `D`。
- 要弄清*哪些*位点和跳跃起作用，请用 [GEMDAT](electrolyte-gemdat.md)；输运给出的是速率，
  不是机理。

```bash
mliport analyze runs/md-800K transport --mobile Li --charge 1 \
  --fit-start-ps 50 --lag-step-ps 5 --lag-stop-ps 500 \
  --temperature-K 800 --drift-reference nonmobile \
  --drift-mode mass_weighted_com --collective-conductivity --jump-diffusion
```

## 比较估计量

验证审计会在*同一条*轨迹上计算若干估计量，并分别报告。它们回答的问题不同，不要求一致：

| 估计量 | 是什么 | 窗口 |
|---|---|---|
| 朴素 OLS 基线 | `D = slope(MSD)/(2d)`，单位换算写明 | 与 kinisi 拟合起点对齐 |
| 原生 MSD 诊断 | 产品自带的显式区间 OLS 诊断（区分扩散区间） | 记录下来的拟合窗口 |
| kinisi 后验 | 带可信区间的示踪扩散 | kinisi 的滞后时间与拟合窗口 |
| Nernst–Einstein | `sigma_NE = n (z e)^2 D_tracer / (k_B T)` | 由示踪估计推出 |

报告里会记录 `kinisi_to_plain_D_ratio`，比值低于 0.5 或高于 2 时给出
`estimator_disagreement_warning`。这是要你去解释差异（窗口、漂移、区间）的提示，不是通过
与否的阈值。单位审计是一个已知答案的检查：`1 Å²/ps = 1e-8 m²/s`，所以三维 MSD 斜率
`6 Å²/ps` 对应 `D = 1e-8 m²/s`。

差异很大通常说明拟合窗口不在扩散区间，所以先看局部 `alpha`：一个"演示、未收敛"的协议
即使轨迹有限而健康，也可能出现 `alpha ≈ 2`。

## 一个经过测试的例子

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns transport \
  --mobile Li --charge 1 --fit-start-ps 100 \
  --lag-step-ps 1 --lag-stop-ps 200
```

对仓库里 1 ns 的 GRACE LGPS 运行，这条命令得到 `D = 1.885e-09 m²/s`，95% 可信区间为
`[1.492e-09, 2.275e-09] m²/s`（拟合窗口 100–200 ps）。轨迹太短时会返回采样不足的状态；
mliport 不会为它们编造扩散系数。
