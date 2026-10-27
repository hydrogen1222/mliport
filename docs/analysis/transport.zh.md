# 输运（kinisi）

`transport` 用 kinisi 从 MD 轨迹估计可移动离子的示踪扩散系数及其误差；对带电离子，还会换算成
Nernst–Einstein 电导率。

```bash
mliport analyze runs/md transport --mobile Li --charge 1 \
  --fit-start-ps 50 --temperature-K 800
```

需要：一条采样轨迹，已知的**保存帧**间隔（外部文件用 `--frame-interval-fs` 给出），可移动离子种类和
它的电荷（以 e 为单位），以及扩散拟合的起始时间。平衡阶段的帧不参与分析。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动离子（必需） |
| `--charge` | 离子电荷，以 e 为单位（必需；用于电导率） |
| `--fit-start-ps` | 扩散拟合的起点（必需） |
| `--lag-step-ps`、`--lag-stop-ps` | 手动设置更稀疏的时间间隔网格；两个要么都写，要么都不写 |
| `--dimensions` | 参与计算的方向，取 `xyz` 的子集（默认 `xyz`） |
| `--temperature-K` | Nernst–Einstein 换算所用的温度 |
| `--drift-reference`、`--drift-mode`、`--drift-indices` | 骨架漂移校正（mode 默认 `arithmetic_mean`） |
| `--collective-conductivity`、`--collective-system-particles` | 同时计算考虑离子关联的集体（Onsager）电导率；后者是 kinisi 按原子编号顺序分的统计组数（默认 1），不是独立的 MD 轨迹 |
| `--jump-diffusion` | 同时运行 kinisi 的跳跃扩散分析，作为参考 |
| `--n-samples`、`--n-walkers`、`--n-burn`、`--n-thin`、`--random-seed` | kinisi 采样控制（默认 1000、32、500、10、0） |
| `--parser-memory-limit-gib` | 估计内存峰值超过该值时拒绝分析（默认 4 GiB） |
| `--allow-reconstructed-fallback` | 仅供诊断：改用 kinisi 未经验证的位移重建方法，结果会标记 `publication_grade=false` |

稀疏网格减少的是时间间隔的个数，轨迹帧并不减少。轨迹很长时，用稀疏网格可以让 kinisi 的协方差矩阵
不超出内存限制。

## 输出

`results.json` 里有 `D` 及其误差、电导率（由示踪扩散得到的 Nernst–Einstein 电导率，要求时还有集体
电导率）、时间间隔网格和拟合区间、采样诊断，以及模型信息和软件 commit。`transport_summary.csv` 把
主要数字放在一行里。

轨迹太短、拟合区间内采样不够、帧间隔未知、坐标约定矛盾、内存估计超限时，都会报错并说明原因。
只有明确允许时才会用位移重建方法，并会记录在结果里。

## 结果怎么看

- 有限长度的轨迹算出的扩散系数，只有在拟合区间处于扩散区间、并给出了采样误差时才可信。要看 kinisi
  给出的误差，不要只看 `D` 的平均值。
- Nernst–Einstein 电导率假设离子之间没有关联，不算集体电导率的话，它不是完整的电导率。报告时要写明
  用的是哪一个。
- 固定体积的晶胞会限制扩散机制，晶胞太小可能使扩散偏慢或失真。请写明晶胞大小以及是否优化过。
- 骨架漂移必须处理，不同的漂移处理会得到不同的 `D`。
- 想知道离子经过*哪些*位点、怎么跳跃，用 [GEMDAT](electrolyte-gemdat.md)。输运分析只给出速率，不揭示
  机理。

```bash
mliport analyze runs/md-800K transport --mobile Li --charge 1 \
  --fit-start-ps 50 --lag-step-ps 5 --lag-stop-ps 500 \
  --temperature-K 800 --drift-reference nonmobile \
  --drift-mode mass_weighted_com --collective-conductivity --jump-diffusion
```

## 几种估计方法的比较

验证时会在*同一条*轨迹上用几种方法估计，分别报告。它们回答的问题不同，结果不要求一致：

| 方法 | 说明 | 拟合区间 |
|---|---|---|
| 简单最小二乘 | `D = slope(MSD)/(2d)`，写明单位换算 | 起点和 kinisi 一致 |
| mliport 的 MSD 诊断 | 在明确区间上的最小二乘（区分扩散区间） | 记录的拟合区间 |
| kinisi 后验 | 示踪扩散系数及其可信区间 | kinisi 的时间间隔和拟合区间 |
| Nernst–Einstein | `sigma_NE = n (z e)^2 D_tracer / (k_B T)` | 由示踪扩散推出 |

报告里记录 `kinisi_to_plain_D_ratio`，比值低于 0.5 或高于 2 时给出 `estimator_disagreement_warning`。这只
是提醒你去找差异的原因（拟合区间、漂移、扩散区间），不是合格线。单位换算用已知结果核对：
`1 Å²/ps = 1e-8 m²/s`，所以三维 MSD 斜率为 `6 Å²/ps` 时 `D = 1e-8 m²/s`。

两者差别很大，通常是因为拟合区间不在扩散区间，先看局部 `alpha`。演示用的短计算即使轨迹本身没问题，
也可能出现 `alpha ≈ 2`。

## 实测例子

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns transport \
  --mobile Li --charge 1 --fit-start-ps 100 \
  --lag-step-ps 1 --lag-stop-ps 200
```

对仓库里 1 ns 的 GRACE LGPS 轨迹，这条命令得到 `D = 1.885e-09 m²/s`，95% 可信区间为
`[1.492e-09, 2.275e-09] m²/s`（拟合区间 100–200 ps）。轨迹太短时会返回"采样不足"，不给出扩散系数。
