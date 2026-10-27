# MSD 与局部指数

`msd` 计算可移动离子的均方位移，以及局部指数 `alpha(t) = d log(MSD) / d log(t)`。它用来判断轨迹处在
哪种扩散区间，扩散系数请用[输运](transport.md)分析。

```bash
mliport analyze runs/md msd --mobile Li --axes xyz
```

`--mobile` 必须写。分析外部轨迹时，要说明坐标是 wrapped 还是 unwrapped（`--positions-convention`）；
文件里没有帧间隔信息时还要给出（`--frame-interval-fs`）。缺少这些信息会报错。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动离子（必需） |
| `--axes` | `x`、`y`、`z` 或组合如 `xy`、`xyz`（默认 `xyz`） |
| `--drift-reference`、`--drift-mode`、`--drift-indices` | 怎样扣除骨架漂移：reference 可选 `none`、`nonmobile`、`all`、`indices`；mode 可选 `none`、`arithmetic_mean`、`mass_weighted_com`（默认） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--method fft\|direct` | MSD 算法（默认 `fft`） |
| `--fit-start-ps`、`--fit-stop-ps` | 拟合区间，会记录在结果里 |
| `--alpha-window-decades` | 局部 log–log 拟合窗口的宽度（默认 0.25 个数量级） |
| `--alpha-min-points`、`--alpha-min-origins` | 窗口内至少几个点（5），以及每个 alpha 点至少需要几个时间原点（8） |
| `--alpha-consistency-band` | `alpha = 1` 附近多宽的范围算作扩散（默认 ±0.2） |
| `--alpha-max-lag-ps` | 估计 alpha 时不用超过这个时间间隔的数据；不影响 MSD 本身 |
| `--alpha-focus-band LOW HIGH`、`--alpha-log-x` | 画图选项：放大 alpha 的范围、时间轴用对数 |

## 怎么读 alpha

`alpha ≈ 1` 对应正常（Fick）扩散，`alpha < 1` 是亚扩散，`alpha > 1` 是超扩散或弹道运动。这些判断只在
你选的时间窗口内成立。超出扩散范围的点标为 `not_diffusive`；时间原点太少的点标为"数据不足"，不给
出指数。看结果时要看整条 alpha 曲线和扩散范围，不要只看某一个数。

另外：

- 扣除漂移会改变长时间的 MSD，要写明用的是哪种漂移处理。
- MSD 是统计估计，应当报告多个拟合区间的结果或给出误差，不要只做一次拟合。
- alpha 不能用来得出电导率。电导率需要考虑电荷、离子之间的关联和 Nernst–Einstein 换算，请用
  [输运](transport.md)。

```bash
mliport analyze runs/md-800K msd --mobile Li --axes xyz \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --fit-start-ps 50 --fit-stop-ps 400 --alpha-focus-band 0.8 1.2
```

## 短轨迹

只有 5 步的测试轨迹也能分析，结果会标明"扩散窗口不足"，alpha 和诊断字段仍是有限值。验收测试把
这当作预期的限制，不当作扩散测量。要得到可用的 MSD，轨迹至少要保存几百帧，拟合区间要落在扩散
区间内。

结果写在 `results.json`，MSD 和 alpha 的表格和图在 `<run>/analysis/msd/<analysis-id>/` 下，同时记录所用
的拟合区间和 alpha 窗口。
