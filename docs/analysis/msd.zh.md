# MSD 与局部指数

`msd` 计算可移动物种的均方位移，以及局部指数 `alpha(t) = d log(MSD) / d log(t)`。请把它
当作"这条轨迹处于哪种扩散区间"的诊断，而不是扩散系数。

```bash
mliport analyze runs/md msd --mobile Li --axes xyz
```

`--mobile` 是必需的。分析外来轨迹时，要说明坐标是 wrapped 还是 unwrapped
（`--positions-convention`），元数据里没有帧间隔时还要给出保存间隔
（`--frame-interval-fs`）；信息缺失时 mliport 会失败，而不是假设。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动物种（必需） |
| `--axes` | `x`、`y`、`z` 或组合如 `xy`、`xyz`（默认 `xyz`） |
| `--drift-reference`、`--drift-mode`、`--drift-indices` | 如何扣除骨架漂移：reference 取 `none`、`nonmobile`、`all` 或 `indices`；mode 取 `none`、`arithmetic_mean` 或 `mass_weighted_com`（默认） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--method fft\|direct` | MSD 算法（默认 `fft`） |
| `--fit-start-ps`、`--fit-stop-ps` | 拟合窗口，随结果一并记录 |
| `--alpha-window-decades` | 局部 log–log 回归窗口的总宽度（默认 0.25） |
| `--alpha-min-points`、`--alpha-min-origins` | 窗口内最少点数（5）和一个 alpha 点要算数所需的最少时间原点数（8） |
| `--alpha-consistency-band` | 围绕 `alpha = 1`、被视为扩散的半宽（默认 0.2） |
| `--alpha-max-lag-ps` | 仅在估计 alpha 时忽略更长的滞后时间；原始 MSD 不受影响 |
| `--alpha-focus-band LOW HIGH`、`--alpha-log-x` | 绘图选项：放大 alpha 图、用对数时间轴 |

## 怎么读 alpha

`alpha ≈ 1` 与 Fick 扩散一致，`alpha < 1` 对应亚扩散，`alpha > 1` 对应超扩散或弹道运动，
但这只在你所拟合的时间窗口内成立。落在一致性带之外的点会被标成 `not_diffusive`；时间
原点太少的点会被报告为"支持不足"，而不是一个指数。请连同一致性带一起读完整条 alpha
曲线，不要只看某一个数。

其他值得注意的事：

- 扣除漂移会改变长时间的 MSD，所以要写明漂移策略。
- MSD 是统计估计量。请报告多个拟合窗口或不确定度，而不是一次拟合。
- alpha 不是电导率的结论。电导率需要电荷、集体关联和 Nernst–Einstein 处理；请用
  [输运](transport.md)。

```bash
mliport analyze runs/md-800K msd --mobile Li --axes xyz \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --fit-start-ps 50 --fit-stop-ps 400 --alpha-focus-band 0.8 1.2
```

## 短轨迹

5 步的冒烟轨迹是被接受的。结果里会带有明确的"扩散窗口不足"状态，alpha 和诊断字段仍然
是有限值。验收套件把它当作预期的局限，绝不当作扩散测量。要得到真正的 MSD，请让这次运
行至少保存几百帧，并让拟合窗口覆盖扩散区间。

结果写在 `results.json` 里，MSD 与 alpha 的表和图在
`<run>/analysis/msd/<analysis-id>/` 下，包括所用的确切拟合窗口与 alpha 窗口。
