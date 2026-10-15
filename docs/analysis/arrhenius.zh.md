# Arrhenius

`arrhenius` 对你从不同温度的独立运行里得到的扩散系数做 Arrhenius 拟合，可选地外推，并给出
对应的活化能。它只拟合你给它的点，绝不会编造缺失的点。

```bash
mliport analyze runs/md-800K arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11
```

每个值都要重复一次选项：温度和扩散系数按位置配对，长度必须相同。你指定的运行或轨迹只是
分析的来源，决定输出写到哪里；数字来自这些选项。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--temperature T` | 温度，单位 K（可重复；必需） |
| `--diffusivity D` | 对应的扩散系数，单位 m²/s（可重复；必需） |
| `--diffusivity-std S` | 每个点可选的不确定度，用于加权（可重复） |
| `--extrapolate-temperature T` | 要外推到的温度，单独记录（可重复） |
| `--source-run-id ID` | 链接回这些点来自的运行（可重复） |

## 输出什么

`results.json` 里有拟合得到的活化能 `Ea`、指前因子、拟合质量、输入以及外推结果，图保存在
`<run>/analysis/arrhenius/<analysis-id>/`。

长度不匹配、取值非正、温度少于两个，或温度范围退化，都会失败。斜率为负的拟合会被报告为
拟合差或无效，绝不会给出负的活化能。

## 怎么读

- 每个扩散系数都必须来自同一个模型标识和同一套 MD 协议（时间步、晶胞、系综、漂移策略），
  否则拟合混合了不同的物理。
- 在测量范围之外外推是预测，也会被记为预测。
- Arrhenius 行为是一种假设。请检查线性程度和拟合质量。

```bash
mliport analyze runs/md series/arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11 \
  --diffusivity-std 0.1e-12 --diffusivity-std 0.2e-12 \
  --diffusivity-std 0.4e-12 --diffusivity-std 0.8e-12 \
  --extrapolate-temperature 300 --extrapolate-temperature 1200
```

## 用 `transport` 的结果作为输入

最自然的输入，就是 `transport` 在各温度下给出的扩散系数。验收运行正是这样做的，用了三个
同引擎的 UMA LGPS 点：

```bash
.venv-grace/bin/mliport analyze results/uma-800K.traj arrhenius \
  --frame-interval-fs 100 --positions-convention unwrapped \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 8.475718e-10 --diffusivity 3.479545e-09 --diffusivity 4.103223e-09 \
  --diffusivity-std 7.356329e-11 --diffusivity-std 2.407381e-10 --diffusivity-std 2.640917e-10
```

源轨迹只是一个锚点；`--positions-convention` 这类轨迹来源选项在这次拟合中会被记为已忽
略。这次运行给出 `Ea = 0.316 eV`（r² = 0.88）。如果成功的输运运行少于两个，流水线会报告
`insufficient_transport_data`，而不是给出活化能。
