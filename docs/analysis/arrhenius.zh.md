# Arrhenius

`arrhenius` 对几个温度下分别算出的扩散系数做 Arrhenius 拟合，给出活化能，也可以外推到其他温度。它只
用你提供的数据点。

```bash
mliport analyze runs/md-800K arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11
```

每个数值前都要写一次选项名。温度和扩散系数按顺序一一对应，个数必须相同。命令里的运行目录只决定结果
写到哪里，拟合用的数字全部来自这些选项。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--temperature T` | 温度，单位 K（可重复；必需） |
| `--diffusivity D` | 对应的扩散系数，单位 m²/s（可重复；必需） |
| `--diffusivity-std S` | 每个点的误差（可选），用作拟合权重（可重复） |
| `--extrapolate-temperature T` | 外推到的温度，结果单独记录（可重复） |
| `--source-run-id ID` | 记录每个点来自哪次计算（可重复） |

## 输出

`results.json` 里有活化能 `Ea`、指前因子、拟合质量、输入数据和外推结果，图保存在
`<run>/analysis/arrhenius/<analysis-id>/`。

两组数个数不一致、有非正值、温度少于两个、或者所有温度都相同，都会报错。如果拟合出的斜率为负，
会标为拟合无效，不会给出负的活化能。

## 结果怎么看

- 所有扩散系数必须用同一个模型、同一套 MD 设置（步长、晶胞、系综、漂移处理）算出来，否则拟合混入了
  不同的条件。
- 外推到测量范围以外的结果是预测值，结果里也按预测值记录。
- Arrhenius 关系本身是一个假设，要检查数据点是否线性、拟合质量如何。

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

通常的做法是把各温度下 `transport` 算出的扩散系数填进来。验收测试就是这样做的，用了三个温度下 UMA
计算的 LGPS 结果：

```bash
.venv-grace/bin/mliport analyze results/uma-800K.traj arrhenius \
  --frame-interval-fs 100 --positions-convention unwrapped \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 8.475718e-10 --diffusivity 3.479545e-09 --diffusivity 4.103223e-09 \
  --diffusivity-std 7.356329e-11 --diffusivity-std 2.407381e-10 --diffusivity-std 2.640917e-10
```

命令里的轨迹只决定输出位置，`--positions-convention` 这类轨迹相关的选项在拟合中不起作用，会记为已
忽略。这次拟合得到 `Ea = 0.316 eV`（r² = 0.88）。如果成功完成的输运计算少于两个，流程会报告
`insufficient_transport_data`，不给出活化能。
