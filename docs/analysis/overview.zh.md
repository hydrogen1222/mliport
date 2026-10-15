# 概览

每一项分析都是一条明确的命令，并且会记下数字是怎么来的：

```bash
mliport analyze <RUN_DIR|trajectory> <task> [task options]
```

分析是诊断，不是结论。MSD 指数、扩散系数或跃迁次数，都必须结合各自页面上关于采样和
模型的注意事项来解读。

## 输入

优先使用 mliport 运行目录：其中的元数据（帧间隔、坐标约定、哪些帧属于平衡阶段）是可
信的。ASE 轨迹（`.traj`）或 XDATCAR 也可以。对这些外来文件，要由你告诉 mliport 它无
从知道的信息：

- `--positions-convention` —— `wrapped`、`unwrapped` 或 `unknown`。如果你声明的约定与
  可信元数据相矛盾，mliport 会拒绝，而不是去猜。
- `--frame-interval-fs` —— **保存下来的**相邻帧之间的时间，它不是 MD 的积分步长。

每个任务还需要各自的选择，例如可移动物种、中心与近邻的物种对，或者温度。

## 输出

每次分析都有自己的目录 `<run>/analysis/<task>/<analysis-id>/`。id 由请求派生出来，
所以对同一条轨迹再问同一个问题，会复用已存的结果；想重新计算就加 `--force`。

- `results.json` —— 请求的参数和结果，
- `request.json` 和 `provenance.json` —— 精确的请求，以及所有信息的来源（软件 commit、
  模型身份、输入产物的哈希），
- 合适时的 CSV 表和 PNG/SVG 图，
- `attempts/` —— 每次尝试的不可变记录，`latest.json` 指向最新的一次。

## 关于采样的常识

- 分析生产阶段的轨迹。除非你用 `--include-equilibration` 明确要求，否则平衡阶段的帧会
  被排除；任务支持时，可以用 `--start-frame` / `--stop-frame` 收窄窗口。
- 清楚保存下来的帧间隔。对外来轨迹，它就是 `--frame-interval-fs`。
- 输运系数比 RDF 或热力学诊断需要更长、平衡得更好的轨迹。
- 换不同的窗口重新分析很正常；每个不同的请求都有自己的 analysis id。

## 没有单独页面的任务

**`validate`** 检查轨迹的语义：坐标约定、帧间隔、哪些帧属于生产阶段，以及它适合做哪些
后续任务。它会报告元数据问题，避免后面的任务从一条不该信任的轨迹里算出一个数。

**`thermo`** 报告温度、能量漂移和压强。`--block-size-frames` 设置分块统计的块大小，对
NVE 运行来说，能量漂移是最先要看的东西。

**`rmsd`** 由周期性位移计算 RMSD 与 RMSF，用来快速检查晶体是否还是晶体。`--species`
把范围限定到某个物种，`--drift-reference none|nonmobile|indices` 选择如何扣除漂移。

**`density`** 生成某个可移动物种的周期性三维密度图，必须给 `--mobile`。`--spacing` 设置
体素大小（默认 0.25 Å），`--smoothing-sigma-A` 加平滑，`--stride` 对帧做抽稀。

## 验收套件运行的命令

LGPS 验收套件在真实轨迹上运行了这些命令：

```bash
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
```

`validate` 和 `thermo` 在短轨迹和长轨迹上都能运行。每个任务都会写出 JSON 加 CSV，必要
时还有 PNG 和 SVG 图，全部在无界面环境下完成。
