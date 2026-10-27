# 概览

每项分析都是一条单独的命令，并且会记录结果是怎么算出来的：

```bash
mliport analyze <RUN_DIR|trajectory> <task> [task options]
```

分析结果是诊断依据，结论还要你自己下。MSD 指数、扩散系数、跳跃次数，都要结合各页面里关于采样和
模型的说明来理解。

## 输入

优先使用 mliport 的运行目录，里面记录了帧间隔、坐标约定、哪些帧属于平衡阶段等信息。也可以用 ASE
轨迹（`.traj`）或 XDATCAR，这时需要你提供程序无法知道的信息：

- `--positions-convention`：`wrapped`、`unwrapped` 或 `unknown`。如果和文件里记录的信息矛盾，程序会
  报错。
- `--frame-interval-fs`：**保存的**相邻两帧之间的时间，不是 MD 的积分步长。

各任务还有自己需要的参数，例如可移动离子、RDF 的中心和近邻元素、温度等。

## 输出

每次分析有单独的目录 `<run>/analysis/<task>/<analysis-id>/`。id 由分析参数决定，同一条轨迹、同样的
参数再分析一次会直接用已有结果；要重新计算就加 `--force`。

- `results.json`：分析参数和结果；
- `request.json` 和 `provenance.json`：完整的分析参数，以及软件 commit、模型信息、输入文件哈希；
- CSV 表格，有图的任务还有 PNG/SVG；
- `attempts/`：每次运行的记录（不会被修改），`latest.json` 指向最近一次。

## 采样方面的注意事项

- 分析采样阶段的轨迹。平衡阶段的帧默认不计入，除非加 `--include-equilibration`；支持的任务可以用
  `--start-frame` / `--stop-frame` 限定范围。
- 要清楚保存帧的时间间隔，外部轨迹用 `--frame-interval-fs` 给出。
- 输运系数比 RDF、热力学量需要更长、平衡得更好的轨迹。
- 用不同的时间窗口反复分析是正常做法，每组参数都有自己的 analysis id。

## 没有单独页面的任务

**`validate`** 检查轨迹的基本信息：坐标约定、帧间隔、哪些帧属于采样阶段，以及适合做哪些分析。发现
信息有问题时会报告出来，避免后续分析用了有问题的轨迹。

**`thermo`** 报告温度、能量漂移和压强。`--block-size-frames` 设置分块统计的块大小。NVE 计算首先要看
能量漂移。

**`rmsd`** 考虑周期性计算 RMSD 和 RMSF，可以快速判断晶体结构是否保持住了。`--species` 只看某种元素，
`--drift-reference none|nonmobile|indices` 选择怎样扣除整体漂移。

**`density`** 生成某种可移动离子的三维密度分布（考虑周期性），必须给 `--mobile`。`--spacing` 设置网格
间距（默认 0.25 Å），`--smoothing-sigma-A` 做平滑，`--stride` 隔帧取样。

## 验收测试用过的命令

LGPS 验收测试在真实轨迹上运行过这些命令：

```bash
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
```

`validate` 和 `thermo` 对长短轨迹都能用。每个任务都输出 JSON 和 CSV，有图的还输出 PNG 和 SVG，不需要
显示器。
