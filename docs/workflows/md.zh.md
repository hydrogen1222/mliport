# 分子动力学

`mliport md` 在你选定的后端上运行 NVE 或 NVT 动力学，并记录下可供
[分析](../analysis/index.md)工具读回的轨迹。

```bash
.venv-mace/bin/mliport md structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --ensemble NVT --temp 300 --timestep 1.0 --steps 1000 \
  --equilibration-steps 500 --save-interval 10 \
  --output runs/md
```

你需要一个周期性结构、一个模型和明确的后端。对分子做 MD 需要分子任务
（`omol` / `molecule`），以及合理的电荷和自旋。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--ensemble` | `NVT`（默认）或 `NVE` |
| `--temp` | 目标温度，单位 K（默认 300） |
| `--timestep` | 积分步长，单位 fs（默认 1.0） |
| `--steps` | 生产阶段步数（默认 1000） |
| `--equilibration-steps` | 生产前在同一系综下的平衡步数（默认 0） |
| `--thermostat` | `LANGEVIN`（默认）、`BUSSI` 或 `NHC` |
| `--friction`、`--bussi-tau`、`--nhc-tdamp`、`--nhc-tchain`、`--nhc-tloop` | 恒温器参数 |
| `--save-interval` | 每 N 步写一帧（默认 10） |
| `--seed` | 随机种子；省略时自动生成并记录 |
| `--velocity-policy`、`--com-policy` | 速度和质心运动的处理方式 |
| `--pre-relax`、`--pre-relax-steps`、`--pre-relax-fmax` | MD 之前一次简短的只动原子位置的弛豫（NVT 默认开启，NVE 默认关闭；50 步，0.1 eV/Å） |
| `--fmax-abort` | 任何力超过这个值（eV/Å）就中止（默认 20） |
| `--write-outcar`、`--write-xdatcar`、`--write-trajectory` | 写出哪些输出 |

### 预弛豫是一道关卡，不是建议

如果预弛豫在 `--pre-relax-steps` 步之内没有达到 `--pre-relax-fmax`，运行会**在正
式 MD 之前停下**。从一个离极小点还很远的结构开始动力学，几乎不会是任何人想要的，
而悄悄地说一句"差不多了"只会把问题藏起来。报错信息会告诉你怎么办：加大步数预算；
在科学上能说得通的前提下放宽阈值；或者明确地关掉预弛豫。已完成的预弛豫的结果记录
在 `md_provenance.pre_relax` 里。

## 步长、保存间隔，以及分析看到什么

`--timestep` 决定运动方程怎么积分，`--save-interval` 决定多久写一帧。步长 1 fs、
`--save-interval 10`，就是每 10 fs 一帧；输运分析需要的是这个**保存下来的**间隔，而
不是积分步长。

不要把平衡阶段混进统计：用 `--equilibration-steps`，并且只分析生产段。不同系综也要
分开：NVE 和 NVT 是不同的动力学，它们的轨迹绝不能平均在一起。

## 恒温器与质心运动

`--thermostat` 选择 NVT 怎么控温。Langevin 是随机的，Bussi 和 NHC 是确定性的。输
运系数取决于这个选择，所以报告扩散系数时请写明。

NHC 需要多说几句。它不能和 FixCom（或其他任何）约束一起使用。默认的
`--com-policy auto`（等同于 `initialize_only`）会在初始化速度之后把质心动量清零一
次，NHC 随后会保持它守恒。因此这次运行有 `3N-3` 个动能自由度，恒温器的设定温度会
乘上 `(3N-3)/3N`（记录为 `nhc_set_temperature_K`）。如果用 `--com-policy none`，随
机的质心动量会保留，整个晶胞会做匀速漂移，除非分析时把漂移扣掉，否则它会主导可移
动离子的 MSD。

约束和固定原子是通过计算器和运行器施加的；报告扩散时请把它们写清楚。

## 接着跑

`md` 没有 `--resume` 参数。要继续一次运行，读取一个带动量的已保存帧，并从它出发开
始一次**新的**运行：

```bash
.venv-mace/bin/mliport md last_frame.traj --model mace-omat-0-medium.model \
  --model-type mace --velocity-policy preserve --no-pre-relax --output runs/md-part2
```

新的运行保持你给的位置和速度，不会重新抽取 Maxwell–Boltzmann 分布。它是一个独立的
运行目录，有自己的记录，它的第一帧等于上一段的最后一帧（验收套件检查的正是这一
点）。

## 什么会让运行停下

任何力超过 `--fmax-abort` 或出现 NaN 都会中止运行，而不是写出损坏的帧。写输出时
失败同样会中止，而不是悄悄丢帧。缺少模型或结构，或者选项组合不被支持，会在动力学
开始前就失败。

验收矩阵在四个后端的环境里各跑了 5 步的 NVE，以及 5 步的 Langevin、Bussi、NHC 冒
烟测试。这些运行检查的是工作流和恒温器的接线，不能说明温度已经平衡。

## 一个接近生产的例子

```bash
.venv/bin/mliport md structure.cif --model uma.pt --model-type uma --task omat \
  --ensemble NVT --temp 800 --timestep 1.0 \
  --equilibration-steps 5000 --steps 50000 --save-interval 10 \
  --thermostat NHC \
  --output runs/md-800K
```

它产生的文件在[输出文件](../outputs.md)里说明：`raw/` 下的标准轨迹和 `md.csv`，
`vasp/` 下的 VASP 风格文件，以及列出它们（附单位与哈希）的 `artifacts.json`。
