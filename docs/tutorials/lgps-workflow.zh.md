# LGPS 固态电解质完整工作流

本教程演示 Li₁₀GeP₂S₁₂（LGPS）的完整 MLIP 模拟与分析流程——LGPS 是目前已知传导率最高的固态锂
离子导体之一。完成后你将得到：

- 优化后的 LGPS 结构
- 600 K NVT 1 ns 轨迹
- Li 扩散系数 *D*_Li 及估算的离子电导率

本教程使用 **MACE** 后端；将 `mliport-mace` 替换为 `mliport-dpa` 等即可切换引擎。

## 前提条件

已安装 mliport + MACE，并准备好 LGPS 的 CIF 文件。如需结构，可从 Materials Project（mp-696128）
获取 ICSD 实验几何构型（空间群 *P4₂/nmc*，a ≈ 8.72 Å，c ≈ 12.70 Å）。

```bash
./scripts/install_mliport.sh --engines mace --device auto
./bin/mliport-mace doctor --engine mace
```

## 第一步 — 几何优化

在 MD 之前先弛豫 LGPS 结构，消除实验构型中残留的力不平衡。

```bash
./bin/mliport-mace opt \
  --structure lgps.cif \
  --model-type mace \
  --model-file path/to/mace-checkpoint.pt \
  --fmax 0.01 \
  --max-steps 500 \
  --cell-opt \
  --output runs/lgps-opt
```

优化后的结构写入 `runs/lgps-opt/CONTCAR`。MD 运行使用此文件，而非原始 CIF。

## 第二步 — NVT 分子动力学

在 600 K 运行 1 ns（1 000 000 步，步长 1 fs）NVT-MD。设置 50 ps 平衡阶段，待系统达到热平衡后
再记录生产轨迹。

```bash
./bin/mliport-mace md \
  --structure runs/lgps-opt/CONTCAR \
  --model-type mace \
  --model-file path/to/mace-checkpoint.pt \
  --ensemble NVT \
  --temp 600 \
  --timestep 1.0 \
  --steps 1000000 \
  --equilibration-steps 50000 \
  --thermostat BUSSI \
  --bussi-tau 1000 \
  --output runs/lgps-nvt-600K
```

!!! note "恒温器选择"
    `BUSSI`（随机速度重标定）可提供正则采样，同时对动力学扰动最小，是输运计算的推荐选项。
    `LANGEVIN` 同样有效，但引入的摩擦力会轻微展宽 VACF 峰。

轨迹写入 `runs/lgps-nvt-600K/trajectory/`。

## 第三步 — 验证轨迹

分析前务必验证，以排除截断或损坏的轨迹。

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K validate
```

报告列出可以安全运行的分析任务。若 `fixed_cell`、`uniform_time_axis` 和 `task_msd` 均通过，
则可以继续。

## 第四步 — 热力学诊断

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K thermo
```

检查：
- 平均温度接近 600 K，方差合理（~400 原子体系约 10–20 K）
- 生产阶段无单调能量漂移

## 第五步 — 结构稳定性（RMSD）

确认 LGPS 框架在 MD 过程中未熔化或发生相变。

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K rmsd \
  --drift-reference nonmobile
```

稳定晶体的 RMSD 应趋于平台（< 1 Å）。若无界增长，说明温度过高或模型迁移性较差。

## 第六步 — 均方位移

从生产轨迹中提取 Li 扩散系数。

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K msd \
  --mobile Li \
  --axes xyz \
  --drift-reference nonmobile \
  --drift-mode mass_weighted_com
```

输出文件 `msd.csv` 包含 MSD(t)（单位 Å²）。扩散系数 *D* 由线性（扩散）区间的斜率得到；
`summary.json` 报告拟合值。

## 第七步 — 离子电导率（transport）

`transport` 任务使用 kinisi 对 MSD 进行贝叶斯拟合，并传播不确定度。

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K transport \
  --mobile Li \
  --frame-interval-fs 1.0 \
  --drift-reference nonmobile \
  --drift-mode mass_weighted_com \
  --temperature 600
```

`summary.json` 报告 *D*（m² s⁻¹）、95% 可信区间以及通过 Nernst–Einstein 关系估算的离子电导率
σ（S cm⁻¹）。

## 第八步 — Li–S 径向分布函数

验证局部配位环境的物理合理性。

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K rdf \
  --center Li \
  --neighbor S \
  --rmax 6.0 \
  --bins 200
```

约 2.5 Å 处的第一个峰对应 Li–S 最近邻距离。

## 第九步 — 三维密度图

可视化 Li 扩散路径。

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K density \
  --mobile Li \
  --spacing 0.2 \
  --smoothing-sigma-A 0.3 \
  --stride 5
```

输出 `density.npz` 包含占据概率和数密度网格。用 NumPy 读取：

```python
import numpy as np
data = np.load("runs/lgps-nvt-600K/analysis/density/<id>/density.npz")
prob = data["occupancy_probability"]
cell = data["cell_A"]
```

将网格后处理为 Cube 文件后，可在 VESTA 中可视化 LGPS 的一维通道。

## 汇总

| 步骤 | 命令 | 主要输出 |
|---|---|---|
| 优化 | `mliport opt` | `CONTCAR` |
| MD | `mliport md --ensemble NVT --temp 600` | `trajectory/` |
| 验证 | `mliport analyze ... validate` | 所有分析的前置检查 |
| 热力学 | `mliport analyze ... thermo` | 能量漂移、温度稳定性 |
| RMSD | `mliport analyze ... rmsd` | 结构稳定性 |
| MSD | `mliport analyze ... msd --mobile Li` | *D*_Li |
| 输运 | `mliport analyze ... transport --mobile Li` | *D* + σ（含置信区间） |
| RDF | `mliport analyze ... rdf --center Li --neighbor S` | Li–S g(r) |
| 密度图 | `mliport analyze ... density --mobile Li` | 三维占据概率 |

## 另请参阅

- [MD 工作流](../workflows/md.md)
- [MSD 分析](../analysis/msd.md)
- [输运分析](../analysis/transport.md)
- [文献基准测试教程](literature-benchmark.md)
