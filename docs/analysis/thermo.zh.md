# 热力学诊断

`thermo` 任务从轨迹中提取温度、能量、压力和体积的时间序列，并计算汇总统计量和分块平均。
对于 NVE 模拟，还会通过线性拟合报告总能量漂移。

```bash
mliport analyze runs/md thermo
mliport analyze runs/md thermo --block-size-frames 500
```

## 输出文件

| 文件 | 内容 |
|---|---|
| `thermo.csv` | 所有可用列的时间序列 |
| `thermo_*.png` | 各列的图表（温度、能量、压力、体积） |

### CSV 列

标注处为逐原子（per-atom）值：

| 列 | 单位 | 说明 |
|---|---|---|
| `time_ps` | ps | 相对于产出起始的时间 |
| `absolute_time_ps` | ps | 绝对模拟时间 |
| `step` | — | MD 步数（如可用） |
| `phase` | — | `equilibration` 或 `production` |
| `temperature_K` | K | 瞬时温度 |
| `potential_energy_eV_atom` | eV/atom | 逐原子势能 |
| `kinetic_energy_eV_atom` | eV/atom | 逐原子动能 |
| `total_energy_eV_atom` | eV/atom | 逐原子总能量 |
| `pressure_GPa` | GPa | 瞬时压力 |
| `volume_A3` | Å³ | 晶胞体积 |

轨迹中不可用的列会被省略。

## 选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--include-equilibration` | 关 | 包含平衡帧（默认仅产出帧） |
| `--start-frame` | — | 起始帧索引 |
| `--stop-frame` | — | 终止帧索引（不包含） |
| `--block-size-frames` | — | 分块统计的块大小（帧数） |

## NVE 能量漂移

当源模拟使用 NVE 系综时，`thermo` 对总能量（eV/atom）对时间做线性拟合，结果包括：

| 键 | 单位 | 说明 |
|---|---|---|
| `nve_total_energy_drift_eV_atom_ps` | eV/atom/ps | 线性拟合斜率 |
| `nve_total_energy_drift_meV_atom_ps` | meV/atom/ps | 同上，meV 单位 |
| `nve_fitted_total_drift_eV_atom` | eV/atom | 拟合线的端到端漂移 |
| `nve_relative_drift_over_production` | — | 漂移相对于初始能量的比值 |

## 数据完整性

统计量仅从有限值计算——NaN 和 ±Inf 被排除。为了透明，汇总包含 `data_integrity` 部分，
逐列报告：

- 总帧数和有限帧数
- 非有限值比例
- 哪些列包含非有限值

如果轨迹的 `raw/md.csv` 中有空白或无法解析的单元格，它们会变成 NaN。均值仍然看起来合理，
但 `data_integrity` 会揭示缺失的数据。

## 分块统计

使用 `--block-size-frames N` 时，每个热力学列被划分为 *N* 帧的不重叠块。块均值及其标准差
可以指示平衡质量：较大的标准差或块均值的系统性趋势表明体系尚未平衡。

块内的非有限值被排除；无有限帧的块不贡献均值。输出记录了多少块可用以及多少帧被跳过。

## 提示

- 在运行任何 transport 分析之前检查 `thermo` 输出——漂移的温度或压力会使扩散系数
  失效。
- 对于 NVE，能量漂移应远小于动能涨落。漂移超过 ~1 meV/atom/ps 通常表示积分问题。
- 使用 `--block-size-frames` 评估平衡。如果块均值显示系统性趋势，考虑延长平衡期。
