# 文献基准测试

本教程展示如何将 MLIP 计算的离子电导率与已发表的实验或 DFT 数据进行对比。工作流程为：

1. 在多个温度下运行 NVT-MD
2. 用 `transport` 提取扩散系数
3. 用 Arrhenius 拟合得到活化能 *E*_a
4. 将 *D*(T) 和 *E*_a 与已发表数据比较

## 为什么要做基准测试

MLIP 对训练集势能面的重现精度很高，但离子电导率对势垒高度的微小误差非常敏感。在正式研究之前，
先用标准材料（LGPS、LLZO、LATP 等）与已知实验值对比，能让你对力场的输运性质可信度有清晰判断。

## 第一步 — 规划温度点

在 1/T 坐标下等间距选取温度（而非在 T 上等间距），这样在 Arrhenius 图中间距近似均匀。
从 400 K 到 1000 K 的四点网格通常足够；发表级别的拟合建议五点或更多。

| 运行 | 温度 |
|---|---|
| `runs/md-400K` | 400 K |
| `runs/md-600K` | 600 K |
| `runs/md-800K` | 800 K |
| `runs/md-1000K` | 1000 K |

每次运行的时长要足以让 MSD 进入线性（扩散）区间。LGPS 在 400 K 通常需要 1–2 ns；在 1000 K
可能只需 100–200 ps。

## 第二步 — 在各温度下运行 MD

```bash
for T in 400 600 800 1000; do
  ./bin/mliport-mace md \
    --structure runs/lgps-opt/CONTCAR \
    --model-type mace \
    --model-file path/to/mace-checkpoint.pt \
    --ensemble NVT \
    --temp $T \
    --timestep 1.0 \
    --steps 1000000 \
    --equilibration-steps 50000 \
    --thermostat BUSSI \
    --output runs/md-${T}K
done
```

## 第三步 — 提取各温度扩散系数

对每个温度运行 `transport`，从 `summary.json` 中记录 *D*。

```bash
for T in 400 600 800 1000; do
  ./bin/mliport-mace analyze runs/md-${T}K transport \
    --mobile Li \
    --frame-interval-fs 1.0 \
    --drift-reference nonmobile \
    --drift-mode mass_weighted_com \
    --temperature $T
done
```

从各运行的 `summary.json` 收集扩散系数：

```bash
for T in 400 600 800 1000; do
  python -c "
import json, glob
path = sorted(glob.glob('runs/md-${T}K/analysis/transport/*/summary.json'))[-1]
d = json.load(open(path))
print(f'T={T}K  D={d[\"diffusivity_m2_s\"]:.3e}  sigma={d.get(\"conductivity_S_cm\",\"N/A\")}')
"
done
```

## 第四步 — Arrhenius 拟合

`arrhenius` 任务通过命令行参数接受扩散系数值。按相同顺序传入每对（温度，扩散系数）：

```bash
mliport analyze runs/md-1000K arrhenius \
  --temperature 400 --temperature 600 \
  --temperature 800 --temperature 1000 \
  --diffusivity 1.5e-14 --diffusivity 3.2e-13 \
  --diffusivity 2.8e-12 --diffusivity 1.1e-11 \
  --extrapolate-temperature 300
```

!!! note "参数语法"
    每个 `--temperature` 和 `--diffusivity` 参数重复传入一次，顺序一一对应。第 *i* 个温度与
    第 *i* 个扩散系数配对。命令中指定的运行目录仅决定输出位置，不读取轨迹数据。

输出：`summary.json` 包含 *E*_a（eV）、*D*₀（m² s⁻¹），以及若设置了 `--extrapolate-temperature 300`
则包含 *D*(300 K)。

## 第五步 — 与文献对比

LGPS 部分参考值：

| 物理量 | 实验值 | MLIP 基准目标 |
|---|---|---|
| σ（300 K） | ~12 mS cm⁻¹（Kamaya 等，2011） | 50% 以内 |
| *E*_a | 0.21–0.24 eV | ± 0.05 eV |
| *D*_Li（600 K） | ~10⁻¹² m² s⁻¹（模拟值） | — |

若你的 *E*_a 在 0.20–0.26 eV 之间，外推得到的室温电导率在 5–20 mS cm⁻¹ 范围内，说明
MACE 模型对 LGPS 输运行为的重现效果良好。

若偏差显著（*D* 相差超过 10 倍，*E*_a 偏差 > 0.1 eV），说明 MLIP 在 LGPS 输运方面
校准不足，结果需谨慎对待。

## 汇报前的核查清单

- [ ] 至少 3 个温度，温度跨度 ≥ 2 倍
- [ ] 每个 MSD 在至少一个时间量级上明显呈线性
- [ ] `thermo` 检查无能量漂移、温度稳定
- [ ] `rmsd` 检查确认任何温度下晶体均未熔化
- [ ] *E*_a 报告时附上 kinisi 拟合的统计不确定度
- [ ] 注明所用恒温器类型（结果依赖恒温器）
- [ ] 与至少一个独立基准进行比较

## 另请参阅

- [LGPS 工作流教程](lgps-workflow.md) — 单温度工作流
- [输运分析](../analysis/transport.md)
- [Arrhenius 分析](../analysis/arrhenius.md)
- [验证](../validation.md)
