# 轨迹分析

`mliport analyze <运行目录或轨迹> <任务>` 读取一个 mliport 运行目录或 ASE 轨迹，把结果
写到 `<run>/analysis/<task>/<analysis-id>/` 下，格式为 JSON、CSV，必要时还有 PNG 和
SVG 图。绘图在无界面环境下完成，不需要显示器。

| 任务 | 它告诉你什么 | 页面 |
|---|---|---|
| `validate` | 轨迹能不能用、能用于哪些任务 | [概览](overview.md) |
| `thermo` | 温度、能量漂移和压强 | [概览](overview.md) |
| `rmsd` | 基于周期性位移的 RMSD 与 RMSF | [概览](overview.md) |
| `density` | 可移动离子的三维密度图 | [概览](overview.md) |
| `rdf` | 偏径向分布与配位数 | [RDF](rdf.md) |
| `msd` | 分方向的均方位移及其局部指数 | [MSD](msd.md) |
| `vacf` | 速度自相关 | [VACF 与谱](vacf-spectrum.md) |
| `spectrum` | 由 VACF 得到的振动谱 | [VACF 与谱](vacf-spectrum.md) |
| `transport` | 示踪扩散与 Nernst–Einstein 电导率（kinisi） | [输运](transport.md) |
| `electrolyte` | 位点、跃迁与渗流（GEMDAT） | [GEMDAT](electrolyte-gemdat.md) |
| `arrhenius` | 对多个温度下的扩散系数做 Arrhenius 拟合 | [Arrhenius](arrhenius.md) |

短轨迹是合法输入，但如果某个任务会因此被过度解读，它会明说，返回的是明确的"窗口不
足"或"采样不足"状态，而不是一个数。
