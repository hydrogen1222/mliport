# 轨迹分析

`mliport analyze <运行目录或轨迹> <任务>` 读取 mliport 的运行目录或 ASE 轨迹，结果写到
`<run>/analysis/<task>/<analysis-id>/` 下，包括 JSON、CSV，有图的任务还会输出 PNG 和 SVG。画图不需要
显示器。

| 任务 | 用途 | 页面 |
|---|---|---|
| `validate` | 检查轨迹是否可用，以及适合做哪些分析 | [概览](overview.md) |
| `thermo` | 温度、能量漂移和压强 | [概览](overview.md) |
| `rmsd` | 考虑周期性的 RMSD 和 RMSF | [概览](overview.md) |
| `density` | 可移动离子的三维密度图 | [概览](overview.md) |
| `rdf` | 分元素对的径向分布函数和配位数 | [RDF](rdf.md) |
| `msd` | 各方向的均方位移及其局部指数 | [MSD](msd.md) |
| `vacf` | 速度自相关 | [VACF 与谱](vacf-spectrum.md) |
| `spectrum` | 由 VACF 得到的振动谱 | [VACF 与谱](vacf-spectrum.md) |
| `transport` | 示踪扩散与 Nernst–Einstein 电导率（kinisi） | [输运](transport.md) |
| `electrolyte` | 位点、跳跃和渗流（GEMDAT） | [GEMDAT](electrolyte-gemdat.md) |
| `arrhenius` | 对多个温度下的扩散系数做 Arrhenius 拟合 | [Arrhenius](arrhenius.md) |

短轨迹也可以分析。如果轨迹太短、不足以支撑某项结果，程序会返回"窗口不足"或"采样不足"的状态，
不会硬给一个数。
