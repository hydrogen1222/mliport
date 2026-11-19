# 轨迹分析

`mliport analyze <运行目录或轨迹> <任务>` 读取 mliport 的运行目录或独立的 ASE 轨迹，
结果写到 `<run>/analysis/<task>/<analysis-id>/` 下。每项分析都输出 JSON 摘要；大多数还会输出 CSV 或 NPZ 数据文件。有图的任务还会输出 PNG 和 SVG。不需要图形显示器。

## 可用任务

| 任务 | 物理量 | 页面 |
|---|---|---|
| `validate` | 轨迹完整性：坐标约定、帧间隔、采样就绪状态 | [验证](validate.md) |
| `thermo` | 温度、能量漂移和压强随时间的变化 | [热力学](thermo.md) |
| `rmsd` | 均方根位移和均方根涨落（考虑周期性） | [RMSD](rmsd.md) |
| `density` | 可移动物种的三维数密度 | [密度](density.md) |
| `rdf` | 径向分布函数 g(r) 和配位数 | [RDF](rdf.md) |
| `msd` | 均方位移和局部指数 α(t) | [MSD](msd.md) |
| `vacf` | 速度自相关函数 | [VACF 与振动谱](vacf-spectrum.md) |
| `spectrum` | 由 VACF 得到的振动态密度 | [VACF 与振动谱](vacf-spectrum.md) |
| `transport` | 示踪扩散系数和 Nernst–Einstein 电导率（通过 kinisi） | [输运](transport.md) |
| `electrolyte` | 晶体位点映射、跳跃分析和渗流（通过 GEMDAT） | [GEMDAT](electrolyte-gemdat.md) |
| `arrhenius` | 扩散系数对温度的 Arrhenius 拟合 | [Arrhenius](arrhenius.md) |

## 我该用哪个分析？

选择哪项分析取决于你对 MD 轨迹提出的问题。

```
运行 MD 之后
│
├─ "模拟是否健康？"
│   └─ validate → thermo → rmsd
│
├─ "有没有离子输运？"
│   └─ msd → transport（α ≈ 1 时）
│
├─ "离子电导率是多少？"
│   └─ transport → arrhenius（多个温度）
│
├─ "离子怎么运动的？"
│   └─ electrolyte（位点映射、跳跃分析）
│       └─ density（可视化离子通道）
│
├─ "局域结构是什么样的？"
│   └─ rdf（对关联、配位数）
│
└─ "振动模式是什么？"
    └─ vacf → spectrum
```

典型的固态电解质工作流程是 `validate` → `thermo` → `rmsd` → `msd` → `transport`
→ `electrolyte`，在三个以上温度下取得扩散系数后再做 `arrhenius`。

## 采样注意事项

短轨迹也可以分析 — mliport 不会拒绝。但如果轨迹太短、不足以得到可靠结果，分析会
返回明确的**"采样不足"**或**"窗口不足"**状态，而不是硬给一个不可信的数值。

一般指导：

- **热力学量**（`thermo`、`rmsd`）：几百帧就有用，适合快速检查。
- **结构信息**（`rdf`、`density`）：需要足够帧数做统计平均，通常几千帧即可。
- **输运性质**（`msd`、`transport`、`electrolyte`）：需要长且充分平衡的轨迹。所
  需长度取决于扩散系数 — 扩散越慢，需要的轨迹越长。固态电解质在每个温度下至少
  规划 100 ps 的产出阶段。
- **Arrhenius**：需要 ≥ 3 个温度下收敛的扩散系数。每个温度对应一次独立的 MD 运行。

各任务通用的输入约定和输出格式见[概览](overview.md)。
