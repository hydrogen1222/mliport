# VACF 与振动谱

`vacf` 计算速度自相关函数，`spectrum` 对它做傅里叶变换，得到振动谱密度。两者都用于诊断动力学行为。

```bash
mliport analyze runs/md vacf --species Li
mliport analyze runs/md spectrum --species Li --taper one-sided-cosine \
  --normalization normalized_area
```

两者都要求轨迹里保存了速度，并且帧间隔均匀；程序不会用位置去反推速度。输出 CSV、PNG 和 SVG，
不需要显示器。

## 物理背景

### 速度自相关函数（VACF）

VACF 度量一个原子"记住"自己速度方向的时间。在 *t* = 0 时 VACF 等于每原子平均动能（速度平方
单位）；随着碰撞使运动方向随机化，它逐渐衰减：

$$
C_v(t) = \left\langle \mathbf{v}_i(t_0) \cdot \mathbf{v}_i(t_0 + t) \right\rangle_{i, t_0}
$$

归一化后的 VACF *C*_v(*t*) / *C*_v(0) 从 1 开始衰减到 0 附近。衰减方式揭示动力学特征：

- **指数衰减** — 类液体的扩散碰撞。
- **振荡并出现负值** — 原子被局部笼子部分束缚并反弹（固态电解质和致密液体的特征）。
- **缓慢振荡衰减** — 振动模式占主导，体系行为更接近晶体。

### 从 VACF 到振动谱

VACF 的傅里叶余弦变换给出**速度功率谱**（经典极限下的振动态密度）：

$$
S(\nu) = \int_0^{\infty} C_v(t) \, \cos(2 \pi \nu t) \, dt
$$

mliport 实际使用实数 FFT（rFFT）对单侧、加窗、归一化的 VACF 进行变换。结果是谱密度，峰对应
体系的振动频率。

!!! warning "这不是声子态密度"
    经典 MD 轨迹得到的 VACF 谱**不是**定量的声子态密度。它缺少量子核效应（零点运动、
    玻色-爱因斯坦统计）、非谐修正，以及中子散射截面或红外活性的适当权重。它是经典动力学的
    诊断工具——适合比较模型、检查人为现象、识别特征频率，但不能直接和非弹性中子散射或
    红外/拉曼实验对比。

### 奈奎斯特频率

谱能分辨的最高频率是奈奎斯特频率，由**保存帧的间隔**（不是积分步长）决定：

$$
\nu_\text{Nyquist} = \frac{1}{2 \, \Delta t_\text{save}}
$$

如果保存间隔为 10 fs，奈奎斯特频率就是 50 THz ≈ 1668 cm⁻¹。高于此频率的信号会产生混叠。
要分辨 O–H 高频伸缩振动（~100 THz），需要每 5 fs 或更短保存一帧。mliport 在每条 VACF 结果中
同时报告 THz 和 cm⁻¹ 的奈奎斯特频率。

### 窗函数（taper）

VACF 的长度是有限的。在轨迹末端的截断会引入**谱泄漏**——边界处的不连续性在谱中产生人为振荡。
窗函数（taper）使截断更平滑。mliport 的默认选项是**单侧余弦窗**（从 lag 0 的 1 到最后一个 lag
的 0 的升余弦），它在几乎不影响峰位的情况下抑制泄漏。

`none`（矩形窗）可用于对比：峰更尖锐，但泄漏更严重。

### 恒温器的人为影响

恒温器会修改动力学。Langevin 恒温器引入的摩擦会使 VACF 衰减更快、谱峰展宽。Nosé–Hoover 链
恒温器对动力学的干扰较小，但如果耦合频率靠近物理模式的频率，可能引入低频振荡。NVE 轨迹给出
最干净的 VACF，但能量漂移可能限制可用长度。

比较谱时，务必使用**相同的恒温器和步长设置**。一个峰在 Langevin 下出现而在 NHC 下消失（或反之），
更可能是人为现象而非物理模式。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--species SPECIES` | 计算哪种元素（不写则用全部原子） |
| `--method fft\|direct` | 自相关的算法（默认 `fft`） |
| `--taper one-sided-cosine\|none` | 傅里叶变换前加的窗函数（`spectrum`；默认 `one-sided-cosine`） |
| `--normalization normalized_area\|raw_spectrum` | 谱的归一化方式（`spectrum`；默认 `normalized_area`） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--positions-convention`、`--frame-interval-fs` | 外部轨迹需要补充的信息 |

### VACF 方法

- **`fft`**（默认） — 通过 FFT 自相关实现，复杂度 O(*N* log *N*)。正常使用就选这个。
- **`direct`** — O(*N*²) 的参考实现。可用于在短轨迹上验证 FFT 结果；太慢，不适合生产使用。

两种方法都计算完整的时间原点平均 VACF，结果完全相同（浮点精度内）。

### 谱的归一化

- **`normalized_area`**（默认） — 谱在频率上的有符号积分等于 1。这使不同轨迹长度的谱
  在形状上可以比较。
- **`raw_spectrum`** — rFFT 实部乘以帧间隔。在需要绝对标度时使用。

## 输出文件

| 文件 | 内容 |
|---|---|
| `results.json` | VACF 和谱的数组、奈奎斯特频率、负值比例、溯源信息 |
| `vacf.csv` | 延迟时间（fs）和原始/归一化 VACF |
| `spectrum.csv` | 频率（THz、cm⁻¹）和谱密度 |
| `vacf.png`、`vacf.svg` | VACF 图 |
| `spectrum.png`、`spectrum.svg` | 谱图 |

## 结果解读

### 应报告的信息

1. **保存帧的间隔** — 它决定奈奎斯特极限和频率分辨率。10 fs 间隔给出 50 THz 奈奎斯特和
   约 1/(N×Δt) 的分辨率。
2. **窗函数** — `one-sided-cosine` 还是 `none`。
3. **轨迹长度** — 更长的轨迹给出更窄、更好分辨的峰。
4. **恒温器** — 它影响峰的形状和宽度。
5. **元素** — 按元素分别做谱可以区分骨架振动和可移动离子的动力学。

### 负的谱值

谱保留负值估计。`results.json` 中的 `negative_fraction` 很小是正常噪声；负值比例过大说明
VACF 没有充分收敛或窗函数选择不当。

### 比较谱

只在**相同模型、相同步长、相同恒温器、相同保存间隔**的条件下比较谱。同样条件下两个模型之间
的峰位移是有意义的；不同保存间隔之间的差异是人为现象。

```bash
mliport analyze runs/md-300K spectrum --species O --method fft \
  --taper one-sided-cosine --normalization normalized_area
```

验收测试在 1 ns 的 GRACE 轨迹上用默认参数运行过：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns vacf
.venv-grace/bin/mliport analyze results/lgps-1ns spectrum
```
