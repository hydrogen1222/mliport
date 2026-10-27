# VACF 与振动谱

`vacf` 计算速度自相关函数，`spectrum` 对它做傅里叶变换，得到振动态密度。两者都用于定性诊断动力学。

```bash
mliport analyze runs/md vacf --species Li
mliport analyze runs/md spectrum --species Li --taper one-sided-cosine \
  --normalization normalized_area
```

两者都要求轨迹里保存了速度，并且帧间隔均匀，程序不会用位置去反推速度。输出 CSV、PNG 和 SVG，不需要
显示器。谱的面积归一化在 NumPy 1.x 和 2.x 下都能用。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--species SPECIES` | 计算哪种元素（不写则用全部原子） |
| `--method fft\|direct` | 自相关的算法（默认 `fft`） |
| `--taper one-sided-cosine\|none` | 傅里叶变换前加的窗函数（`spectrum`；默认 `one-sided-cosine`） |
| `--normalization normalized_area\|raw_spectrum` | 谱的归一化方式（`spectrum`；默认 `normalized_area`） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--positions-convention`、`--frame-interval-fs` | 外部轨迹需要补充的信息 |

## 注意事项

决定奈奎斯特频率的是**保存帧的间隔**，不是积分步长。如果帧间隔太大、覆盖不了要求的频率范围，程序
会报告采样不足，避免出现混叠。

谱的形状取决于帧间隔、窗函数和轨迹长度，这三项都要写明。经典 MLIP 轨迹得到的谱不能代替定量的声子
计算，也不包含原子核的量子效应。只比较同一个模型、同样步长设置下的谱。短轨迹得到的谱噪声大，其中
的峰不能当作收敛的振动模式。

```bash
mliport analyze runs/md-300K spectrum --species O --method fft \
  --taper one-sided-cosine --normalization normalized_area
```

验收测试在 1 ns 的 GRACE 轨迹上用默认参数运行过：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns vacf
.venv-grace/bin/mliport analyze results/lgps-1ns spectrum
```
