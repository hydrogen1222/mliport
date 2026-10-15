# VACF 与振动谱

`vacf` 计算速度自相关函数，`spectrum` 计算它的傅里叶变换，也就是振动谱密度。请把它们当作
动力学的诊断。

```bash
mliport analyze runs/md vacf --species Li
mliport analyze runs/md spectrum --species Li --taper one-sided-cosine \
  --normalization normalized_area
```

两者只需要一条可读的轨迹。速度按与保存帧一致的方式导出，都可以无界面运行，并写出 CSV 以及
PNG 和 SVG。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--species SPECIES` | 要做相关的物种（省略时为全部原子） |
| `--method fft\|direct` | 相关算法（默认 `fft`） |
| `--taper one-sided-cosine\|none` | 变换前加的窗函数（`spectrum`；默认 `one-sided-cosine`） |
| `--normalization normalized_area\|raw_spectrum` | 谱的归一化方式（`spectrum`；默认 `normalized_area`） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--positions-convention`、`--frame-interval-fs` | 外来轨迹的元数据 |

## 注意什么

决定奈奎斯特频率的是**保存帧间隔**，而不是积分时间步。如果它对你要求的谱范围来说太粗，
mliport 会报告采样不足，而不是悄悄混叠。

谱取决于帧间隔、窗函数和轨迹长度，三者都请报告。经典 MLIP 轨迹给出的谱不是定量的声子计
算，也不含核量子效应。只在同一个模型和同一套时间步协议内比较谱。短轨迹会给出有噪声但有限
的谱；不要把其中的频率当作收敛的振动模式。

```bash
mliport analyze runs/md-300K spectrum --species O --method fft \
  --taper one-sided-cosine --normalization normalized_area
```

验收套件在 1 ns 的 GRACE 轨迹上运行最朴素的形式：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns vacf
.venv-grace/bin/mliport analyze results/lgps-1ns spectrum
```
