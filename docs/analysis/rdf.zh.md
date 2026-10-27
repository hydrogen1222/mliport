# RDF

`rdf` 计算两种元素之间的径向分布函数，并给出配位数。

```bash
mliport analyze runs/md rdf --center Li --neighbor S --rmax 6.0 --bins 200
```

中心元素和近邻元素都必须写。对 LGPS，常看的是 Li–Li、Li–S 和 P–S。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--center SPECIES`、`--neighbor SPECIES` | 中心元素和近邻元素（都必需） |
| `--rmax` | 最大半径，单位 Å |
| `--bins` | 直方图格数（默认 200） |
| `--cn-cutoff` | 计算配位数所用的截断半径，单位 Å |
| `--stride` | 每隔 N 帧取一帧（默认 1） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--positions-convention`、`--frame-interval-fs` | 外部轨迹需要补充的信息 |

长轨迹可以用 `--start-frame`、`--stop-frame` 和 `--stride` 减少计算量：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns rdf \
  --center Li --neighbor S --start-frame 0 --stop-frame 2000 --stride 5
```

## 输出

分元素对的 RDF、配位数，以及 `<run>/analysis/rdf/<analysis-id>/` 下的 CSV、PNG 和 SVG。元素不存在或
有效帧太少会报错；外部轨迹需要说明坐标约定。

## 结果怎么看

RDF 是结构的统计平均，不能直接当作成键判据。报告时要写明配位数的截断半径和取平均的帧范围。单个
快照只能作为粗略参考，收敛的 RDF 需要足够多的帧。

```bash
mliport analyze runs/md-800K rdf --center Li --neighbor O \
  --rmax 8 --bins 240 --cn-cutoff 3.0 --stride 5
```
