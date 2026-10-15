# RDF

`rdf` 计算中心物种与近邻物种之间的径向分布函数，并给出配位数。

```bash
mliport analyze runs/md rdf --center Li --neighbor S --rmax 6.0 --bins 200
```

两个物种都必须给出。对 LGPS，有用的组合是 Li–Li、Li–S 和 P–S。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--center SPECIES`、`--neighbor SPECIES` | 物种对（都必需） |
| `--rmax` | 最大半径，单位 Å |
| `--bins` | 直方图格数（默认 200） |
| `--cn-cutoff` | 计算配位数所用的截断半径，单位 Å |
| `--stride` | 每隔 N 帧取一帧（默认 1） |
| `--start-frame`、`--stop-frame`、`--include-equilibration` | 分析窗口 |
| `--positions-convention`、`--frame-interval-fs` | 外来轨迹的元数据 |

`--start-frame`、`--stop-frame` 和 `--stride` 能在长轨迹上压低计算量：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns rdf \
  --center Li --neighbor S --start-frame 0 --stop-frame 2000 --stride 5
```

## 输出什么

偏 RDF、配位数，以及 `<run>/analysis/rdf/<analysis-id>/` 下的 CSV、PNG 和 SVG。物种不存
在或有效帧太少会失败；外来轨迹需要说明坐标约定。

## 怎么读

RDF 是结构上的平均，不是键的定义。请报告配位数用的截断半径和做平均的窗口。单个快照只是
诊断；收敛的 RDF 需要足够多的帧和均匀的晶胞。

```bash
mliport analyze runs/md-800K rdf --center Li --neighbor O \
  --rmax 8 --bins 240 --cn-cutoff 3.0 --stride 5
```
