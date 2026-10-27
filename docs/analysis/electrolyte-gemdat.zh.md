# GEMDAT 电解质分析

`electrolyte` 用 GEMDAT 把可移动离子的位置归到各个位点上，统计位点之间的跳跃，并寻找渗流通道。它
回答的是*机理*问题：离子停在哪些位点、怎样跳跃。扩散快慢请用 `transport` 分析。

位点有两种给法，二选一：用你自己的位点文件，或者从可移动离子的密度分布中自动识别。

```bash
mliport analyze runs/md electrolyte --mobile Li --sites sites.yaml --temperature-K 800
mliport analyze runs/md electrolyte --mobile Li --discover-sites-from-density
```

位点定义必须和结构及原子顺序一致，不一致会报错。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动离子（必需） |
| `--sites PATH` 或 `--discover-sites-from-density` | 位点文件，或从密度自动识别（必须二选一） |
| `--resolution-A`、`--background-level` | 密度网格和位点识别的参数（默认 0.5 Å、0.1） |
| `--site-radius-A` | 位点的半径，离子进入这个范围算作占据 |
| `--minimal-residence` | 至少停留多少帧才算一次占据（默认 0） |
| `--jump-dimensions` | 跳跃分析的维度：1、2 或 3（默认 3） |
| `--percolation-axes` | 检查哪些方向上的渗流（默认 `xyz`） |
| `--temperature-K` | 计算衍生量用的温度 |
| `--drift-reference`、`--drift-mode`、`--drift-indices` | 骨架漂移校正（mode 默认 `arithmetic_mean`） |

## 输出

`results.json` 里有位点列表、占据和跳跃次数、跳跃方向和停留时间统计、渗流诊断，以及软件和模型信息。

GEMDAT 本身出错时，结果标为**后端失败**（`backend_error`），和"零次跳跃"区分开。只有分析正常完成且确实
没有跳跃时，才会报告零次跳跃。位点定义为空或无效、指定的可移动离子在结构里不存在，都会报错。

## 结果怎么看

- 划分位点本身是一种建模选择，位点定义不同，跳跃次数就不同，要写明位点是怎么来的。
- 出现 `backend_error` 时不能得出"没有扩散"的结论，先看报错信息。
- 跳跃次数受 `--minimal-residence`、漂移处理和帧间隔影响，这些都要一起报告。
- 沿某几个方向能渗流，不等于三维长程导通，要写明检查了哪些维度和方向。

```bash
mliport analyze runs/md-800K electrolyte --mobile Li \
  --discover-sites-from-density --resolution-A 0.2 --minimal-residence 2 \
  --jump-dimensions 3 --percolation-axes xyz
```

## 计算量

在 1 ns、400 个原子的完整轨迹上识别位点很耗时。验收测试只取其中 30 ps，并加上
`--frame-interval-fs 100`：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns electrolyte \
  --mobile Li --discover-sites-from-density
```

运行中关于部分占据的警告来自 GEMDAT 本身，不影响你指定的位点。
