# GEMDAT 电解质分析

`electrolyte` 借助 GEMDAT 把可移动离子的位置归到位点上，统计位点之间的跳跃，并寻找渗流
通道。它关心的是*机理*：离子待在哪里、怎么跳，而不是整体扩散得多快。

GEMDAT 需要位点的定义，且只能二选一：你自己的位点文件，或者从可移动物种的密度中发现的
位点。

```bash
mliport analyze runs/md electrolyte --mobile Li --sites sites.yaml --temperature-K 800
mliport analyze runs/md electrolyte --mobile Li --discover-sites-from-density
```

位点定义必须与结构及其原子顺序一致；不一致是配置错误，不能忽略。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--mobile SPECIES` | 可移动物种（必需） |
| `--sites PATH` 或 `--discover-sites-from-density` | 显式位点文件，或基于密度的发现（二者必须恰好给一个） |
| `--resolution-A`、`--background-level` | 密度与位点发现的参数（默认 0.5 Å、0.1） |
| `--site-radius-A` | 位点周围的捕获半径 |
| `--minimal-residence` | 记作一次位点访问所需的最短停留帧数（默认 0） |
| `--jump-dimensions` | 跳跃分析的维度：1、2 或 3（默认 3） |
| `--percolation-axes` | 检验渗流的轴（默认 `xyz`） |
| `--temperature-K` | 派生量所用的温度 |
| `--drift-reference`、`--drift-mode`、`--drift-indices` | 骨架漂移校正（mode 默认 `arithmetic_mean`） |

## 输出什么

`results.json` 里有位点列表、访问与跳跃次数、跳跃向量与停留统计、渗流诊断，以及每个分析
都会带的来源信息。

如果 GEMDAT 本身出错，会报告为**后端失败**（`backend_error`），绝不会写成"零次跳跃"。
只有分析确实运行了、并且没发现跳跃时，才会报告零次跳跃。位点定义为空或无效、可移动物种
的选择没有匹配到任何原子，这些情形都会失败。

## 怎么读结果

- 位点映射是对输运机理的一种建模。不同的位点定义会改变跳跃次数，所以请写明位点的来源。
- `backend_error` 不等于"没有扩散"。解读任何机理之前，先读错误信息。
- 跳跃次数取决于 `--minimal-residence`、漂移策略和帧间隔，请一并报告。
- 沿所选轴的渗流不自动等于三维的长程输运。请写明检验了哪些维度和轴。

```bash
mliport analyze runs/md-800K electrolyte --mobile Li \
  --discover-sites-from-density --resolution-A 0.2 --minimal-residence 2 \
  --jump-dimensions 3 --percolation-axes xyz
```

## 计算开销

在完整的 1 ns、400 原子轨迹上发现位点很贵。验收套件只取 30 ps 的一段，并加上
`--frame-interval-fs 100`：

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns electrolyte \
  --mobile Li --discover-sites-from-density
```

GEMDAT 的部分占据警告来自上游库，不会改变你所要求的位点集合。
